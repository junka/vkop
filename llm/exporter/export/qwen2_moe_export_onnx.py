"""Qwen1.5-MoE (Qwen2MoeForCausalLM) → ONNX，为「专家权重按需装载」改造。

== 为什么需要这个文件 ==

qwen2_export_onnx.py 导出的稠密 Qwen2 里，MLP 是一个普通 nn.Module，权重是几个
平铺的 initializer。MoE 不是：HF 把**全部 60 个专家**存成两个融合参数

    gate_up_proj  (60, 2816, 2048)   前 1408 行是 gate，后 1408 行是 up
    down_proj     (60, 2048, 1408)

于是任何「按专家换字节」的方案都**没有可寻址的槽位**：60 个专家挤在两个
[60,...] 的参数里，切不出「第 7 个专家的 gate_proj」。

== 两个阶段（必须分开跑）==

    # 阶段 1：trace 模型，产出 llm_moe.onnx —— 图 + 槽位契约的底座
    python3 export/qwen2_moe_export_onnx.py

    # 阶段 2：纯离线 protobuf 改写。拆槽位、接线、声明 —— 不读张量字节
    python3 make_moe_variant.py -i llm_moe.onnx -o llm_moe_paged.onnx

== 内存/磁盘预算（这台机器实测出来的约束，不是猜的）==

模型 fp16/bf16 都是 26.67 GiB，宿主 MemAvailable ~19 GB 且 swap 已被别的进程
吃满 —— `from_pretrained` 整模型驻留必然 OOM（第一版两次被内核杀在 47%/59%，
一次 16.6 GB、一次 19.9 GB anon-rss）。同时 `/` 只剩 35 GB，checkpoint 已占
27 GB：fp16 专家池再落一份盘（23.2 GB）+ 最终 vkopbin blob（26.7 GB）根本放不下。

所以本导出器：

  1. **模型结构建在 meta 设备上**（0.79 GB / 0.03 s，387 个张量全是占位），
     然后用 `safe_open` **逐分片、逐张量**填充真正需要的参数：非专家权重
     （339 个键，~3.5 GB）+ 每层接线的 top_k 个专家（96 × 17.3 MB ≈ 1.66 GB）。
     峰值 RSS ~7-8 GB。填充期间的 bf16→fp16 转换与 from_pretrained(dtype=fp16)
     完全同型（同一份 bf16 位型 → 同一个 fp16 舍入）。
  2. **导出后把所有「参数出身的」initializer 改写成指向 checkpoint 分片**：
     location=分片绝对路径、offset=safetensors 头算出的绝对偏移、dtype=BFLOAT16、
     raw_data 清空。张量字节从此不经过 torch —— 转换期由 onnx2vkop 的
     weight_stream 直接从分片流式读出并转 fp16（bf16 pass-through，见
     weight_stream.py / converter.py 的 `_PASS_THROUGH_DATA_TYPES`）。
  3. 唯一在产物里保留内联字节的，是常量折叠出的 288 个接线专家权重
     （24 层 × 4 槽 × 3 矩阵，~1.66 GB fp16）：它们让 llm_moe.onnx 今天就能
     作为「4 专家稠密模型」转换、跑通 llm_chat（计划第 1 步的验证）。
     阶段 2 会把这些折叠权重删掉，换成指向分片的 60 槽位 initializer。

因此**没有 llm_moe.weights.bin**；llm.weights.bin 等一切 canonical 文件不被触碰。

== I/O 契约与稠密版逐字一致 ==

    in : inputs_embeds (B,q,2048) fp16 | position_ids (B,q) int64
         attention_bias (B,1,q,kv) fp16 | past_key_values_{i} (B,2,16,kv,128) fp16
    out: logits (B,q,151936) fp16      | present_key_values_{i} (B,2,16,kv,128) fp16

llm_chat.cpp 不用改。

== 与稠密导出器的四个实质差异（每一个都能静默出错）==

1. **专家段不能用 HF 的 forward**。Qwen2MoeExperts.forward 里有
   `nonzero()` / `torch.where()` / `index_add_()` 三条数据相关的控制流与散射，
   导出会退化成循环展开或直接失败。所以稠密导出器的 `h = layer.mlp(h)` 一行换成
   `SparseMoeOnnx`（整体替换 mlp 容器）：router 用原模块（本来就是普通 Linear），
   专家逐槽位展开成固定 4 条分支。**必须是换容器而不是补丁 forward** —— 实例级
   `mlp.forward = <函数>` 会让 torch 2.10 的 tracer 在 alias_analysis 里内部
   崩溃（见 SparseMoeOnnx 的 docstring，实测结论）。

2. **槽位求和顺序与 HF 不同**。HF 是「按专家 id 升序依次 index_add_」，本导出器是
   「按 top-k 槽位顺序依次相加」。两者数学上等价（加法可交换，选中集合与每个
   (token,专家) 的系数都相同），但**中间张量顺序不同**，所以逐位不保证一致。
   这是精度考虑，不是正确性问题。

3. **silu 的算子顺序**。HF 写 `down(act(gate(x)) * up(x))`，展开是 Silu→Mul。我把
   mul 的第一个操作数放 silu 结果，与 HF 一致。

4. **attention 逐字复用**稠密导出器的写法（eager、手写 reshape/transpose，不靠
   HF 的 attn_implementation）—— 注意力部分**没有**可复用的离线原型，而这条写法
   已有 ORT 对齐的先例。

== 常驻集不写进产物 ==

实测的 Qwen1.5-MoE 真实路由（4 个 prompt）：静态热集是**死的**（top-4/8/12/16 命中
0.0%，最热专家只占 4.0%），唯一可用的信号是 lag-1 时域局部性（全集合命中
64.8–72.0%）。所以「主图含热门专家的副本」里的"热门"**必须**解成运行期的 lag-1
工作集，而不是从文件里读一张频次表。本文件产出的图里接线的 4 个槽位只是一个
**让文件今天能作为稠密模型正确加载**的默认接线（槽位号 0..3 ↔ 专家 0..3）；真正
接哪些由 make_moe_variant.py --resident 决定。

用法:
    MODEL_PATH=/home/gwm/.cache/modelscope/hub/models/Qwen/Qwen1___5-MoE-A2___7B \
        python3 export/qwen2_moe_export_onnx.py
    EXPORT_PATH=/tmp/llm_moe.onnx python3 export/qwen2_moe_export_onnx.py
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_PATH = os.path.join(
    os.path.expanduser("~"),
    ".cache/modelscope/hub/models/Qwen/Qwen1___5-MoE-A2___7B")
MODEL_PATH = os.environ.get("MODEL_PATH") or DEFAULT_PATH
print(f"[load] model = {MODEL_PATH}")

from moe_ckpt import is_expert_key, non_expert_keys, offset_table  # noqa: E402

import transformers  # noqa: E402
from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM  # noqa: E402
from transformers.models.qwen2_moe.modeling_qwen2_moe import (  # noqa: E402
    Qwen2MoeRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)

OPSET = 17

# 这两个属性名是阶段 2（make_moe_variant.py）与转换器（onnx2vkop）之间的契约：
# 挂在**声明槽位的 graph input** 上（ONNX 的 initializer 不能带属性）。阶段 1 的
# 产物里还没有槽位（槽位在阶段 2 诞生），这里只为三方契约留名。
MOE_GROUP_ATTR = "vkop_moe_group"       # "3.gate_proj" —— 槽位标识
MOE_EXPERT_ATTR = "vkop_moe_expert_id"  # "42" —— 该 entry 属于哪一号专家

# checkpoint 在 modelscope 缓存里，全部张量 BF16（见 moe_ckpt.py 的说明）。
# 导出器自己读的字节（非专家参数 + 接线专家）在填充时转 fp16；其余 1344 个专家
# 张量的字节**从不进本进程**，声明成 bf16 external 引用，转换期流式转 fp16。
_ckpt_table = offset_table(MODEL_PATH)
_non_expert_keys = sorted(non_expert_keys(_ckpt_table))

cfg = Qwen2MoeConfig.from_pretrained(MODEL_PATH, local_files_only=True)

# **结构建在 meta 设备上**。_from_config 不读任何 safetensors 字节；387 个张量
# 以 meta 占位存在（0 字节），后续要么被真张量替换（非专家）、要么永远不被
# forward 触碰（融合专家参数 —— 槽位字节走 _tensors 的 side cache）。
with torch.device("meta"):
    model = Qwen2MoeForCausalLM._from_config(cfg, attn_implementation="eager")
model.eval()

config = model.config

NLAYERS = config.num_hidden_layers
HIDDEN = config.hidden_size
NUM_HEADS = config.num_attention_heads
NUM_KV_HEADS = config.num_key_value_heads
HEAD_DIM = getattr(config, "head_dim", None) or HIDDEN // NUM_HEADS
NUM_KV_GROUPS = NUM_HEADS // NUM_KV_HEADS
VOCAB = config.vocab_size
SCALING = model.model.layers[0].self_attn.scaling

NUM_EXPERTS = config.num_experts
TOP_K = config.num_experts_per_tok
MOE_INTER = config.moe_intermediate_size
SHARED_INTER = config.shared_expert_intermediate_size
NORM_TOPK = bool(config.norm_topk_prob)

print(f"[config] layers={NLAYERS} hidden={HIDDEN} heads={NUM_HEADS}/"
      f"{NUM_KV_HEADS} head_dim={HEAD_DIM} scaling={SCALING:.6f} vocab={VOCAB}")
print(f"[config] experts={NUM_EXPERTS} top_k={TOP_K} moe_inter={MOE_INTER} "
      f"shared_inter={SHARED_INTER} norm_topk_prob={NORM_TOPK} "
      f"decoder_sparse_step={config.decoder_sparse_step}")
print(f"[config] tie_word_embeddings={config.tie_word_embeddings} "
      f"use_sliding_window={config.use_sliding_window} "
      f"max_window_layers={config.max_window_layers}")


def _check_exportable():
    """导出前的硬检查。每一条都对应一个会**静默出错**的假设。"""
    qm = getattr(config, "quantization_config", None)
    if qm is not None:
        raise SystemExit(
            f"[fatal] checkpoint 带量化配置 {qm}: 专家权重不是 fp16 nn.Parameter，"
            f"按需分页槽位会写进错误字节。请先反量化到 fp16。")

    if config.decoder_sparse_step not in (None, 1) or config.mlp_only_layers:
        # decoder_sparse_step==1 且 mlp_only_layers 为空时才「每层都是稀疏 MoE」。
        raise SystemExit(
            f"[fatal] decoder_sparse_step={config.decoder_sparse_step}, "
            f"mlp_only_layers={config.mlp_only_layers}: 不是每层都是稀疏 MoE 块。"
            f"本导出器假定所有层同构。")

    if config.use_sliding_window:
        raise SystemExit(
            "[fatal] use_sliding_window=true: 各层窗口策略不同，attention_bias 的"
            "形状不再统一。本导出器只支持全注意力（Qwen1.5-MoE 的 config 里就是 "
            "false，其 sliding_window 不生效）。")

    if config.tie_word_embeddings:
        raise SystemExit(
            "[fatal] tie_word_embeddings=true: lm_head 与 embed_tokens 共享一份张量，"
            "I/O 语义与假定不同。Qwen1.5-MoE 是 false；换了模型请先确认。")

    # 融合专家参数的**形状**契约（dtype 是 meta 的，无从检查；真正的 dtype 断言
    # 在 _tensors 读字节处）。槽位顺序的完整证据链见 _tensors 的 docstring。
    e0 = model.model.layers[0].mlp.experts
    want_gu = (NUM_EXPERTS, 2 * MOE_INTER, HIDDEN)
    want_dp = (NUM_EXPERTS, HIDDEN, MOE_INTER)
    if tuple(e0.gate_up_proj.shape) != want_gu or \
       tuple(e0.down_proj.shape) != want_dp:
        raise SystemExit(
            f"[fatal] 融合专家参数形状 {tuple(e0.gate_up_proj.shape)}/"
            f"{tuple(e0.down_proj.shape)}，期望 {want_gu}/{want_dp}。"
            f"专家槽位顺序会静默错位，拒绝导出。")

    bad = [k for k in _non_expert_keys if _ckpt_table[k]["dtype"] != "BF16"]
    if bad:
        raise SystemExit(
            f"[fatal] checkpoint 里 {len(bad)} 个非专家张量不是 BF16（如 {bad[:3]}）。"
            f"填充转换与槽位字节声明都按 BF16 假设写，换 dtype 请显式核对本文件。")

    print(f"[experts] 融合参数 gate_up_proj{want_gu} down_proj{want_dp} "
          f"(meta 占位，由阶段 2 拆槽位)；checkpoint {len(_ckpt_table)} 键，"
          f"非专家 {len(_non_expert_keys)}")


_check_exportable()

lm = model.model
layers = lm.layers
norm = lm.norm
rotary_emb = lm.rotary_emb
lm_head = model.lm_head


# ---------------------------------------------------------------------------
# 结构填充：非专家参数逐张量从 checkpoint 换成真 fp16。meta 模型的参数是占位，
# `copy_` 不可用（meta 无存储），所以直接**替换参数对象**——nn.Module 的参数就是
# 一个属性，setattr 换掉即可，模块树/名字/前缀全部不动。
# ---------------------------------------------------------------------------

def _resolve_param(key):
    """'model.layers.3.self_attn.q_proj.weight' -> (module, 'weight')。"""
    if key.startswith("model."):
        obj, attr = model.model, key[len("model."):]
    else:
        obj, attr = model, key  # lm_head.weight
    path, _, pname = attr.rpartition(".")
    for p in path.split("."):
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    return obj, pname


def fill_non_expert():
    """逐分片 safe_open 填充。峰值内存 = 一个张量（最大 embed 604 MB）的 bf16+fp16。"""
    import safetensors
    from collections import defaultdict

    by_shard = {}
    for key in _non_expert_keys:
        by_shard.setdefault(os.path.basename(_ckpt_table[key]["location"]), []) \
            .append(key)

    filled = set()
    for shard, keys in sorted(by_shard.items()):
        path = os.path.join(MODEL_PATH, shard)
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for key in keys:
                mod, pname = _resolve_param(key)
                real = f.get_tensor(key).to(torch.float16)
                setattr(mod, pname, torch.nn.Parameter(real, requires_grad=False))
                filled.add(key)
        print(f"[fill] {shard}: {len(keys)} tensor(s) -> fp16")

    if filled != set(_non_expert_keys):
        missing = sorted(set(_non_expert_keys) - filled)[:5]
        raise SystemExit(f"[fatal] 填充覆盖不完整，缺 {missing}（少一层都会在这里"
                         f"报出来，而不是留着一层随机权重静默跑）")
    # 转换确实发生了吗？—— 采样断言，而不是相信。
    for probe in ("model.embed_tokens.weight", "lm_head.weight",
                  "model.layers.0.self_attn.q_proj.weight"):
        mod, pname = _resolve_param(probe)
        if getattr(mod, pname).dtype != torch.float16:
            raise SystemExit(f"[fatal] {probe} 未转成 fp16 —— 填充映射有错")

    # rotary 的 inv_freq 是非持久 buffer，checkpoint 里没有、meta 上是占位。
    # 用同 config 新建的 CPU 实例取真值，保证与 from_pretrained 的模型逐位一致。
    lm.rotary_emb.inv_freq = Qwen2MoeRotaryEmbedding(config=config).inv_freq
    if lm.rotary_emb.inv_freq.is_meta:
        raise SystemExit("[fatal] inv_freq 仍是 meta —— forward 会把整条链变回 meta")


fill_non_expert()


class Qwen2MoeLLMOnnx(nn.Module):
    """decoder-only wrapper，attention 与稠密 Qwen2 逐字相同，mlp 换成稀疏 MoE。"""

    def __init__(self):
        super().__init__()
        self.embed_tokens = lm.embed_tokens
        self.layers = layers
        self.norm = norm
        self.rotary_emb = rotary_emb
        self.lm_head = lm_head
        self.num_heads = NUM_HEADS
        self.num_kv_heads = NUM_KV_HEADS
        self.num_kv_groups = NUM_KV_GROUPS
        self.head_dim = HEAD_DIM
        self.hidden = HIDDEN
        self.scaling = SCALING

    def decoder_layer(self, layer, hidden, cos, sin, attention_bias, past_kv):
        B, q_len, _ = hidden.shape
        residual = hidden
        h = layer.input_layernorm(hidden)

        attn = layer.self_attn
        q = attn.q_proj(h).view(B, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        cs = (B, q_len, self.num_kv_heads, self.head_dim)
        k = attn.k_proj(h).view(cs).transpose(1, 2)
        v = attn.v_proj(h).view(cs).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # past_kv: (B, 2, NKV, past_len, HD) -> combined: (B, 2, NKV, past+q, HD)
        k_new = torch.cat([past_kv[:, 0], k], dim=2)
        v_new = torch.cat([past_kv[:, 1], v], dim=2)

        k_r = repeat_kv(k_new, self.num_kv_groups)
        v_r = repeat_kv(v_new, self.num_kv_groups)

        attn_w = torch.matmul(q, k_r.transpose(2, 3)) * self.scaling
        attn_w = attn_w + attention_bias
        attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn_w, v_r)
        out = out.transpose(1, 2).reshape(B, q_len, self.num_heads * self.head_dim)
        hidden = residual + attn.o_proj(out)

        residual = hidden
        h = layer.post_attention_layernorm(hidden)
        hidden = residual + layer.mlp(h)

        return hidden, torch.stack([k_new, v_new], dim=1)

    def forward(self, inputs_embeds, position_ids, attention_bias, *past_kvs):
        hidden = inputs_embeds
        cos, sin = self.rotary_emb(hidden, position_ids)

        presents = []
        for idx, layer in enumerate(self.layers):
            hidden, pk = self.decoder_layer(
                layer, hidden, cos, sin, attention_bias, past_kvs[idx])
            presents.append(pk)

        hidden = self.norm(hidden)
        return (self.lm_head(hidden), *presents)


# ---------------------------------------------------------------------------
# 稀疏 MoE 一层。
#
# HF 的 Qwen2MoeSparseMoeBlock.forward 只保留**非专家**部分（router / shared
# expert / shared_expert_gate），专家那一段整体换成我们展开的固定 4 条分支。
# ---------------------------------------------------------------------------

from safetensors import safe_open  # noqa: E402

_shard_handles = {}


def _expert_key(layer_idx, expert_id, proj):
    return f"model.layers.{layer_idx}.mlp.experts.{expert_id}.{proj}.weight"


def _shard_open(shard):
    h = _shard_handles.get(shard)
    if h is None:
        h = safe_open(os.path.join(MODEL_PATH, shard), framework="pt", device="cpu")
        _shard_handles[shard] = h
    return h


def _load_expert(layer_idx, expert_id):
    """(gate, up, down)，各 fp16 contiguous。gate/up 来自 checkpoint 的独立张量
    （[1408,2048]，out×in 布局）—— checkpoint **没有**融合键；融合只发生在 HF
    加载器的 WeightConverter（MergeModulelist(dim=0) + Concatenate(dim=1)）里，
    那条路把 state_dict 按 dot_natural_key 数值序排好再 append，所以融合参数的
    第 0 轴 = 专家 id 升序。这里逐专家读独立张量，顺序天然就是 id 序。"""
    parts = []
    for proj in ("gate_proj", "up_proj"):
        key = _expert_key(layer_idx, expert_id, proj)
        e = _ckpt_table.get(key)
        if e is None:
            raise SystemExit(f"[fatal] checkpoint 缺键 {key}")
        t = _shard_open(os.path.basename(e["location"])).get_tensor(key)
        if list(t.shape) != [MOE_INTER, HIDDEN]:
            raise SystemExit(f"[fatal] {key} 形状 {list(t.shape)}，期望 "
                             f"[{MOE_INTER},{HIDDEN}]")
        parts.append(t.to(torch.float16))
    # gate/up 各自独立且 contiguous，融合布局（前 gate 后 up）只存在于 HF 的
    # 内存参数里 —— 对逐专家读取毫无必要，这里保持两块分离。
    dkey = _expert_key(layer_idx, expert_id, "down_proj")
    d = _ckpt_table.get(dkey)
    if d is None:
        raise SystemExit(f"[fatal] checkpoint 缺键 {dkey}")
    dtp = _shard_open(os.path.basename(d["location"])).get_tensor(dkey)
    if list(dtp.shape) != [HIDDEN, MOE_INTER]:
        raise SystemExit(f"[fatal] {dkey} 形状 {list(dtp.shape)}，期望 "
                         f"[{HIDDEN},{MOE_INTER}]")
    parts.append(dtp.to(torch.float16))
    return tuple(p.contiguous() for p in parts)


def _gate_weights(topv):
    """槽位加权系数。

    norm_topk_prob=false ⇒ 系数就是 softmax 后的 topv 本身。若换模型要开
    norm_topk_prob，除了这里改成 `topv / (topv.sum(-1, keepdim=True) + 1e-6)`，
    **还必须同时改运行期** —— 这是整层唯一一处「gate 的绝对尺度影响输出」的地方。
    """
    if NORM_TOPK:
        return topv / (topv.sum(dim=-1, keepdim=True) + 1e-6)
    return topv


class SparseMoeOnnx(nn.Module):
    """一层稀疏 MoE 的导出形态，整体**替换** Qwen2MoeSparseMoeBlock。

    为什么是换模块而不是改 mlp.forward：实例级 forward 补丁（原方案）会让
    torch 2.10 的 TorchScript tracer 在 alias_analysis.cpp:623 内部断言崩溃
    （aten::view 参数类型不合法）—— 同样的数学走原始类 forward（router /
    shared expert / 本类）都能 trace，唯独走过补丁 forward 的图不行。换容器
    唯一的代价是 state_dict 前缀：本文件不按 state_dict 装载（参数全部
    setattr 填充），wired 专家本来就是新参数，无迁移问题。

    HF 的 router（Qwen2MoeTopKRouter.forward）：
        router_logits = F.linear(h, weight)
        probs = softmax(logits, dtype=float32)
        topv, topi = topk(probs, top_k)
        if norm_topk_prob: topv /= topv.sum(-1, keepdim=True)
    Qwen1.5-MoE 是 norm_topk_prob=false，所以**不**重新归一化 —— 这一点在导出时
    定死（图上没有这个分支），事后改 config 不会影响已有产物。
    """

    def __init__(self, old, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        # router / shared expert / shared_expert_gate 原样接管（它们的参数已在
        # fill_non_expert 填成真 fp16，模块对象原封不动，参数身份不变）。
        self.gate = old.gate
        self.shared_expert = old.shared_expert
        self.shared_expert_gate = old.shared_expert_gate
        # 接线的 top_k 个专家成为**本模块自己的参数**。旧 fused-experts 子模块
        # （meta 占位，26.7 GiB）随 old 一起被丢弃，从未进入 trace。
        for slot in range(TOP_K):
            g, u, d = _load_expert(layer_idx, slot)
            setattr(self, f"wired_gate_{slot}",
                    nn.Parameter(g, requires_grad=False))
            setattr(self, f"wired_up_{slot}",
                    nn.Parameter(u, requires_grad=False))
            setattr(self, f"wired_down_{slot}",
                    nn.Parameter(d, requires_grad=False))

    def forward(self, x):
        B, S, _ = x.shape
        flat = x.reshape(-1, HIDDEN)                      # (B*S, H)

        if self.gate.num_experts != NUM_EXPERTS or self.gate.top_k != TOP_K:
            raise SystemExit(f"[fatal] layer {self.layer_idx} 的 router 是 "
                             f"{self.gate.num_experts}/{self.gate.top_k}，"
                             f"与 config 的 {NUM_EXPERTS}/{TOP_K} 不符")

        # router 模块没被替换：它的 forward 就是 HF 的三行，且都是数据无关的
        # 常规算子。软化的 topv 是 fp32（softmax(dtype=float32)），加权时按 HF
        # 的 `.to(dtype)` 转回 fp16 —— 显式 to(fp16)，免得图上留下一个 fp32 乘法
        # 把后续张量全抬成 fp32（那会改掉 MatMul 的 dispatch）。
        _logits, topv, _topi = self.gate(flat)            # 已经 softmax + topk 过
        gate_w = _gate_weights(topv).to(flat.dtype)       # (B*S, K) fp16

        out = None
        for slot in range(TOP_K):
            # 导出阶段每个槽位接**槽位号本身**（0..K-1）：产出的图自洽（4 条分
            # 支各读一个真实专家），阶段 2 只换 initializer 名，结构不动。
            g = getattr(self, f"wired_gate_{slot}")
            u = getattr(self, f"wired_up_{slot}")
            d = getattr(self, f"wired_down_{slot}")
            w = gate_w[:, slot:slot + 1]                  # (B*S, 1)
            h = F.silu(F.linear(flat, g)) * F.linear(flat, u)
            y = F.linear(h, d)                            # (B*S, H)
            out = y * w if out is None else out + y * w

        se = self.shared_expert
        sh = se.down_proj(F.silu(se.gate_proj(flat)) * se.up_proj(flat))
        sg = torch.sigmoid(self.shared_expert_gate(flat))  # (B*S, 1)

        return (out + sh * sg).reshape(B, S, HIDDEN)


# 整体替换每层 mlp。旧 Qwen2MoeSparseMoeBlock（含 26.7 GiB 的 meta 融合占位）
# 在这里失去最后的引用，被 gc 掉。
for _idx, _layer in enumerate(layers):
    _layer.mlp = SparseMoeOnnx(_layer.mlp, _idx)

wrapper = Qwen2MoeLLMOnnx()
wrapper.eval()

B = 1
q_len = 32
kv_len = q_len

inputs_embeds = torch.randn(B, q_len, HIDDEN, dtype=torch.float16)
position_ids = torch.arange(q_len, dtype=torch.long).unsqueeze(0).expand(B, -1)
attention_bias = torch.triu(
    torch.full((q_len, kv_len), torch.finfo(torch.float16).min, dtype=torch.float16),
    diagonal=1).unsqueeze(0).unsqueeze(0)

past_kvs = tuple(
    torch.zeros(B, 2, NUM_KV_HEADS, 0, HEAD_DIM, dtype=torch.float16)
    for _ in range(NLAYERS))
inputs = (inputs_embeds, position_ids, attention_bias, *past_kvs)

input_names = ["inputs_embeds", "position_ids", "attention_bias"]
input_names += [f"past_key_values_{i}" for i in range(NLAYERS)]
output_names = ["logits"] + [f"present_key_values_{i}" for i in range(NLAYERS)]

dynamic_axes = {
    "inputs_embeds": {0: "batch", 1: "seq"},
    "position_ids": {0: "batch", 1: "seq"},
    "attention_bias": {0: "batch", 2: "q_len", 3: "kv_len"},
    "logits": {0: "batch", 1: "seq"},
}
for i in range(NLAYERS):
    dynamic_axes[f"past_key_values_{i}"] = {0: "batch", 3: "kv_len"}
    dynamic_axes[f"present_key_values_{i}"] = {0: "batch", 3: "kv_len"}

EXPORT_PATH = os.environ.get("EXPORT_PATH", "llm_moe.onnx")

# torch.onnx.export 会把 >2 GiB 的权重写成散文件；这份目录快照在 export **之前**
# 拍，清理时按差集删 —— 也保证 llm.weights.bin 等 pre-existing 文件永远不会被
# 本文件的清理逻辑碰到。
_pre = set(os.listdir("."))

print(f"[export] {NLAYERS} layers × {TOP_K}/{NUM_EXPERTS} experts, "
      f"{len(input_names)} in / {len(output_names)} out, opset={OPSET} "
      f"-> {EXPORT_PATH}")

with torch.no_grad():
    torch.onnx.export(
        wrapper, inputs, EXPORT_PATH,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
print(f"[ok] traced -> {EXPORT_PATH}")

# trace 结束，释放参数侧的真张量：图 proto 里已经拿走了它需要的字节（内联或
# external 文件）。模块级名字 lm/layers/... 还抓着模型本体不放，一并删掉；
# 否则 ~5.2 GB 的 fp16 参数（含 26.7 GiB 的 meta 占位引用）会一直活到进程退出。
del wrapper, model, lm, layers, norm, rotary_emb, lm_head
del _shard_handles
import gc  # noqa: E402

gc.collect()

# ---------------------------------------------------------------------------
# initializer 重定向。torch 的常量折叠把每个 F.linear 的权重换成了**转置副本**，
# 命名 onnx::MatMul_*（对 llm.onnx 的实测：311 个 initializer 里 197 个是
# onnx:: 前缀，0 个 Linear 权重保留 FQN —— 只有 Gather/Mul 消费的 embed/norm
# 保留参数名）。所以产物里的 initializer 分四类：
#
#   A. 参数名保留（72 个 qkv bias + 49 个 RMSNorm）：被 Gather/Mul 直接消费、
#      无 Transpose。checkpoint 字节是 BF16，而图上的激活是 fp16 —— bf16/fp16
#      混接的 ONNX 是非法类型（ORT 拒绝加载），所以这里**转成 fp16**（与
#      onnx2vkop `_bf16_bytes_to_fp16` 同一配方）进侧文件。~150 KB，微不足道。
#   B. 折叠出的**接线专家**权重（288 个，形状只可能是 (2048,1408)/(1408,2048)，
#      全图唯一）：保持内联 fp16（~1.66 GB，proto < 2 GiB 上限）。阶段 2 会把
#      它们删掉换成指向分片的槽位 initializer，所以内联只是过渡。
#   C. 折叠出的**非专家** Linear 权重（q/k/v/o、shared expert、lm_head；
#      ~169 个，~3.1 GB）：字节是 fp16 转置副本，checkpoint 里没有对应布局，
#      也不能内联（总量超 2 GiB）→ 写进旁边的 fp16 侧文件
#      llm_moe.linweights.bin，initializer 指 external。fp16 + 只被 MatMul 消费
#      ⇒ 转换期走 weight_stream 原样流式，不占转换内存。
#   D. 小常量（< 1 MiB）：router / shared_expert_gate 的折叠块等，内联。
#
# 产物里**没有任何 bf16**，external 只指向侧文件 —— phase 2 的槽位才引用
# checkpoint（bf16，由转换器的流式转换处理）。
#
# 危险边界（务必看清）：本文件**只写** EXPORT_PATH / LINWEIGHTS_PATH 指定的文件，
# 绝不触碰 llm.weights.bin / llm.onnx 等 canonical 产物。
# ---------------------------------------------------------------------------
import onnx  # noqa: E402
from onnx import TensorProto  # noqa: E402

LINWEIGHTS_NAME = os.environ.get("LINWEIGHTS_PATH", "llm_moe.linweights.bin")

m = onnx.load(EXPORT_PATH, load_external_data=True)
g = m.graph

# 参数出身的名字集合：wrapper 的真参数 FQN。融合专家参数是 meta 且从未被 forward
# 触碰，不会出现在图里 —— 这一点由「非参数 initializer 必须是折叠专家」断言兜底。
# 采样过的 FQN：embed_tokens.weight / layers.N.self_attn.q_proj.weight /
# norm.weight / lm_head.weight / layers.N.mlp.gate.weight / shared_expert.*。
_param_names = set()
for _name in _non_expert_keys:
    if _name == "lm_head.weight":
        _param_names.add("lm_head.weight")
    else:
        _param_names.add(_name[len("model."):])

expert_sized = {tuple(e["shape"]) for e in
                (_ckpt_table[_expert_key(0, 0, p)] for p in
                 ("gate_proj", "up_proj", "down_proj"))}

rewritten = folded = side_n = tiny = 0
side_off = 0
side_f = open(LINWEIGHTS_NAME, "wb")
try:
    for t in g.initializer:
        name = t.name
        if name in _param_names:
            # A 类：未转置参数（qkv bias + norms）。checkpoint 字节是 BF16，而图
            # 上的激活是 fp16 —— ONNX 里 Mul(MatMul) 混接 bf16/fp16 是非法类型
            # （ORT 直接拒绝加载）。这些张量总共只有 ~150 KB，转成 fp16 与折叠
            # 权重一起进侧文件，产物里不再有任何 bf16 initializer。
            key = name if name.startswith("lm_head") else "model." + name
            e = _ckpt_table.get(key)
            if e is None:
                raise SystemExit(f"[fatal] initializer '{name}' 找不到 checkpoint "
                                 f"键 '{key}' —— 命名映射漂移，拒绝静默错位")
            if list(t.dims) != e["shape"]:
                raise SystemExit(f"[fatal] initializer '{name}' dims "
                                 f"{list(t.dims)} != checkpoint {e['shape']}")
            import numpy as np
            with open(e["location"], "rb") as f:
                f.seek(e["offset"])
                raw = f.read(e["length"])
            if len(raw) != e["length"]:
                raise SystemExit(f"[fatal] {key} 短读 {len(raw)}/{e['length']}")
            u16 = np.frombuffer(raw, dtype=np.uint16)
            f32 = (u16.astype(np.uint32) << 16).view(np.float32)
            with np.errstate(over="ignore"):
                fp16 = np.ascontiguousarray(f32.astype(np.float16)).tobytes()
            side_f.write(fp16)
            t.data_type = TensorProto.FLOAT16
            t.data_location = TensorProto.EXTERNAL
            t.raw_data = b""  # bytes 字段不支持 del 切片，赋空是唯一清法
            t.external_data.clear()
            for k, v in (("location", LINWEIGHTS_NAME), ("offset", str(side_off)),
                         ("length", str(len(fp16)))):
                kv = t.external_data.add()
                kv.key, kv.value = k, v
            side_off += len(fp16)
            rewritten += 1
        elif len(t.raw_data) >= 1024 * 1024:
            dims = tuple(int(d) for d in t.dims)
            if t.data_type != TensorProto.FLOAT16:
                raise SystemExit(f"[fatal] 折叠 initializer '{name}' dtype "
                                 f"{t.data_type}，期望 fp16")
            if dims in expert_sized:
                folded += 1  # B 类：内联 fp16，阶段 2 删
            else:
                # C 类：fp16 转置副本 → 侧文件
                side_f.write(t.raw_data)
                t.data_location = TensorProto.EXTERNAL
                t.external_data.clear()
                for k, v in (("location", LINWEIGHTS_NAME),
                             ("offset", str(side_off)),
                             ("length", str(len(t.raw_data)))):
                    kv = t.external_data.add()
                    kv.key, kv.value = k, v
                side_off += len(t.raw_data)
                t.raw_data = b""
                side_n += 1
        else:
            tiny += 1  # D 类：router/shared_gate 折叠块等小常量，内联
finally:
    side_f.close()

want_folded = NLAYERS * TOP_K * 3
if folded != want_folded:
    raise SystemExit(f"[fatal] 折叠专家权重 {folded} 个，期望 {want_folded} "
                     f"（{NLAYERS} 层 × {TOP_K} 槽 × 3 矩阵）。_moe_forward 的展开"
                     f"结构变了，阶段 2 的槽位接线会静默错位。")
# 图上 FQN 保留的参数 = 不被 F.linear 消费的：q/k/v bias（MatMul 的 C 输入，
# 3N）+ 每层两个 RMSNorm + model.norm（Mul 消费，N+1）+ 0。embed_tokens 不在
# 图里 —— I/O 契约吃 inputs_embeds（与稠密导出器一致）。
want_a = 5 * NLAYERS + 1
if rewritten != want_a:
    raise SystemExit(f"[fatal] 参数 initializer 重写 {rewritten} 个，期望 "
                     f"{want_a}（qkv bias 3N + layernorm 2N + model.norm）。"
                     f"有参数没被 trace 到，或 torch 的折叠行为变了 —— 图不完整。")
# F.linear 权重全部变成折叠转置副本：≥1MiB 的（q/k/v/o、shared×3、lm_head）
# 进侧文件，router/shared_expert_gate（<1MiB）内联。
want_side = 7 * NLAYERS + 1
want_tiny = 2 * NLAYERS
if side_n != want_side or tiny != want_tiny:
    raise SystemExit(f"[fatal] 侧文件 {side_n} / 小常量 {tiny}，期望 "
                     f"{want_side} / {want_tiny} —— 折叠分类与预期不符，"
                     f"拒绝继续（侧文件契约失效）")

print(f"[rewrite] {rewritten} 参数 -> fp16 侧文件; "
      f"{side_n} 非专家 Linear -> {LINWEIGHTS_NAME} (共 {side_off / 1e9:.2f} GB fp16); "
      f"{folded} 折叠专家权重内联 fp16; {tiny} 小常量")

# 与稠密导出器相同的声明：attention_bias 输入就是标准 causal bias（转换器据此
# 走 fused attention / 免回读路径）。
m.metadata_props.add(key="vkop_causal_mask:attention_bias", value="1")

# TopK（router）输出的类型声明。torch 这次导出没写任何 value_info；转换器对无
# 声明的输出只能猜 dtype —— indices 被猜成 fp16，而 vkop 的 TopK kernel 只写
# int32（ops/Topk.hpp 的硬门，运行期构造即崩）。**value_info 会被 onnxoptimizer
# 整体丢弃**（见 converter 对 metadata 的同一处理），所以两个输出都进
# graph.output（I/O 声明在优化全程保留，converter.py 的输出查找也查它）。
# indices 在本图里没有数据消费者（分支接线固定 0..3），改声明零数学影响；
# M1 的聚合回读要的正是 int32 的 top-k id。
from onnx import helper as onnx_helper  # noqa: E402

_topk_n = 0
for _n in g.node:
    if _n.op_type != "TopK":
        continue
    vals, idxs = _n.output[0], _n.output[1]
    # TopK(x [bs, 60], k=4) 的两个输出都是 [bs, 4] —— values 的宽度是 **k**，
    # 不是专家数（第一版写成了 NUM_EXPERTS，runtime 按声明的 [1,60] 预分配，
    # 路由权重就读错了张量 —— logits 相关但不对，正是这个错）。
    # 只有 **indices** 进 graph.output（int32，M1 聚合回读要的就是它）：
    # values 若也声明成 graph output，会被加载器按 precision_ 预绑定成 fp16，
    # 而 kernel 按 fp32 输入写 fp32 —— dtype/形状两头都对不上。values 走普通
    # 节点输出路径，runtime 按 fp32 输入自己推出 fp32。
    vi = onnx_helper.make_tensor_value_info(idxs, TensorProto.INT32,
                                            ["bs", TOP_K])
    g.value_info.append(vi)
    g.output.append(vi)
    _topk_n += 1
if _topk_n != NLAYERS:
    raise SystemExit(f"[fatal] TopK 节点 {_topk_n} 个，期望 {NLAYERS}")
print(f"[vi] {_topk_n} 个 TopK 输出声明为 values fp32 / indices int32")

blob = m.SerializeToString()
with open(EXPORT_PATH, "wb") as f:
    f.write(blob)
del m, blob
gc.collect()

removed = 0
for f_ in sorted(os.listdir(".")):
    if f_ in _pre or f_ in (EXPORT_PATH, LINWEIGHTS_NAME):
        continue
    # `norm.weight` 在上面已被重写进侧文件，是真正的残留。但 layers.*.bias 不是：
    # 阶段 1 的硬校验要求所有 external 都指向侧文件，所以那份布局里的每个散文件
    # 都会**先**被下面的 save_model 合并进 LINWEIGHTS_NAME，之后再清才安全 ——
    # 而下面那条路径会 SystemExit（只允许 LINWEIGHTS_NAME 这一处 external）。
    # torch.onnx.export 只在单张量 >2 GiB 时写散文件，一次也没有触发过，所以
    # 这一批本身就是死后代码；先给次级产物留条命，别删掉将来那份合并布局要读的
    # 东西。删掉本循环（以及它依赖的 _pre 快照）是安全的。
    if f_.endswith(".bias"):
        continue
    if f_.endswith(".weight") or f_.startswith("onnx__") or \
       f_.endswith(".bin") or f_.endswith(".data"):
        os.remove(f_)
        removed += 1

# 硬校验：重读（不碰字节），external 只允许指向侧文件（本产物的图不含 bf16，
# 所以也不该再有任何 checkpoint 引用 —— 那些在阶段 2 的槽位里）。
_back = onnx.load(EXPORT_PATH, load_external_data=False)
_ok_locs = {os.path.realpath(LINWEIGHTS_NAME)}
_ext = 0
for _t in _back.graph.initializer:
    if _t.HasField("data_location") and _t.data_location == onnx.TensorProto.EXTERNAL:
        _ext += 1
        for _kv in _t.external_data:
            if _kv.key == "location" and \
               os.path.realpath(_kv.value) not in _ok_locs:
                raise SystemExit(f"[fatal] initializer '{_t.name}' external 指向 "
                                 f"{_kv.value}，只允许 {LINWEIGHTS_NAME}")
print(f"[ok] {EXPORT_PATH}: {len(_back.graph.initializer)} initializer(s), "
      f"{_ext} external; 清理 {removed} 个 torch 散文件")
print(f"[next] python3 make_moe_variant.py -i {EXPORT_PATH} -o llm_moe_paged.onnx "
      f"--ckpt {MODEL_PATH}")
