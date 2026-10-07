#!/usr/bin/env python3
"""导出 GLM-Edge-1.5B-Chat (`GlmForCausalLM`, model_type=glm) 的 LLM 到 ONNX。

与 qwen3_export_onnx.py / phi4_export_onnx.py 保持**完全相同的张量 I/O 契约**，
这样 C++ 驱动 (llm_chat) 不用为第三个架构改加载逻辑：

  入：inputs_embeds      (B, q, HIDDEN)   fp16
      position_ids       (B, q)           int64
      attention_bias     (B, 1, q, kv)    fp16      # 加法 bias，仅 causal
      past_key_values_{0..NLAYERS-1}       (B, 2, NKV, kv, HD)   fp16
  出：logits             (B, q, VOCAB)   fp16
      present_key_values_{0..NLAYERS-1}    (B, 2, NKV, kv, HD)   fp16

GLM-Edge 的结构（已对着 safetensors 头逐张量核过，227 个张量、每层 8 个）：
朴素 pre-norm decoder-only —— 分离 q/k/v 投影（**无** QK-norm、**无** norm_head、
**无** rel-position）、fused gate_up_proj（同 Phi）、SiLU、RMSNorm、tied embeddings、
head_dim 128 且 hidden 2048 / 16 q 头 / 4 kv 头（GQA 4 组）、max_ctx 8192、
config 里没有 sliding_window / layer_types 字段 → 纯 causal。

与 Qwen3 / Phi-4 的唯一实质差异：**RoPE 的配对方式是交错（interleaved）的**。
`modeling_glm.apply_rotary_pos_emb` 用 `x[..., 0::2] / x[..., 1::2]` 配对，并把
cos/sin `repeat_interleave(2)`，即维度 (2i, 2i+1) 共享角度 i；而 Qwen/Phi 的
`rotate_half` 配对的是 (i, i+64)。runtime 的 RotaryEmbedding 内核和转换器的
fuse_rotary_embedding pattern 都按半切约定实现，所以这里把交错形式**改写成半切
形式 + 固定列置换**（脚本会先自证两者逐位等价再导出）：

    对 x 施加置换 P = [偶数列 ‖ 奇数列]，做半切 RoPE，再施加 P⁻¹。
    HF 的 rotary_emb 输出 cos = cat(f, f) 正好等于半切形式所需的 cos 宽度，
    所以不需要动权重、也不需要给内核加新模式。
代价是 q/k 各多两个 Reshape+Transpose（成对抵消，present_key_values 与 HF 同基，
逐层 KV 对齐检查照常可比）。

上下文：max_position_embeddings 8192，与驱动侧 KV 预分配上界一致；rotary 是
`rope_type=default`（无 longrope/llama3 那种数据相关分支），inv_freq 常量，
attention_scaling 1.0 —— 脚本在导出前逐条断言，不接受「大概是默认」这类假设。

注意力**不按 HF 那样 repeat_kv 展开 K/V**，而是把同一组的 g 个 q 头并进矩阵乘的
m 维（`attention_gqa`，导出前用 `_selfcheck_gqa` 断言两条路等价）。展开在图上等于
每步 decode 重写 g 倍历史长度 fp16（GLM 是 56 个 Expand），实测是 ms/token 随
past_len 线性增长的主要来源之一。代价是 attention_bias 要沿 q 轴平铺 g 份
（比 K/V 的展开小两个数量级），以及 q/out 各多一次 5-D permute 的搬数。

用法:
  python3 glm_edge_export_onnx.py            # → text_glm_edge/llm.onnx + llm.weights.bin
  MODEL_PATH=/path/to/glm-edge-1.5b-chat EXPORT_PATH=llm.onnx python3 glm_edge_export_onnx.py
"""

import json
import os
import sys
import torch

MODEL_PATH = os.environ.get("MODEL_PATH") or os.path.join(
    os.path.expanduser("~"), ".cache/modelscope/models/"
    "ZhipuAI--glm-edge-1.5b-chat/snapshots/master")

# 产物目录与 text_qwen3/ text_phi4/ 平级，避免多套模型的 llm.onnx 互相覆盖
_DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "text_glm_edge")
EXPORT_PATH = os.environ.get("EXPORT_PATH") or os.path.join(_DEFAULT_DIR, "llm.onnx")
WEIGHTS_BIN = os.path.basename(EXPORT_PATH).replace(".onnx", ".weights.bin")

print(f"[load] model = {MODEL_PATH}")

from transformers.models.glm.modeling_glm import (  # noqa: E402
    GlmForCausalLM,
    apply_rotary_pos_emb,
    repeat_kv,
)

OPSET = 17

model = GlmForCausalLM.from_pretrained(
    MODEL_PATH,
    attn_implementation="eager",
    torch_dtype=torch.float16,
    local_files_only=os.path.isdir(os.path.expanduser(MODEL_PATH)),
)
model.eval()

config = model.config
NLAYERS = config.num_hidden_layers           # 28
HIDDEN = config.hidden_size                  # 2048
NUM_HEADS = config.num_attention_heads       # 16
NUM_KV_HEADS = config.num_key_value_heads    # 4
HEAD_DIM = getattr(config, "head_dim", None) or HIDDEN // NUM_HEADS   # 128
NUM_KV_GROUPS = NUM_HEADS // NUM_KV_HEADS    # 4
INTERMEDIATE = config.intermediate_size      # 6144
VOCAB = config.vocab_size                    # 59264
SCALING = model.model.layers[0].self_attn.scaling          # 128**-0.5

print(f"[config] layers={NLAYERS} hidden={HIDDEN} heads={NUM_HEADS} "
      f"kv_heads={NUM_KV_HEADS} head_dim={HEAD_DIM} groups={NUM_KV_GROUPS} "
      f"scaling={SCALING:.6f} vocab={VOCAB} inter={INTERMEDIATE}")

# ---------------------------------------------------------------------------
# 导出前置校验：把「这份 checkpoint 就是朴素 decoder-only + 默认 RoPE」变成硬约束
# ---------------------------------------------------------------------------
raw_cfg = json.load(open(os.path.join(MODEL_PATH, "config.json")))
rop = config.rope_parameters or {}
partial = float(rop.get("partial_rotary_factor", 1.0))

assert rop.get("rope_type", "default") == "default", \
    f"预期默认 RoPE，得到 {rop}"
assert partial == 1.0, f"partial_rotary_factor={partial}：RoPE 只旋转部分维度，" \
    f"下面的置换改写（rotary_dim==head_dim）不再成立，需要按 Phi 的 partial 路径处理"
assert HEAD_DIM % 2 == 0, f"head_dim {HEAD_DIM} 必须是偶数（半切/交错都要成对）"
assert NUM_HEADS % NUM_KV_HEADS == 0, "GQA 组数不整除"
assert "sliding_window" not in raw_cfg and "layer_types" not in raw_cfg, \
    "config 里出现了滑窗字段：纯 causal 的 attention_bias 契约不再成立，需要重做 mask"
assert raw_cfg.get("attention_bias") is False, "attention_bias 预期 False（投影无 bias）"
assert config.hidden_act == "silu", f"hidden_act={config.hidden_act}"

attn0 = model.model.layers[0].self_attn
assert not hasattr(attn0, "q_norm") and not hasattr(attn0, "k_norm"), \
    "出现了 QK-norm，需要按 Qwen3 的路径加 norm"
for mod in (attn0.q_proj, attn0.k_proj, attn0.v_proj, attn0.o_proj,
            model.model.layers[0].mlp.gate_up_proj, model.lm_head):
    assert mod.bias is None, "投影带 bias，契约里没有它"

rotary = model.model.rotary_emb
assert rotary.attention_scaling == 1.0, \
    f"attention_scaling={rotary.attention_scaling} 会乘进 cos/sin，必须复用 HF rotary_emb"
assert rotary.inv_freq.numel() == HEAD_DIM // 2, \
    f"inv_freq 长度 {rotary.inv_freq.numel()} != head_dim/2"
_manual_inv = 1.0 / (rop["rope_theta"] **
                     (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
assert torch.allclose(rotary.inv_freq.detach().float().cpu(), _manual_inv, atol=1e-9), \
    "inv_freq 与默认 RoPE 不一致"
print(f"[rope] 默认分支已确认：inv_freq == 1/theta^(2i/d)，attention_scaling=1.0，"
      f"配对方式=交错")

# tied embeddings：驱动侧只取一张 embed 表，图里的 lm_head 用的是它自己的副本
with torch.no_grad():
    _tied = torch.equal(model.lm_head.weight, model.model.embed_tokens.weight)
assert raw_cfg["tie_word_embeddings"] == _tied, \
    f"config 说 tie={raw_cfg['tie_word_embeddings']}，实际张量比较结果 {_tied}"
print(f"[tie] lm_head == embed_tokens: {_tied}")


# ---------------------------------------------------------------------------
# 交错 RoPE → 半切 RoPE + 固定列置换
# ---------------------------------------------------------------------------
def _to_halfsplit_basis(x):
    """[..., D] -> [偶数列 ‖ 奇数列]（即置换 P）。"""
    *lead, d = x.shape
    return x.reshape(*lead, d // 2, 2).transpose(-1, -2).reshape(*lead, d)


def _from_halfsplit_basis(y):
    """_to_halfsplit_basis 的逆变换：[a ‖ b] -> [a0, b0, a1, b1, ...]。"""
    *lead, d = y.shape
    return y.reshape(*lead, 2, d // 2).transpose(-1, -2).reshape(*lead, d)


def _rotate_half_standard(x):
    """transformers 里 qwen3/llama 版的 rotate_half（前半/后半配对）。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def rope_halfsplit_equivalent(q, k, cos, sin):
    """与 modeling_glm.apply_rotary_pos_emb 逐位等价的半切写法。

    cos/sin 是 HF rotary_emb 的输出 (B, q, head_dim) == cat(f, f)，
    在半切基下正是需要的 [c_0..c_31 ‖ c_0..c_31] 形式，无需再变换。
    """
    c = cos.unsqueeze(1)                      # (B, 1, q, D)，与 HF 的 unsqueeze(1) 同形
    s = sin.unsqueeze(1)
    qp = _to_halfsplit_basis(q)
    kp = _to_halfsplit_basis(k)
    qr = qp * c + _rotate_half_standard(qp) * s
    kr = kp * c + _rotate_half_standard(kp) * s
    return _from_halfsplit_basis(qr), _from_halfsplit_basis(kr)


def _selfcheck_rope():
    """把等价性变成断言，而不是靠推导：随机 q/k + 真实 cos/sin，与 HF 实现比。"""
    torch.manual_seed(0)
    B, S, H, HKV, D = 2, 7, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM
    q = torch.randn(B, H, S, D, dtype=torch.float32)
    k = torch.randn(B, HKV, S, D, dtype=torch.float32)
    pos = torch.arange(S, dtype=torch.long).unsqueeze(0).expand(B, -1)
    x = torch.zeros(B, S, HIDDEN, dtype=torch.float32)
    cos, sin = model.model.rotary_emb(x, pos)
    q_hf, k_hf = apply_rotary_pos_emb(q, k, cos, sin)      # interleaved（HF 原样）
    q_alt, k_alt = rope_halfsplit_equivalent(q, k, cos, sin)
    dq = (q_hf - q_alt).abs().max().item()
    dk = (k_hf - k_alt).abs().max().item()
    print(f"[rope-check] fp32 maxdiff q={dq:.3e} k={dk:.3e}")
    assert dq < 1e-6 and dk < 1e-6, "半切改写与 HF 的交错 RoPE 不等价，不要继续导出"


_selfcheck_rope()


# ---------------------------------------------------------------------------
# GQA 注意力：K/V 一份都不复制，把同组的 q 头并进矩阵乘的 m 维
# ---------------------------------------------------------------------------
def attention_gqa(q, k, v, attention_bias, scaling, num_kv_groups):
    """与「repeat_kv 后再算」等价的注意力，返回 (B, q_len, nq*head_dim)。

    HF 的 repeat_kv(k, g) 把每组 K/V 展开成 nq 份 —— 在图上就是每步 decode 都要
    整段重写 g 倍历史长度的 fp16（GLM 的图里是 56 个 Expand）。但同一 kv 头对这
    g 个 q 头本来就是同一份 K/V，所以把这 g 个 q 头并进 m 维即可：
        q (B, nq, q, hd) -> (B, nkv, g*q, hd)     # h = kv*g + j，正是 repeat_kv 的头序
    每个输出元素读的 K/V 数值不变，变的只是 GEMM 的 m 维从 q_len 变成 g*q_len，
    于是 fp32 累加顺序可能不同（fp16 下是最后几位），不是数学差异。
    """
    B, nq, q_len, hd = q.shape
    nkv = k.shape[1]
    kv_len = k.shape[2]
    q_g = q.reshape(B, nkv, num_kv_groups * q_len, hd)
    # 与 HF 的 eager_attention_forward 同序：先 matmul、再 *scaling、再 +bias
    scores = torch.matmul(q_g, k.transpose(2, 3)) * scaling
    # scores 的行是 (group, query) 合并的，group 在外层，所以 bias 要平铺 g 份
    bias = attention_bias.unsqueeze(2).expand(
        B, 1, num_kv_groups, q_len, kv_len).reshape(
        B, 1, num_kv_groups * q_len, kv_len)
    scores = scores + bias
    w = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(w, v)                       # (B, nkv, g*q, hd)
    out = out.reshape(B, nkv, num_kv_groups, q_len, hd)
    return out.permute(0, 3, 1, 2, 4).reshape(B, q_len, nq * hd)


def attention_repeat_kv(q, k, v, attention_bias, scaling, num_kv_groups):
    """HF 原样（repeat_kv 展开 K/V），只用来自证上面那条路与它等价。"""
    B, nq, q_len, hd = q.shape
    k_r = repeat_kv(k, num_kv_groups)
    v_r = repeat_kv(v, num_kv_groups)
    scores = torch.matmul(q, k_r.transpose(2, 3)) * scaling
    w = torch.nn.functional.softmax(
        scores + attention_bias, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(w, v_r)
    return out.transpose(1, 2).reshape(B, q_len, nq * hd)


def _selfcheck_gqa():
    """等价性必须是断言：随机 q/k/v + 真因果 bias，两条路在 fp16 下比。"""
    torch.manual_seed(0)
    g = NUM_KV_GROUPS
    for B, S, KV in ((2, 7, 11), (1, 1, 32)):        # prefill 与 decode 两种形状
        q = torch.randn(B, NUM_HEADS, S, HEAD_DIM, dtype=torch.float16)
        k = torch.randn(B, NUM_KV_HEADS, KV, HEAD_DIM, dtype=torch.float16)
        v = torch.randn(B, NUM_KV_HEADS, KV, HEAD_DIM, dtype=torch.float16)
        causal = torch.triu(torch.full((S, KV), torch.finfo(torch.float16).min,
                                      dtype=torch.float16), diagonal=1)
        bias = causal.unsqueeze(0).unsqueeze(0)
        a = attention_gqa(q, k, v, bias, SCALING, g)
        b = attention_repeat_kv(q, k, v, bias, SCALING, g)
        rel = ((a.float() - b.float()).abs().max()
               / b.float().abs().max().clamp(min=1e-6)).item()
        print(f"[gqa-check] B={B} q={S} kv={KV} bitwise_equal={torch.equal(a, b)} "
              f"maxrel={rel:.3e}")
        assert rel < 1e-3, "GQA 的 reshape 写法与 repeat_kv 不等价，不要继续导出"


_selfcheck_gqa()

lm = model.model
embed_tokens = lm.embed_tokens
layers = lm.layers
norm = lm.norm
rotary_emb = lm.rotary_emb
lm_head = model.lm_head


class GlmEdgeLLMOnnx(torch.nn.Module):
    """手写 forward：显式 KV cache 张量 I/O + 加法 attention bias，绕开 HF 的
    DynamicCache 和 create_causal_mask。子模块全部复用 HF 的，保证权重与算子顺序
    一致；只有 RoPE 用上面自证等价的半切写法。"""

    def __init__(self):
        super().__init__()
        self.embed_tokens = embed_tokens
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
        self.inter = INTERMEDIATE

    def decoder_layer(self, layer, hidden, cos, sin, attention_bias, past_kv):
        """单层前向，逐 op 对应 GlmDecoderLayer.forward。返回 (hidden, present_kv)。"""
        B, q_len, _ = hidden.shape
        residual = hidden
        h = layer.input_layernorm(hidden)

        attn = layer.self_attn
        # HF: q/k/v 各自 proj().view(hidden_shape).transpose(1, 2)，hidden_shape 用
        # kv 头数还是 q 头数由各自投影决定 —— 逐个 view 与 HF 一致
        q = attn.q_proj(h).view(B, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = attn.k_proj(h).view(B, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = attn.v_proj(h).view(B, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 交错 RoPE 的半切等价形式（_selfcheck_rope 已证与 HF 逐位一致）
        q, k = rope_halfsplit_equivalent(q, k, cos, sin)

        # concat past KV: past_kv (B, 2, nkv, past, hd)
        past_k = past_kv[:, 0]
        past_v = past_kv[:, 1]
        k_new = torch.cat([past_k, k], dim=2)
        v_new = torch.cat([past_v, v], dim=2)

        # GQA：不 repeat_kv（_selfcheck_gqa 已证等价），K/V 在图里只有一份
        out = attention_gqa(q, k_new, v_new, attention_bias,
                            self.scaling, self.num_kv_groups)
        out = attn.o_proj(out)

        hidden = residual + out

        residual = hidden
        h = layer.post_attention_layernorm(hidden)
        mlp = layer.mlp
        # HF 用 chunk(2, -1) 拆 gate/up；这里等价地用常量边界切片，避开 chunk 生成的
        # 整数 shape 链被融成 FUSED_ELEMWISE 后 Slice.ends 不再是 int64 的那类崩溃
        # （见 README「跑通 Phi 期间修掉的三个静默错误」）。
        up_states = mlp.gate_up_proj(h)
        gate = up_states[..., :self.inter]
        up_states = up_states[..., self.inter:]
        hidden = residual + mlp.down_proj(up_states * mlp.activation_fn(gate))

        present_kv = torch.stack([k_new, v_new], dim=1)
        return hidden, present_kv

    def forward(self, inputs_embeds, position_ids, attention_bias, *past_kvs):
        hidden = inputs_embeds
        # 复用 HF rotary_emb：cos/sin = (B, q, head_dim)，配对信息在列序里
        cos, sin = self.rotary_emb(hidden, position_ids)

        presents = []
        for idx, layer in enumerate(self.layers):
            hidden, pk_new = self.decoder_layer(
                layer, hidden, cos, sin, attention_bias, past_kvs[idx])
            presents.append(pk_new)

        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        return (logits, *presents)


wrapper = GlmEdgeLLMOnnx()
wrapper.eval()

# ---------------------------------------------------------------------------
# 构造 prefill dummy inputs
# ---------------------------------------------------------------------------
B = 1
L = int(os.environ.get("PREFILL_LEN", "32"))
q_len = kv_len = L

inputs_embeds = torch.randn(B, q_len, HIDDEN, dtype=torch.float16)
position_ids = torch.arange(L, dtype=torch.long).unsqueeze(0).expand(B, -1)
# causal mask：上三角填 finfo(fp16).min（-65504），与 HF create_causal_mask 实测一致；
# 不用 -inf —— fp16 softmax 需要留一点区分度
causal = torch.triu(
    torch.full((q_len, kv_len), torch.finfo(torch.float16).min, dtype=torch.float16),
    diagonal=1)
attention_bias = causal.unsqueeze(0).unsqueeze(0)  # (1, 1, q, kv)

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

print(f"[export] GlmEdge-onnx: {NLAYERS} layers, {len(input_names)} inputs / "
      f"{len(output_names)} outputs, opset={OPSET} ...")

os.makedirs(os.path.dirname(os.path.abspath(EXPORT_PATH)), exist_ok=True)

with torch.no_grad():
    torch.onnx.export(
        wrapper, inputs, EXPORT_PATH,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,  # legacy TorchScript 导出器，与 qwen3/phi4 保持一致
    )
print(f"[✓] exported → {EXPORT_PATH}")

# ---------------------------------------------------------------------------
# 合并散权重
# ---------------------------------------------------------------------------
import onnx  # noqa: E402

out_dir = os.path.dirname(os.path.abspath(EXPORT_PATH))
print("[consolidate] checking external data ...")
# 先不带外部数据读一遍，只为了拿到 torch.onnx.export 撒下来的散文件名单（合并后要删掉，
# 否则目录里同时留着 226 个散文件和一份单文件权重，下一步转换时容易拿错）。
m_refs = onnx.load(EXPORT_PATH, load_external_data=False)
def _ext_loc(t):
    for kv in t.external_data:
        if kv.key == "location":
            return kv.value
    return None


m_refs = onnx.load(EXPORT_PATH, load_external_data=False)
old_locs = sorted({_ext_loc(t) for t in m_refs.graph.initializer} - {None})

m = onnx.load(EXPORT_PATH, load_external_data=True)
n_init = len(m.graph.initializer)
n_ext = sum(1 for t in m.graph.initializer
            if t.HasField("data_location") and t.data_location == 1)
print(f"[consolidate] initializers={n_init} external={n_ext}")

# onnx 的 location 是**相对模型文件**的路径，但它内部用 os.path.exists(location) 查重，
# 那是相对 cwd 的 —— 不切到产物目录就会撞上别处同名的 llm.weights.bin（FileExistsError）。
os.chdir(out_dir)
onnx.save_model(
    m, os.path.basename(EXPORT_PATH),
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location=WEIGHTS_BIN,
    convert_attribute=True,
)
removed = 0
for f in old_locs:
    if f != WEIGHTS_BIN and os.path.exists(f):
        os.remove(f)
        removed += 1
print(f"[✓] consolidated → {out_dir}/{os.path.basename(EXPORT_PATH)} + {WEIGHTS_BIN} "
      f"(删掉 {removed}/{len(old_locs)} 个散权重文件)")
