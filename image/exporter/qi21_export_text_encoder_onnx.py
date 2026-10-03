"""把 Qwen-Image-2.1 文生图的**文本编码器**导出成 onnx2vkop 友好的 ONNX。

模型来源是 `text_encoder/`，那是一份完整的 `Qwen3VLForConditionalGeneration`
（8.77B 参数 / bf16 / 17.5 GB），但文生图只用它的 `model.language_model`
（36 层 / hidden 4096 / 32 q 头 + 8 kv 头 / head_dim 128 / vocab 151936）。所以本
脚本只把文本塔读进 fp16（15.1 GB 含查表，进图 13.9 GB），`lm_head`（1.24 GB）和视
觉塔（~1.1 GB）一个字节都不碰 —— 既省内存，也不会出现在图里。

    # 真权重 -> text_encoder.onnx + text_encoder.weights.bin
    /Users/doudou/qi21-env/bin/python qi21_export_text_encoder_onnx.py
    # N 层随机权重，只看图结构（秒级，不写 13.9 GB）
    /Users/doudou/qi21-env/bin/python qi21_export_text_encoder_onnx.py --tiny 2
    # 顺带用 fp16 eager 跑一条真实 prompt，产出可与 encode_prompt_real.py 对拍的文件
    ... --prompt "a red cube" --emit-ref ref_te

和 LLM 那条线（llm/exporter/qwen3vl_export_onnx.py）的三处不同：

  1. **输出是 hidden_states，不是 logits**：砍掉 lm_head，而且**不过最后那个
     RMSNorm** —— 官方 pipeline 用 forward hook 把 `text_model.norm` 绕开（transformers
     5.x 会把 `hidden_states[-1]` 绑到已归一化的 last_hidden_state，坑的来龙去脉记在
     encode_prompt_real.py 开头）。本 wrapper 结构上就没有 norm 这一步，不存在"忘了绕开"。
  2. **没有 KV cache**：文本塔每条 prompt 只整段跑一次，所以图里没有
     past/present_key_values_*，也没有为它们服务的 Concat。
  3. **形状全静态**：序列长度固定 `P = drop_idx + prefix_len`（当前权重/模板下
     14 + 64 = 78）。换 prefix_len 要重导（和 DiT 的 static 图同一套取舍）。

I/O 契约（text_encoder.onnx）：
  入：inputs_embeds  (1,P,4096) fp16   # 模板+prompt 逐 token 查表；补位行喂零即可
  出：hidden_states  (1,P,4096) fp16   # 未过 final norm 的最后一层输出

rope 表和因果 mask 都**折进图里当常量**，所以整张图只有一个输入。这不是省事，是必须的：
`rotary_emb` 里 mrope 的 section 重排是 `freqs_thw[..., idx] = freq[dim][..., idx]` 这种
**原地分片赋值**，legacy 导出器会写成 4 个 ScatterND（每轴一次 × cos/sin），而 ScatterND
正是 vkop 会**静默 pass-through** 那一类 op。纯文本时 position_ids 三行恒等（都是
`arange(P)`），那次重排在数学上是恒等的，所以直接按标准 rope 算 cos/sin、并**当场断言
它和 HF 的 rotary_emb 逐位相等**（`check_rope_tables`）——等价性由实测兜住，而不是由
"我觉得三行相等所以一样" argument 出来。位置编号与 prompt 长短无关，故连 `position_ids`
输入也不需要了。

补位行的账：真实 prompt 短于 prefix_len 时，第 `drop_idx+valid_len` 行往后是补位。
因果注意力保证第 i 行只看 ≤i 的 key，所以补位**不会**污染真实行，它们自己算出来的是
垃圾、由驱动丢弃；驱动取 `[drop_idx, drop_idx+valid_len)` 再右补**精确零**到
prefix_len，与 encode_prompt_real.py 逐行一致。同理文本塔内部只需要纯因果 mask
（屏蔽补位列只影响同样是补位的行）；DiT 侧那个"屏蔽 prompt 补零列"的 bias 是另一张图。

精度口径：checkpoint 是 bf16，参考脚本 encode_prompt_real.py 也在 bf16 下前向，本图是
fp16。bf16→fp16 的**权重**转换在数值范围内无损（8 位尾数 → 11 位），差别在**激活**的
舍入路径，所以 `--emit-ref` 与 bf16 参考比对时预期 cos≈0.999 而不是逐位相同；真要逐位
对齐，得把 encode_prompt_real.py 也换成 fp16 前向。
"""

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

import onnx
import torch
from safetensors import safe_open
from transformers import Qwen3VLProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextConfig,
    Qwen3VLTextModel,
    apply_rotary_pos_emb,
    repeat_kv,
)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from encode_prompt_real import MS_ROOT, drop_idx_of, tokenize_prompt  # noqa: E402
# 大权重（>2 GB）的 external data 只能自己流式合并，理由见 qi21_export_onnx.merge_scattered
# 的 docstring：onnx.save_model 的 all_tensors_to_one_file 在 32 层真图上写出过坏账。
from qi21_export_onnx import ext_check, merge_scattered  # noqa: E402

OPSET = 17
CKPT = MS_ROOT / "text_encoder"
KEY_PREFIX = "model.language_model."
EMBED = "embed_tokens.weight"           # 不进图：驱动侧查表，和 llm 线的 embed_tokens.bin 同形
FP16_MIN = torch.finfo(torch.float16).min
IN_NAMES = ["inputs_embeds"]
OUT_NAME = "hidden_states"


# ---------------------------------------------------------------------------
# 权重：只取文本塔，逐张量 copy_ 进目标 fp16 缓冲
# ---------------------------------------------------------------------------
def text_config(layers: int = 0) -> Qwen3VLTextConfig:
    raw = json.loads((CKPT / "config.json").read_text())
    assert raw["dtype"] == "bfloat16", raw["dtype"]
    assert raw["architectures"] == ["Qwen3VLForConditionalGeneration"], raw["architectures"]
    tc = Qwen3VLTextConfig(**raw["text_config"])
    if layers:
        tc.num_hidden_layers = layers
    return tc


def build_text_model(tc: Qwen3VLTextConfig) -> Qwen3VLTextModel:
    """在 fp16 默认 dtype 下构造，让参数**生下来就是** fp16。

    不这么做就得靠 `.to(fp16)` 转换，那一刻"旧的一份 + 新的一份"同时在场
    （15.1 GB×2 = 30 GB），36 GB 统一内存直接进交换区。构造完立刻断言 dtype：错的话
    在这一步炸，而不是导出一个 fp32 图、等到转换/运行期才静默失配。
    """
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        model = Qwen3VLTextModel(tc)
    finally:
        torch.set_default_dtype(prev)
    bad = [n for n, p in model.named_parameters() if p.dtype != torch.float16]
    assert not bad, f"{len(bad)} 个参数不是 fp16（例：{bad[:3]}）"
    return model


def load_text_tower(model: Qwen3VLTextModel) -> None:
    """把 `model.language_model.*` 从分片逐个搬进 `model` 的参数（原地 copy_）。

    逐张量、不 `load_file` 整片：目标缓冲已 15.1 GB，再叠一份整片字典（单片最大
    5 GB）就把这台机器推进交换区；`safe_open` 走 mmap，`copy_` 边读边把 bf16 转 fp16。
    覆盖性靠 `remaining` 断言：少一层都会在这里报出来，而不是留着一层随机权重静默跑。
    """
    index = json.loads((CKPT / "model.safetensors.index.json").read_text())["weight_map"]
    own = dict(model.named_parameters())
    remaining = set(own)
    by_shard = collections.defaultdict(list)
    for ck, shard in index.items():
        sub = ck.removeprefix(KEY_PREFIX)
        if ck.startswith(KEY_PREFIX) and sub in remaining:
            by_shard[shard].append((ck, sub))
            remaining.discard(sub)
    assert not remaining, f"checkpoint 里缺文本塔参数: {sorted(remaining)[:5]}"
    t0 = time.time()
    for shard in sorted(by_shard):
        moved = 0
        with safe_open(CKPT / shard, framework="pt") as f:
            for ck, sub in by_shard[shard]:
                p = own[sub]
                p.data.copy_(f.get_tensor(ck))
                moved += p.numel() * p.element_size()
        print(f"[load] {shard}: {moved / 1e9:.2f} GB", flush=True)
    total = sum(p.numel() * p.element_size() for p in own.values())
    print(f"[load] 文本塔 {len(own)} 个参数 / {total / 1e9:.2f} GB fp16，用时 {time.time() - t0:.0f}s")


# ---------------------------------------------------------------------------
# 导出 wrapper：N 层 decoder，无 norm、无 lm_head、无 KV
# ---------------------------------------------------------------------------
def positions(P: int) -> torch.Tensor:
    """MRoPE 的三轴位置：纯文本恒为 arange(P) 复制三份。"""
    return torch.arange(P, dtype=torch.int64).view(1, 1, P).expand(3, 1, P).contiguous()


def rope_tables(rotary, P: int, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """按**标准 1D rope** 算 cos/sin (1,P,head_dim)，绕开 mrope 的分片赋值。

    与 `Qwen3VLTextRotaryEmbedding.forward` 的 fp32 路径逐 op 对齐：fp32 外积 -> cos/sin
    -> 乘 attention_scaling -> cat(f,f) -> 最后一步才落到 `dtype`。少了"最后一步才降
    精度"这一条，cos/sin 就会和 HF 差一个 fp16 ulp，rope 的误差会一路乘进注意力分数。
    """
    inv = rotary.inv_freq.float()                       # (64,) fp32，构造时就是 fp32
    pos = torch.arange(P, dtype=torch.float32)          # (P,)
    freqs = (inv[:, None] @ pos[None, :]).transpose(0, 1)      # (P,64)
    scaling = float(rotary.attention_scaling)
    emb = torch.cat((freqs, freqs), dim=-1)             # (P,head_dim)
    return ((emb.cos() * scaling).to(dtype).unsqueeze(0),
            (emb.sin() * scaling).to(dtype).unsqueeze(0))


def check_rope_tables(rotary, P: int, dtype) -> None:
    """断言上面那条"纯文本时 mrope 重排是恒等的"确实成立（逐位，不是 cos 近似）。

    HF 侧走的是 `freqs[3,B,P,64]` -> 按 mrope_section 选轴 -> cat。三行位置相同时选轴
    结果与 freq[0] 一致，但这必须是**测出来**的：换成带图像的位置、或 transformers 改了
    重排方式，这里就会炸，而不是导出一张 rope 静默错位的图。
    """
    hf_cos, hf_sin = rotary(torch.zeros(1, P, 8, dtype=dtype), positions(P))
    mine_cos, mine_sin = rope_tables(rotary, P, dtype)
    for name, a, b in (("cos", mine_cos, hf_cos), ("sin", mine_sin, hf_sin)):
        assert a.shape == b.shape, (name, tuple(a.shape), tuple(b.shape))
        d = (a.float() - b.float()).abs().max().item()
        print(f"[rope] {name} {tuple(a.shape)} vs HF rotary: maxdiff={d:g}")
        assert d == 0.0, f"{name} 与 HF rotary 不等（maxdiff={d}）：mrope 恒等的前提不成立"


class TextTowerOnnx(torch.nn.Module):
    """自己写层循环，逐 op 贴 HF 的 Qwen3VLTextDecoderLayer（与 llm 线那套一字同源）。

    cos/sin/因果 bias 都以 buffer 形式带在模块上（见模块 docstring：它们进不了 ScatterND
    那条路，而且对这张图是常量）。`apply_rotary_pos_emb` 内部会 unsqueeze(1) 把
    (1,P,hd) 广播成 (1,1,P,hd)，外面不要再加维度。
    """

    def __init__(self, text_model: Qwen3VLTextModel, P: int):
        super().__init__()
        self.layers = text_model.layers
        cfg, attn = text_model.config, text_model.layers[0].self_attn
        self.hidden, self.nh = cfg.hidden_size, cfg.num_attention_heads
        self.nkv, self.hd = cfg.num_key_value_heads, attn.head_dim
        self.groups, self.scaling = attn.num_key_value_groups, attn.scaling
        cos, sin = rope_tables(text_model.rotary_emb, P, torch.float16)
        self.register_buffer("cos", cos.contiguous())
        self.register_buffer("sin", sin.contiguous())
        self.register_buffer("bias", causal_bias(P).contiguous())

    def forward(self, inputs_embeds):
        hidden = inputs_embeds
        B, q_len, _ = hidden.shape
        hs, kvs = (B, q_len, self.nh, self.hd), (B, q_len, self.nkv, self.hd)
        for layer in self.layers:
            residual = hidden
            h = layer.input_layernorm(hidden)
            a = layer.self_attn
            # HF 顺序：view(B,q,heads,hd) -> q_norm -> transpose(1,2)。颠倒会让注意力分数错位。
            q = a.q_norm(a.q_proj(h).view(hs)).transpose(1, 2)
            k = a.k_norm(a.k_proj(h).view(kvs)).transpose(1, 2)
            v = a.v_proj(h).view(kvs).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, self.cos, self.sin)
            k, v = repeat_kv(k, self.groups), repeat_kv(v, self.groups)
            w = torch.matmul(q, k.transpose(2, 3)) * self.scaling + self.bias
            w = torch.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
            o = torch.matmul(w, v).transpose(1, 2).reshape(B, q_len, self.hidden)
            hidden = residual + a.o_proj(o)

            residual = hidden
            h = layer.post_attention_layernorm(hidden)
            m = layer.mlp
            hidden = residual + m.down_proj(m.act_fn(m.gate_proj(h)) * m.up_proj(h))
        return hidden                                            # 不过 norm、不乘 lm_head


def causal_bias(P: int, dtype=torch.float16) -> torch.Tensor:
    return torch.zeros(1, 1, P, P, dtype=dtype).masked_fill(
        torch.triu(torch.ones(P, P, dtype=torch.bool), 1), FP16_MIN)


def make_inputs(P: int, hidden: int, dtype=torch.float16) -> tuple:
    return ((torch.randn(1, P, hidden) * 0.02).to(dtype),)


# ---------------------------------------------------------------------------
# 结构检查：onnx2vkop 的 fold 认具体节点形态，导出器换写法要在这一步先响
# ---------------------------------------------------------------------------
def graph_check(m, path: Path, loc: str, P: int, hidden: int, n_layers: int):
    ops = collections.Counter(node.op_type for node in m.graph.node)
    print(f"\n[{path.name}] nodes={sum(ops.values())} inputs={len(m.graph.input)} "
          f"outputs={len(m.graph.output)}")
    for op, cnt in sorted(ops.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {op:<22} {cnt}")

    def shape(t):
        return [d.dim_value if d.HasField("dim_value") else d.dim_param
                for d in t.type.tensor_type.shape.dim]
    ins = [(t.name, shape(t), t.type.tensor_type.elem_type) for t in m.graph.input]
    want = [("inputs_embeds", [1, P, hidden], onnx.TensorProto.FLOAT16)]
    outs = [(t.name, shape(t), t.type.tensor_type.elem_type) for t in m.graph.output]
    print(f"  in : {ins}")
    print(f"  out: {outs}")
    assert ins == want, ins
    assert outs == [(OUT_NAME, [1, P, hidden], onnx.TensorProto.FLOAT16)], outs

    for banned in ("If", "Loop", "NonZero", "ScatterND", "GatherND", "Clip", "Compress",
                   "SequenceEmpty", "SequenceAt", "SequenceInsert", "CumSum", "Range"):
        assert ops[banned] == 0, f"出现 onnx2vkop 没有的 op: {banned}={ops[banned]}"
    # 每层恰好一个 attention softmax（没有 KV cache 就不该多出第二个）；Neg/Slice/Concat
    # 来自 rotate_half，是 runtime 认的那条 rope 形态（rope 表本身已折成常量）。
    assert ops["Softmax"] == n_layers, (ops["Softmax"], n_layers)
    assert ops["Neg"] == 2 * n_layers, (ops["Neg"], n_layers)
    assert ops["Slice"] and ops["Concat"], dict(ops)
    ext_check(path, loc, m)


# ---------------------------------------------------------------------------
# --emit-ref：fp16 eager 跑真实 prompt，产出与 encode_prompt_real 同形的文件
# ---------------------------------------------------------------------------
def check_vs_hf(model, wrapper, embeds: torch.Tensor, P: int) -> None:
    """同 dtype、同权重、同输入下比"手写层循环"和 HF `Qwen3VLTextModel.forward`。

    这条才是抓"逻辑错位"的：bf16 参考（encode_prompt_real.py）与本图的 fp16 激活必然
    差在舍入路径上（cos≈0.999），那种量级里看不出 q_norm 的 view/transpose 顺序写反、
    GQA 组数弄错这类真错。两边都在 fp16 下跑，差的只有 attention 实现细节（HF 走 sdpa、
    mask 值由 create_causal_mask 生成），所以门槛取 cos>0.9999：逻辑错会掉到 0.9 以下。
    照官方口径用 forward hook 把 `model.norm` 绕开，否则比的是归一化后的张量。
    """
    handle = model.norm.register_forward_hook(lambda m, a, o: a[0])
    try:
        with torch.no_grad():
            ref = model(inputs_embeds=embeds, position_ids=positions(P),
                        attention_mask=torch.ones(1, P, dtype=torch.long),
                        use_cache=False).last_hidden_state
    finally:
        handle.remove()
    with torch.no_grad():
        mine = wrapper(embeds)
    a, b = mine.float(), ref.float()
    cos = float((a * b).sum() / (a.norm() * b.norm() + 1e-12))
    print(f"[hf] 手写层循环 vs HF forward（都 fp16）: cos={cos:.6f} "
          f"maxabs={float((a - b).abs().max()):.4f} rms={float(a.pow(2).mean().sqrt()):.4f}")
    assert cos > 0.9999, f"与 HF 不等价（cos={cos}）：先查 q_norm 顺序 / GQA / rope 布局"


def emit_ref(model, wrapper, proc, ids: torch.Tensor, prefix_len: int, drop_idx: int,
             out: Path, ref_embeds: Path):
    """fp16 eager 跑一段真实 token 序列，产出与 encode_prompt_real.py 同形的文件。

    `ids` 是**模板+prompt 的完整** token 序列（含前 drop_idx 个 system 段 token）。
    对拍参考产物时要用 `--prompt-ids` 直接喂参考那次的 token id：同一段 token 才谈得上
    逐位比，靠 prompt 字符串重新分词会把"分词是否一致"混进"图是否等价"。
    """
    total = int(ids.shape[0])
    real_len = total - drop_idx
    if real_len > prefix_len:
        raise SystemExit(f"序列实际 {real_len} token > prefix_len={prefix_len}，换短的")
    P = drop_idx + prefix_len
    embeds = model.embed_tokens(ids).to(torch.float16).unsqueeze(0)    # (1,total,H)
    if total < P:                       # 补位行喂零：真实行不受影响（因果），见模块 docstring
        embeds = torch.cat([embeds, torch.zeros(1, P - total, embeds.shape[-1],
                                                dtype=torch.float16)], 1)
    check_vs_hf(model, wrapper, embeds, P)
    with torch.no_grad():
        hid = wrapper(embeds)
    out.mkdir(parents=True, exist_ok=True)
    hid.contiguous().numpy().tofile(out / "te_hidden.raw")             # fp16 (1,P,H)
    emb = hid[0, drop_idx:drop_idx + real_len]
    emb = torch.cat([emb, torch.zeros(prefix_len - real_len, emb.shape[-1],
                                      dtype=torch.float16)], 0)        # 右补精确零
    emb.reshape(1, prefix_len, -1).contiguous().numpy().tofile(out / "prompt_embeds_te.raw")
    (out / "prompt_valid_len.txt").write_text(f"{real_len}\n")
    print(f"[ref] drop_idx={drop_idx} total={total} valid={real_len} "
          f"-> padded {prefix_len}（图长 P={P}）")
    surf = [proc.tokenizer.decode([int(i)]) for i in ids[drop_idx:drop_idx + real_len]]
    print(f"[ref] 尾部 6 个 surface: {surf[-6:]}")
    print(f"[ref] hidden absmax={hid.abs().max():.4f} rms={hid.float().pow(2).mean().sqrt():.4f}")
    if not ref_embeds.exists():
        print(f"[ref] 没有 {ref_embeds}，跳过对拍（先跑 encode_prompt_real.py --out-dir {ref_embeds.parent}）")
        return
    import numpy as np
    a = np.fromfile(out / "prompt_embeds_te.raw", dtype=np.float16).astype(np.float32)
    b = np.fromfile(ref_embeds, dtype=np.float16).astype(np.float32)
    if a.shape != b.shape:
        print(f"[ref] 形状不符，跳过：本图 {a.shape} vs 参考 {b.shape}")
        return
    cos = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    print(f"[ref] vs {ref_embeds.name}（bf16 参考）cos={cos:.6f} "
          f"maxabs={float(np.abs(a - b).max()):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-len", type=int, default=64, help="DiT 侧消费的文本段长度")
    ap.add_argument("--tiny", type=int, default=0, metavar="N",
                    help="只用 N 层随机权重看图结构（产物带 _tiny 后缀，不读 checkpoint）")
    ap.add_argument("--prompt", default=None, help="配合 --emit-ref：真实 prompt")
    ap.add_argument("--prompt-ids", default=None,
                    help="配合 --emit-ref：直接给完整的模板+prompt token id（逗号分隔），"
                         "用来逐位复现 encode_prompt_real.py 某一次的产物")
    ap.add_argument("--emit-ref", metavar="DIR", default=None,
                    help="额外用 fp16 eager 跑 --prompt/--prompt-ids，把 embeds 写进 DIR 并与参考对拍")
    ap.add_argument("--ref-embeds", default=str(HERE / "ref_real" / "prompt_embeds.raw"),
                    help="encode_prompt_real.py 的产物，用于 cos 对拍")
    ap.add_argument("--out", default=str(HERE), help="输出目录")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    tc = text_config(args.tiny)
    print(f"[config] layers={tc.num_hidden_layers} hidden={tc.hidden_size} "
          f"heads={tc.num_attention_heads} kv={tc.num_key_value_heads} "
          f"head_dim={tc.head_dim} vocab={tc.vocab_size} rms_eps={tc.rms_norm_eps}")
    model = build_text_model(tc)
    if args.tiny:
        torch.manual_seed(0)
        with torch.no_grad():
            for _, p in model.named_parameters():
                if p.ndim == 1 and p.numel() in (tc.head_dim, tc.hidden_size):
                    p.normal_(1.0, 0.05)      # RMSNorm 权重：均值 1，避免整层输出被压成 0
                else:
                    p.normal_(0.0, 0.02)
        print(f"[tiny] {args.tiny} 层随机权重（seed=0），没读 checkpoint")
    else:
        load_text_tower(model)

    proc = Qwen3VLProcessor.from_pretrained(str(MS_ROOT / "processor"))
    # 序列长度 = system 段 + prefix_len。drop_idx 从 processor 现算，和 encode_prompt_real
    # 同源（写死一个数会在换 tokenizer 时静默错位）。tiny 模式下照抄，只为看形状。
    drop_idx = drop_idx_of(proc)
    P = drop_idx + args.prefix_len
    print(f"[plan] drop_idx={drop_idx} prefix_len={args.prefix_len} -> 图长 P={P}")

    check_rope_tables(model.rotary_emb, P, torch.float16)
    wrapper = TextTowerOnnx(model, P)
    with torch.no_grad():
        probe = wrapper(*make_inputs(P, tc.hidden_size))
    # trace 之前先 eager 跑一遍：层循环里任何 shape/广播错都该在这里以异常形式炸，
    # 而不是被导出器折成一张形状对、数值错的图。
    print(f"[eager] {OUT_NAME} {tuple(probe.shape)} {probe.dtype} "
          f"finite={bool(torch.isfinite(probe).all())} absmax={probe.abs().max():.4f}")
    assert probe.shape == (1, P, tc.hidden_size) and torch.isfinite(probe).all()

    if args.emit_ref:
        assert not args.tiny, "--emit-ref 要用真权重，tiny 的随机权重对拍不出任何东西"
        if args.prompt_ids:
            ids = torch.tensor([int(x) for x in args.prompt_ids.split(",")], dtype=torch.long)
        else:
            assert args.prompt is not None, "--emit-ref 需要 --prompt 或 --prompt-ids"
            enc = tokenize_prompt(proc, args.prompt)
            total = int(enc.attention_mask[0].sum())
            # batch=1 时 padding 不生效；真出现左填充的话，[drop_idx:] 的切法和位置编号都会
            # 错位，宁可在这里报错，也不要产出"看起来对"的 embeds。
            assert total == enc.input_ids.shape[1], \
                f"分词带 {enc.input_ids.shape[1] - total} 个 padding，本脚本按无 padding 处理"
            ids = enc.input_ids[0]
        emit_ref(model, wrapper, proc, ids, args.prefix_len, drop_idx,
                 Path(args.emit_ref).resolve(), Path(args.ref_embeds))

    suffix = "_tiny" if args.tiny else ""
    out_dir = Path(args.out).resolve()            # resolve 必须在 chdir 之前，否则相对 --out 失效
    path, loc = out_dir / f"text_encoder{suffix}.onnx", f"text_encoder{suffix}.weights.bin"
    # legacy 导出器按**相对 CWD** 的名字写散权重，合并也按同名读回去 —— 整段必须在 out_dir 里跑。
    prev_cwd = os.getcwd()
    os.chdir(out_dir)
    try:
        print(f"[trace] {path.name}: {len(IN_NAMES)} 输入 / 1 输出，{tc.num_hidden_layers} 层，P={P}")
        t0 = time.time()
        with torch.no_grad():
            torch.onnx.export(wrapper, make_inputs(P, tc.hidden_size), path.name,
                              input_names=IN_NAMES, output_names=[OUT_NAME],
                              opset_version=OPSET, do_constant_folding=True,
                              dynamo=False)       # 同 llm/DiT：legacy 导出器，避开 dynamo guard
        print(f"[trace] 用时 {time.time() - t0:.0f}s")
        print("[merge] 散权重 -> 单个 external data 文件")
        m = onnx.load(path.name, load_external_data=False)
        merge_scattered(m, loc)
        onnx.save(m, path.name)
        graph_check(m, path, loc, P, tc.hidden_size, tc.num_hidden_layers)
    finally:
        os.chdir(prev_cwd)
    size = (out_dir / loc).stat().st_size
    print(f"\n[done] {path}\n[done] {out_dir / loc}（{size / 1e9:.2f} GB）")
    print("[next] 转 vkopbin：>13 GB 的图必须跳过 optimizer（见 convert_dit_to_vkop.py），"
          "然后按 image/exporter/BASELINE.md 的口径与 ORT 对拍")


if __name__ == "__main__":
    main()
