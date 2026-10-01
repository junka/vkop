#!/usr/bin/env python3
"""导出 Phi-4-mini-instruct (Phi3ForCausalLM) 的 LLM 到 ONNX。

与 qwen3_export_onnx.py 保持**完全相同的张量 I/O 契约**，这样 C++ 驱动
(llm_chat) 不用为第二个架构改加载逻辑：

  入：inputs_embeds      (B, q, HIDDEN)   fp16
      position_ids       (B, q)           int64
      attention_bias     (B, 1, q, kv)    fp16      # 加法 bias，仅 causal
      past_key_values_{0..NLAYERS-1}       (B, 2, NKV, kv, HD)   fp16
  出：logits             (B, q, VOCAB)   fp16
      present_key_values_{0..NLAYERS-1}    (B, 2, NKV, kv, HD)   fp16

与 Qwen3 的三处架构差异（都已逐 op 贴 HF modeling_phi3 实现）：
  1. fused 投影：qkv_proj 一个 MatMul 出 q/k/v，gate_up_proj 一个 MatMul 出
     gate/up。这里不解融合权重，直接照 HF 用 slice / chunk 拆**输出**，
     图上只多 Split/Slice 节点，数值路径与 HF 一字不差。
  2. partial RoPE：rotary_dim = head_dim * partial_rotary_factor = 128*0.75 = 96，
     后 32 维不参与旋转直通。HF 的 apply_rotary_pos_emb 用 cos.shape[-1] 反推
     rotary_dim 再 Slice+Concat，本脚本复用同一个函数。
  3. longrope：Phi3RotaryEmbedding 把 attention_scaling（此处 1.190238）乘进
     cos/sin，所以**必须**复用 HF 的 rotary_emb 而不是手写 RoPE —— 自己算会
     漏掉这个因子，注意力分数差 1.190238^2 = 1.4167 倍。

上下文限制：rotary_emb 的 longrope 分支按 `torch.max(position_ids)+1 > 4096`
（original_max_position_embeddings）在 short_factor / long_factor 之间切换。这是一
个数据相关分支，trace 时就被固化成导出时走的那一支。Phi-4-mini 的 short_factor
全是 1.0（等价于普通 RoPE），本脚本在导出前显式校验这一点，并只支持
position_ids < 4096 的场景；超过 4096 需要重导出（改走 long_factor）。

用法:
  python3 phi4_export_onnx.py                      # 默认导出到 llm/exporter/text_phi4/
  MODEL_PATH=/path/to/Phi-4-mini-instruct EXPORT_PATH=llm.onnx \
    python3 phi4_export_onnx.py
"""

import os
import sys
import torch

MODEL_PATH = os.environ.get("MODEL_PATH") or os.path.join(
    os.path.expanduser("~"), ".cache/modelscope/models/"
    "LLM-Research--Phi-4-mini-instruct/snapshots/master")

# 产物目录：与 Qwen 的 text_qwen3/ 平级，避免两套模型的 llm.onnx 互相覆盖
_DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "text_phi4")
EXPORT_PATH = os.environ.get("EXPORT_PATH") or os.path.join(_DEFAULT_DIR, "llm.onnx")
WEIGHTS_BIN = os.path.basename(EXPORT_PATH).replace(".onnx", ".weights.bin")

print(f"[load] model = {MODEL_PATH}")

from transformers.models.phi3.modeling_phi3 import (  # noqa: E402
    Phi3ForCausalLM,
    apply_rotary_pos_emb,
    repeat_kv,
)

OPSET = 17
# ONNX 里位置编码的上限：超过它 HF 会换 long_factor，导出的图就不成立了。
SHORT_FACTOR_MAX_CTX = 4096

model = Phi3ForCausalLM.from_pretrained(
    MODEL_PATH,
    attn_implementation="eager",
    torch_dtype=torch.float16,
    local_files_only=os.path.isdir(os.path.expanduser(MODEL_PATH)),
)
model.eval()

config = model.config
NLAYERS = config.num_hidden_layers          # 32
HIDDEN = config.hidden_size                 # 3072
NUM_HEADS = config.num_attention_heads      # 24
NUM_KV_HEADS = config.num_key_value_heads   # 8
HEAD_DIM = getattr(config, "head_dim", None) or HIDDEN // NUM_HEADS   # 128
NUM_KV_GROUPS = NUM_HEADS // NUM_KV_HEADS   # 3
INTERMEDIATE = config.intermediate_size     # 8192
VOCAB = config.vocab_size                   # 200064
PARTIAL_ROPE = config.rope_parameters.get("partial_rotary_factor", 1.0)  # 0.75
ROTARY_DIM = int(HEAD_DIM * PARTIAL_ROPE)   # 96
# attention 的 1/sqrt(head_dim)；longrope 的 attention_scaling 在 cos/sin 里，
# 不在这里，两者不能合并（合并就和 HF 不是同一条数值路径了）。
SCALING = model.model.layers[0].self_attn.scaling

print(f"[config] layers={NLAYERS} hidden={HIDDEN} heads={NUM_HEADS} "
      f"kv_heads={NUM_KV_HEADS} head_dim={HEAD_DIM} rotary_dim={ROTARY_DIM} "
      f"scaling={SCALING:.6f} vocab={VOCAB}")

# ---------------------------------------------------------------------------
# 导出前置校验：把「只有 short_factor 这一支被固化进图」这件事变成硬约束
# ---------------------------------------------------------------------------
rotary = model.model.rotary_emb
assert ROTARY_DIM % 2 == 0, f"rotary_dim {ROTARY_DIM} 必须是偶数（半切旋转）"
assert rotary.rope_type == "longrope", f"预期 longrope，得到 {rotary.rope_type}"
short_factor = config.rope_parameters["short_factor"]
assert all(abs(f - 1.0) < 1e-12 for f in short_factor), (
    "short_factor 非全 1：导出的 inv_freq 与默认 RoPE 不等价，"
    "需要把 short/long 两支都验一遍再放行")
_manual_inv = 1.0 / (config.rope_parameters["rope_theta"] **
                     (torch.arange(0, ROTARY_DIM, 2, dtype=torch.float32) / ROTARY_DIM))
assert torch.allclose(rotary.inv_freq.detach().float().cpu(), _manual_inv, atol=1e-9), \
    "rotary.inv_freq 与默认（short_factor=1）inv_freq 不一致，trace 走的是 long 分支"
print(f"[rope] short 分支已确认：inv_freq == 默认，attention_scaling="
      f"{rotary.attention_scaling:.9f}（会乘进 cos/sin）")

lm = model.model
embed_tokens = lm.embed_tokens
layers = lm.layers
norm = lm.norm
rotary_emb = lm.rotary_emb
lm_head = model.lm_head


class Phi4LLMOnnx(torch.nn.Module):
    """手写 forward：显式 KV cache 张量 I/O + 加法 attention bias，绕开 HF 的
    DynamicCache 和 create_sliding_window_causal_mask（内部 torch.diff，opset17
    trace 不过去）。子模块全部复用 HF 的，保证权重与算子顺序一致。"""

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
        self.q_size = NUM_HEADS * HEAD_DIM           # 3072
        self.kv_size = NUM_KV_HEADS * HEAD_DIM       # 1024
        self.inter = INTERMEDIATE                    # 8192

    def decoder_layer(self, layer, hidden, cos, sin, attention_bias, past_kv):
        """单层前向，逐 op 对应 Phi3DecoderLayer.forward。返回 (hidden, present_kv)。"""
        B, q_len, _ = hidden.shape
        residual = hidden
        h = layer.input_layernorm(hidden)

        attn = layer.self_attn
        # HF: qkv = qkv_proj(h)，再按 [q_size, kv_size, kv_size] 连续切片
        qkv = attn.qkv_proj(h)
        q = qkv[..., :self.q_size].view(B, q_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = qkv[..., self.q_size:self.q_size + self.kv_size]
        k = k.view(B, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = qkv[..., self.q_size + self.kv_size:]
        v = v.view(B, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # partial RoPE：cos/sin (B, q, rotary_dim)，apply_rotary_pos_emb 内部
        # unsqueeze(1) 广播到 (B, heads, q, rotary_dim)，再 Slice+Concat 拼回 128 维
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # concat past KV: past_kv (B, 2, nkv, past, hd)
        past_k = past_kv[:, 0]
        past_v = past_kv[:, 1]
        k_new = torch.cat([past_k, k], dim=2)
        v_new = torch.cat([past_v, v], dim=2)

        k_r = repeat_kv(k_new, self.num_kv_groups)
        v_r = repeat_kv(v_new, self.num_kv_groups)

        # attention（手写，纯标准 ONNX op；与 eager_attention_forward 同序）
        attn_w = torch.matmul(q, k_r.transpose(2, 3)) * self.scaling
        attn_w = attn_w + attention_bias
        attn_w = torch.nn.functional.softmax(
            attn_w, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn_w, v_r)
        out = out.transpose(1, 2).reshape(B, q_len, self.num_heads * self.head_dim)
        out = attn.o_proj(out)

        # HF 这里过 resid_attn_dropout，eval 下是恒等，直接加
        hidden = residual + out

        residual = hidden
        h = layer.post_attention_layernorm(hidden)
        mlp = layer.mlp
        # HF: gate_up_proj 一次出 2*inter，chunk(2, -1) 拆成 gate / up。这里等价地
        # 用常量边界切片：chunk 会被 trace 成 Shape→Gather→Add→Div→Mul 的**整数**
        # shape 链，而转换器的 fuse_elemwise_chain 会把链尾的 Div 融成
        # FUSED_ELEMWISE —— 那个内核只有 fp16/fp32 变体，融掉之后 /Slice 的 ends
        # 不再是 int64 张量，runtime 的 as_tensor<int64_t> 空转一下就是 SIGSEGV
        # （实测崩在 Tensor<long long>::copyToCPU，this=0）。常量切片既与 HF 逐位
        # 等价，又少掉每层 4 个动态 shape 节点。
        up_states = mlp.gate_up_proj(h)
        gate = up_states[..., :self.inter]
        up_states = up_states[..., self.inter:]
        hidden = residual + mlp.down_proj(up_states * mlp.activation_fn(gate))

        present_kv = torch.stack([k_new, v_new], dim=1)
        return hidden, present_kv

    def forward(self, inputs_embeds, position_ids, attention_bias, *past_kvs):
        hidden = inputs_embeds
        # 复用 HF rotary_emb：cos/sin = (B, q, rotary_dim)，且已乘 longrope 的
        # attention_scaling。它内部按 max(position_ids) 选 short/long factor，是
        # 数据相关分支 → trace 固化；上面已校验当前走的是 short 分支。
        cos, sin = self.rotary_emb(hidden, position_ids)

        presents = []
        for idx, layer in enumerate(self.layers):
            hidden, pk_new = self.decoder_layer(
                layer, hidden, cos, sin, attention_bias, past_kvs[idx])
            presents.append(pk_new)

        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        return (logits, *presents)


wrapper = Phi4LLMOnnx()
wrapper.eval()

# ---------------------------------------------------------------------------
# 构造 prefill dummy inputs
# ---------------------------------------------------------------------------
B = 1
L = int(os.environ.get("PREFILL_LEN", "32"))
q_len = kv_len = L
assert L <= SHORT_FACTOR_MAX_CTX, f"导出长度 {L} 超过 short 分支上限 {SHORT_FACTOR_MAX_CTX}"

inputs_embeds = torch.randn(B, q_len, HIDDEN, dtype=torch.float16)
position_ids = torch.arange(L, dtype=torch.long).unsqueeze(0).expand(B, -1)
# causal mask：上三角填 finfo(fp16).min，与 HF create_causal_mask 一致；不用 -inf
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

os.makedirs(os.path.dirname(os.path.abspath(EXPORT_PATH)), exist_ok=True)

print(f"[export] Phi4-onnx: {NLAYERS} layers, {len(input_names)} inputs / "
      f"{len(output_names)} outputs, opset={OPSET} → {EXPORT_PATH} ...")

with torch.no_grad():
    torch.onnx.export(
        wrapper, inputs, EXPORT_PATH,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,  # legacy TorchScript 导出器：dynamo 在 longrope 的
                       # max(position_ids) 数据相关分支上会 trigger Guard
    )
print(f"[✓] exported → {EXPORT_PATH}")

# ---------------------------------------------------------------------------
# 合并散权重（>2GB 一定触发 external data）
# ---------------------------------------------------------------------------
import onnx  # noqa: E402

out_dir = os.path.dirname(os.path.abspath(EXPORT_PATH))
print("[consolidate] checking external data ...")
m = onnx.load(EXPORT_PATH, load_external_data=True)
n_init = len(m.graph.initializer)
n_ext = sum(1 for t in m.graph.initializer
            if t.HasField("data_location") and t.data_location == 1)
print(f"[consolidate] initializers={n_init} external={n_ext}")

# lm_head 与 embed_tokens 共享权重（tie_word_embeddings=true），图里只有一份 MatMul
# 权重；embed_tokens.bin 由 dump_embed_tokens_phi4.py 单独给 C++ 驱动用。
onnx.save_model(
    m, EXPORT_PATH,
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location=WEIGHTS_BIN,
    convert_attribute=True,
)

removed = 0
for f in os.listdir(out_dir):
    if f == WEIGHTS_BIN or not (f.endswith(".weight") or f.startswith("onnx__MatMul_")):
        continue
    os.remove(os.path.join(out_dir, f))
    removed += 1
print(f"[✓] consolidated into {WEIGHTS_BIN}, removed {removed} scattered files")

m2 = onnx.load(EXPORT_PATH, load_external_data=False)
locs = set()
for t in m2.graph.initializer:
    if t.HasField("data_location") and t.data_location == 1:
        for kv in t.external_data:
            if kv.key == "location":
                locs.add(kv.value)
print(f"[verify] external locations: {locs} (expect {{'{WEIGHTS_BIN}'}})")
assert locs == {WEIGHTS_BIN}, f"unexpected external locations: {locs}"

print(f"\ndone. {EXPORT_PATH} ({os.path.getsize(EXPORT_PATH)/1e6:.1f} MB)")
print("下一步：导出 embed_tokens 权重表")
print("  MODEL_PATH=", MODEL_PATH)
print("  python3 llm/exporter/dump_embed_tokens_phi4.py "
      f"{os.path.join(out_dir, 'embed_tokens.bin')}")
