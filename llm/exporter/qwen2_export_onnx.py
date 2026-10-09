"""导出纯文本 Qwen2.5 系列 (Qwen2ForCausalLM) 的 LLM 到 ONNX。

qwen3_export_onnx.py 的 Qwen2 版本。区别:
  · Qwen2 没有 q_norm / k_norm (Qwen3 有),attention 更简单
  · Qwen2ForCausalLM 而非 Qwen3ForCausalLM
  · RoPE 仍是标准 2D (B, q),与 Qwen3 文本一致

支持模型 (从 ModelScope / HuggingFace 路径加载):
  · Qwen/Qwen2.5-0.5B-Instruct   (896 hidden, 24 layers, 14 heads, 2 kv heads)
  · Qwen/Qwen2.5-1.5B-Instruct
  · Qwen/Qwen2.5-7B-Instruct

I/O 契约 (llm.onnx):
  入: inputs_embeds      (B, q, HIDDEN)   fp16
      position_ids       (B, q)          int64
      attention_bias     (B, 1, q, kv)   fp16      # 加法 bias,仅 causal
      past_key_values_{0..NLAYERS-1}    (B, 2, NKV, kv, HD)   fp16
  出: logits             (B, q, VOCAB)   fp16
      present_key_values_{0..NLAYERS-1} (B, 2, NKV, kv, HD)   fp16

用法:
  python3 qwen2_export_onnx.py                 # 默认 Qwen2.5-0.5B
  MODEL_PATH=... python3 qwen2_export_onnx.py  # 指定模型
  VKOP_KV_FP8=1 python3 qwen2_export_onnx.py   # fp8 KV cache (同 qwen3)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_PATH = os.environ.get("MODEL_PATH") or DEFAULT_PATH
print(f"[load] model = {MODEL_PATH}")

from transformers import Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import (
    apply_rotary_pos_emb,
    repeat_kv,
)

OPSET = 17

model = Qwen2ForCausalLM.from_pretrained(
    MODEL_PATH,
    attn_implementation="eager",
    torch_dtype=torch.float16,
    local_files_only=os.path.isdir(os.path.expanduser(MODEL_PATH)),
)
model.eval()

config = model.config
NLAYERS = config.num_hidden_layers
HIDDEN = config.hidden_size
NUM_HEADS = config.num_attention_heads
NUM_KV_HEADS = config.num_key_value_heads
# Qwen2Config has no head_dim field (Qwen3 does); derive from hidden/heads.
HEAD_DIM = getattr(config, "head_dim", None) or HIDDEN // NUM_HEADS
NUM_KV_GROUPS = NUM_HEADS // NUM_KV_HEADS
VOCAB = config.vocab_size
SCALING = model.model.layers[0].self_attn.scaling

print(f"[config] layers={NLAYERS} hidden={HIDDEN} heads={NUM_HEADS} "
      f"kv_heads={NUM_KV_HEADS} head_dim={HEAD_DIM} scaling={SCALING:.6f} "
      f"vocab={VOCAB}")

lm = model.model
embed_tokens = lm.embed_tokens
layers = lm.layers
norm = lm.norm
rotary_emb = lm.rotary_emb
lm_head = model.lm_head


class Qwen2LLMOnnx(nn.Module):
    """纯 decoder-only wrapper,每层手写 Q/K/V proj + RoPE + GQA + attention + MLP;
    KV 显式传递。Qwen2 无 q_norm/k_norm。"""

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

    def decoder_layer(self, layer, hidden, cos, sin, attention_bias, past_kv):
        B, q_len, _ = hidden.shape
        residual = hidden
        h = layer.input_layernorm(hidden)

        attn = layer.self_attn
        # Qwen2: no q_norm/k_norm (unlike Qwen3).
        hidden_shape = (B, q_len, self.num_heads, self.head_dim)
        q = attn.q_proj(h).view(hidden_shape).transpose(1, 2)

        kv_shape = (B, q_len, self.num_kv_heads, self.head_dim)
        k = attn.k_proj(h).view(kv_shape).transpose(1, 2)
        v = attn.v_proj(h).view(kv_shape).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        past_k = past_kv[:, 0]
        past_v = past_kv[:, 1]
        k_new = torch.cat([past_k, k], dim=2)
        v_new = torch.cat([past_v, v], dim=2)

        k_r = repeat_kv(k_new, self.num_kv_groups)
        v_r = repeat_kv(v_new, self.num_kv_groups)

        attn_w = torch.matmul(q, k_r.transpose(2, 3)) * self.scaling
        attn_w = attn_w + attention_bias
        attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn_w, v_r)
        out = out.transpose(1, 2).reshape(B, q_len, self.num_heads * self.head_dim)
        out = attn.o_proj(out)

        hidden = residual + out

        residual = hidden
        h = layer.post_attention_layernorm(hidden)
        mlp = layer.mlp
        h = mlp.down_proj(mlp.act_fn(mlp.gate_proj(h)) * mlp.up_proj(h))
        hidden = residual + h

        present_kv = torch.stack([k_new, v_new], dim=1)
        return hidden, present_kv

    def forward(self, inputs_embeds, position_ids, attention_bias, *past_kvs):
        hidden = inputs_embeds
        cos, sin = self.rotary_emb(hidden, position_ids)

        presents = []
        for idx, layer in enumerate(self.layers):
            hidden, pk_new = self.decoder_layer(
                layer, hidden, cos, sin, attention_bias, past_kvs[idx])
            presents.append(pk_new)

        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        return (logits, *presents)


wrapper = Qwen2LLMOnnx()
wrapper.eval()

B = 1
L = 32
q_len = L
kv_len = L

inputs_embeds = torch.randn(B, q_len, HIDDEN, dtype=torch.float16)
position_ids = torch.arange(L, dtype=torch.long).unsqueeze(0).expand(B, -1)
causal = torch.triu(
    torch.full((q_len, kv_len), torch.finfo(torch.float16).min, dtype=torch.float16),
    diagonal=1)
attention_bias = causal.unsqueeze(0).unsqueeze(0)

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

print(f"[export] Qwen2-onnx: {NLAYERS} layers, {len(input_names)} inputs / "
      f"{len(output_names)} outputs, opset={OPSET} ...")

EXPORT_PATH = os.environ.get("EXPORT_PATH", "llm.onnx")

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
print(f"[ok] exported -> {EXPORT_PATH}")

import onnx
from onnx import helper, TensorProto

print("[consolidate] checking external data ...")
m = onnx.load(EXPORT_PATH, load_external_data=True)


def insert_fp8_kv_cache(model, scale=0.1):
    g = model.graph
    nlayers = 0
    while any(o.name == f"present_key_values_{nlayers}" for o in g.output):
        nlayers += 1
    if nlayers == 0:
        print("[kv-fp8] no present_key_values_* outputs found, skipping")
        return
    fp8_dt = TensorProto.FLOAT8E4M3FN
    scale_name = "kv_cache_fp8_scale"
    scale_init = helper.make_tensor(scale_name, TensorProto.FLOAT, [], [scale])
    g.initializer.append(scale_init)
    for i in range(nlayers):
        past_name = f"past_key_values_{i}"
        pres_name = f"present_key_values_{i}"
        past_shape = None
        for vi in list(g.input):
            if vi.name == past_name:
                past_shape = [
                    d.dim_value if d.HasField("dim_value") else d.dim_param
                    for d in vi.type.tensor_type.shape.dim
                ]
                vi.type.tensor_type.elem_type = fp8_dt
                break
        past_fp16 = f"{past_name}_fp16"
        # Declare the DQ output as FLOAT16. Without a value_info entry, ONNX
        # shape-inference types the DequantizeLinear output as float32 (DQ's
        # default when no output_dtype attribute is present) and propagates
        # fp32 down the entire K/V concat chain. The runtime then builds those
        # Concat/Gather/Expand tensors with fp32 containers, and the fp32
        # word-mover shaders reinterpret the fp16 (and fp8) bytes as 32-bit
        # words -- garbage K/V and NaN attention. Forcing the bridge to fp16
        # makes shape-inference propagate fp16 to every consumer. The sibling
        # present_*_fp16 needs no entry: it is the K/V concat output, whose
        # inputs are now correctly fp16.
        g.value_info.append(helper.make_tensor_value_info(
            past_fp16, TensorProto.FLOAT16, past_shape))
        # Rewire consumers BEFORE appending the DQ node: iterating a protobuf
        # repeated field yields fresh Python wrappers, so an `is`-identity skip
        # of the new node does NOT work and the DQ would rewrite its own input
        # into a self-loop (the converter then reports a dependency cycle).
        for node in g.node:
            for k in range(len(node.input)):
                if node.input[k] == past_name:
                    node.input[k] = past_fp16
        dq = helper.make_node(
            "DequantizeLinear",
            inputs=[past_name, scale_name],
            outputs=[past_fp16],
            name=f"DequantizeLinear_kv_{i}",
        )
        g.node.append(dq)
        pres_fp16 = f"{pres_name}_fp16"
        for node in g.node:
            for k in range(len(node.output)):
                if node.output[k] == pres_name:
                    node.output[k] = pres_fp16
        q = helper.make_node(
            "QuantizeLinear",
            inputs=[pres_fp16, scale_name],
            outputs=[pres_name],
            name=f"QuantizeLinear_kv_{i}",
        )
        g.node.append(q)
        for o in g.output:
            if o.name == pres_name:
                o.type.tensor_type.elem_type = fp8_dt
                break
    print(f"[kv-fp8] rewrote {nlayers} layer(s): past/present_key_values "
          f"now E4M3 fp8 (per-tensor scale={scale})")


if os.environ.get("VKOP_KV_FP8") == "1":
    kv_scale = float(os.environ.get("VKOP_KV_SCALE", "0.1"))
    insert_fp8_kv_cache(m, scale=kv_scale)

n_init = len(m.graph.initializer)
n_ext = sum(1 for t in m.graph.initializer
            if t.HasField("data_location") and t.data_location == 1)
print(f"[consolidate] initializers={n_init} external={n_ext}")

if n_ext > 0:
    print("[consolidate] saving as single llm.weights.bin ...")
    onnx.save_model(
        m, EXPORT_PATH,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="llm.weights.bin",
        convert_attribute=True,
    )
    removed = 0
    for f in os.listdir("."):
        if f.endswith(".weight") or f.startswith("onnx__MatMul_"):
            if f != "llm.weights.bin":
                os.remove(f)
                removed += 1
    print(f"[ok] consolidated into llm.weights.bin, removed {removed} scattered files")
else:
    print("[consolidate] no external data; saving with external data to be safe ...")
    onnx.save_model(
        m, EXPORT_PATH,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="llm.weights.bin",
        convert_attribute=True,
    )
    print("[ok] saved with external data (llm.weights.bin)")

print(f"\ndone. llm.onnx ({os.path.getsize(EXPORT_PATH)/1e6:.1f} MB)")
print("\n下一步:导出 embed_tokens 权重表")
print("  MODEL_PATH=", MODEL_PATH)
