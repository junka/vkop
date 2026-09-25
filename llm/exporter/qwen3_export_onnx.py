"""导出纯文本 Qwen3 系列 (Qwen3ForCausalLM) 的 LLM 到 ONNX。

与 qwen3vl_export_onnx.py (Qwen3-VL 多模态) 的区别：
  · 无视觉塔，只导出一个 llm.onnx（纯 decoder-only LM）
  · 无 deepstack 注入 / image_pad_mask / visual.onnx
  · position_ids 用标准 2D RoPE (B, q)，不是 3D MRoPE (3, B, q)
  · attention_bias 只有 causal mask（加法 bias）

支持的模型（从 ModelScope / HuggingFace 路径加载均可）：
  · Qwen/Qwen3-4B-Instruct
  · Qwen/Qwen3-8B-Instruct
  · Qwen/Qwen3-32B-Instruct

I/O 契约（llm.onnx）：
  入：inputs_embeds      (B, q, HIDDEN)   fp16
      position_ids       (B, q)          int64
      attention_bias     (B, 1, q, kv)   fp16      # 加法 bias，仅 causal
      past_key_values_{0..NLAYERS-1}    (B, 2, NKV, kv, HD)   fp16
  出：logits             (B, q, VOCAB)   fp16
      present_key_values_{0..NLAYERS-1} (B, 2, NKV, kv, HD)   fp16

用法:
  python3 qwen3_export_onnx.py                 # 默认导出 Qwen3-8B
  MODEL_PATH=~/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Instruct \
    python3 qwen3_export_onnx.py               # 指定模型路径
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 模型路径：环境变量 MODEL_PATH 优先，否则尝试 ModelScope / HF 常见位置
# ---------------------------------------------------------------------------
DEFAULT_PATH = "Qwen/Qwen3-4B-Instruct-2507"

MODEL_PATH = os.environ.get("MODEL_PATH") or DEFAULT_PATH

print(f"[load] model = {MODEL_PATH}")

# 根据模型类名选择对应的 Qwen3 文本模型（自动适配 HF transformers 版本）
# Qwen3ForCausalLM 在 transformers.models.qwen3.modeling_qwen3
# 它的 rotary_emb 实现是标准 RoPE（2D position_ids）
try:
    from transformers import Qwen3ForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import (
        apply_rotary_pos_emb,
        repeat_kv,
    )
except ImportError:
    # 兼容 transformers < 某个版本的 fallback 路径（部分版本用 qwen2.5 路径）
    from transformers import AutoModelForCausalLM
    Qwen3ForCausalLM = None
    print("[warn] Qwen3ForCausalLM not found, falling back to AutoModelForCausalLM")

OPSET = 17

# ---------------------------------------------------------------------------
# 加载模型（fp16，eager attention —— 绕开 FlashAttention / SDPA 等不可 trace 实现）
# ---------------------------------------------------------------------------
if Qwen3ForCausalLM is not None:
    model = Qwen3ForCausalLM.from_pretrained(
        MODEL_PATH,
        attn_implementation="eager",
        torch_dtype=torch.float16,
        local_files_only=os.path.isdir(os.path.expanduser(MODEL_PATH)),
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        attn_implementation="eager",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=os.path.isdir(os.path.expanduser(MODEL_PATH)),
    )
model.eval()

# ---------------------------------------------------------------------------
# 从 config 读取架构参数 —— 全部动态，不硬编码
# ---------------------------------------------------------------------------
config = model.config
NLAYERS = config.num_hidden_layers          # e.g. 32 (8B), 36 (4B), 28 (2B)
HIDDEN = config.hidden_size                 # e.g. 4096 (8B), 2560 (4B)
NUM_HEADS = config.num_attention_heads      # e.g. 32 (8B), 20 (4B)
NUM_KV_HEADS = config.num_key_value_heads   # e.g. 8 (8B), 10 (4B)
HEAD_DIM = config.head_dim or HIDDEN // NUM_HEADS
NUM_KV_GROUPS = NUM_HEADS // NUM_KV_HEADS
VOCAB = config.vocab_size
# scaling: 从第一层 attention 读，对齐 HF 实际值
SCALING = model.model.layers[0].self_attn.scaling

print(f"[config] layers={NLAYERS} hidden={HIDDEN} heads={NUM_HEADS} "
      f"kv_heads={NUM_KV_HEADS} head_dim={HEAD_DIM} scaling={SCALING:.6f} "
      f"vocab={VOCAB}")

lm = model.model  # Qwen3Model
embed_tokens = lm.embed_tokens
layers = lm.layers
norm = lm.norm
rotary_emb = lm.rotary_emb
lm_head = model.lm_head

# ---------------------------------------------------------------------------
# 自定义 Qwen3 LLM ONNX wrapper —— 复用 HF 层权重，手写 forward 层循环
# 绕开 HF 内部不可 trace 的部分：
#   · Cache 对象（past_key_values 动态 list append）
#   · create_causal_mask（内部 torch.diff，opset17 不支持）
# 把 KV cache 和 attention mask 暴露为显式张量 I/O
# ---------------------------------------------------------------------------
class Qwen3LLMOnnx(nn.Module):
    """纯 decoder-only wrapper，每层手写 Q/K/V proj + q/k_norm + RoPE +
    GQA + attention + MLP；KV 显式传递。"""

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
        """单层前向。返回 (hidden, present_kv)。"""
        B, q_len, _ = hidden.shape
        residual = hidden
        h = layer.input_layernorm(hidden)

        attn = layer.self_attn
        # HF 顺序（严格对齐 modeling_qwen3.py 的实现）：
        #   q = q_norm(q_proj(h).view(B,q,heads,hd)).transpose(1,2)
        #   k = k_norm(k_proj(h).view(B,q,nkv,hd)).transpose(1,2)
        #   v = v_proj(h).view(B,q,nkv,hd).transpose(1,2)       # v 无 norm
        hidden_shape = (B, q_len, self.num_heads, self.head_dim)
        q = attn.q_proj(h).view(hidden_shape)
        q = attn.q_norm(q)
        q = q.transpose(1, 2)

        kv_shape = (B, q_len, self.num_kv_heads, self.head_dim)
        k = attn.k_norm(attn.k_proj(h).view(kv_shape))
        k = k.transpose(1, 2)
        v = attn.v_proj(h).view(kv_shape).transpose(1, 2)

        # RoPE: cos/sin 形状 (B, q, head_dim)；apply_rotary_pos_emb 内部
        # 会 unsqueeze(1) 广播到 q/k 的 (B, heads, q, hd)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # concat past KV: past_kv (B,2,nkv,past,hd)
        past_k = past_kv[:, 0]
        past_v = past_kv[:, 1]
        k_new = torch.cat([past_k, k], dim=2)
        v_new = torch.cat([past_v, v], dim=2)

        # GQA repeat
        k_r = repeat_kv(k_new, self.num_kv_groups)
        v_r = repeat_kv(v_new, self.num_kv_groups)

        # attention（手写，纯标准 ONNX op）
        attn_w = torch.matmul(q, k_r.transpose(2, 3)) * self.scaling
        attn_w = attn_w + attention_bias
        attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn_w, v_r)
        out = out.transpose(1, 2).reshape(B, q_len, self.num_heads * self.head_dim)
        out = attn.o_proj(out)

        hidden = residual + out

        # MLP (gate/up/down)
        residual = hidden
        h = layer.post_attention_layernorm(hidden)
        mlp = layer.mlp
        h = mlp.down_proj(mlp.act_fn(mlp.gate_proj(h)) * mlp.up_proj(h))
        hidden = residual + h

        present_kv = torch.stack([k_new, v_new], dim=1)
        return hidden, present_kv

    def forward(self, inputs_embeds, position_ids, attention_bias, *past_kvs):
        # 复用 HF rotary_emb —— 输入 position_ids (B, q) 得到 cos/sin (B, q, hd)
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


wrapper = Qwen3LLMOnnx()
wrapper.eval()

# ---------------------------------------------------------------------------
# 构造 prefill dummy inputs（纯文本，无视觉相关 I/O）
# ---------------------------------------------------------------------------
B = 1
L = 32          # prefill 序列长度（随便，只要 < 模型训练长度）
q_len = L
kv_len = L      # prefill 时 kv_len = q_len，无 past

inputs_embeds = torch.randn(B, q_len, HIDDEN, dtype=torch.float16)
position_ids = torch.arange(L, dtype=torch.long).unsqueeze(0).expand(B, -1)
# causal mask: 上三角 = finfo(float16).min (-65504)，与 HF create_causal_mask 一致
# 不用 -inf：fp16 softmax 需要微小区分度
causal = torch.triu(
    torch.full((q_len, kv_len), torch.finfo(torch.float16).min, dtype=torch.float16),
    diagonal=1)
attention_bias = causal.unsqueeze(0).unsqueeze(0)  # (1, 1, q, kv)

# past_key_values: NLAYERS 个空张量 (B, 2, NKV, 0, HD)
past_kvs = tuple(
    torch.zeros(B, 2, NUM_KV_HEADS, 0, HEAD_DIM, dtype=torch.float16)
    for _ in range(NLAYERS))

inputs = (inputs_embeds, position_ids, attention_bias, *past_kvs)

# ---------------------------------------------------------------------------
# ONNX 导出
# ---------------------------------------------------------------------------
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

print(f"[export] Qwen3-onnx: {NLAYERS} layers, {len(input_names)} inputs / "
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
        dynamo=False,  # legacy TorchScript 导出器；dynamo 对 RoPE 的数据依赖
                        # cos/sin 可能 trigger GuardOnDataDependentSymNode
    )
print(f"[✓] exported → {EXPORT_PATH}")

# ---------------------------------------------------------------------------
# 合并散权重（如果 >2GB 触发 torch external-data）
# ---------------------------------------------------------------------------
import onnx

print("[consolidate] checking external data ...")
m = onnx.load(EXPORT_PATH, load_external_data=True)
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
    # 删除散文件
    removed = 0
    for f in os.listdir("."):
        if f.endswith(".weight") or f.startswith("onnx__MatMul_"):
            if f != "llm.weights.bin":
                os.remove(f)
                removed += 1
    print(f"[✓] consolidated into llm.weights.bin, removed {removed} scattered files")

    # 验证
    m2 = onnx.load(EXPORT_PATH, load_external_data=False)
    locs = set()
    for t in m2.graph.initializer:
        if t.HasField("data_location") and t.data_location == 1:
            for kv in t.external_data:
                if kv.key == "location":
                    locs.add(kv.value)
    print(f"[verify] external locations: {locs} (expect {{'llm.weights.bin'}})")
    assert locs == {"llm.weights.bin"}, f"unexpected external locations: {locs}"
else:
    # 无 external data 但模型可能 >2GB protobuf 限制 —— 强制 external 保存
    print("[consolidate] model has no external data but may exceed 2GB protobuf limit, "
          "saving with external data ...")
    onnx.save_model(
        m, EXPORT_PATH,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="llm.weights.bin",
        convert_attribute=True,
    )
    print("[✓] saved with external data (llm.weights.bin)")

print(f"\ndone. llm.onnx ({os.path.getsize(EXPORT_PATH)/1e6:.1f} MB)")

# embed_tokens 权重表不在 llm.onnx 图里（图输入是 inputs_embeds），需要单独导出：
#   python3 dump_embed_tokens_qwen3.py    (本脚本同目录，自动适配模型路径)
print("\n下一步：导出 embed_tokens 权重表")
print("  MODEL_PATH=", MODEL_PATH)
print("  python3 llm/exporter/dump_embed_tokens_qwen3.py [embed_tokens.bin]")
