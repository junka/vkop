"""导出 Qwen3-VL-2B 的视觉编码器和语言模型到 ONNX。

相比旧版的三处改进：
  1. **权重合并**：llm.onnx 超 2GB 会触发 torch 自动 external-data（一 initializer
     一文件，255 个散文件）。导出后用 onnx.save_model 合并成单个 llm.weights.bin
     并删除散文件。
  2. **deepstack 端到端**：视觉塔在第 5/11/17 层产出 3 个 deepstack 特征，文本模
     型在对应层把它们加到 hidden_states 的 image_pad 位置。visual.onnx 现导出
     pooler_output + 3 个 deepstack_features；llm.onnx 接收 3 个 deepstack_embeds
     + image_pad_mask，内部按层注入。
  3. **LLM 带 KV cache**：自定义 wrapper 把 past_key_values 作为显式输入/输出
     （每层 (B,2,nkv,kv_len,hd)），绕开 HF 的 Cache 对象（不可 trace）与
     create_causal_mask（含 opset17 不支持的 torch.diff），改传预计算的加法
     attention_bias。纯标准 ONNX op，无 TRT plugin。

I/O 契约（llm.onnx）：
  入：inputs_embeds (B,q,2048) fp16
      position_ids  (3,B,q) int64        # MRoPE 的 t/h/w，纯文本时三行相等
      attention_bias (B,1,q,kv) fp16      # 加法 bias，含因果与 padding
      deepstack_embeds_{0,1,2} (n_img,2048) fp16
      image_pad_mask (B,L) bool
      past_key_values_{0..27} (B,2,8,kv,128) fp16
  出：logits (B,q,151936) fp16
      present_key_values_{0..27} (B,2,8,kv,128) fp16
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.vision_utils import (
    get_vision_cu_seqlens,
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)

MODEL_PATH = os.path.expanduser("~/.cache/modelscope/hub/models/Qwen/Qwen3-VL-2B-Instruct")
IMAGE_TOKEN_ID = 151655  # <|image_pad|>
OPSET = 17

print(f"[load] {MODEL_PATH}")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, attn_implementation="eager", torch_dtype=torch.float16)
model.eval()

visual = model.model.visual
lm = model.model.language_model
text_config = model.config.text_config

# 真实参数（从 config 读，不硬编码）。
NUM_LAYERS = text_config.num_hidden_layers          # 28
HIDDEN = text_config.hidden_size                    # 2048
NUM_HEADS = text_config.num_attention_heads         # 16
NUM_KV_HEADS = text_config.num_key_value_heads      # 8
HEAD_DIM = text_config.head_dim or HIDDEN // NUM_HEADS  # 128
NUM_KV_GROUPS = NUM_HEADS // NUM_KV_HEADS           # 2
SCALING = lm.layers[0].self_attn.scaling            # 1/sqrt(head_dim)
VOCAB = text_config.vocab_size                       # 151936
DEEPSTACK_LAYERS = list(visual.deepstack_visual_indexes)  # [5,11,17] 视觉层
# 注意：deepstack 在「文本模型」的注入位置是 layer_idx ∈ range(3) = [0,1,2]，
# 即前 3 个文本层（HF modeling_qwen3_vl.py:81 的 `layer_idx in range(len(deepstack_visual_embeds))`）。
# [5,11,17] 是「视觉塔」产出 deepstack 特征的视觉层号，与文本注入位置不同。
TEXT_DEEPSTACK_LAYERS = list(range(len(DEEPSTACK_LAYERS)))  # [0,1,2]
SPATIAL_MERGE = visual.config.spatial_merge_size    # 2

print(f"[config] layers={NUM_LAYERS} hidden={HIDDEN} heads={NUM_HEADS} kv_heads={NUM_KV_HEADS} "
      f"head_dim={HEAD_DIM} scaling={SCALING:.6f} deepstack={DEEPSTACK_LAYERS}")


# ---------------------------------------------------------------------------
# 工具：把 (n_img, H) 的 deepstack_embed 加到 hidden 的 image_pad 位置。
# 等价 HF _deepstack_process: hidden[mask,:] += embed。opset17 的 ScatterND 无 add
# reduction，故用 gather+add+scatter 覆盖实现。
# ---------------------------------------------------------------------------
def scatter_add_visual(hidden, embed, mask):
    B, L, H = hidden.shape
    flat_mask = mask.reshape(-1)
    idx = torch.nonzero(flat_mask, as_tuple=False).squeeze(-1)  # (n_img,)
    flat = hidden.reshape(-1, H).clone()
    gathered = flat[idx] + embed                                # 在视觉位置相加
    idx2 = idx.unsqueeze(1).expand(-1, H)
    flat = torch.scatter(flat, 0, idx2, gathered)               # 覆盖回原位
    return flat.reshape(B, L, H)


# ---------------------------------------------------------------------------
# 1. 视觉编码器：导出 pooler_output + 3 个 deepstack_features
# ---------------------------------------------------------------------------
class VisualExport(nn.Module):
    """直接调 HF visual forward，返回 (pooler_output, *deepstack_features)。

    HF 视觉 forward 里的三个 grid 预计算（pos_embed 插值索引/权重、rotary
    position_ids、attention 的 cu_seqlens）用了 `torch.repeat_interleave(x, tensor)`
    和 cumsum，legacy 导出器会把它 emit 成 ONNX 控制流（Loop / SequenceEmpty /
    SequenceAt / SequenceInsert / SplitToSequence / ConcatFromSequence / CumSum），
    而 vkop runtime 没有这些算子（且对未知 op 是静默 pass-through）。这三个 helper
    都支持从 kwargs 里取预算好的值——官方给 trace/export 用的逃生口——所以在 trace
    前用 eager 把它们算成常量张量传进去，图上就只剩静态数据流。
    代价：这些常量与 grid_thw 一起被固化，换导出尺寸必须重跑本脚本（本来如此，
    见下面 EXPORT_IMG_SIZE 的说明）。
    """

    def __init__(self, visual, grid_thw):
        super().__init__()
        self.visual = visual
        cfg = visual.config
        interp_indices, interp_weights = get_vision_interpolation_indices_and_weights(
            grid_thw,
            num_grid_per_side=int(cfg.num_position_embeddings ** 0.5),
            mode=visual.interpolation_mode,
            align_corners=visual.interpolation_align_corners,
            spatial_merge_size=cfg.spatial_merge_size,
        )
        self.grid_consts = {
            "interp_indices": interp_indices,
            "interp_weights": interp_weights,
            "position_ids": get_vision_position_ids(grid_thw, cfg.spatial_merge_size),
            "cu_seqlens": get_vision_cu_seqlens(grid_thw),
        }
        print("[visual] precomputed grid consts: " + " ".join(
            f"{k}{tuple(v.shape)}" for k, v in self.grid_consts.items()))

    def forward(self, pixel_values, grid_thw):
        # visual.forward 会 pop 这些 kwargs，每次调用传一份浅拷贝。
        out = self.visual(pixel_values, grid_thw, **dict(self.grid_consts))
        # out.pooler_output: (n_patches, out_hidden)
        # out.deepstack_features: list[3] of (n_patches, out_hidden)
        deepstack = out.deepstack_features
        return out.pooler_output, deepstack[0], deepstack[1], deepstack[2]


patch_size = visual.patch_embed.patch_size
temporal_patch_size = visual.patch_embed.temporal_patch_size
in_chans = visual.patch_embed.in_channels
row = in_chans * temporal_patch_size * patch_size * patch_size
# 导出尺寸从环境变量 EXPORT_IMG_SIZE 读，默认 224×224。注意：visual.onnx 的整张图
# 都是「导出时固化」的——grid_thw 及其派生量（pos_embed 插值索引/权重、rotary
# position_ids、cu_seqlens）全部折成常量，输入输出也不标 dynamic_axes。
# 要换尺寸就改 EXPORT_IMG_SIZE 重新导出一个 visual.onnx，不同尺寸不能共用一个图。
EXPORT_IMG_SIZE = int(os.environ.get("EXPORT_IMG_SIZE", "224"))
height = width = EXPORT_IMG_SIZE
assert height % patch_size == 0 and width % patch_size == 0, \
    f"导出尺寸 {height}×{width} 必须是 patch_size={patch_size} 的整数倍"
grid_t, grid_h, grid_w = 1, height // patch_size, width // patch_size
assert grid_h % SPATIAL_MERGE == 0 and grid_w % SPATIAL_MERGE == 0, \
    f"grid {grid_h}×{grid_w} 必须能被 spatial_merge_size={SPATIAL_MERGE} 整除"
seq_len = grid_t * grid_h * grid_w
grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.int32)
pixel_values = torch.randn(seq_len, row, dtype=visual.dtype)
print(f"[visual] img={height}×{width} patch={patch_size} seq_len={seq_len} "
      f"row={row} grid_thw={grid_thw.tolist()}")

visual_out_names = ["image_features", "deepstack_features_0",
                    "deepstack_features_1", "deepstack_features_2"]

with torch.no_grad():
    torch.onnx.export(
        VisualExport(visual, grid_thw), (pixel_values, grid_thw), "visual.onnx",
        input_names=["pixel_values", "grid_thw"],
        output_names=visual_out_names,
        # 不给 dynamic_axes：grid 相关量已全部固化成常量，标成动态只会骗人
        # ——runtime 侧必须按导出尺寸喂图。grid_thw 仍留作输入位（llm_chat 不绑定它），
        # 以免改动 C++ 侧的输入契约。
        opset_version=OPSET,
        dynamo=False,  # 用 legacy TorchScript 导出器：dynamo 对 visual 的
        # fast_pos_embed_interpolate (torch.linspace 含数据相关长度) 会 guard 失败。
    )
print("[✓] visual.onnx exported (pooler_output + 3 deepstack_features, 无控制流)")

if os.environ.get("EXPORT_VISUAL_ONLY") == "1":
    # 只调视觉时用：llm.onnx 导出要几分钟且会重写 3.4GB 权重文件。
    print("[visual-only] 跳过 llm.onnx 导出")
    raise SystemExit(0)


# ---------------------------------------------------------------------------
# 2. 语言模型：带 KV cache + deepstack 的自定义 wrapper
# ---------------------------------------------------------------------------
class Qwen3VLLMOnnx(nn.Module):
    """复用 HF 层权重，自己写 forward 层循环，绕开 Cache 与 create_causal_mask。

    每层手写 Q/K/V proj + qk_norm + RoPE + concat past KV + GQA repeat +
    标准 attention + MLP；KV 作为显式输入/输出张量。deepstack 在 [5,11,17] 层后
    加到 image_pad 位置。
    """

    def __init__(self, full_model):
        super().__init__()
        self.lm = full_model.model.language_model
        self.embed_tokens = self.lm.embed_tokens
        self.layers = self.lm.layers
        self.norm = self.lm.norm
        self.rotary_emb = self.lm.rotary_emb
        self.lm_head = full_model.lm_head

        self.num_heads = NUM_HEADS
        self.num_kv_heads = NUM_KV_HEADS
        self.num_kv_groups = NUM_KV_GROUPS
        self.head_dim = HEAD_DIM
        self.hidden = HIDDEN
        self.scaling = SCALING
        self.deepstack_layers = TEXT_DEEPSTACK_LAYERS  # 文本注入位置 [0,1,2]

    def decoder_layer(self, layer, hidden, cos, sin, attention_bias, past_kv):
        """单层前向。返回 (hidden, present_kv)。"""
        B, q_len, _ = hidden.shape
        residual = hidden
        h = layer.input_layernorm(hidden)

        attn = layer.self_attn
        # HF 顺序（严格对齐 modeling_qwen3_vl.py:472-474）：
        #   hidden_shape = (B, q, heads, head_dim)
        #   q = q_norm(q_proj(h).view(hidden_shape)).transpose(1,2)   # q_norm 在 view 后、transpose 前
        #   k = k_norm(k_proj(h).view(hidden_shape)).transpose(1,2)
        #   v = v_proj(h).view(hidden_shape).transpose(1,2)           # v 无 norm
        hidden_shape = (B, q_len, self.num_heads, self.head_dim)
        q = attn.q_proj(h).view(hidden_shape)
        q = attn.q_norm(q)                  # RMSNorm over head_dim (last dim)
        q = q.transpose(1, 2)               # (B, heads, q, head_dim)
        kv_shape = (B, q_len, self.num_kv_heads, self.head_dim)
        k = attn.k_norm(attn.k_proj(h).view(kv_shape))
        k = k.transpose(1, 2)
        v = attn.v_proj(h).view(kv_shape).transpose(1, 2)

        # RoPE: cos/sin 形状 (B, q, head_dim)。apply_rotary_pos_emb 内部会
        # unsqueeze(1) 成 (B,1,q,head_dim) 广播到 q/k 的 (B,heads,q,head_dim)。
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # concat past KV: past_kv (B,2,nkv,past,hd) -> K=past_kv[:,0], V=past_kv[:,1]
        past_k = past_kv[:, 0]  # (B, nkv, past, hd)
        past_v = past_kv[:, 1]
        k_new = torch.cat([past_k, k], dim=2)  # (B, nkv, past+q, hd)
        v_new = torch.cat([past_v, v], dim=2)

        # GQA repeat
        k_r = repeat_kv(k_new, self.num_kv_groups)
        v_r = repeat_kv(v_new, self.num_kv_groups)

        # attention
        attn_w = torch.matmul(q, k_r.transpose(2, 3)) * self.scaling  # (B, heads, q, kv)
        attn_w = attn_w + attention_bias
        attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn_w, v_r)  # (B, heads, q, hd)
        out = out.transpose(1, 2).reshape(B, q_len, self.hidden)
        out = attn.o_proj(out)

        hidden = residual + out

        # MLP
        residual = hidden
        h = layer.post_attention_layernorm(hidden)
        mlp = layer.mlp
        h = mlp.down_proj(mlp.act_fn(mlp.gate_proj(h)) * mlp.up_proj(h))
        hidden = residual + h

        present_kv = torch.stack([k_new, v_new], dim=1)  # (B, 2, nkv, past+q, hd)
        return hidden, present_kv

    def forward(self, inputs_embeds, position_ids, attention_bias,
                deepstack_0, deepstack_1, deepstack_2, image_pad_mask, *past_kvs):
        hidden = inputs_embeds
        cos, sin = self.rotary_emb(hidden, position_ids)
        deepstacks = [deepstack_0, deepstack_1, deepstack_2]

        presents = []
        for idx, layer in enumerate(self.layers):
            hidden, pk_new = self.decoder_layer(
                layer, hidden, cos, sin, attention_bias, past_kvs[idx])
            presents.append(pk_new)
            # deepstack 注入：真实层 5/11/17 后加对应 deepstack_embed
            if idx in self.deepstack_layers:
                ds_idx = self.deepstack_layers.index(idx)
                hidden = scatter_add_visual(hidden, deepstacks[ds_idx], image_pad_mask)

        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        return (logits, *presents)


# --- 构造 prefill dummy inputs ---
n_img = seq_len // (SPATIAL_MERGE ** 2)  # 196 / 4 = 49
L = n_img + 8  # 49 image_pad + 8 文本 token

# input_ids 含 image_pad（占位）+ 文本，用 get_rope_index 预算 3D position_ids。
input_ids = torch.tensor(
    [[IMAGE_TOKEN_ID] * n_img + [1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
attention_mask_1d = torch.ones(1, L, dtype=torch.long)
# mm_token_type_ids: image=1, text=0
mm_token_type_ids = torch.zeros(1, L, dtype=torch.int32)
mm_token_type_ids[0, :n_img] = 1
image_grid_thw = grid_thw  # (1,3)

with torch.no_grad():
    position_ids, _ = model.model.get_rope_index(
        input_ids=input_ids,
        mm_token_type_ids=mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask_1d,
    )  # (3, B, L) int64
print(f"[llm] n_img={n_img} L={L} position_ids shape={tuple(position_ids.shape)}")

# inputs_embeds: embed 后把 image_pad 位置换成随机 image_features（模拟视觉注入）
with torch.no_grad():
    inputs_embeds = lm.embed_tokens(input_ids).float().to(torch.float16)
    image_features = torch.randn(n_img, HIDDEN, dtype=torch.float16)
    image_pad_mask = torch.zeros(1, L, dtype=torch.bool)
    image_pad_mask[0, :n_img] = True
    flat_mask = image_pad_mask.reshape(-1)
    idx = torch.nonzero(flat_mask, as_tuple=False).squeeze(-1)
    flat = inputs_embeds.reshape(-1, HIDDEN).clone()
    flat[idx] = image_features
    inputs_embeds = flat.reshape(1, L, HIDDEN)

# 3 个 deepstack_embed（随机，长度 = n_img）
ds0 = torch.randn(n_img, HIDDEN, dtype=torch.float16)
ds1 = torch.randn(n_img, HIDDEN, dtype=torch.float16)
ds2 = torch.randn(n_img, HIDDEN, dtype=torch.float16)

# attention_bias: causal 加法 mask (B,1,q,kv)，kv=q（prefill 无 past）。
# 用 finfo(float16).min（≈-65504）与 HF create_causal_mask 实测一致；用 -1e4 会因
# fp16 softmax 区分度不足产生偏差。
kv_len = L
causal = torch.triu(
    torch.full((L, kv_len), torch.finfo(torch.float16).min, dtype=torch.float16),
    diagonal=1)
attention_bias = causal.unsqueeze(0).unsqueeze(0)  # (1,1,L,kv)

# past_key_values: 28 个空张量 (B,2,nkv,0,hd)
past_kvs = tuple(
    torch.zeros(1, 2, NUM_KV_HEADS, 0, HEAD_DIM, dtype=torch.float16)
    for _ in range(NUM_LAYERS))

inputs = (inputs_embeds, position_ids, attention_bias,
          ds0, ds1, ds2, image_pad_mask, *past_kvs)

input_names = ["inputs_embeds", "position_ids", "attention_bias",
               "deepstack_embeds_0", "deepstack_embeds_1", "deepstack_embeds_2",
               "image_pad_mask"]
input_names += [f"past_key_values_{i}" for i in range(NUM_LAYERS)]
output_names = ["logits"] + [f"present_key_values_{i}" for i in range(NUM_LAYERS)]

dynamic_axes = {
    "inputs_embeds": {0: "batch", 1: "seq"},
    "position_ids": {1: "batch", 2: "seq"},
    "attention_bias": {0: "batch", 2: "q_len", 3: "kv_len"},
    "image_pad_mask": {0: "batch", 1: "seq"},
    "logits": {0: "batch", 1: "seq"},
}
for i in range(3):
    dynamic_axes[f"deepstack_embeds_{i}"] = {0: "num_image_patches"}
for i in range(NUM_LAYERS):
    dynamic_axes[f"past_key_values_{i}"] = {0: "batch", 3: "kv_len"}
    dynamic_axes[f"present_key_values_{i}"] = {0: "batch", 3: "kv_len"}

print(f"[llm] exporting {NUM_LAYERS} layers, {len(input_names)} inputs / {len(output_names)} outputs ...")
with torch.no_grad():
    torch.onnx.export(
        Qwen3VLLMOnnx(model), inputs, "llm.onnx",
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,  # 同视觉：legacy TorchScript 导出器，避开 dynamo guard
    )
print("[✓] llm.onnx exported (KV cache + deepstack)")


# ---------------------------------------------------------------------------
# 3. 合并 llm.onnx 的散权重为单文件 llm.weights.bin，删除散文件
# ---------------------------------------------------------------------------
import onnx

print("[consolidate] loading llm.onnx with external data (scattered files) ...")
m = onnx.load("llm.onnx", load_external_data=True)
n_init = len(m.graph.initializer)
n_ext = sum(1 for t in m.graph.initializer if t.HasField("data_location") and t.data_location == 1)
print(f"[consolidate] initializers={n_init} external={n_ext}")

print("[consolidate] saving as single llm.weights.bin ...")
onnx.save_model(
    m, "llm.onnx",
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location="llm.weights.bin",
    convert_attribute=True,
)

# 删除散文件（保留 llm.onnx、llm.weights.bin、visual.onnx、脚本等）
removed = 0
for f in os.listdir("."):
    if f.endswith(".weight") or f.startswith("onnx__MatMul_"):
        if f != "llm.weights.bin":
            os.remove(f)
            removed += 1
print(f"[✓] consolidated into llm.weights.bin, removed {removed} scattered files")

# 验证：重载确认 location 全指向 llm.weights.bin
m2 = onnx.load("llm.onnx", load_external_data=False)
locs = set()
for t in m2.graph.initializer:
    if t.HasField("data_location") and t.data_location == 1:
        for kv in t.external_data:
            if kv.key == "location":
                locs.add(kv.value)
print(f"[verify] external locations: {locs} (expect {{'llm.weights.bin'}})")
assert locs == {"llm.weights.bin"}, f"unexpected external locations: {locs}"
print("done.")

# embed_tokens 权重表不由本脚本导出：它不在 llm.onnx 图里（图输入是
# inputs_embeds），C++ generate loop (llm_chat) 用独立脚本 dump_embed_tokens.py
# 单独抽取为 embed_tokens.bin，无需重跑本 ONNX 导出。
