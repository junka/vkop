# Qwen3-VL-2B ONNX 导出与推理

把 HuggingFace `Qwen3VLForConditionalGeneration` 导出成两个 ONNX 图
（`visual.onnx` + `llm.onnx`），用于脱离 PyTorch 的 ONNX Runtime 推理。本文档
说明导出策略、与 HF 原模型的**算子级对齐方式**、KV cache 契约，以及**端到端
一致性验证**（ONNX 推理 vs HF `Qwen3VLInference`）。

```
┌─────────────────┐  pixel_values (seq_len,1536)   ┌──────────────┐
│ HF image proc   │ ─────────────────────────────▶ │ visual.onnx  │
│ + tokenizer     │                                 │ (deepstack)  │
└─────────────────┘  input_ids, grid_thw, mask     └──────┬───────┘
        │                                              pooler_output + 3×deepstack
        │ get_rope_index (3D pos_ids)                           │
        │ causal attention_bias                                ▼
        ▼          ┌──────────────────────────────────────────────────┐
   embed_tokens    │ llm.onnx (KV cache + deepstack 注入)             │
   scatter image   │  inputs_embeds + past_kv + pos_ids + attn_bias   │
   → inputs_embeds │  → logits + present_kv                           │
                   └──────────────────────────────────────────────────┘
        │                          ▲
        └──── greedy decode loop ──┘  (KV 在步间传递)
```

## 组件

| 文件 | 作用 |
|---|---|
| [qwen3vl_export_onnx.py](qwen3vl_export_onnx.py) | 导出 visual.onnx / llm.onnx（含 deepstack、KV cache、权重合并）。 |
| [infer.py](infer.py) | ONNX 端到端推理驱动（`OnnxQwen3VL`：visual + LLM prefill/decode + greedy 生成）。 |
| [cases.py](cases.py) | 一致性测试用例构造（合成 PIL 图、prompt 模板、`CASES` 列表，供 tests 复用）。 |
| [tests/](tests/) | pytest 测试套件（数值对齐 + 端到端 token 一致性）。 |
| [qwen3vl_infer.py](qwen3vl_infer.py) | HF 原生推理封装（`Qwen3VLInference`），作为对比基准。 |
| `visual.onnx` / `llm.onnx` / `llm.weights.bin` | 导出产物（`llm.weights.bin` 是 llm.onnx 的外部权重单文件）。 |

## 构建

```bash
python3 qwen3vl_export_onnx.py   # 产出 visual.onnx / llm.onnx / llm.weights.bin
pytest                            # 跑全部测试（数值对齐 + 端到端一致性）
pytest tests/test_numeric.py        # 仅数值对齐
pytest tests/test_consistency.py -k ocr   # 仅 OCR 用例
```

依赖：`torch`、`transformers`、`onnx`、`onnxruntime`、`pytest`、`qwen_vl_utils`。

> `infer.py` 也可 `python3 infer.py` 直接跑一个单图样例（ONNX vs HF 对比），
> 主要逻辑在 `OnnxQwen3VL` 类，被 tests 复用。

---

## 导出策略与算子对齐

### 1. 视觉编码器（visual.onnx）

直接包一层 `VisualExport` 调 HF `visual(pixel_values, grid_thw)`，导出
`pooler_output` + 3 个 `deepstack_features`（来自视觉层 5/11/17）。

- **为什么用 `dynamo=False`**：torch 2.9 默认 dynamo 导出器对视觉的
  `fast_pos_embed_interpolate`（`torch.linspace` 含数据相关长度）会触发
  `GuardOnDataDependentSymNode`。改用 legacy TorchScript 导出器（`dynamo=False`）绕开。
- **grid_thw 被常量折叠**：`fast_pos_embed_interpolate` 内部 `grid_thw.tolist()`
  把 grid_thw 转成 Python list，trace 后被当常量折进图，故 `visual.onnx` 的输入
  **只有 `pixel_values`**（grid_thw 在导出时固化）。这是 HF 视觉实现的特性，导出
  尺寸固定。默认 224×224 → patch_size=16 → 14×14=196 个 patch（grid_h=grid_w=14，
  即每个维度 14 个 patch，不是 patch_size=14）。不同图像尺寸需重新导出
  （`EXPORT_IMG_SIZE=336 python3 qwen3vl_export_onnx.py`）。
  > 注意区分：`patch_size`（=16，config 里读，每个 patch 的像素边长）与
  > `grid_h/grid_w`（=14，图像在每维切出的 patch 数）。224÷16=14。
- 视觉 attention 是 eager（无 `create_causal_mask`），不触 `torch.diff`，安全 trace。

### 2. 语言模型（llm.onnx）—— 自定义 wrapper，绕开 HF 内部

HF 的 `Qwen3VLTextModel.forward` 有两处 ONNX 不可导出：
- `past_key_values` 是 `Cache` 对象（`DynamicCache` 内部 list append），不可 trace。
- `create_causal_mask` → `find_packed_sequence_indices` 用 `torch.diff`，opset17 不支持。

故写 `Qwen3VLLMOnnx(nn.Module)`，**复用 HF 的层权重**，但自己写 forward 层循环，
把 KV cache 与 attention mask 暴露成显式张量 I/O，纯标准 ONNX op。

#### I/O 契约

| 方向 | 名称 | 形状 | dtype | 说明 |
|---|---|---|---|---|
| 入 | `inputs_embeds` | (B, q, 2048) | fp16 | embed 后已 scatter image_features |
| 入 | `position_ids` | (3, B, q) | int64 | MRoPE 的 t/h/w 位置 |
| 入 | `attention_bias` | (B, 1, q, kv) | fp16 | 加法 mask（因果+padding） |
| 入 | `deepstack_embeds_{0,1,2}` | (n_img, 2048) | fp16 | 视觉 deepstack 特征 |
| 入 | `image_pad_mask` | (B, L) | bool | image_pad 位置 |
| 入 | `past_key_values_{0..27}` | (B, 2, 8, kv, 128) | fp16 | 每层 K/V，prefill 时 kv=0 |
| 出 | `logits` | (B, q, 151936) | fp16 | |
| 出 | `present_key_values_{0..27}` | (B, 2, 8, kv, 128) | fp16 | 更新后的 K/V |

#### 每层 forward（与 HF 逐 op 对齐）

```python
# 1. Q/K/V proj + q_norm/k_norm（严格按 HF modeling_qwen3_vl.py:472-474 顺序）
hidden_shape = (B, q, num_heads, head_dim)
q = q_norm(q_proj(h).view(hidden_shape)).transpose(1,2)   # q_norm 在 view 后、transpose 前
k = k_norm(k_proj(h).view((B,q,num_kv_heads,hd))).transpose(1,2)
v = v_proj(h).view((B,q,num_kv_heads,hd)).transpose(1,2)

# 2. RoPE：复用 HF apply_rotary_pos_emb（rotate_half 版，非 interleaved；
#    interleaved 已在 rotary_emb.apply_interleaved_mrope 内完成）
q, k = apply_rotary_pos_emb(q, k, cos, sin)   # cos/sin 形状 (B,q,hd)，内部 unsqueeze(1)

# 3. concat past KV（绕开 Cache 对象）
k_new = torch.cat([past_kv[:,0], k], dim=2)   # (B, nkv, past+q, hd)
v_new = torch.cat([past_kv[:,1], v], dim=2)

# 4. GQA repeat（HF repeat_kv，interleave 方式）
k_r = repeat_kv(k_new, 2); v_r = repeat_kv(v_new, 2)

# 5. 标准 attention（手写，绕开 create_causal_mask）
attn = matmul(q, k_r.T) * scaling             # scaling = 1/sqrt(128) = 0.0884
attn = attn + attention_bias                  # 加法 mask，外部预算
attn = softmax(attn, -1, dtype=fp32).to(fp16)
out = matmul(attn, v_r)
out = o_proj(out.transpose(1,2).reshape(B, q, hidden))

# 6. MLP（gate/up/down，SiLU）
h = down_proj(silu(gate_proj(x)) * up_proj(x))

# 7. deepstack 注入（仅文本层 0/1/2）
present_kv = stack([k_new, v_new], dim=1)
```

**关键对齐点**（踩过的坑）：
1. **RoPE 的 unsqueeze**：`apply_rotary_pos_emb` 内部会 `cos.unsqueeze(1)`，**不要**
   外部再 unsqueeze，否则产生多余维度导致 `torch.cat` 维度不匹配（4 vs 5）。
2. **q_norm/k_norm 顺序**：必须 `view(B,q,heads,hd)` → `q_norm` → `transpose(1,2)`，
   与 HF 完全一致。顺序错会导致 logits maxdiff 达 18。
3. **deepstack 注入位置**：是**文本层 [0,1,2]**，不是视觉层 [5,11,17]。
   `[5,11,17]` 是**视觉塔**产出 deepstack 的层号；文本模型在 `layer_idx in range(3)`
   处把它们加到 hidden（HF `modeling_qwen3_vl.py:81`）。这个误判曾导致 logits maxdiff=18。
4. **attention_bias 的 mask 值**：用 `torch.finfo(float16).min`（≈-65504），与 HF
   `create_causal_mask` 实测一致（probe 到 HF 传给 attention 的 mask min=-65504）。
   用 `-1e4` 会因 fp16 下 softmax 区分度不足产生偏差。
5. **MRoPE position_ids**：形状 `(3,B,q)`，纯文本时三行相等退化为标准 RoPE；含图像时
   t/h/w 不同（图像 token 用 3D 位置）。由调用方用 HF `get_rope_index` 预算，wrapper 不算位置。

#### deepstack 注入（opset17 兼容）

HF `_deepstack_process`：`hidden[mask,:] += embed`。opset17 的 `ScatterND` 无 add
reduction，用 gather+add+scatter 覆盖实现：

```python
def scatter_add_visual(hidden, embed, mask):
    B, L, H = hidden.shape
    idx = nonzero(mask.reshape(-1)).squeeze(-1)   # (n_img,)
    flat = hidden.reshape(-1, H).clone()
    flat[idx] = flat[idx] + embed                  # 加
    return torch.scatter(flat, 0, idx.unsqueeze(1).expand(-1,H), flat[idx]).reshape(B,L,H)
```

验证：单独测 deepstack 注入 vs HF，**maxdiff=0.0**（完全一致）。

### 3. 权重合并（llm.weights.bin）

llm.onnx 权重 ~2.3GB 超 protobuf 2GB 上限，`torch.onnx.export` 自动 external-data
成 255 个散文件（`lm.*.weight`、`onnx__MatMul_8XXX`）。导出后用
`onnx.save_model(..., all_tensors_to_one_file=True, location="llm.weights.bin")`
合并成单文件并删除散文件。移动 `llm.onnx` 只需带 `llm.weights.bin` 一个文件。

---

## 一致性验证

### 数值级（tests/test_numeric.py，onnxruntime vs HF，fp16）

| 测试 | maxdiff | mean | 判定 |
|---|---|---|---|
| `test_visual_pooler_and_deepstack` pooler_output | 0.19 | 3e-3 | OK |
| `test_visual_pooler_and_deepstack` deepstack_0/1/2 | ≤0.05 | ≤1.7e-3 | OK |
| `test_llm_prefill_logits_and_kv` logits | 0.20 | 1.2e-2 | OK |
| `test_llm_prefill_logits_and_kv` present_kv[0] | 0.50 | 5.7e-4 | OK |
| `test_llm_decode_logits` | 0.06 | 1.0e-2 | OK |

判定用「绝对均值差 < 2e-2」（maxdiff 受 fp16 CPU ort 累积舍入影响偶达 ~0.2，
但均值差极小；逻辑错时 mean 通常 >0.5 或几十，能可靠区分）。

### 端到端 token 级（tests/test_consistency.py，ONNX 推理 vs HF `model.generate`）

`infer.py` 的 `OnnxQwen3VL` 用 `visual.onnx` + `llm.onnx` 跑完整生成（prefill +
greedy decode 循环，KV 在步间传递），与 HF `Qwen3VLForConditionalGeneration.generate(do_sample=False)`
对比生成的 token id 序列。用例来自 [cases.py](cases.py)（10 个业界通用场景）：

| case | ONNX 输出 | HF 输出 | 一致 |
|---|---|---|---|
| 纯文本 "Count 1 to 5" | `1, 2, 3, 4,` | `1, 2, 3, 4,` | ✅ |
| 纯文本 "translate hello→中文" | `你好` | `你好` | ✅ |
| 纯文本 "capital of France" | `Paris` | `Paris` | ✅ |
| 红/绿/蓝/黄图 "what color" | `red`/`green`/`blue`/`yellow` | 同 | ✅ |
| OCR "42" | `42` | `42` | ✅ |
| OCR "HI" | `hi` | `hi` | ✅ |
| 长文本 "3 primary colors" | `Red, Blue, Yellow` | 同 | ✅ |

ONNX 端 greedy 生成的 token id 与 HF greedy 完全一致（含多模态图像理解 + OCR）。
`pytest tests/test_consistency.py` 全 10 用例 PASS。

### 与 Qwen3VLInference 对比

[qwen3vl_infer.py](qwen3vl_infer.py) 的 `Qwen3VLInference` 是 HF 原生推理封装
（`processor.apply_chat_template` + `model.generate(do_sample=True)`）。
`infer.py` 复用同一 `AutoProcessor` 生成 `input_ids`/`pixel_values`/`grid_thw`/
`mm_token_type_ids`，区别仅在 LLM 推理路径（ONNX 图 vs PyTorch）。

- **greedy（do_sample=False）**：ONNX 与 HF token id 逐个一致（见上表）。
- **采样（do_sample=True）**：`Qwen3VLInference` 默认 temperature=0.7 采样，ONNX 驱动
  目前只做 greedy；采样一致性需在 ONNX 端实现同样的 temperature/top_p 采样逻辑
  （logits 加噪 + top-p 过滤），未在本驱动覆盖。数值上 ONNX logits 与 HF 对齐
  （prefill/decode maxdiff ≤0.2），故接入相同采样器后结果分布一致。

> 注：本机无 CUDA，验证在 CPU ort fp16 上跑。GPU 上 fp16 精度通常更好（maxdiff 更小）。

---

## 局限

- **视觉尺寸固定**：`visual.onnx` 的 grid_thw 被常量折叠，导出尺寸（224×224）固定。
  不同图像尺寸需重新导出，或改用 TRT-edge 的 `Qwen3VLVisionModelPatch`（grid_thw 作输入）。
- **prefill/decode 共用一图**：单图两用，decode 时 q_len=1 仍走完整 attention 路径，
  效率非最优但正确。生产环境可拆 prefill/decode 双图。
- **采样未实现**：ONNX 驱动仅 greedy；采样需自行加 temperature/top_p。
- **MRoPE 由调用方预算**：wrapper 不算位置，调用方用 HF `get_rope_index`。纯文本场景
  三行相等，简单；多模态需正确传 image_grid_thw + mm_token_type_ids。
