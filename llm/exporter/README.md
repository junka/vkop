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
| [qwen3_export_onnx.py](qwen3_export_onnx.py) | 纯文本 Qwen3（`Qwen3ForCausalLM`）导出 → `text_qwen3/`。 |
| [phi4_export_onnx.py](phi4_export_onnx.py) | Phi-4-mini（`Phi3ForCausalLM`）导出 → `text_phi4/`；与 Qwen 共用同一套张量 I/O 契约。 |
| [glm_edge_export_onnx.py](glm_edge_export_onnx.py) | GLM-Edge-1.5B-Chat（`GlmForCausalLM`）导出 → `text_glm_edge/`；同一套张量 I/O 契约。 |
| [greedy_ref.py](greedy_ref.py) | CPU ORT 的 greedy 逐步解码参考（架构参数全部从图形状读），用 `ONNX`/`EMBED`/`MODEL_PATH` 选模型，和 GPU 逐 token 对拍。 |
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
6. **转换器折掉「升秩 Unsqueeze」**：`fuse_unsqueeze_eliminate` 会删掉只加一个 size-1 轴的
   `Unsqueeze`（字节排布确实不变），但删掉之后消费者在 runtime 拿到的是 view **前**的秩。
   `/rotary_emb` 的 `Unsqueeze(axes=[-1])×3 → Concat(axis=-1)` 就这样被折掉，`ScatterND`
   把 `indices.shape[-1]=21` 当成 index_rank（应为 3），20 个 scatter 目标全部越出 256B 的
   输出 buffer，按数据决定的地址把 fp32 的 rope cos/sin 写进相邻的 fp16 权重显存
   （权重里出现 `0xfffa` = NaN），于是 decode 从某轮起全 NaN。**这就是「最早支持的模型
   莫名其妙坏了」的真相：模型文件一个字没改，是转换器新增的折叠规则把它的地基挖空了。**
   现在折叠走白名单（fail closed）：只有逐元素一元 op、以及「插入轴严格晚于拼接轴」的
   Concat 才折；其余 view 节点留给 runtime 的 `SqueezeUnsqueeze`（纯 GPU 别名，不读回）。
   runtime 侧同时补了不变量检查：`ScatterND`/`ScatterElements` 的形状不满足 ONNX 契约就
   直接抛错并打印三边形状，shader 丢弃越界写 —— 越界不再静默。

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

---

## vkop C++ 驱动 (llm_chat)

`llm_chat.cpp` 是 vkop（Vulkan ONNX runtime）的端到端对话驱动：文本输入 →
tokenize → prefill（q_len=L）→ 逐 token greedy decode（q_len=1）→ detokenize 输出。
不依赖 PyTorch / ONNX Runtime，只用 `llm.vkopbin`（转换后的 vkop 图）+
`embed_tokens.bin`（host 端 embedding 查表）+ tokenizer。

支持**多模态**（`--image` + `--visual`）：加载图片 → C++ 图像预处理（`image_preproc.hpp`）
→ 跑 `visual.vkopbin` 取 image_features + 3 个 deepstack 特征 → scatter 进
`inputs_embeds` 的 `<|image_pad|>` 位置 → `get_rope_index` 算 3 轴 MRoPE
position_ids + rope_delta。decode 用 `position_ids = past_len + rope_delta`。
纯文本场景（无 `--image`）三轴位置都取 `0..L-1`，`image_pad_mask` 全 false，
`deepstack_embeds` 全零，rope_delta=0。

**多模态输出已验证与 ORT 参考逐 token 一致**：合成 224×224 绿色图 +
"What color is the image?" → `[6176, 151645]` = " green"，rope_delta=-42
（单图 grid [1,14,14]）。详见 memory `int8-dtype-propagation-multimodal`。

### 构建

已合并进主 CMake（`ENABLE_LLM_CHAT`，默认 ON）：

```bash
make -C build llm_chat        # 仅构建 llm_chat
make -C build                 # 全量构建（含 llm_chat）
```

链接 `vkop + vload + tokenizer`（都来自主 CMake 的 target）。改了 `libvkop.a` 或
`llm_chat.cpp` 后重跑 `make -C build llm_chat` 即可，无需手动 g++ 脚本。
vulkan 在 `libvkop` 内部 `dlopen` 加载，链接期不需要 `-lvulkan`。

### 运行

```bash
./build/llm_chat <llm.vkopbin> <embed_tokens.bin> <tokenizer.bin> [max_new] \
  [--image <img> --visual <visual.vkopbin>]
# 然后在 stdin 输入 prompt，回车提交，Ctrl-D 退出
```

纯文本（从仓库根）：

```bash
./build/llm_chat \
  llm/exporter/llm.vkopbin \
  llm/exporter/embed_tokens.bin \
  llm/tokenizer/qwen3_vl.bin
```

多模态（单图；`--image` 可重复，多图属于第一轮）：

```bash
./build/llm_chat \
  llm/exporter/llm.vkopbin \
  llm/exporter/embed_tokens.bin \
  llm/tokenizer/qwen3_vl.bin \
  --image llm/exporter/synth224.png \
  --visual llm/exporter/visual.vkopbin
```

| 参数 | 含义 |
|---|---|
| `llm.vkopbin` | 转换后的 vkop LLM 图（~3.4GB） |
| `embed_tokens.bin` | token→hidden 查表（fp16 `[151936,2048]`，~590MB） |
| `tokenizer.bin` | BBPE 词表（`tokenizer_to_bin.py` 产物） |
| `max_new`（可选） | 最大生成 token 数（含 prefill 后的全部 decode），默认 **64** |
| `--image <img>` | 输入图片路径（多模态；需配合 `--visual`）。可重复，按给出顺序与 prompt 里的图像占位一一对应；这些图算在第一轮头上，之后的轮次通过历史上下文继续「看到」它们 |
| `--visual <vkopbin>` | 视觉编码器 vkop 图（`visual.vkopbin`，~770MB）。多模态必需 |

> 多模态约束：图片尺寸必须是 `patch_size*merge = 16*2 = 32` 的整数倍
> （224×224 → grid [1,14,14] → n_img=49 个图像 token），否则 `image_preproc`
> 报 "not divisible by patch*merge"。不同尺寸需重新导出 `visual.onnx` 并转换。

启动时加载模型 + embedding + tokenizer（~2s），之后每轮 prefill/decode 约 1.5–2s
（Intel ARL，buffer backend，fp16）。遇到结束符自动停止当前轮。

### 多轮上下文（Conversation）

同一个 REPL 进程里，每条输入都带着之前的对话：`llm/exporter/conversation.hpp`
存 token 级的轮次历史（user 轮含角色前后缀与展开后的图像占位；assistant 轮含
**生成时的原始 ids**），`render()` 把整段历史 + assistant 引导串拼成一次 prefill
的输入，**每轮 KV 从零重新算**。两条不变量：

- assistant 的回复绝不重新分词。BBPE 的 decode→encode 不保证可逆（词首空格标记
  在 decode 时被丢），重新 encode 会把模型「自己写下的那段历史」改成另一个序列。
- 上下文策略全在上层：`trimToBudget()` 按 token 预算整对（user+assistant）丢掉最旧
  的轮次，预算默认 `MAX_KV - max_new`（KV 预分配上界 8192），`VKOP_MAX_CTX` 可覆盖。
  序列 + `max_new` 超出上界时直接报错退出，不静默扩大 buffer。

跨轮**续用** KV（跳过已算前缀）是纯加速层，前提是新序列是上一轮的严格前缀扩展，
校验点已经留在 `Conversation::prefixMatch()`；vkop 目前没有 paged KV / block
table，所以这条还没接（也就没有 vLLM/SGLang 那种跨请求 prefix caching）。

验证（纯文本 Qwen3-4B，`llm/exporter/text_qwen3/`）：

```bash
printf '请先记住：我的名字叫小明，我喜欢打篮球。只需要回复"好的"。\n我的名字是什么？我喜欢什么运动？\n' | \
  ./build/llm_chat llm/exporter/text_qwen3/llm.vkopbin \
  llm/exporter/text_qwen3/embed_tokens.bin llm/exporter/text_qwen3/tokenizer.bin 28
# [prompt] 28 tokens (history 1 turns) → 好的
# [prompt] 46 tokens (history 3 turns) → 我的名字是小明，我喜欢打篮球。
# 单独问第二个问题（新进程、无历史）则答不出名字 —— 上下文确实生效。
```

### 环境变量

| 变量 | 作用 |
|---|---|
| `VKOP_RAW_PROMPT=1` | 跳过 chat template **和多轮历史**，每行输入独立（对齐参考 `dump_llm_decode.py`，不走对话格式） |
| `VKOP_MAX_CTX=<tokens>` | 多轮上下文的 token 预算，默认 `8192 - max_new`；超预算整对丢掉最旧的一问一答 |
| `VKOP_CHATDBG=1` | 打印每轮 KV cache 反馈形状 + logits top5 |
| `VKOP_PROFILE=1` | 开启性能分析：每轮结束输出 prefill/decode 的 tokens/s、延迟百分位 (p50/p90/p99)、KV cache 利用率；会话结束时打印全局汇总统计 |
| `VKOP_LOAD_MMAP=1` | 载入走旧的 mmap 路径。默认（不设）走**从文件 pread 直灌 staging buffer**：多 GB initializer 逐页 fault 实测 ~1.4 GB/s，大块 pread 快得多且省掉 host→host memcpy。三个模型的 vkopbin 上 warm cache 差距只有 ~10%，冷启动才明显；仅影响 `LoadModel`，数值路径逐 token 不变 |
| `VKOP_DUMP_TENSORS='*'` | dump 所有命名中间张量（fp16 hex + fp32 dec；配合 `VKOP_DUMP_INT64=1` 看 int64） |
| `VKOP_DUMP_OFF='name:offset'` | 只 dump 某张量 offset 起 16 个元素 |
| `VKOP_DUMP_INT64=1` | dump int64 张量（默认跳过，因为体积大） |

### Profiling 性能分析

设置 `VKOP_PROFILE=1` 后，`llm_chat` 会在每轮结束和会话结束时输出详细的性能指标：

```bash
VKOP_PROFILE=1 ./build/llm_chat llm/exporter/llm.vkopbin \
  llm/exporter/embed_tokens.bin llm/tokenizer/qwen3_vl.bin 64
```

**每轮输出示例：**
```
[profile] round 1:
  prefill: 9 tokens in 113.6ms (79.3 tok/s)
  decode:  4 tokens in 393.0ms (10.2 tok/s, avg 98.2ms/token)
  kv cache: 13/8192 (0.2%)
```

**会话汇总示例：**
```
[profile] session summary (1 rounds):
  total prompt tokens: 9
  total generated tokens: 4
  avg prefill tok/s: 79.3
  avg decode tok/s: 10.2
  p50 decode latency: 98.3ms
  p90 decode latency: 99.1ms
  p99 decode latency: 99.4ms
```

指标说明：
- **prefill tok/s**：prompt 处理速度，随历史长度增长而下降（需重新处理整段上下文）
- **decode tok/s**：生成速度，相对稳定但受 past_len 影响（attention_bias 维度增大）
- **p50/p90/p99 延迟**：decode 单步延迟的百分位，帮助发现偶发抖动（GC、内存分配等）
- **kv cache 利用率**：当前序列长度占 MAX_KV (8192) 的比例，用于调优 `VKOP_MAX_CTX`

实现细节见 [project-llm-profiling](../../memory/project-llm-profiling.md)。

### 示例

```bash
# 对话模式（走 chat template）
printf 'What is the capital of France?' | \
  ./build/llm_chat llm/exporter/llm.vkopbin \
  llm/exporter/embed_tokens.bin llm/tokenizer/qwen3_vl.bin 12
# → The capital of France is Paris.<|im_end|>

# raw 模式（对齐 ORT 参考）
echo "Hello" | VKOP_RAW_PROMPT=1 ./build/llm_chat \
  llm/exporter/llm.vkopbin llm/exporter/embed_tokens.bin \
  llm/tokenizer/qwen3_vl.bin 6
# → 358,1184,311,3270,264,2805  ("I need to write a short")

# 多模态（单图问答）
printf 'What color is the image? Answer in one word.' | \
  ./build/llm_chat llm/exporter/llm.vkopbin \
  llm/exporter/embed_tokens.bin llm/tokenizer/qwen3_vl.bin 8 \
  --image llm/exporter/synth224.png --visual llm/exporter/visual.vkopbin
# → [prompt] 70 tokens (image tokens 4..52)
#   position_ids ok (rope_delta=-42)
#   [prefill] → token 6176   green
#   [r1] pos=28 → 151645  <|im_end|>
#   green<|im_end|>
```

### decode 轮数限制

`max_new` 默认 64，对应代码：

```cpp
int max_new = (argc > 4) ? std::atoi(argv[4]) : 64;   // llm_chat.cpp:259
...
for (int step = 1; step < max_new; ++step) {          // decode 循环
    if (next_id == IM_END) break;
    ...
}
```

注意 `max_new` 是**总输出预算**：prefill 占 1 个（step 从 1 起），所以实际 decode
token 数 = `max_new - 1`。默认 64 → prefill 后最多再生成 63 个 decode token，遇
`<|im_end|>` 提前停。这是**纯软限制**，不是 vkop 的硬约束——KV cache 每轮通过
`feedback_kv` 调 `ResizeInput` 按实际 `kv_len` 动态增长（`past_len += 1`），无固定上限
的 buffer 预分配，所以理论上可以无限 decode 下去。

**放开限制的方式**（按推荐度排序）：

1. **命令行传更大的 `max_new`**（最简单，无需改码）：
   ```bash
   ./build/llm_chat ... 4096    # 放到 4k token
   ./build/llm_chat ... 100000  # 实质无限制（靠 IM_END 自然停）
   ```

2. **改默认值**：把 `llm_chat.cpp:259` 的 `: 64` 改成更大的数（如 `: 2048`）。
   适合不想每次传参的场景。

3. **完全去掉上界，只靠 IM_END 停**：把 `for (int step = 1; step < max_new; ++step)`
   改成 `for (int step = 1; ; ++step)`，循环内只保留 `if (next_id == IM_END) break;`。
   风险：若模型不输出 IM_END（坏采样 / 陷入循环），会无限生成；建议加一个
   `step < HARD_CAP` 的安全上界（如 32768，Qwen3-VL 的训练上下文长度），超过则强制截断。

**放开后的实际约束**（非 vkop 限制，是模型/硬件层面）：

- **模型上下文长度**：Qwen3-VL 训练长度通常 32768。超过后 attention 质量下降
  （不崩，但生成可能变乱）。KV cache 此时占 `NLAYERS × 2 × NKV × kv_len × 128 × 2B`
  = `28 × 2 × 8 × 32768 × 128 × 2` ≈ 3.75GB 显存（fp16），需确保 GPU 内存够。
- **显存**：KV cache 随 `kv_len` 线性增长，加上 attention 的 `(q, kv_len)` 中间张量也
  线性增长。超显存会 Vulkan 内存分配失败（vkop 报错，不崩进程）。
- **延迟**：每轮 decode 的 attention 计算量随 `kv_len` 线性增长，后期 token 越来越慢。

简言之：**放开就是改 `max_new`**，vkop 侧无硬墙；真正的上界是 GPU 显存 + 模型训练长度。

---

## 第二个架构：Phi-4-mini-instruct（`text_phi4/`）

Phi-4-mini 是 `Phi3ForCausalLM`：32 层 / hidden 3072 / 24 q 头 8 kv 头（GQA 3 组）/
head_dim 128 / intermediate 8192 / vocab 200064 / `tie_word_embeddings=true`。
导出物与 Qwen **共用同一套张量 I/O 名字与形状**，所以 `llm_chat` / `kv_cache` /
`conversation` 的加载逻辑一行都不用改；架构差异全部消化在导出脚本和 runtime 算子层。

```bash
# 1) 导出 ONNX（权重：ModelScope LLM-Research/Phi-4-mini-instruct）
python3 phi4_export_onnx.py            # → text_phi4/llm.onnx + llm.weights.bin
python3 dump_embed_tokens_phi4.py      # → text_phi4/embed_tokens.bin (200064x3072 fp16)
# 2) 数值对齐（ORT fp16 CPU vs HF）：prefill logits + 逐层 present_kv + decode 单步
python3 phi4_check_onnx.py
# 3) 转 vkopbin —— 多 GB 图不要加 -u（UnifiedMeta 的偏移是 int32，会溢出）
python3 -m onnx2vkop.cli -i text_phi4/llm.onnx -o text_phi4/llm.vkopbin
# 4) tokenizer bin（MODEL_DIR / OUTPUT_BIN 两个环境变量决定输入与产物）
MODEL_DIR=~/.cache/modelscope/models/LLM-Research--Phi-4-mini-instruct/snapshots/master \
  OUTPUT_BIN=../tokenizer/phi4_mini.bin python3 ../tokenizer/tokenizer_to_bin.py
# 5) 三方 token 对齐（HF generate / ONNX Runtime / vkop GPU）
python3 greedy_ref.py "用一个词回答：天空是什么颜色？"   # 默认指 text_phi4/
```


**与 Qwen3 的四处架构差异**（都在导出脚本里逐 op 贴 HF 实现）：

1. **fused 投影**：`qkv_proj` 一个 MatMul 出 q/k/v，`gate_up_proj` 一个 MatMul 出
   gate/up。不动权重手术，照 HF 用切片拆**输出**（`qkv[..., :3072]` 等），数值路径
   与 HF 一字不差。gate/up 用常量边界切片而**不是** `chunk(2, -1)` —— 见下面坑 1。
2. **partial RoPE**：`rotary_dim = head_dim x 0.75 = 96`，每个 head 后 32 维不参与
   旋转直通。runtime 的 `RotaryEmbedding` 原本假设 cos/sin 行宽 == head_dim，
   现已按 cos 的实际行宽取 `rotary_dim`（等于 head_dim 时行为完全不变）。
3. **longrope**：`attention_scaling = 1.190238` 由 HF 的 `rotary_emb` 乘进 cos/sin，
   所以必须复用 HF 的 rotary_emb 而不是手写 RoPE；自己算会漏掉它，注意力分数差
   1.190238^2 = 1.4167 倍。
4. **模板与停止符**：Phi 的每轮壳是 `<|role|>...<|end|>`（无换行），且
   `add_generation_prompt=False` 时模板会给整段对话补一个 eos。当时为它加的
   `im_end_token_id()` 双字面量探测，到 GLM 那一族已经不够用（三个停止 id、模板还
   没有轮末标签），现在统一成 `stop_token_ids()`，见下一节。

**跑通 Phi 期间修掉的三个静默错误**（都会让输出变成全 0 或错位，且都不报错）：

- **转换器的 Unsqueeze 消除会吃掉 GQA 的升秩**。`fuse_unsqueeze_eliminate` 无条件把
  单轴 Unsqueeze 折进消费者；Expand 的广播是「shape 列表右对齐到输入秩」，少一维就
  把复制轴安错位置 —— Phi 的 `repeat_kv` 复制了 8 份而不是 3 份，尺寸对不上之后
  下游 MatMul 直接读到一个没被写过的全 0 缓冲区。现在 Expand 作为消费者的 Unsqueeze
  一律保留，交给 runtime 真执行 view。
- **`chunk(2, -1)` 导出的整数 shape 链被融成 FUSED_ELEMWISE**。那条链是
  Shape→Gather→Add→Div→Mul→Slice（全 int64），而 FUSED_ELEMWISE 只有 fp16/fp32 变体；
  融掉之后 Slice 的 `ends` 不再是 int64 张量，`SliceBuffer::execute` 里
  `as_tensor<int64_t>` 得到空指针，直接 SIGSEGV（崩在
  `Tensor<long long>::copyToCPU`，this=0）。导出侧改成常量边界切片，顺带每层少 4 个
  动态 shape 节点。
- **超过 1GB 的张量上传会静默失败**。staging pool 上限 1GB，而 Phi 是
  tie_word_embeddings，`lm_head` 就是那张 200064x3072 的 fp16 表 = 1.23GB；
  一次分配失败后原代码只 `printf` 一句就 return，SSBO 保持全 0 —— 症状是 logits 全 0、
  argmax 恒等于 id 0（输出变成一串 `!`）。现在按 64MB 分块上传，常规尺寸仍走单次快速路径。

**三方 token 一致（greedy，prompt「用一个词回答：天空是什么颜色？」）**：

```
HF generate = ONNX Runtime = vkop GPU = [72721, 4472, 788, 200020]  →  '蓝色。' + 轮末符
```

`phi4_check_onnx.py` 的数值口径：prefill logits 相对均值差 1.5e-3（fp16+CPU ORT 的
累积舍入），32 层 present_key_values 全部 < 1.4e-3，decode 单步 argmax 一致，
每轮 GPU 约 330ms/token。

---

## 第三个架构：GLM-Edge-1.5B-Chat（`text_glm_edge/`）

`ZhipuAI/glm-edge-1.5b-chat`（ModelScope）是 `GlmForCausalLM`：28 层 / hidden 2048 /
16 q 头 4 kv 头（GQA 4 组）/ head_dim 128 / intermediate 6144 / vocab 59264 /
`tie_word_embeddings=true` / `max_position_embeddings=8192`。导出物同样复用那套张量
I/O 名字与形状，`llm_chat` / `kv_cache` 的加载逻辑一行没改；架构差异全部落在导出脚本、
分词器和 `Conversation` 的轮边界处理里。

```bash
# 1) 导出 ONNX（朴素 pre-norm decoder，分离 q/k/v 投影 + fused gate_up_proj）
python3 glm_edge_export_onnx.py          # → text_glm_edge/llm.onnx + llm.weights.bin
python3 dump_embed_tokens_glm_edge.py    # → text_glm_edge/embed_tokens.bin (59264x2048 fp16, 243MB)
# 2) 数值对齐（ORT fp16 CPU vs HF）：prefill logits + 逐层 present_kv + decode 单步
python3 glm_edge_check_onnx.py
# 3) 转 vkopbin —— 权重 2.94GB，同样不能加 -u（UnifiedMeta 偏移是 int32）
python3 -m onnx2vkop.cli -i text_glm_edge/llm.onnx -o text_glm_edge/llm.vkopbin
# 4) tokenizer bin
MODEL_DIR=~/.cache/modelscope/models/ZhipuAI--glm-edge-1.5b-chat/snapshots/master \
  OUTPUT_BIN=../tokenizer/glm_edge.bin python3 ../tokenizer/tokenizer_to_bin.py
# 5) 三方 token 对齐（ONNX Runtime 这一路；HF 与 GPU 见下）
ONNX=text_glm_edge/llm.onnx EMBED=text_glm_edge/embed_tokens.bin \
MODEL_PATH=~/.cache/modelscope/models/ZhipuAI--glm-edge-1.5b-chat/snapshots/master \
  python3 greedy_ref.py "用一个词回答：天空是什么颜色？"
# 6) GPU
../../build/llm_chat text_glm_edge/llm.vkopbin text_glm_edge/embed_tokens.bin \
  ../tokenizer/glm_edge.bin 48
```

**与 Qwen3 / Phi-4 的三处差异**（每一处都是「切错了不报错」的那类）：

1. **RoPE 的配对方式是交错（interleaved）的**：`modeling_glm.apply_rotary_pos_emb` 用
   `x[..., 0::2] / x[..., 1::2]` 配对并 `repeat_interleave(2)` cos/sin，即维度 (2i, 2i+1)
   共享角度 i；Qwen/Phi 的 `rotate_half` 配的是 (i, i+64)。runtime 的
   `RotaryEmbedding` 内核和转换器的 `fuse_rotary_embedding` 都只按半切实现，所以导出时
   把交错形式**改写成半切 + 固定列置换**：对 x 施加 `P = [偶数列 ‖ 奇数列]`、做半切 RoPE、
   再施加 `P⁻¹`。HF 的 `cos = cat(f, f)` 正好等于半切所需宽度，于是权重不动、内核不加模式；
   代价是 q/k 各两个 Reshape+Transpose（成对抵消，present_key_values 仍与 HF 同基，
   逐层 KV 对齐照常可比）。脚本导出前先自证这两种写法逐位等价。
2. **分词器是 GPT-2 扫描器的第三种口径**：GLM 的 pre_tokenizer 与 Qwen3-VL 只差数字段
   ——`\p{N}` 对 `\p{N}{1,3}`（字母分支完全是 GPT-2 口径），而 Phi 的 o200k 是按大小写
   形状切分。所以 flags 加 **bit2 = 数字段按 1~3 个切**（`tokenizer.cpp` 的
   `pre_tokenize(..., digit_max)`），GLM 的 bin 是 `flags=6`（bit1 无 NFC + bit2）。
   写端逐分支比对已知正则表、表外形状直接报错，读端拒绝任何未定义的 flag 位：span 切错
   只会静默产出错 token，整条数值链路跑完都看不出来。早先用
   `"\p{N}{1,3}" in 全文` 判 Phi 就是把 GLM-Edge 误判成 Phi 的那次事故。
3. **模板没有轮末标签，停止符是一组 id**：GLM 的 chat template 每轮渲染成
   `<|user|>\n{content}`，`eos_token_id = [59246, 59253, 59255]`（依次是
   `` / `<|user|>` / `<|observation|>`，见 tokenizer.json 的 `added_tokens`）—— 一轮是由**下一轮的开头标签**封口的。于是
   `im_end_token_id()` 换成 `stop_token_ids()`（按字面量查、命中任意一个即停），
   `Conversation` 的收尾串改由 `turn_tail_literal()` 从模板剥出（GLM 剥出来是空串）。
   空收尾串就是这一族的信号：assistant 轮里生成出来的那个 `<|user|>` **必须丢掉**，
   因为下一轮的 `apply_chat_template` 还会再渲染一次 —— 留着就成了双份角色标签，
   历史序列不再是 HF 能渲染出来的那个串。ChatML/Phi 那一族正相反，轮末标签就是收尾串本身，
   必须保留。

**三方 token 一致（greedy，prompt「用一个词回答：天空是什么颜色？」）**：

```
HF generate (fp16 与 bf16 同解) = ONNX Runtime = vkop GPU
  = [21499, 372, 7390, 9235, 326, 59253]  →  '天空是蓝色的。<|user|>'
```

prompt 侧也逐 token 相同：13 个 id `[59253, 10, 551, …]`（GPU 侧用 `VKOP_CHATDBG=1`
打印）。`glm_edge_check_onnx.py` 的数值口径：prefill logits 相对均值差 1.5e-3、argmax
一致率 1.000，28 层 present_key_values 全部 < 1.7e-3，decode 单步 argmax 相同；
GPU 上 prefill 约 90ms、decode 约 110ms/token。多轮：第二轮的历史是 37 个 token =
第一轮 user 13 + assistant 正文 11 + 第二轮 user 13，正好是「丢掉末尾那个 `<|user|>`
停止符」之后的长度（留着就是 38、并且角色标签成双），模型答 `小明。`。把它和 HF 整段
（3 轮）渲染重 encode 的 36 个 token 对齐比：唯一差别在 assistant 正文里那一个 token
（生成时是 `…13127, 552…` 两段，重 encode 同一段文本只出 10 段）—— 这正是
`Conversation` 绝不重分词 assistant 轮的原因；两个 role 标签、换行和轮边界逐 token 相同。

上下文上界是 8192（config 的 `max_position_embeddings`，与驱动侧 KV 预分配上界一致）；
rope 是 `rope_type=default`、`attention_scaling=1.0`、inv_freq 常量，没有 Phi-4 那种
数据相关的 longrope 分支。config 里没有 `sliding_window` / `layer_types` 字段，导出与
对齐脚本都就此断言一次 —— 有滑窗的话加法 causal bias 不再与 HF 等价。

### 局限（llm_chat）

- **仅 greedy**：无 temperature/top-p 采样，与 `Qwen3VLInference` 的 `do_sample=True`
  路径不等价。logits 已与 ORT 对齐（见 memory `llm-chat-generate-loop`），接入采样器
  后分布一致，但驱动本身未实现采样。
- **单图整会话**：`--image` 指定的图片对整个会话（所有轮次）生效；不支持中途换图，
  也不支持一轮里多张图。多图需扩展 `expand_image_token` + 多组 deepstack 特征。
- **视觉尺寸固定**：图片必须是 32 的整数倍（224×224 默认），且 `visual.vkopbin`
  的 grid_thw 在导出时固化（见上文「视觉编码器」节）。不同尺寸需重新导出 + 转换。
- **多轮 = 每轮重算 KV**：会话历史在 token 层累积后整体重新 prefill（见上文
  「多轮上下文」一节）；跨轮续用 KV（跳过已算前缀）还没接——vkop 没有 paged KV /
  block table，`Conversation::prefixMatch()` 留着校验点。
- **Phi-4-mini 上下文 ≤ 4096**：longrope 在 `rotary_emb` 里按
  `max(position_ids) > original_max_position_embeddings(=4096)` 选 short/long factor，
  那是数据相关分支，trace 时被固化成导出时走的那一支（short factor 全 1，等价普通
  RoPE，脚本导出前会断言这一点）。要超 4096 得用 long_factor 重新导出。
