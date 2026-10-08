# FP8 KV Cache 全链路实现笔记

> 临时设计文档。目标:让 vkop 的 LLM 推理支持 E4M3 FP8 KV cache,把每层
> `past/present_key_values` 的存储从 fp16 (2B/elem) 降到 fp8 (1B/elem),
> 显存减半。attention 计算仍走 fp16 —— 在 cache 边界做 fp8↔fp16 量化/反量化。

## 现状关键事实(已核对)

- **KV cache 管理器** `llm/exporter/kv_cache.hpp`:
  - 构造函数 `as_tensor<uint16_t>(pin)->preallocate_buffer(dev, kv_elems)`,
    `kv_elems = 2 * nkv * max_kv * hd` —— 硬编码 fp16。
  - `reset_for_prefill()` / `upload()` / `feedback()` 全用 `as_tensor<uint16_t>`。
  - `feedback()` 用 `swap_gpu_buffer_with` 交换 present→past 的 VkBuffer(dtype 无关)。
- **LLM 驱动** `llm/exporter/llm_chat.cpp`:
  - `infer_model_arch` 用 `as_tensor<uint16_t>(pk0)` 读 NKV/HD。
  - `FP16_MIN = 0xFBFF` 是 causal mask 填充值(attention_bias 输入,不是 cache)。
  - `fill_fp16_input` / `upload_input` 按 uint16_t 走 cache。
- **DType** `core/DType.hpp` / `core/DType.cpp`:
  - `kFloat8E4M3FN` / `kFloat8E5M2` 已定义,`elem_kind_supported` 返回 true
    (但注释说"weight-only MatMul only")。
  - `storage_matches_kind` 对 fp8 返回 false —— 没有对应 C++ 类型,走 int8_t 容器。
  - `require_word_movable_elem` 对 ≤8-bit 一律 throw(Concat/Slice/Gather/Transpose
    /Expand/Reshape 全挡)。
- **运行时输入创建** `core/runtime.cpp:148-196`:
  - switch 覆盖 int64/int32/int8/bool/float32/float16;fp8 落到 `default:` throw。
- **导出器** `llm/exporter/qwen3_export_onnx.py:155-183`:
  - `present_kv = torch.stack([k_new, v_new], dim=1)` —— cache 是 fp16。
  - cache shape `(B, 2, NKV, kv_len, HD)`,dim 1 = {K, V}。
- **转换器** `model/pypi/onnx2vkop/dag.py`: ONNX dtype 17/19 → "float8e4m3fn"/"float8e5m2"。
- **量化器** `model/pypi/onnx2vkop/optimizer.py:4976-5119` `quantize_to_fp8_weight_only`:
  - per-output-column fp32 scale = amax / max_finite (E4M3=448)。
  - `_fp8_encode` RNE 编码;overflow 检查 Inf/NaN 字节。
- **fp8 shader 解码** `shaders/buffer/matmul.comp:137-149`:
  - 手工字段重组 e4m3_val/e5m2_val(无 VK_EXT_shader_float8)。
- **word-mover 闸门** `ops/Concat.hpp:376-379` 等 fail-closed。

## 设计决策

### 方案:cache 存 fp8,concat 边界反量化回 fp16

每层 cache 存 E4M3 字节 + 每层一个 fp32 per-tensor scale。图内:
- **写 cache 前**:新算出的 K/V (fp16) → QuantizeLinear → fp8 bytes 存入 present。
- **读 cache 后**:past (fp8) → DequantizeLinear → fp16,喂给 attention。

attention 全程 fp16 不变,现有 kernel 零改动。

### scale 策略:第一版用 per-tensor 静态 scale

- 从校准数据(或直接用模型权重的统计)离线算一个 fp32 标量,作为图常量。
- 省掉 GPU amax kernel + 每轮 readback。
- 风险:长对话 V 侧值域漂移可能超 ±448。用 ORT 逐轮比对验证;不行再上动态 scale。

### 因果掩码

`FP16_MIN` 是 attention_bias 输入的填充,不是 cache —— 保持 fp16 不动。
cache 自身不需要掩码值。

## 实施步骤(按依赖序)

### 步骤 1:运行时 fp8 字节容器

**文件**: `core/runtime.cpp` 输入创建 switch (148-196)

加 `case ElemKind::kFloat8E4M3FN:` / `case ElemKind::kFloat8E5M2:`:
- `Tensor<int8_t>` 承载字节(和 int8/bool 同容器)。
- `set_elem_kind(kind)` 记录语义。
- `as_storage_buffer(dev)`。
- 注释:fp8 是 payload,shader 解码,不参与 elem_kind_of_storage 推断。

输出创建 (208-222) 暂不动 —— present_key_values 在 fp8 方案里仍是 fp16 图输出?
**否**:如果 cache 存 fp8,present 也必须是 fp8(否则 swap 不了)。需要让输出也按
graph dtype 走。但 runtime 输出现在按 `precision_` (fp16/fp32) 二选一。
**决策**:输出也加 dtype 感知 —— 当输出 dtype 字符串是 fp8 时,创建 int8_t 容器。

### 步骤 2:KVCache 参数化存储类型

**文件**: `llm/exporter/kv_cache.hpp`

构造函数加 `ElemKind cache_kind = ElemKind::kFloat16` 参数:
- fp16: `as_tensor<uint16_t>`, `kv_elems = 2*nkv*max_kv*hd`。
- fp8: `as_tensor<int8_t>`, `kv_elems = 2*nkv*max_kv*hd` (1B/elem,总数不变)。
- 用模板/重载避免到处 `if`。

`reset_for_prefill` / `upload` / `feedback` 按 cache_kind 分派到对应 `as_tensor<T>`。

### 步骤 3:LLM 驱动读 cache dtype

**文件**: `llm/exporter/llm_chat.cpp`

`infer_model_arch` 从 `past_key_values_0` 的 elem_kind() 读 cache dtype(不再假设 uint16)。
KVCache 构造传入该 dtype。
`upload_input` 已按 dtype 分派(int8_t 路径走 fp8)。

### 步骤 4:QuantizeLinear / DequantizeLinear buffer op

**文件**: 新建 `ops/QuantizeLinear.hpp` + `shaders/buffer/quantize_linear.comp`

- QuantizeLinear: fp16/fp32 input → fp8 output + fp32 scale。
  - shader 复用 matmul.comp 的 e4m3 编码逻辑(反向)。
  - per-tensor scale 从 push constant 传(或 scale input tensor)。
- DequantizeLinear: fp8 input + fp32 scale → fp16/fp32 output。
  - shader 复用 matmul.comp 的 e4m3_val 解码。

### 步骤 5:Concat byte build(cache 追加)

**文件**: `ops/Concat.hpp` + `shaders/buffer/concat.comp`

cache 方案里,Concat 仍作用于 fp16(past 反量化后 + 新 K/V),不是 fp8。
**重新审视**:如果 cache 存 fp8,那么"past fp8 + 新 K/V fp16 → present fp8"的 Concat
跨 dtype,ONNX 不允许。所以要么:
- (A) Concat 在 fp16 域:读 past 时先 DequantLinear→fp16,Concat fp16,再 QuantLinear→fp8 存 present。每轮 2 次额外 dispatch。
- (B) Concat 在 fp8 域:需要 fp8 Concat byte build(1B/elem,4-per-word 打包或 1-per-thread)。

**决策**:方案 A 更简单,复用现有 fp16 Concat;额外 dispatch 成本在 decode (~260ms/轮)
里可忽略。先做 A。

→ 步骤 5 降级为"确认 fp16 Concat 路径不变",无需新 shader。

### 步骤 6:导出器插 Quant/Dequant

**文件**: `llm/exporter/qwen3_export_onnx.py` 加 `--kv-fp8` 分支

在 `decoder_layer` 里:
- 读 past: `past_kv (fp8) → DequantLinear → fp16 past_k/past_v`。
- 写 present: `present_kv (fp16) → QuantLinear → fp8`。

scale 用 per-tensor 常量(从权重统计或固定值)。
past_key_values 输入 dtype 改 float8e4m3fn。

### 步骤 7:转换器/加载器确认  ✅ 已核对

- `dag.py` `_DATA_TYPE_MAP` 已映射 dtype 17/19 → "float8e4m3fn"/"float8e5m2"。
- `converter.py` `_io_dtype_name` 对 graph input/output 用 `_DATA_TYPE_MAP` 解析,
  **格式无 name 时 HARD-ERROR**(防止静默回退 fp16)。fp8 有 name,放行。
- `dag.py` `_dtype_to_str` 对已是字符串的 dtype 原样返回,fp8 拼写直达 FlatBuffer
  `ShapeRef.dtype`。
- 加载器 `require_supported_elem` 对 fp8 返回 true。
- runtime.cpp 输入路径(197-205)/输出路径(226-236)按 `elem_kind_from_name` →
  `kFloat8E4M3FN` 创建 `Tensor<int8_t>` + `set_elem_kind`。

## 验证

### 单元测试  ✅ 3/3 通过 (byte-exact vs torch Float8_e4m3fn)

`tests/QuantizeLinearTest.cpp`:
- `RoundTripFp16Fp8Fp16`: unit scale,fp8 字节逐个 == torch `Float8_e4m3fn` cast,
  round-trip fp16 也 byte-exact。覆盖 normal/subnormal/saturation(±448)。
- `ScaledRoundTrip`: scale=0.1,对 torch 自身 fp8 round-trip(apply 同款 val/scale)
  byte-exact。E4M3 网格噪声(val~10→code 96,dequant 9.6)是预期,不是 bug。
- `OddTotal`: 5 元素,验证 word-packing 边界(4 vals/fp8 word,2 vals/fp16 word)。

全量回归:`all_tests -MatMulTest.*:NmsTest.*` → 153 passed / 1 skipped / 0 failed
(MatMul/Nms 在 Intel ARL 上 flaky,与本次改动无关)。

### 端到端验证  ⏳ 待用户执行

- `VKOP_KV_FP8=1 python3 qwen3_export_onnx.py` 导出 fp8 KV cache 版 llm.onnx。
- onnx2vkop 转换(用户手动,VSCode 在 3.4GB 转换时崩溃)。
- ORT 逐轮比对:6/6 decode rounds MATCH。
- NANSCAN:无 NaN/Inf。
- 显存:KV cache 部分减半。

## 待确认

- 静态 per-tensor scale 的取值:默认 0.1(VKOP_KV_SCALE 可调),覆盖 Qwen3 K/V 经
  RMSNorm 后的典型值域。长对话 V 侧漂移超 ±448 时再上动态 per-round GPU amax。
- 第一版静态 scale 已就绪;动态 scale 是降级备选,非阻塞。
