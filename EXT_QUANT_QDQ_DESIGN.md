# 外部量化 (QDQ int8) 导入支持 — 设计笔记

> 临时设计文档。目标:让 vkop 能导入图里已经带 QuantizeLinear/DequantizeLinear
> 节点(int8 + scale + zero_point,非对称)的 ONNX 模型 —— 即"外部量化",区别
> 于转换器自带的 `-q int8` 内部 weight-only 量化。
>
> 用途:权重 int8 量化省显存(activation 仍 fp16/fp32)。
> 测试模型:ModelScope 下载 Qwen2.5-0.5B-Instruct(纯文本,~943MB fp16),用
> ORT quantize_dynamic 给它加 QDQ 节点,或直接导出一个带 QDQ 的 ONNX。

## 现状关键事实(已核对)

### 已有(内部 weight-only 量化,工作正常)

- **转换器** `model/pypi/onnx2vkop/optimizer.py:4142 Quantizer`:
  - `quantize_to_int8_weight_only`(4353):per-output-column fp32 scale,int8 字节。
  - `quantize_to_fp8_weight_only` / `_4bit` / `_nf4` / `_nvfp4` 类似。
  - CLI:`onnx2vkop -i model.onnx -q int8`。把 fp16/fp32 权重在转换时量化。
- **MatMul** `ops/Matmul.hpp`:weight-only int8 路径吃 `[A_fp, B_int8字节, scale(N)]`,
  着色器 `matmul.comp:w8_val` 内部 dequant(b = byte - 128,symmetric)。
  - **symmetric only**(w8_val = `float(b ^ 0x80) - 128.0`,即 byte 当无符号偏移 128)。
  - 无 zero_point 概念。
- **Conv2d** `ops/Conv2d.hpp`:同样 weight-only int8 + scale。

→ 这条线**完整且工作正常**,但是是"转换器内部量化",不是"导入外部 QDQ 图"。

### 没有(外部 QDQ 导入)

- **转换器** `converter.py`:node-passthrough。读每个 ONNX 节点的 op_type 原样写
  FlatBuffer,不认识 Q/DQ 语义,不折叠,不做 shape 推断。QDQ 节点会进 vkopbin,
  runtime 需要真正的 Q/DQ op 执行。
  - 但 `converter.py:_io_dtype_name`(42)对 graph I/O 的 elem_type 会 hard-error
    如果没 name;int8/uint8 有 name("int8"/"uint8"),放行。
  - `dag.py:_DATA_TYPE_MAP`:2→"uint8",3→"int8"。`_DTYPE_BYTES`:2→1,3→1。
- **runtime** `core/runtime.cpp`:
  - 输入容器 switch(149):`kInt8`/`kBool` → Tensor<int8_t>(165);**没有 kUInt8
    单独 case**(uint8 落到哪?需查 —— 见待办)。
  - 节点输出 dtype_marker(642):`QUANTIZE_LINEAR`→`_i8_`(硬编码,本意 fp8),
    `DEQUANTIZE_LINEAR`→`_f16_`(硬编码)。对 int8 QDQ 不对:
    - int8 QuantizeLinear 输出 elem_kind 应是 kInt8(不是 fp8)。
    - int8 DequantizeLinear 输出应跟精度(fp16 或 fp32),不是固定 fp16。
- **QuantizeLinear.hpp** + `shaders/buffer/quantize_linear.comp`:
  - **fp8 E4M3 专用**。硬编码 e4m3_encode/e4m3_decode 字段重组。
  - **忽略 zero_point 输入**(只读 inputs[2] 的 scale,从不读 zero_point)。
  - 方向靠 `inputs[0]->dtype()` 推(uint16→quant,int8→dequant)—— 对 fp8 OK,
    对 int8 QDQ 也碰巧能分(uint16/fp32→quant,int8/uint8→dequant),但语义不同。
- **MatMul**:**只支持 weight 端 int8**(B 是 int8 容器 + appended scale)。
  不支持 activation 端 int8。QDQ 图若 DQ 把 weight 展开成 fp16 再喂 MatMul,
  走普通 fp16 路径(失去省显存优势,见下)。

### 工具链(利好)

- `~/.venv/bin/python`:onnx 1.19 / torch 2.10 / transformers 5.3 /
  **onnxruntime 1.22(含量化 API `quantize_dynamic`)** / modelscope 1.37。
- ModelScope 网络可达(HF 不行)。Qwen2.5-0.5B-Instruct ~943MB,可下。
- ORT `quantize_dynamic` 能给任意 ONNX 加 QDQ 节点(QuantFormat.QDQ, QInt8/QUInt8)。

## 核心设计权衡:QDQ 路径 vs weight-only 折叠

标准 QDQ 图长这样(ORT dynamic quantize 产出):
```
   past_kv / activation (fp16)
        │
   DequantizeLinear(weight_int8, scale, zero_point) ──→ weight_fp16
        │
        ▼
     MatMul(A_fp16, B_fp16)   ← 普通 fp16 MatMul
```
**问题**:DQ 节点把 int8 权重**展开成 fp16**,运行时仍是 fp16 权重,**不省显存**。
省的只是 ONNX 文件大小(磁盘上 int8),VRAM 没省。

要真正省 VRAM,必须让权重在 GPU 上**保持 int8 字节**,在 MatMul 着色器内部 dequant。
即:转换器把 `DequantizeLinear → MatMul` **折叠**成 weight-only 形式:
```
   MatMul(A_fp16, B_int8字节, scale)   ← 复用现有 weight-only int8 路径
```
这正好是 vkop 已有的 `[A, B_int8, scale]` MatMul。**但现有路径是 symmetric
(w8_val = byte - 128),不支持 zero_point(非对称)**。

### 决策:分两阶段

**阶段 1(本阶段)—— QDQ 正确性:让 DQ 节点能正确执行(int8→fp16)**
- 扩展 QuantizeLinear/DequantizeLinear op 支持 int8/uint8 + scale + zero_point(非对称)。
- DQ 输出 fp16,走普通 fp16 MatMul。**不省 VRAM,但能跑通外部 QDQ 图**。
- 这是"导入"的基础:任何带 QDQ 的 ONNX 都能正确执行,精度对齐 ORT。
- 适用:activation 量化、或权重量化但不在乎 VRAM 的场景。

**阶段 2(后续)—— QDQ→weight-only 折叠:省 VRAM**
- 转换器加 pass:`DequantizeLinear(weight_const, scale, zp) → MatMul` 折叠成
  `MatMul(A, weight_int8, scale)`,要求 weight 是 Constant(可离线 dequant 验证)。
- MatMul weight-only int8 路径扩展支持 zero_point(非对称):`w8_val = (byte - zp) * scale`
  或等价。现有 `w8_val = float(b^0x80) - 128` 改成读 push-constant 里的 zp。
- 这才真正省 VRAM(权重存 int8 字节)。
- 需要处理 zero_point 的 per-tensor vs per-channel。

→ **本设计文档聚焦阶段 1**。阶段 2 在阶段 1 验证通过后另起。

## ⚠️ 先修的 bug:现有 fp8 op 读错 scale 输入位

**核对发现**:`ops/QuantizeLinear.hpp:79` 读 `inputs[2]` 当 scale,但 ONNX 规范
`QuantizeLinear/DequantizeLinear` 输入是 `[x, scale, zero_point]` —— **scale 是
inputs[1],inputs[2] 是 zero_point**。

证据:
- ONNX `helper.make_node("QuantizeLinear", inputs=["x","scale","zp"])` →
  input 顺序 `[x, scale, zero_point]`。
- KV cache 图 `insert_fp8_kv_cache`:`DequantizeLinear(inputs=[past_name, scale_name])`
  只有 2 输入,scale 在 inputs[1],**无 inputs[2]**。
- op 读 inputs[2] → nullptr → fall back `scale=1.0`。
- 单元测试 `onExecute({input, nullptr, scale_t})` 把 scale 放第 3 位(inputs[2]),
  碰巧喂进了 op 读的位置 —— 测试通过但**掩盖了 bug**。

**后果**:真实 KV cache 图里 scale_name=0.1 **从未被 op 读到**,fp8 量化实际用
scale=1.0。round-trip 仍 byte-exact(encode/decode 都用 1.0 自洽),但 K/V 值域
没被 0.1 缩放 → 超过 ±448 的值被饱和截断,丢精度。单元测试没覆盖(测试自己
把 scale 塞到 inputs[2])。

**修法**(步骤 1 一并做):op 读 scale 改成 `inputs[1]`(ONNX 规范位),zero_point
读 `inputs[2]`(若存在)。同步修测试 `onExecute({input, scale_t, nullptr})` 或
`onExecute({input, scale_t})`。fp8 路径无 zero_point,inputs[2] 为空 → 不读 zp。

→ **这是阶段 1 的前置修复**,因为它正好是"QDQ op 要正确读 scale/zero_point"的
同一处代码。重构时一并修对。

## 阶段 1 实施:通用 int8/uint8 QDQ

### 关键约束

- **zero_point 必须支持**:ORT dynamic 默认非对称(uint8,zp 通常非 0)。
  现有 fp8 op 完全忽略 zero_point,必须补。
- **scale 形状**:per-tensor(scalar)或 per-axis(1-D,长度 = 某轴)。
  ONNX DequantizeLinear 有 `axis` 属性指定 scale 广播轴。
  - 阶段 1 先做 **per-tensor**(scalar scale + scalar zp),覆盖 dynamic quantize 的
    权重量化(它对每个权重张量给一个 scale+zp)。
  - per-axis 留后续(静态量化常用 per-channel)。
- **dtype 组合**:
  - QuantizeLinear:fp16/fp32 input → int8/uint8 output + fp32 scale + int8/uint8 zp。
  - DequantizeLinear:int8/uint8 input + fp32 scale + int8/uint8 zp → fp16/fp32 output。
  - fp8(E4M3/E5M2)路径**保留**(KV cache 用),不能破坏。

### 步骤 1:重构 QuantizeLinear.hpp — 多格式分派

**文件**: `ops/QuantizeLinear.hpp`

当前:单一 fp8 路径,方向靠 `inputs[0]->dtype()` 推。

改为:根据 **input dtype + zero_point 存在性** 分派三种模式:
1. **fp8 quant**(fp16/fp32 → E4M3 fp8):现有路径,保留。触发条件:input 是 float,
   且无 zero_point 输入(或 zp 类型是 fp8)。**或** 用一个明确的 elem_kind 标志。
2. **int8/uint8 quant**(fp16/fp32 → int8/uint8 + scale + zp):新增。
3. **int8/uint8 dequant**(int8/uint8 → fp16/fp32):新增。
4. **fp8 dequant**(fp8 → fp16/fp32):现有路径,保留。

**分派依据**(清晰且不依赖隐式约定):
- 看 `inputs[0]->elem_kind()`(数据输入):
  - `kFloat16`/`kFloat32` → quant 方向。
    - 看 `outputs[0]->elem_kind()`:
      - `kFloat8E4M3FN`/`kFloat8E5M2` → fp8 quant(现有)。
      - `kInt8`/`kUInt8` → int8 quant(新)。
  - `kInt8`/`kUInt8`/`kFloat8E4M3FN`/`kFloat8E5M2` → dequant 方向。
    - 输出按精度(fp16/fp32)。
- zero_point 是 `inputs[2]`(QuantizeLinear)或 `inputs[2]`(DequantizeLinear,
  ONNX: x, scale, [zero_point])。**注意**:fp8 路径里 scale 是 inputs[2] 而
  zero_point 不存在 —— 但 ONNX 标准里 inputs[1]=scale, inputs[2]=zero_point。
  **现有 fp8 op 把 scale 当 inputs[2] 是个 bug/偏移**(对照 ONNX 规范)。

  → 查 ONNX QuantizeLinear 签名:`(x, y_scale, y_zero_point)`,scale 是 **inputs[1]**。
    现有代码读 `inputs[2]` 当 scale —— 这跟 ONNX 不一致(可能测试里 scale 传在第3位)。
    **int8 路径必须按 ONNX 规范 inputs[1]=scale, inputs[2]=zero_point**。
    fp8 路径保持现状(兼容现有测试/KV cache 图)。

**Push constant 扩展** `quant::QuantPC`:
```cpp
struct alignas(16) QuantPC {
    int mode;       // 0=quant, 1=dequant
    int total;      // element count
    float scale;    // per-tensor scale (阶段1; per-axis 走 binding)
    int zp;         // zero_point as int (per-tensor; 阶段1)
    int fmt;        // 0=fp8_e4m3, 1=fp8_e5m2, 2=int8, 3=uint8
    int out_fp32;   // 0=fp16 output, 1=fp32 output (dequant 用)
    int _pad[2];
};
```

### 步骤 2:重写 quantize_linear.comp — 多格式着色器

**文件**: `shaders/buffer/quantize_linear.comp`

当前:硬编码 e4m3_encode/e4m3_decode,fp16↔fp8 word packing。

改为:按 `uPC.fmt` 分派:
- **fmt 0/1 (fp8)**:现有 e4m3/e5m2 编解码(加 e5m2,见之前讨论)。fp16↔fp8 packing。
- **fmt 2 (int8)**:
  - quant: `q = round(x / scale) + zp`,clamp [-128,127],cast int8。
  - dequant: `x = (float(q) - zp) * scale`。
  - packing: int8 4字节/word;fp16 2值/word;fp32 1值/word。
- **fmt 3 (uint8)**:
  - quant: `q = round(x / scale) + zp`,clamp [0,255]。
  - dequant: `x = (float(q) - zp) * scale`。
  - 容器仍是 int8_t(无 uint8_t 容器,按无符号解释)。

**zero_point 处理**:per-tensor zp 进 push constant(int)。per-axis(后续)走 binding。

**word packing 重做**:现有 fp8 路径假设 fp16 input(2/word)+ fp8 output(4/word)。
int8 路径要支持:
- quant: fp16/fp32 input → int8/uint8 output。
- dequant: int8/uint8 input → fp16/fp32 output。
- 通用化:1 thread per output-word,int8/uint8 output 4字节/word;
  fp16 output 2值/word;fp32 output 1值/word。input 侧对称。

### 步骤 3:runtime.cpp dtype_marker 修正

**文件**: `core/runtime.cpp:642-651`

当前:`QUANTIZE_LINEAR`→`_i8_`,`DEQUANTIZE_LINEAR`→`_f16_`(硬编码)。

改为:
- `QUANTIZE_LINEAR`:输出 dtype 由 **output 的 ONNX elem_type** 决定(转换器记录在
  `out_shape.dtype`)。若是 int8/uint8/fp8 → `_i8_` 容器 + 对应 elem_kind;
  否则(float)→ `_f16_`/`_f32_`。
- `DEQUANTIZE_LINEAR`:输出 dtype 由 output elem_type 决定(float16→`_f16_`,
  float32→`_f32_`),**不硬编码 fp16**。

同时确认 elem_kind 设置(777 附近)对 int8/uint8 输出正确设 `kInt8`/`kUInt8`
(而非 fp8)。

### 步骤 4:转换器 — QDQ 节点 passthrough 验证 + zero_point initializer

**文件**: `model/pypi/onnx2vkop/converter.py` / `dag.py`

转换器已是 node-passthrough,QDQ 节点会原样进 vkopbin。需确认:
- `QuantizeLinear`/`DequantizeLinear` 节点的 scale/zero_point 通常是 **Constant
  initializer**(per-tensor 标量)。它们走 `dag_model.initializers` 正常路径
  (fp32 scale + int8/uint8 zp),`_init_byte_len`/`_init_bytes` 已支持。
- 节点的 `axis` 属性(per-axis scale)需序列化 —— 阶段1 per-tensor 可忽略,
  但要确认 passthrough 不丢属性。
- 节点输出 dtype:converter 已记录 `out_dtype = tensor_type.elem_type`(converter.py:320),
  但**前提是图里有该输出的 ValueInfoProto**。ORT 量化后图通常有 value_info,但需验证。
  若无,runtime 的 dtype_marker fallback 要能工作。

→ 阶段1 转换器**可能零改动**,只需验证。若 value_info 缺失导致 dtype 推断错,
  补一个 QDQ 专用 dtype 推断(看 zero_point 的 dtype 推 output 容器 dtype)。

### 步骤 5:kUInt8 容器路径 + elem_kind_supported

**文件**: `core/DType.hpp:140` `elem_kind_supported` / `core/runtime.cpp` 输入 switch(149)

**核对发现**:
- `kUint8` 在 `elem_kind_supported` 返回 **false**(DType.hpp:140,不在 true 列表)。
  → `require_supported_elem("uint8")` 会抛 "recognized, no kernel"。
  → **uint8 QDQ 图在 loader 就被拒**。
- `kInt8` 在列表(true),int8 QDQ 能过 loader。
- runtime 输入 switch(149)**无 kUint8 case**,落到 default throw。

**问题**:ORT `quantize_dynamic` 默认对权重产生 **QUInt8**(uint8,非对称)。
所以 uint8 必须支持,否则最常见的 ORT 量化图进不来。

**修法**:
- `elem_kind_supported` 加 `case ElemKind::kUint8: return true;`(配注释:QDQ op 读)。
- runtime 输入 switch 加 `case ElemKind::kUint8:` → `Tensor<int8_t>` + `set_elem_kind(kUint8)`
  (uint8 复用 int8_t 容器,elem_kind 区分语义,同 bool/fp8 模式)。
- `storage_matches_kind`(DType.cpp:78)kUint8 已返回 false(走 int8_t 容器),OK。

### 步骤 6:测试模型 + QDQ 图生成

**模型**: Qwen2.5-0.5B-Instruct(ModelScope,~943MB)。

**生成 QDQ 测试图**(两种):
1. **小合成图**(开发用,秒级):手写一个 ONNX,MatMul 的 weight 用 QDQ 包
   (Constant int8 weight + scale + zp → DQ → MatMul)。对照 ORT 算参考值。
2. **真实 LLM 子图**:把 Qwen2.5-0.5B 的 llm.onnx 用 ORT `quantize_dynamic`
   生成 QDQ 版,转 vkopbin,逐层对比 ORT。

**单元测试** `tests/QuantizeLinearTest.cpp` 扩展:
- `Int8RoundTrip`:fp16→int8→fp16,带非零 zero_point,对照 numpy/ort。
- `UInt8AsymmetricRoundTrip`:uint8 + 非对称 zp(模拟 ORT dynamic uint8)。
- `PerTensorScale`:多个 scale 值验证。
- 现有 fp8 测试保留不破坏。

## 阶段 1 验证

- 单元测试:int8/uint8 QDQ round-trip byte-exact vs ORT/numpy。
- 端到端:Qwen2.5-0.5B QDQ 图,逐层 ORT 对比(activation 中间值 + logits)。
- NANSCAN:无 NaN/Inf。
- 注:阶段1 **不验证 VRAM 省减**(DQ 展开 fp16),只验证正确性。

## 阶段 2(后续,省 VRAM)— 提纲

- 转换器 pass:`DequantizeLinear(weight_const, scale, zp) → MatMul` 折叠成
  `MatMul(A, weight_int8, scale, zp)`,weight 必须是 Constant。
- MatMul weight-only int8 路径加 zero_point 支持:`w8_val = (byte - zp) * scale`,
  zp 进 push constant(per-tensor)或 binding(per-channel)。
- 此时权重在 VRAM 存 int8 字节,真正省 50%。
- 处理 per-axis scale(静态量化 per-channel)。

## 待确认

- ORT `quantize_dynamic` 对 LLM 的 MatMul weight 产生 per-tensor 还是 per-channel?
  (影响阶段1是否必须支持 per-axis)
- Qwen2.5-0.5B 的导出器:复用 `qwen3_export_onnx.py` 改 Qwen2.5,还是新写?
  (Qwen2.5 vs Qwen3 架构差异:Qwen2.5 无 q_norm/k_norm,RoPE 实现略不同)
- zero_point 容器:ONNX 允许 zp 是 int8/uint8,与 weight 同 dtype。确认 runtime
  创建 zp initializer 时按 elem_kind 走对应容器。
