
### 项目介绍

vkop 是一个基于 Vulkan 实现的迷你AI推理引擎, 仅在GPU上运行.

### 如何使用

#### 1. 依赖安装
首先需要安装项目的依赖项shaderc或者vulkan sdk:
```bash
wget https://sdk.lunarg.com/sdk/download/latest/linux/vulkan-sdk.tar.gz
tar xvf vulkan-sdk.tar.gz
source path/to/VulkanSDK/setup-env.sh
export PATH=$VULKAN_SDK/x86_64_bin:$PATH
```
对于模型转换
```
export CMAKE_POLICY_VERSION_MINIMUM=3.5
pip install onnx onnx-simplifier onnxsim onnxruntime
```
对于压测模型下载
```
pip install torchvision
```

对于测试依赖libtorch, cmake过程自动下载解压

#### 2. 环境设置
设置 Vulkan ICD 加载器，以 NVIDIA 为例：
```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
```

确保使用正确的 Vulkan 版本：
```bash
source path/to/VulkanSDK/setup-env.sh
```

#### 3. 编译项目

```
cmake .. -DENABLE_TESTS=ON -DUSE_VALIDATION_LAYERS=ON -DENABLE_ASAN=OFF -DUSE_DEBUG_LAYERS=OFF -DUSE_MEASURE_TIME=OFF -DPython3_EXECUTABLE=$(which python3)
```
如果是交叉编译，需要设置交叉编译环境变量，借鉴参考toolchain.cmake
```
cmake .. -DCMAKE_TOOLCHAIN_FILE=../toolchain.cmake -DENABLE_TESTS=OFF
```

##### macOS (Apple Silicon) 构建

必须使用 Homebrew LLVM 的 clang++（`/opt/homebrew/opt/llvm/bin/clang++`）：

- Apple clang 17 编译 `core/function.cpp` / `core/runtime.cpp` 时前端在
  `TransformCXXFoldExpr` 处无限递归段错误，不可用；
- GCC + Apple libc++ 桥接 ABI 不兼容（std::string/流运行时输出乱码），不可用。

安装依赖并配置：
```bash
brew install cmake shaderc vulkan-loader vulkan-headers molten-vk \
  glfw utf8proc pkgconf googletest libomp
cmake .. -DENABLE_TESTS=ON \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ \
  -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang \
  -DCMAKE_PREFIX_PATH="/opt/homebrew;/opt/homebrew/opt/llvm" \
  -DPython3_EXECUTABLE=$(which python3)
```

运行时 `VulkanLib.cpp` 通过 dlopen 加载 Vulkan loader，必须设置环境变量：
```bash
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
export VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json
```

代码中已处理的 clang/macOS 兼容性点：
- `core/Tensor.hpp` fp16 内联汇编的 `h0`/`s0` 寄存器别名是 GCC 专有扩展，
  已通过 `!defined(__clang__)` 守卫，clang 走可移植路径；
- macOS 上 `size_t`（unsigned long）≠ `uint64_t`（unsigned long long），
  `Tensor` 标量构造函数需显式接受 `std::size_t`，否则
  `Tensor<int64_t>(v.size())` 会误配到 `Tensor(bool)` 构造出空张量；
- `VK_EXT_host_image_copy` 在 Vulkan 1.4 晋升为核心功能后，MoltenVK 的
  loader 对 EXT 后缀入口点 dispatch 为 NULL，
  `vulkan/VulkanImage.cpp` 已改为优先解析无后缀核心名；
- int8/fp8 权重 WIDE GEMV 的 kernel MoltenVK 编不出来
  （`vkCreateComputePipelines` 返回 `VK_ERROR_INITIALIZATION_FAILED`），已拆成
  `shaders/buffer/matmul.comp` 的第二个 fp16 模块（`-DW8_WIDE`，host 侧见
  `MatMulBuffer::w8_wide_pipeline`）单独探测；留在通用模块里时，它一次建流失败
  会拖垮本机**所有** buffer MatMul。能走到该 kernel 的形状
  （单行 A 且 N ≥ 896）由 `MatMulInt8WeightOnlyBuffer`/
  `MatMulFp8WeightOnlyBuffer` 覆盖；
- 融合 KV cache 的复制长度写成 `4 / sizeof(T)`，8 字节 dtype 的模板实例里
  这个比值折成 0，clang 会按实例常量折叠并报
  `-Werror,-Wdivision-by-zero`——即使运行时根本不会传到那个 dtype。
  `ops/Attention.hpp` 已显式声明该路径只接受 2/4 字节的 Q。

#### 4. 模型转换
```bash
python3 -m onnx2vkop.cli -i resnet18-v2-7.onnx
```

模型文件使用 FlatBuffers 格式（file identifier `VKOP`，version 1）。旧版
`struct.pack` 格式不再生成；已有的旧 `.vkopbin` 需用本命令重新转换。

- 支持量化：fp16, int8 对称量化
- 支持指定batch size
- 支持针对3D,4D NCHW to RGBA转换到模型
- 支持tensor合并，以便节约内存
```
usage: cli.py [-h] [-q QUANT] -i INPUT [-u] [-b BATCH] [-r]

options:
  -h, --help         show this help message and exit
  -q, --quant QUANT  Override input_model
  -i, --input INPUT  input_model file
  -u, --unify        convert initializers to a single memory block
  -b, --batch BATCH  batch size for inference
  -r, --rgba         nchw to rgba conversion for initializers

```


#### 4. 运行程序
```
./benchmark/vkbench ../resnet18-v2-7.vkopbin dog.jpeg
```
支持将postproc 手动注册到gpu 处理，比如softmax，topk减少CPU与GPU间的内存吞吐

#### 5. LLM 推理（llm/exporter/driver/llm_chat.cpp）

- KV cache 已 GPU 化，而且**不再有任何拷贝**：converter 把每层的 past/present
  合成同一个 buffer（`fuse_kv_cache_append`，`Attention` 带 `has_cache`），
  decode 每轮由 kernel 就地在容量步长上追加一行，present 就是 past，既无
  device→device 拷贝也无 CPU 往返（原 `KV_INPLACE_PLAN` 已完成并删除）。
  当前 `llm/exporter/llm.vkopbin` 的 28 层全是这个形态。
- GPU shape-meta：张量带 `shape_ssbo_` 侧信道（产出方填充），binary 广播
  shader 的 broadcast==2 SSBO 路径 + `dispatch_from_shape` 间接派发已落地，
  由 `VKOP_GPU_SHAPE` 开关控制（默认关，走 CPU dims 回退）。稳态 readback
  主要通过 Reshape/Expand 的自动学习缓存（LEARNING→CONFIRMING→STABLE）和 host-authoritative int64/int32 initializer 跳过回读来消除（原
  `PHASE2_4_PLAN` 已完成/被替代并删除）。

##### MatMul 优化经验（实测）

host 选路按 `attn_softmax → coopmat_i8(默认关) → coopmat → tiled → ksplit(+WIDE) → fused → reduce`
第一个命中即返回（`ops/Matmul.hpp`）：prefill 走 coopmat，decode GEMV 走 ksplit+WIDE。
以下结论在 Intel ARL + Qwen2.5-0.5B 上实测得出，完整实验记录见 git 历史（原 `MATMUL_OPTIMIZATION.md`）：

- 已证伪、勿重试：改 coopmat tile 形状（BM=32 无可测差异——瓶颈是延迟/occupancy，
  不是权重 DRAM 带宽）；削减权重带宽（同一结论）；ksplit 放宽到 prefill（1510ms vs coopmat 253ms）；
  tiled 替代 coopmat（慢 1.3–1.4x）；量化提速 prefill（quantized coopmat 慢 3x、W8A8 慢 ~2x）；
  `ksplit_q=32`（相对 16 回退）。
- NVIDIA Ada 不支持 8×8×16 coopmat 组合（静默算错），必须走 16×16×16 变体 + 运行时选组合。
- 未试候选：把偶数 N 的 pack 折回 coopmat kernel（省 fp32 scratch 往返 + 一次 dispatch）；
  K 循环软件流水（每个 barrier 做 2 个 K-tile）。
- 测量方法学：iGPU 跑间噪声 2–3x，单次运行无意义，min-of-N 也不可靠（幸运快跑会造假结论）；
  必须交替成对比较（A,B,A,B,...）看逐对胜负；低于 ~20% 的差异量不出来。

调参/诊断开关：

| 变量 | 作用 |
|---|---|
| `VKOP_NO_COOP=1` | 关掉 coopmat，回落 tiled |
| `VKOP_COOP_I8=1` | 开 W8A8 SINT8 coopmat（默认关） |
| `VKOP_KSPLIT_Q=<n>` | 覆盖 ksplit 的列-quad 数（调优用） |
| `VKOP_KSPLIT_WIDE=0/1` | 关/强制 WIDE |
| `VKOP_KSPLIT_WIDE_N=<n>` | 设 fp16+byte 两个 WIDE 阈值 |
| `VKOP_SUBMIT_PROF=1` | 每个 op 的 GPU 时间戳查询（权威） |
| `VKOP_OPPROF=1` | record 侧 wall time |

#### 6. 文生图（image/exporter/image_gen）

一条 prompt 直接出 PNG，**运行期不读任何 python/ORT 中间产物**：C++ 分词 + 原始 ChatML
模板 + `drop_idx` 截取 + padding 屏蔽，以及 DiT 的 joint rope 频率表和 FlowMatch σ 调度
都在驱动内现算（`--ref DIR` 只用于数值对拍，可选）。

```
cd image/exporter
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib \
VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json \
../../build/image_gen dit_prefill_static.vkopbin dit_decode_static.vkopbin \
  "A red fox sitting on a wooden bench in a sunlit park" 40 7
```

四份图按阶段顺序装载、用完立刻释放（单份 ~14 GB，任意两份同时驻留就超 36 GB 统一内存）：
文本塔 `text_encoder.vkopbin` + 1.24 GB 输入查表 `text_encoder_embeds.bin`（mmap，一条
prompt 只碰其中几十行）、DiT prefill/decode、VAE `vae_decoder_512.vkopbin`；分词器复用
LLM 的 `llm/tokenizer/qwen3_vl.bin`。产物路径可用 `TEXT_ENCODER_VKOPBIN` /
`TEXT_ENCODER_EMBEDS` / `TOKENIZER_BIN` / `VAE_VKOPBIN` 覆盖。512×512 / 40 步实测冷机 138 秒、
热机 152 秒（decode 冷启动 2.6 s/步、爬升后稳在 2.9–3.4 s/步，文本塔前向 0.27 s，VAE 4.8 s）。
中文 prompt 直接可用，但静态文本段只有 64 token（约 42 个汉字），超长会报错而不是截断。
导出、转换与逐位对拍口径见
`image/exporter/BASELINE.md`。

---

### Project Introduction

vkop is a mini AI inference engine based on Vulkan, with runtime logic under 1000 lines of code.

### How to Use

#### 1. Dependency Installation
First, install the required dependencies, such as shaderc or Vulkan SDK:
```bash
wget https://sdk.lunarg.com/sdk/download/latest/linux/vulkan-sdk.tar.gz
tar xvf vulkan-sdk.tar.gz
source path/to/VulkanSDK/setup-env.sh
export PATH=$VULKAN_SDK/x86_64_bin:$PATH
```
For model conversion:
```bash
export CMAKE_POLICY_VERSION_MINIMUM=3.5
pip install onnx onnx-simplifier onnxsim onnxruntime
```
For benchmarking models:
```
pip install torchvision
```

For testing dependencies, libtorch is downloaded and extracted automatically during the cmake process.

#### 2. Environment Setup
Set up the Vulkan ICD loader, using NVIDIA as an example:
```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
```

Ensure the correct Vulkan version is used:
```bash
source path/to/VulkanSDK/setup-env.sh
```

#### 3. Compilation
```bash
cmake .. -DENABLE_TESTS=ON -DUSE_VALIDATION_LAYERS=OFF -DENABLE_ASAN=OFF -DUSE_DEBUG_LAYERS=OFF -DUSE_MEASURE_TIME=OFF
```
If you are cross-compiling, set up the cross-compilation environment variables, based on toolchain.cmake:
```
cmake .. -DCMAKE_TOOLCHAIN_FILE=../toolchain.cmake -DENABLE_TESTS=OFF
```

##### macOS (Apple Silicon) build

Homebrew LLVM clang++ (`/opt/homebrew/opt/llvm/bin/clang++`) is required:

- Apple clang 17 segfaults in its frontend (infinite recursion in
  `TransformCXXFoldExpr`) when compiling `core/function.cpp` /
  `core/runtime.cpp`;
- GCC + Apple libc++ has a broken ABI (garbled std::string/stream output at
  runtime).

Install dependencies and configure:
```bash
brew install cmake shaderc vulkan-loader vulkan-headers molten-vk \
  glfw utf8proc pkgconf googletest libomp
cmake .. -DENABLE_TESTS=ON \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ \
  -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang \
  -DCMAKE_PREFIX_PATH="/opt/homebrew;/opt/homebrew/opt/llvm" \
  -DPython3_EXECUTABLE=$(which python3)
```

`VulkanLib.cpp` loads the Vulkan loader via dlopen at runtime, so these
environment variables must be set:
```bash
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
export VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json
```

macOS/clang compatibility points already handled in the code:
- The `h0`/`s0` register aliases in the fp16 inline asm of
  `core/Tensor.hpp` are GCC-only extensions, guarded by
  `!defined(__clang__)`; clang takes the portable path;
- On macOS `size_t` (unsigned long) is NOT `uint64_t` (unsigned long long),
  so the `Tensor` scalar ctor must explicitly accept `std::size_t`, otherwise
  `Tensor<int64_t>(v.size())` silently resolves to `Tensor(bool)` and yields
  an empty tensor;
- Once `VK_EXT_host_image_copy` was promoted to Vulkan 1.4 core, the MoltenVK
  loader leaves the EXT-suffixed entry points with a NULL dispatch;
  `vulkan/VulkanImage.cpp` resolves the unsuffixed core names first;
- MoltenVK's SPIR-V → MSL backend cannot compile the int8/fp8 weight-only WIDE
  GEMV (`vkCreateComputePipelines` → `VK_ERROR_INITIALIZATION_FAILED`), so it
  lives in a second fp16 module of `shaders/buffer/matmul.comp` (`-DW8_WIDE`,
  see `MatMulBuffer::w8_wide_pipeline`). While it sat in the general module its
  failed pipeline creation aborted **every** buffer MatMul on this GPU;
  `MatMulInt8WeightOnlyBuffer`/`MatMulFp8WeightOnlyBuffer` cover the shape range
  that reaches it (N ≥ 896 with one A row);
- the fused KV-cache copy sizes itself with `4 / sizeof(T)`, which folds to 0 in
  the 8-byte instantiations, and clang reports the resulting
  `-Werror,-Wdivision-by-zero` per template instantiation even though no such
  dtype ever reaches that branch; `ops/Attention.hpp` now states the contract
  (the cache path takes a 2- or 4-byte Q).

#### 4. Model Conversion
```bash
python3 -m onnx2vkop.cli -i resnet18-v2-7.onnx
```

Model files use the FlatBuffers format (file identifier `VKOP`, version 1). The
legacy `struct.pack` format is no longer produced; existing old `.vkopbin`
files must be reconverted with this command.

- Supports quantization: fp16, int8 symmetric quantization
- Supports specifying batch size
- Supports 3D/4D NCHW to RGBA model conversion
- Supports tensor merging to save memory
```
usage: cli.py [-h] [-q QUANT] -i INPUT [-u] [-b BATCH] [-r]

options:
    -h, --help         show this help message and exit
    -q, --quant QUANT  Override input_model
    -i, --input INPUT  input_model file
    -u, --unify        convert initializers to a single memory block
    -b, --batch BATCH  batch size for inference
    -r, --rgba         nchw to rgba conversion for initializers

```

#### 5. Running the Program
```bash
./benchmark/vkbench ../resnet18-v2-7.vkopbin dog.jpeg
```
Supports manually registering post-processing operations like softmax and top-k on the GPU to reduce memory throughput between CPU and GPU.

#### 6. LLM inference (llm/exporter/driver/llm_chat.cpp)

- KV cache is GPU-resident and, since `fuse_kv_cache_append`, **not copied at
  all**: each layer's past and present are one buffer (`Attention` carries
  `has_cache`), so a decode round appends the new row in place at the bound
  buffer's capacity pitch and hands back the same tensor — no device→device
  copy, no CPU round-trip (the former `KV_INPLACE_PLAN` is done and removed).
- GPU shape-meta: tensors carry a `shape_ssbo_` side-channel (populated by
  the producing op); the broadcast==2 SSBO path in binary shaders plus
  `dispatch_from_shape` indirect dispatch are wired up behind the
  `VKOP_GPU_SHAPE` env flag (off by default, CPU dims fallback). Steady-state
  readbacks are instead eliminated via the Reshape/Expand auto-learning cache
  (LEARNING→CONFIRMING→STABLE) and host-authoritative int64/int32
  initializers skipping readback (the former `PHASE2_4_PLAN` is done or
  superseded and removed).

##### MatMul optimization notes (measured)

Host path selection returns on first match in
`attn_softmax → coopmat_i8 (off by default) → coopmat → tiled → ksplit(+WIDE) → fused → reduce`
(`ops/Matmul.hpp`): prefill goes coopmat, decode GEMV goes ksplit+WIDE.
Conclusions below were measured on Intel ARL + Qwen2.5-0.5B; the full
experiment log lives in git history (the former `MATMUL_OPTIMIZATION.md`):

- Disproven, do not retry: changing coopmat tile shape (BM=32 made no
  measurable difference — the bottleneck is latency/occupancy, not weight DRAM
  bandwidth); cutting weight DRAM traffic (same conclusion); relaxing ksplit
  into prefill (1510 ms vs coopmat 253 ms); tiled instead of coopmat
  (1.3–1.4x slower); quantization for prefill speed (quantized coopmat 3x
  slower, W8A8 ~2x); `ksplit_q=32` (regresses vs 16).
- NVIDIA Ada does not support the 8×8×16 coopmat combination (silently wrong);
  the 16×16×16 variant plus runtime selection is required.
- Untried candidates: folding the even-N pack back into the coopmat kernel
  (saves an fp32 scratch round-trip plus a dispatch); software pipelining of
  the K loop (2 K-tiles per barrier).
- Measurement methodology: iGPU run-to-run noise is 2–3x; single runs are
  meaningless and min-of-N is unreliable (one lucky fast run fabricates
  conclusions); use alternating paired comparisons (A,B,A,B,...) judged
  per-pair; effects below ~20% are unmeasurable on this machine.

Tuning/diagnostic switches:

| Variable | Effect |
|---|---|
| `VKOP_NO_COOP=1` | disable coopmat, fall back to tiled |
| `VKOP_COOP_I8=1` | enable W8A8 SINT8 coopmat (off by default) |
| `VKOP_KSPLIT_Q=<n>` | override ksplit column-quad count (tuning) |
| `VKOP_KSPLIT_WIDE=0/1` | disable/force WIDE |
| `VKOP_KSPLIT_WIDE_N=<n>` | set both WIDE auto thresholds |
| `VKOP_SUBMIT_PROF=1` | per-op GPU timestamp queries (authoritative) |
| `VKOP_OPPROF=1` | record-side wall time |

#### 7. Text-to-image (image/exporter/image_gen)

One prompt in, one PNG out, and the driver reads **no python/ORT intermediate
artifacts at runtime**: tokenization, the raw ChatML template, `drop_idx`
slicing and padding masking happen in C++, and so do the DiT joint-rope
frequency table and the FlowMatch σ schedule (`--ref DIR` exists only for
numeric alignment and is optional).

```
cd image/exporter
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib \
VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json \
../../build/image_gen dit_prefill_static.vkopbin dit_decode_static.vkopbin \
  "A red fox sitting on a wooden bench in a sunlit park" 40 7
```

The four graphs load one stage at a time and are released right after use (each
is ~14 GB; any two together exceed 36 GB of unified memory): the text tower
`text_encoder.vkopbin` plus its 1.24 GB input table `text_encoder_embeds.bin`
(mmapped — a prompt touches a few dozen of its rows), DiT prefill/decode, and
the VAE `vae_decoder_512.vkopbin`. Tokenization reuses the LLM's
`llm/tokenizer/qwen3_vl.bin`. Override paths with `TEXT_ENCODER_VKOPBIN`,
`TEXT_ENCODER_EMBEDS`, `TOKENIZER_BIN`, `VAE_VKOPBIN`. Measured 512×512 at 40
steps: 138 s cold / 152 s warm (decode 2.6 s/step on a cold start, settling at
2.9–3.4 s/step once the SoC throttles; 0.27 s text-tower forward; 4.8 s VAE).
Chinese prompts work, but the static text segment holds only 64 tokens (~42
Chinese characters) and the driver errors rather than truncating. Export,
conversion and the bit-wise alignment protocol are in
`image/exporter/BASELINE.md`.

