
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
cmake .. -DENABLE_TESTS=ON -DUSE_VALIDATION_LAYERS=ON -DENABLE_ASAN=OFF -DUSE_DEBUG_LAYERS=OFF -DUSE_FP16=OFF -DUSE_MEASURE_TIME=OFF -DPython3_EXECUTABLE=$(which python3)
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
  `vulkan/VulkanImage.cpp` 已改为优先解析无后缀核心名。

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

#### 5. LLM 推理（llm/exporter/llm_chat）

- KV cache 已 GPU 化：decode 每轮 present→past 为单条命令缓冲内的
  device→device 拷贝，无 CPU 往返（原 `KV_INPLACE_PLAN` 已完成并删除）。
- GPU shape-meta：张量带 `shape_ssbo_` 侧信道（产出方填充），binary 广播
  shader 的 broadcast==2 SSBO 路径 + `dispatch_from_shape` 间接派发已落地，
  由 `VKOP_GPU_SHAPE` 开关控制（默认关，走 CPU dims 回退）。稳态 readback
  主要通过 Reshape/Expand 的自动学习缓存（LEARNING→CONFIRMING→STABLE）和
  host-authoritative int64/int32 initializer 跳过回读来消除（原
  `PHASE2_4_PLAN` 已完成/被替代并删除）。

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
cmake .. -DENABLE_TESTS=ON -DUSE_VALIDATION_LAYERS=OFF -DENABLE_ASAN=OFF -DUSE_DEBUG_LAYERS=OFF -DUSE_FP16=OFF -DUSE_MEASURE_TIME=OFF
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
  `vulkan/VulkanImage.cpp` resolves the unsuffixed core names first.

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

#### 6. LLM inference (llm/exporter/llm_chat)

- KV cache is GPU-resident: each decode round copies present→past
  device→device inside one command buffer, with no CPU round-trip (the
  former `KV_INPLACE_PLAN` is done and removed).
- GPU shape-meta: tensors carry a `shape_ssbo_` side-channel (populated by
  the producing op); the broadcast==2 SSBO path in binary shaders plus
  `dispatch_from_shape` indirect dispatch are wired up behind the
  `VKOP_GPU_SHAPE` env flag (off by default, CPU dims fallback). Steady-state
  readbacks are instead eliminated via the Reshape/Expand auto-learning cache
  (LEARNING→CONFIRMING→STABLE) and host-authoritative int64/int32
  initializers skipping readback (the former `PHASE2_4_PLAN` is done or
  superseded and removed).

