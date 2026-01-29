
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
cmake .. -DENABLE_TESTS=ON -DUSE_VALIDATION_LAYERS=ON -DENABLE_ASAN=OFF -DUSE_DEBUG_LAYERS=OFF -DUSE_FP16=OFF -DUSE_MEASURE_TIME=OFF
```
如果是交叉编译，需要设置交叉编译环境变量，借鉴参考toolchain.cmake
```
cmake .. -DCMAKE_TOOLCHAIN_FILE=../toolchain.cmake -DENABLE_TESTS=OFF
```

#### 4. 模型转换
```bash
python3 model/onnx2vkop.py -i resnet18-v2-7.onnx
```

- 支持量化：fp16, int8 对称量化
- 支持指定batch size
- 支持针对3D,4D NCHW to RGBA转换到模型
- 支持tensor合并，以便节约内存
```
usage: onnx2vkop.py [-h] [-q QUANT] -i INPUT [-u] [-b BATCH] [-r]

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

#### 4. Model Conversion
```bash
python3.13 tools/onnx2vkop.py -i resnet18-v2-7.onnx
```

- Supports quantization: fp16, int8 symmetric quantization
- Supports specifying batch size
- Supports 3D/4D NCHW to RGBA model conversion
- Supports tensor merging to save memory
```
usage: onnx2vkop.py [-h] [-q QUANT] -i INPUT [-u] [-b BATCH] [-r]

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

