
### 项目介绍

vkop 是一个基于 Vulkan 的实现的AI推理引擎，简化运行时逻辑。

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
对于测试依赖torch
```
pip install torch
```

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
cmake .. -DENABLE_TESTS=ON -DPython3_EXECUTABLE=.venv/bin/python3.13 -DUSE_VALIDATION_LAYERS=ON -DENABLE_ASAN=OFF -DUSE_DEBUG_LAYERS=OFF -DUSE_FP16=OFF -DUSE_MEASURE_TIME=OFF
```

#### 4. 运行程序
```
./benchmark/vkbench ../resnet18-v2-7.vkopbin dog.jpeg
```
支持将postproc 手动注册到gpu 处理，比如softmax，topk减少内存带宽

---

### Project Introduction

vkop is an AI inference engine based on Vulkan, designed to provide high-performance computing capabilities.

### How to Use

#### 1. Dependency Installation
First, install the required dependencies:
```bash
wget https://sdk.lunarg.com/sdk/download/latest/linux/vulkan-sdk.tar.gz
tar xvf vulkan-sdk.tar.gz
source path/to/VulkanSDK/setup-env.sh
export PATH=$VULKAN_SDK/x86_64_bin:$PATH
```

#### 2. Environment Setup
Set up the Vulkan ICD loader, using NVIDIA as an example:
```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
```

#### 3. Compilation
(Specific compilation steps need to be added here)

#### 4. Running the Program
(Specific execution instructions need to be added here)