
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
  （`-q fp16` 产出真正的 fp16 模型：`fp16_graph.py` 算出整图的 FP16 域，把激活声明和
  权重字节一起改成 fp16。只有「shader 按张量 dtype 选择」且「确实有 fp16 构建」的 op
  能进这个域；进不了的 op 连同它读写的张量整体留在 fp32（这个判定会沿数据流传播，
  Cast 是唯一允许横跨两种宽度的节点），日志按节点名给出原因。运行时按数据输入选
  shader，半宽权重配 fp32 shader 是错模型，不是省空间）
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

按节点/作用域排除量化（定位精度损失用）：`--exclude-node`（节点名 glob）、
`--exclude-scope`（模块路径前缀，按 `/` 和 `.` 切 token，故 `layers.0` 不会命中
`layers.01`）、`--exclude-downstream-of`（命中节点及其全部下游）。被点名的权重保持
原始 dtype、不追加 scale，其余照常量化，因此两次转换之间只有那一层的精度不同。
只对 byte 格式（`-q int8/fp8e4m3/fp8e5m2/int4/nf4/nvfp4/awq`）生效：`-q fp16` 下
单个 fp32 权重需要一个 Cast 节点，不是精度开关，会显式告警。匹配不到任何节点的
pattern 直接失败——悄悄选空的 flag 会交出未排除的模型却报告成功。详见
`model/pypi/onnx2vkop/exclusion.py`。

```bash
python3 -m onnx2vkop.cli -i llm.onnx -q int4 --exclude-scope model.layers.0
python3 -m onnx2vkop.cli -i llm.onnx -q awq --awq-format nf4 \
    --exclude-node '*/layers.1/attn/*' --exclude-downstream-of /embed
```

#### 5. 运行程序
```
./benchmark/vkbench ../resnet18-v2-7.vkopbin dog.jpeg
```
支持将postproc 手动注册到gpu 处理，比如softmax，topk减少CPU与GPU间的内存吞吐。
注册的后处理算子不在图节点里，转换器的 archive 裁剪（按图 op 普查）不会带上
它们的 kernel，需要在转换时用 `--extra-shaders` 显式声明（逗号分隔，ONNX 拼写），
否则 LoadModel 建 pipeline 时 archive 缺 key 直接失败：

```bash
# LLM 驱动的 GPU argmax（RegisterPostProcess TopK k=1）
python3 -m onnx2vkop.cli -i llm.onnx -q int8 --extra-shaders TopK
# resnet 式分类器（softmax + topk 都手动注册时）
python3 -m onnx2vkop.cli -i resnet18.onnx --extra-shaders Softmax,TopK
```

#### 6. 精度回环（tools/vkop_vs_ort.py）

转换「能跑」不等于「算对」：每个 op 都有 kernel、图能 load、run 能结束，输出仍然可能
是错的数字。回环把 GPU 写出的张量和 onnxruntime 对同一输入的参考逐输出对比：

```bash
python3 benchmark/resnet.py resnet18.onnx dog.jpg --dump-input   # dump 真实输入
python3 -m onnx2vkop.cli -i resnet18.onnx -q int8                # 转换
python3 tools/vkop_vs_ort.py resnet18.vkopbin --inputs DIR --topk 5
```

结论可信靠三个前提：

- **同一个输入数组，不是两个长得像的。** 数组在这里造一次（`--inputs DIR/<name>.npy`，
  float 输入可 `--random-fill` + `--seed`），原样喂给 ORT，同时写进 vkop-bench 用
  `--load-inputs` 读的目录；两边都不再 resize/归一化，差异才只属于 kernel。跑 JPEG 必须
  先 `--dump-input`：PIL 的 bilinear 和运行时的 stb triangle 不是同一个 resize，跨它们
  比较测的是 resize。`--dump-input` 会把三种布局写在模型同目录，所以给回环的 `DIR` 里
  只放 `py_input_nchw.npy`（单输入模型 + 目录里只有一个 .npy 时不必和输入同名）。
  `.vkopbin` 的 dtype/shape 优先，不符会打印 cast/reshape。
- **对 GPU 写出的字节没有歧义读法。** `vkop-bench --save-outputs DIR` 写 raw bytes 加
  `DIR/manifest.tsv`（name / bytes-as / dims / nbytes，取最后一次 run 之后的真实形状），
  回环断言 `prod(dims) × itemsize == nbytes`，不符该输出直接判失败。从 `.vkopbin` 反推
  形状会让错形状或错 reinterpret 解码成「看起来像舍入误差」的数；输出名是配对契约，
  ORT 侧没有同名输出报 UNMATCHED 而不是猜。整数输出（token id、位置、cache index）精确
  比较——错的 index 没有容差带，`--topk` 对它也无意义。
- **阈值按格式说明白。** gate 全是 flag（`--cos-min` `--rtol` `--atol` `--bad-frac-max`），
  每次 run 把自己的标准打在表头，不继承魔法数。`--rtol` 拿参考的最大元素带宽，不是逐元素
  相对误差：后者在近零值上测的是存储格式的舍入下限，不是 op 的正确性。fp32-vs-fp32 该过
  cos≥0.999 且几乎无越带元素；权重-only int8 合理过不了，测出它漂多远正是目的。

实测（resnet18-v2-7，224×224 单图，`--backend buffer`，默认 gate
cos≥0.999 / atol 1e-3 / rtol 1e-2 / 越带 ≤0.1%）：

| 产物 | maxabs | relL2 | cos | 越带元素 | top-1 | top-5 重合 |
|---|---|---|---|---|---|---|
| `-q fp16`（整图 FP16 域） | 8.811e-03 | 7.449e-04 | 1.000000 | 0.0000% | 1.000 | 1.0000 |
| `-q int8`（权重-only） | 3.678e-01 | 3.923e-02 | 0.999329 | 33.80% | 0.000 | 0.8000 |
| `-q int8 --exclude-scope /layer4/` | 3.766e-01 | 3.785e-02 | 0.999394 | 31.70% | 1.000 | 0.8000 |

第一行过 gate；第二行在默认 gate 下 FAIL 是回环给出的结论，不是回环坏了——maxabs 3.7e-01
已经大到换掉 top-1，把 gate 放宽去「通过」等于放弃这次测量。第三行是它的使用方法：把
layer4 整层留在原 dtype，relL2 只从 3.923e-02 走到 3.785e-02（越带 33.8%→31.7%），说明损失
主要不在这一层；两次转换同输入、同 shader，只差 flag 点名的那层，因此差值可归因。逐层做
同样的比较，就是 `model/pypi/onnx2vkop/exclusion.py` 存在的理由。

退出码 0 全部通过 / 1 至少一个输出没过 gate / 2 回环本身跑不起来（缺 binary、模型、ORT 或
崩溃）。onnxruntime 不是转换器依赖（不在 wheel 的 requires 里），GPU 也不是这里的；两者都
在开始前检查，并给出修复方式而不是 traceback。`--onnx` 默认取 `.vkopbin` 的同名 `.onnx`，
`--bench` 默认 `build/benchmark/vkop-bench`（再退到 `$PATH`）；LLM 那种几十个输出的 KV dump
用 `--limit N` 只看前 N 行，`--keep` 或 `--work-dir` 把 dump 留在盘上。

和它对得上的是转换期的 `--numeric-audit` / `VKOP_NUMERIC_AUDIT`（只报警，见
`model/pypi/onnx2vkop/numeric_audit.py`）：权重-only 格式下 kernel 读的是
`decode(encode(W))`，所以每个权重自己的 relL2 不用 GPU 就能算。resnet18 的 21 个真实权重
（按 GEMM 视图）中位数是 fp16 2.07e-04、int8 1.27e-02，本回环端到端是 7.449e-04 与
3.923e-02，即 3.6× 与 3.1×——若 21 层各把自己的权重误差全额传到输出，平方和的上限是
√21 ≈ 4.6×，实测在其之下。两边明显偏离时先查转换期那个数：端到端的差里可能混进了 kernel。

#### 7. LLM 推理（llm/exporter/driver/llm_chat.cpp）

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
| `VKOP_KSPLIT512=1` | ksplit 走 512-lane 变体（同 q、K 切深翻倍；探测失败自动回落 256-lane） |
| `VKOP_SUBMIT_PROF=1` | 每个 op 的 GPU 时间戳查询（权威） |
| `VKOP_OPPROF=1` | record 侧 wall time |

#### 8. 文生图（image/exporter/image_gen）

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
  (`-q fp16` produces a real FP16 artifact: `fp16_graph.py` computes the graph's
  FP16 domain and rewrites the activation declarations together with the weight
  bytes. Only an op whose shader build follows the tensor dtype, and that has a
  real dual FP16 build, may enter it; an op that cannot keeps itself and every
  tensor it touches at FP32 -- that verdict propagates along the data flow, and
  Cast is the only node allowed between the two widths -- and the run names the
  op and the reason. The runtime picks each op's shader from its data input, so
  an FP16 weight behind an FP32 shader is a wrong model, not a smaller one)
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

Per-node / per-scope precision exclusion (for localizing an accuracy loss):
`--exclude-node` (a glob on the node name), `--exclude-scope` (a module-path
prefix, tokenized on both `/` and `.`, so `layers.0` covers `layers.0.attn` and
not `layers.01`) and `--exclude-downstream-of` (the matched node plus everything
that reads its output). A named weight keeps the dtype it arrived with and gets
no scale input, while the rest of the model quantizes as usual -- so the two
conversions differ in exactly that layer. Only the byte formats honor it
(`-q int8/fp8e4m3/fp8e5m2/int4/nf4/nvfp4/awq`): under `-q fp16` a lone fp32
weight needs a Cast node rather than being a precision knob, and the run says
so. A pattern that matches no node fails the conversion, because a silently
empty exclusion list would report success on the un-excluded model. See
`model/pypi/onnx2vkop/exclusion.py`.

```bash
python3 -m onnx2vkop.cli -i llm.onnx -q int4 --exclude-scope model.layers.0
python3 -m onnx2vkop.cli -i llm.onnx -q awq --awq-format nf4 \
    --exclude-node '*/layers.1/attn/*' --exclude-downstream-of /embed
```

#### 5. Running the Program
```bash
./benchmark/vkbench ../resnet18-v2-7.vkopbin dog.jpeg
```
Supports manually registering post-processing operations like softmax and top-k on the GPU to reduce memory throughput between CPU and GPU. Registered post-process ops are not graph nodes, so the converter's archive pruning (per-graph op census) will not carry their kernels — declare them explicitly at convert time with `--extra-shaders` (comma-separated, ONNX spelling), e.g. `--extra-shaders TopK` for the LLM driver's GPU argmax, `--extra-shaders Softmax,TopK` for resnet-style classifiers. An archive missing the key fails hard at pipeline creation during LoadModel.

#### 6. Accuracy loop (tools/vkop_vs_ort.py)

A conversion can be mechanically fine -- every op has a kernel, the graph loads,
the run finishes -- and still compute wrong numbers. The loop compares what the
GPU wrote against what ONNX Runtime gives for the same input, output by output:

```bash
python3 benchmark/resnet.py resnet18.onnx dog.jpg --dump-input   # the real input
python3 -m onnx2vkop.cli -i resnet18.onnx -q int8                # convert
python3 tools/vkop_vs_ort.py resnet18.vkopbin --inputs DIR --topk 5
```

Three things make the verdict worth reading:

- **ONE input tensor, not two look-alikes.** The array is built here once (from
  `DIR/<input name>.npy`, or a seeded `--random-fill` for float inputs), fed to
  onnxruntime as-is, and written to the directory vkop-bench reads with
  `--load-inputs`. Neither side resizes or renormalizes afterwards, so a
  divergence belongs to the kernels. For a JPEG dump the tensor first: PIL's
  bilinear and the runtime's stb triangle filter are not the same resize, and a
  comparison across them would measure the resize. `--dump-input` writes its
  three layouts next to the model, so point `DIR` at a directory holding only
  `py_input_nchw.npy` -- with exactly one input and exactly one `.npy`, the file
  name need not match (that is what `--dump-input` leaves behind). The
  `.vkopbin`'s dtype and dims win; a cast or reshape is printed.
- **An unambiguous reading of what the GPU wrote.** `vkop-bench --save-outputs DIR`
  dumps raw bytes plus `DIR/manifest.tsv` (name, bytes-as, dims, nbytes --
  the shape the run actually had), and the loop asserts
  `prod(dims) x itemsize == nbytes`, failing that output if it disagrees.
  Re-deriving the layout from the `.vkopbin` lets a mis-shaped or reinterpreted
  array decode into plausible numbers, which reads as a rounding error rather
  than as the misread it is. Output names are the pairing contract: an output
  with no same-named ORT result is reported UNMATCHED, not guessed. Integer
  outputs (token ids, positions, cache indices) compare exactly -- a wrong index
  has no tolerance band, and `--topk` over one is meaningless.
- **A threshold that means something per format.** The gates are flags
  (`--cos-min`, `--rtol`, `--atol`, `--bad-frac-max`) so every run states the bar
  it was judged against instead of inheriting a magic number. `--rtol` bands the
  outlier count against the reference's largest element, not per element,
  because per-element relative error on near-zero values measures the storage
  format's rounding floor rather than the op's correctness. fp32-vs-fp32 should
  clear cos>=0.999 with almost no outliers; an int8 weight-only model legitimately
  will not, and measuring how far it drifts is the point.

Measured (resnet18-v2-7, one 224x224 image, `--backend buffer`, default gates
cos>=0.999 / atol 1e-3 / rtol 1e-2 / out<=0.1%):

| artifact | maxabs | relL2 | cos | elements beyond band | top-1 | top-5 overlap |
|---|---|---|---|---|---|---|
| `-q fp16` (whole graph in the FP16 domain) | 8.811e-03 | 7.449e-04 | 1.000000 | 0.0000% | 1.000 | 1.0000 |
| `-q int8` (weight-only) | 3.678e-01 | 3.923e-02 | 0.999329 | 33.80% | 0.000 | 0.8000 |
| `-q int8 --exclude-scope /layer4/` | 3.766e-01 | 3.785e-02 | 0.999394 | 31.70% | 1.000 | 0.8000 |

The first row passes; the second failing its gates is the loop's conclusion, not
a broken loop -- a maxabs of 3.7e-01 is already large enough to flip top-1, and
loosening the gate to make it "pass" would give up the measurement. The third row
is how the loop is used: keeping layer4 at its source dtype moves relL2 only from
3.923e-02 to 3.785e-02 (beyond-band 33.8% to 31.7%), so that layer is not where
the loss mainly sits. The two conversions share the input and the shaders and
differ in exactly the layer the flag named, which is what makes the delta
attributable -- repeating it layer by layer is the reason
`model/pypi/onnx2vkop/exclusion.py` exists.

Exit codes: 0 every output passed, 1 at least one failed its gates, 2 the loop
itself could not run (missing binary/model/ORT, crash). onnxruntime is
deliberately not a converter dependency (it is not in the wheel's requires) and
neither is a GPU here; both are checked up front and reported with the fix
rather than as a traceback. `--onnx` defaults to the `.vkopbin`'s own stem with
`.onnx`, `--bench` to `build/benchmark/vkop-bench` (then `$PATH`); an LLM's
dozens of KV outputs are readable with `--limit N`, and `--keep` or `--work-dir`
leaves the dump on disk.

Its convert-time counterpart is `--numeric-audit` / `VKOP_NUMERIC_AUDIT`
(report-only, see `model/pypi/onnx2vkop/numeric_audit.py`): for a weight-only
format the kernel reads `decode(encode(W))`, so each weight's own round-trip
rel-L2 needs no GPU. On resnet18's 21 real weights (in their GEMM view) the
median is 2.07e-04 for fp16 and 1.27e-02 for int8, against 7.449e-04 and 3.923e-02
here at the model output -- 3.6x and 3.1x, under the sqrt(21) ~= 4.6x the sum
would reach if every layer passed its own weight error through in full. When the
two disagree, start with the convert-time number: a GPU run's difference may
contain a kernel, not only a
format.

#### 7. LLM inference (llm/exporter/driver/llm_chat.cpp)

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
| `VKOP_KSPLIT512=1` | route ksplit through the 512-lane build (same q, double K-slice depth; falls back to 256-lane on any refusal) |
| `VKOP_KSPLIT_FLOOR=<n>` | block floor the int8 q search insists on (default 8; raise to split K deeper) |
| `VKOP_SUBMIT_PROF=1` | per-op GPU timestamp queries (authoritative) |
| `VKOP_OPPROF=1` | record-side wall time |

#### 8. Text-to-image (image/exporter/image_gen)

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

