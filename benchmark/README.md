# benchmark

## vkbench — CNN 分类基准

GPU-only 推理（CPU 利用率 < 2%，耗时在 batch cmd 提交），100 轮取平均。

```bash
# 下载并转换模型
python3 model/download_models.py
python3 -m onnx2vkop.cli -i onnx_models/resnet50.onnx [-q int8]

# 运行
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib \
VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json \
./build/benchmark/vkbench onnx_models/resnet50_fp16.vkopbin <image.jpg> benchmark/imagenet_classes.txt
```

macOS 注意：`DYLD_*` 必须写在启动命令上（env 前缀）；经 `/usr/bin/env`、`bash -c` 继承会被 SIP 剥掉，dlopen 失败后 SIGSEGV。vkbench 的 INFO（`avg time` 等）在 stdout，`[graph] …` 在 stderr。

`VKOP_CNN_PROFILE=1` 在基准结束后输出 inference avg/p50/p90/p99、preprocess、postprocess 分解；per-op GPU 计时和 H2D/D2H 统计由 `vkop-bench --profile-out` 提供。

## vkop-bench — 通用驱动

不挑模型、不需要外部数据（dtype 匹配的随机输入，或 `--load-inputs` 的 `.npy`）：

```bash
build/benchmark/vkop-bench <model.vkopbin> [options]
```

| 选项 | 作用 |
|---|---|
| `--iters N` / `--warmup N` | 计时 / 预热轮数（默认 10 / 3） |
| `--precision fp16\|fp32` | 图精度（默认 fp16） |
| `--backend buffer\|image` | 运行时后端（默认 buffer），不匹配报可读错误并提示 `--backend image` |
| `--shape name:d1xd2x...` | 动态输入 resize，可重复 |
| `--seed N` | 随机输入种子（默认 `0xc0ffee`，可复现） |
| `--load-inputs DIR` / `--save-outputs DIR` | 喂 `.npy` / 末轮输出裸 `.bin`（逐字节 A/B 用） |
| `--profile-out FILE` | 逐 op GPU 计时 JSON（隐式开 submit profiling） |
| `--list-devices` / `--device N` | 列出 / 选择物理设备（DISCRETE > INTEGRATED > VIRTUAL > CPU） |
| `--list-inputs` / `--time-load-only` | 打印输入信息 / 只测 `LoadModel` |

相关环境变量（逻辑都在 libvkop）：

| 变量 | 作用 |
|---|---|
| `VK_PIPELINE_CACHE=1` | pipeline 序列化到 `<model>.vkopbin.pcache`（要配 `MESA_SHADER_CACHE_DISABLE=1` 才量得出差别） |
| `VKOP_SHADER_DEBUG=1` | 逐 op 打 shader 来源 hit/miss |
| `VKOP_SUBMIT_PROF=1` | 逐 op GPU 时间到 stderr |

Shader 只从模型内嵌归档来，**运行期没有替换 shader 的开关**：profile 跑的就是部署跑的 kernel，换 shader 必须重转。`Model.kernel_plans` 把 per-op kernel 选择写进文件，Python 筛归档、C++ 查 key 共用一张表，漂移即硬失败。

## bench_matmul — MatMul GEMV 微基准

decode 形状 fp16 buffer 后端（Qwen3-2B 投影权重形状），走真实 MatMul op + replay cache，纯 GPU 计时；附带 256MB memcpy 带宽上限和空提交开销两个参照。

```bash
# 构建：x86_64 g++-14，链 libvkop + libvload，-DVKOP_DEV_SHADER_ARCHIVE 指向
# build/vkop_shaders.bin（libvkop 不带 SPIR-V），完整命令见源码头注释
#
# 运行（x86_64 二进制用 /usr/local 的 Vulkan SDK 栈，不是 /opt/homebrew）：
DYLD_FALLBACK_LIBRARY_PATH=/usr/local/lib \
VK_ICD_FILENAMES=/usr/local/etc/vulkan/icd.d/MoltenVK_icd.json \
VKOP_NO_COOP=1 ./build/bench_matmul          # 或 ./build/bench_matmul K N [reps] 自定义形状
```

2026-10-21 实测（AMD Radeon Pro 560X，MoltenVK x86_64，reps=30，correctness probe 全过）：

| 项 | 耗时 | 带宽 |
|---|---|---|
| memcpy 256MB（读+写） | 7.43 ms | 72.2 GB/s |
| 空提交开销 | 0.024 ms | — |
| q/k/v/o K=2048 N=2048 | 0.311 ms | 27.0 GB/s |
| gate/up K=2048 N=6144 | 0.842 ms | 29.9 GB/s |
| down K=6144 N=2048 | 0.940 ms | 26.8 GB/s |
| lm_head K=2048 N=151936 | 19.50 ms | 31.9 GB/s |

## 跨 GPU 参考数据（resnet，100 轮平均）

| GPU | Model | Operators |Precision | latency (ms) |
| --- | ----- | --------- | -------- | ----------- |
| A2000| alexnet | 13 | fp32 | 6.91 |
| A2000| alexnet | 13 | fp16 | 6.01 |
| A2000| densenet121 | 246 | fp32 | 25.45 |
| A2000| densenet121 | 246 | fp16 | 22.26 |
| A2000| densenet161 | 326 | fp32 | 55.83 |
| A2000| densenet161 | 326 | fp16 | 46.75 |
| A2000| densenet201 | 406 | fp32 | 50.69 |
| A2000| densenet201 | 406 | fp16 | 45.92 |
| A2000| efficientnet_b0 | 124 | fp32 | 12.12 |
| A2000| efficientnet_b0 | 124 | fp16 | 11.94 |
| A2000| inceptionv3 | 120 | fp32 | 27.49 |
| A2000| inceptionv3 | 120 | fp16 | 20.62 |
| A2000| mobilenetv2 | 64 | fp32 | 3.60 |
| A2000| mobilenetv2 | 64 | fp16 | 3.24 |
| A2000| resnet18 | 31 | fp32 | 7.71 |
| A2000| resnet18 | 31 | fp16 | 6.81 |
| A2000| resnet34 | 55 | fp32 | 14.08 |
| A2000| resnet34 | 55 | fp16 | 12.02 |
| A2000| resnet50 | 72 | fp32 | 18.36 |
| A2000| resnet50 | 72 | fp16 | 14.80 |
| A2000| resnet101 | 140 | fp32 | 33.61 |
| A2000| resnet101 | 140 | fp16 | 26.66 |
| A2000| resnet152 | 208 | fp32 | 47.04 |
| A2000| resnet152 | 208 | fp16 | 37.79 |
| A2000| shufflenet_v2_x1_0 | 136 | fp32 | 4.24 |
| A2000| shufflenet_v2_x1_0 | 136 | fp16 | 3.89 |
| A2000| squeezenet1_0 | 38 | fp32 | 4.82 |
| A2000| squeezenet1_0 | 38 | fp16 | 3.96 |
| A2000| vgg16 | 23 | fp32 | 57.85 |
| A2000| vgg16 | 23 | fp16 | 46.56 |
| T2000| resnet18 | 31 | fp16 | 17.04 |
| T2000| resnet34 | 73 | fp16 | 30.03 |
| T2000| resnet50 | 90 | fp16 | 35.30 |
| Tegra Orin| resnet18 | 31 | fp32 | 14.76 |
| Tegra Orin| resnet18 | 31 | fp16 | 11.38 |
| Tegra Orin| resnet18 | 31 | int8 | 10.93 |
| Tegra Orin| resnet50 | 72 | fp16 | 26.61 |
| Tegra Thor| resnet50 | 72 | fp32 | 20.89 |
|Intel ARL| resnet18 | 31 | fp32 | 33.75 |
|Intel ARL| resnet18 | 31 | fp16 | 28.59 |
|Intel ARL| resnet34 | 55 | fp32 | 62.29|
|Intel ARL| resnet34 | 55 | fp16 | 55.75|
|Apple M5 Max (MoltenVK)| resnet18 | 31 | fp32 | 3.80 |
|Apple M5 Max (MoltenVK)| resnet18 | 31 | fp16 | 3.60 |
|Apple M5 Max (MoltenVK)| resnet34 | 55 | fp32 | 6.82 |
|Apple M5 Max (MoltenVK)| resnet34 | 55 | fp16 | 6.41 |
|Apple M5 Max (MoltenVK)| resnet50 | 72 | fp32 | 7.86 |
|Apple M5 Max (MoltenVK)| resnet50 | 72 | fp16 | 7.43 |

## 性能开关要点（Apple M5 Max，resnet50 fp16）

- `VKOP_GRAPH_SUBMIT=0` 是最大项（2.2x），已默认开。提交/屏障路径总共只值 ~4%：`VKOP_MERGE_LEVELS=8`（转换期 level 合并，−3.9%）和 `VKOP_GRAPH_NO_BARRIER=1`（−2.8%）花同一份预算、不叠加。基线剩下的优化空间在 kernel，不在提交路径。
- `VKOP_CONV_WEIGHT_FOLD` 已默认且净收益（关掉 +0.97%）；省的是 VkImage array pitch 而非 payload 字节（r18 fp32 −38%、r50 fp16 −15%），且依赖 kx innermost 循环序。
- `VKOP_REPLAY` 在 graph 模式被显式关闭（CNN 无从生效）；`VKOP_NO_ALIAS_BARRIER` 在 CNN 上 ±0（收益在 LLM decode 的 buffer 回收路径）。
- 方法：**配对轮换**（A,B,A,B...）按 rep 配对看逐对胜负，min-of-N 不可靠；跨批次绝对值会漂 ~4%，只有同批次相对值可信；`VKOP_OPPROF` 只是 host record 开销（resnet50 全图 0.23ms），GPU 时间看 `VKOP_RUN_PROFILE=1`。

## resnet 与 tiled GEMM

resnet 在 vkbench 里跑不到 tiled GEMM：vkbench 不切 buffer 后端，53 个 Conv 全走 image 后端的 naive `grouped_conv2d()`（无 shared memory）；buffer 后端的 conv 同样 naive，而 shared-memory tiled GEMM（`shaders/buffer/matmul.comp`）只服务 MatMul。要有收益只能：把 tiling 移植进 `shaders/image/conv2d.comp`，或 im2col + MatMul（激活膨胀 ~9x + 需要 buffer 后端）。参考：本机 fp16 相对 fp32 仅 5~6%（A2000 上 ~24%），说明 image conv kernel 没吃到 fp16 算力红利。
