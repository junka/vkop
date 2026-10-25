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

### ksplit 512-lane 变体（已证伪，勿重试）

问的是：ksplit 的 shared-memory fold（`ks_flat` + barrier）若真有可榨的尾巴，
把 block 从 256 lane 加到 512（同 `ksplit_q` 列覆盖、K 切深翻倍）应该能看到收益——
这也是「要不要换 subgroup 实现」的前置实验。2026-10-22 实测（560X，`VKOP_KSPLIT512=1`，
17 个 MatMul 单测含 byte-WIDE 形状全过）：

**测量口径（踩过的坑）**：`bench_matmul K N [reps]`，M 来自 `BENCH_M`；不要写成
`M K N reps`——那样会变成 K=1/reps=大数，跑几分钟且量的是空 dispatch。中段 N 用
`reps=20` 只有 ~26ms 采样，方差可达 50%（N=16384 三次跑出 1.42/0.93/1.41）；本表全部
用 `reps=200`（小 N）/`reps=50`（大 N）、A/B 交错、min-of-3。另：同一进程里连跑四个内置
形状时 lm_head 曾出现 13.6ms 的坏点，独立三次与前后的内置跑均为 9.83–9.86ms——**单点
异常一律重测，不要拿一次采样下结论**。

fp16 WIDE，K=2048，min-of-3 交错（ms）：

| N | 256-lane | 512-lane |
|---|---|---|
| 8192 | **1.144** | 1.327 |
| 12288 | 1.660 | 1.650 |
| 16384 | **0.923** | 0.955 |
| 24576 | 1.391 | 1.372 |
| 32768 | 1.872 | 1.900 |
| 49152 | 2.830 | 2.864 |
| 65536 | 3.797 | 3.870 |
| 98304 | 5.935 | 5.947 |
| 131072 | 8.147 | 8.138 |
| 151936 | 9.830 | 9.810 |

int8 WIDE，K=896（真实量化模型路径），min-of-3 交错（ms）：

| N | 256-lane | 512-lane |
|---|---|---|
| 896 | 0.104 | 0.103 |
| 4864（gate/up） | **0.167** | 0.212（+27%） |
| 8192 | **0.283** | 0.314（+11%） |
| 16384 | **0.514** | 0.684（+33%） |
| 32768 | 0.458 | 0.477 |
| 65536 | 0.853 | 0.872 |
| 151936（lm_head） | 2.300 | **2.250** |

结论：**512-lane 没有可用收益，subgroup 归约不立项。**
- 大 N 上两者持平（fp16 151936：+0.2%；int8 151936：−2.2%），都在噪声边缘；fold+barrier
  的尾巴在带宽饱和形状上本就被别的 block 的 streaming 盖住，翻倍切深无油水。
- 中 N 段 512 明显更差（int8 N=4864/16384 掉 27%/33%），因为同 q 下 `n_slice` 翻倍、
  单 lane chunk 减半。
- 中 N 段的真正问题是 **q 下限 32 block 把切分推得太深**：int8 N=16384（q=32、n_slice=8、
  chunk=112）只有 28 GB/s，而 N=32768（q=64、n_slice=4、chunk=224）是 70 GB/s——两倍数据
  反而更快。这是 256-lane 自己的 q 启发式问题，与 lane 宽度无关，可作为独立后续实验
  （放开 32-block 下限、改用「chunk 长度下限」选 q）。
- 开关保留（默认关，探测失败自动回落 256-lane），仅作 fold-tail 对照工具。

### int8 q 选择的 block 下限：32 → 8

上表暴露的「int8 N=16384 只有 28 GB/s、N=32768 反而 70 GB/s」不是 lane 宽度问题，也不是
q 方向问题（方向修复早就在），而是 **block 下限 32 在中段把 K 切得太深**：`q` 搜索的
`while (q > 4 && UP_DIV(n/16, q) < 32) q /= 2` 会把 q 压到让 `n_slice` 翻倍、单 lane
chunk 减半。实测（560X，256-lane，`BENCH_INT8=1`，min-of-3；`VKOP_KSPLIT_FLOOR` 扫
4/8/16/32，ms）：

| N | floor 4 | floor 8 | floor 16 | floor 32（旧） |
|---|---|---|---|---|
| 4864 | 0.152 | **0.127** | 0.180 | 0.158 |
| 8192 | 0.243 | **0.133** | 0.271 | 0.287 |
| 16384 | 0.411 | **0.260** | 0.258 | 0.515 |
| 32768 | 0.474 | 0.473 | 0.479 | 0.461 |
| 65536 | 0.924 | 0.935 | 0.910 | 0.938 |
| 151936 | 2.34 | 2.35 | 2.333 | 2.341 |

大 N 的 ±3% 在 min-of-5 复测下消失（32768：0.533 vs 0.535；65536：0.879 vs 0.877；
151936：2.261 vs 2.242），因为那里 floor 已不再约束（q=256 本身就够 37 个 block）。
所以取 floor=8：中段 −20%…−50%，大 N 不动。floor=16 是陷阱——它在 N=8192 落到 chunk
112（0.271）、在 N=16384 落到 chunk 224（0.258），只覆盖一头。

顺带把 `VKOP_KSPLIT_Q` 变成真正的 override：以前 pin 住的 q 仍会被 floor 走回去，
N=16384 下 q=32/64/128 三个 pin 得到同一个数，就是这个原因。

实机验证（12 token）：输出逐 token 一致；`MatMul` 单测 18 项全过。fp16 WIDE 分支的下限
仍是 32（未动，实测 N=16384 已到 72.7 GB/s，不受此问题影响）。

### tile vs GEMV 交叉点（`tiled_min_m` profile 标定）

2026-10-25 实测（本机双卡：AMD Radeon Pro 560X `BENCH_DEV=0`、Intel UHD
Graphics 630 `BENCH_DEV=1`，均 MoltenVK x86_64，`VKOP_NO_COOP=1`；
`bench_matmul` / `bench_matmul_i8`，每个点 3 轮取最小，单轮仍是平均值，
本机 run-to-run 抖动可达 30–50%，改阈值前应按同一口径复测）。tile 一次固定
算满 `TILE_M=64` 行，所以 m 方向耗时平坦；split-K GEMV 每多一行多流一遍权重。
split-K 只在 `batch*m <= 8` 时启用，这条窗口决定了阈值的上限：

| 设备 | 权重 | K | N | GEMV 每行 | tile | 交叉点 |
|---|---|---|---|---|---|---|
| 560X | int8 | 2048 | 6144 | ~0.30 ms | 2.46 ms | ≈8 |
| 560X | int8 | 2048 | 151936 | ~4.4 ms | 64 ms | ≈14（超出窗口） |
| 560X | fp16 | 2048 | 6144 | ~0.65 ms | 2.26 ms | ≈4 |
| 560X | fp16 | 2048 | 151936 | ~8.3 ms | 48.3 ms | ≈6 |
| UHD630 | int8 | 2048 | 6144 | ~1.4 ms | 21 ms | > 8 |
| UHD630 | int8 | 2048 | 151936 | ~14.8 ms | 500 ms | > 8 |
| UHD630 | fp16 | 2048 | 6144 | ~1.1 ms | 15 ms | > 8 |
| UHD630 | fp16 | 2048 | 151936 | ~22.7 ms | 360 ms | > 8 |

- 560X int8 在窗口内一直是 GEMV 更快：m=8、N=6144 时 2.43 vs tile 2.46（持平），
  N=151936 时 36 vs 64 ms（1.8x）；交叉点 ≈14 已越过窗口，取不到。所以
  `byte_min_m = byte_wide_min_m = 9`——GEMV 吃满 `batch*m <= 8`，tile 从 9 开始。
- 560X fp16 每行权重是双倍字节，交叉点落在窗口内：m=4、N=6144 时 GEMV 2.99 vs
  tile 2.26；m=6、N=151936 时 49.7 vs 48.4。`wide_n=65536` 切出 lm_head 这一档。
- UHD630 的 MSL tile 更差，两个 dtype 的交叉点都在窗口外：int8 m=8 时 GEMV
  10.3 ms vs tile 21.2 ms（N=6144，2x）、119 vs 508（N=151936，4.3x）；
  fp16 m=8 时 7.95 vs 15（2x）、173 vs 366（2.1x）。所以
  `byte_min_m = fp16_min_m = 9`、`wide_n=0`（不需要分档）。
- 窗口外（m ≥ 9）GEMV 不再可用，退化为 column-parallel：560X int8 N=6144
  6.8 ms、N=151936 135 ms，fp16 5.7 / 134 ms，都明显慢于 tile，所以 tile
  必须从 9 接上。UHD630 上 fp16 同样是 tile 更快（26/645 vs 15/360），int8
  N=6144 两者相当（22 vs 21），N=151936 column-parallel 略快（393 vs 500，
  m=9..11 段）——按 dtype 统一取 9，不做第二档。
- 阈值来自 `VulkanDevice::MatmulTileProfile`：`driverID == MOLTENVK` 且
  `vendorID == 0x1002`（AMD）套用上表 560X 值 `byte_min_m=9`、`fp16_min_m=4`、
  `wide_n=65536`（`byte_wide_min_m=9`、`fp16_wide_min_m=6`）；`vendorID ==
  0x8086`（Intel）套用 `byte_min_m=9`、`fp16_min_m=9`、`wide_n=0`。ARL/Ada
  保持默认 `2`/`3`、`wide_n=0`。power-of-two ≥ 64 KiB 行 stride
  （fp16 N=32768/65536/131072、byte N=65536/131072）仍优先走 12，profile
  不覆盖该分支。
- e2e 不受影响：int8、prompt m≈9 时 N=6144 的 tile 3.75 ms 对 column-parallel
  6.5 ms，lm_head 走 tile；变的是 `batch*m <= 8` 的 decode 与极短 prompt。
- 同机 coop 不可用：M=16 时加不加 `VKOP_NO_COOP=1` 都是 2.597 ms（走 tile），
  因此 `8aea65f` 的 coop 几何选择在本机不派发；`matmul_coop16` /
  `matmul_coop_i8` 只是随包占字节。
- 复现（已按 HEAD 重编 libvkop + archive + 两个 bench）。`BENCH_DEV` 选卡，
  窗口内 GEMV 与窗口外 tile 各一条，N=151936 用 reps=8 控制耗时：

```bash
export DYLD_FALLBACK_LIBRARY_PATH=/usr/local/lib \
  VK_ICD_FILENAMES=/usr/local/etc/vulkan/icd.d/MoltenVK_icd.json VKOP_NO_COOP=1

# 560X：GEMV ~2.33 ms / tile ~2.3 ms
BENCH_M=3 BENCH_DEV=0 ./build/bench_matmul 2048 6144 30
BENCH_M=4 BENCH_DEV=0 ./build/bench_matmul 2048 6144 30

# UHD630：同一形状，GEMV ~7.9 ms / tile ~15 ms
BENCH_M=8 BENCH_DEV=1 ./build/bench_matmul 2048 6144 20
BENCH_M=9 BENCH_DEV=1 ./build/bench_matmul 2048 6144 20

# int8 lm_head：560X m=8 GEMV ~35 ms，UHD630 m=8 GEMV ~119 ms
./build/bench_matmul_i8 8 2048 151936 8
BENCH_DEV=1 ./build/bench_matmul_i8 8 2048 151936 5
```

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
- Conv 权重 image 的 fold 布局（`weight_pack`）现在由转换器写在节点上、默认开启，运行时的 `VKOP_CONV_WEIGHT_FOLD` A/B 开关已删除。原开关的收益是省 VkImage array pitch 而非 payload 字节（r18 fp32 −38%、r50 fp16 −15%），且依赖 kx innermost 循环序。
- `VKOP_REPLAY` 在 graph 模式被显式关闭（CNN 无从生效）；`VKOP_NO_ALIAS_BARRIER` 在 CNN 上 ±0（收益在 LLM decode 的 buffer 回收路径）。
- 方法：**配对轮换**（A,B,A,B...）按 rep 配对看逐对胜负，min-of-N 不可靠；跨批次绝对值会漂 ~4%，只有同批次相对值可信；`VKOP_OPPROF` 只是 host record 开销（resnet50 全图 0.23ms），GPU 时间看 `VKOP_RUN_PROFILE=1`。

## resnet 与 tiled GEMM

resnet 在 vkbench 里跑不到 tiled GEMM：vkbench 不切 buffer 后端，53 个 Conv 全走 image 后端的 naive `grouped_conv2d()`（无 shared memory）；buffer 后端的 conv 同样 naive，而 shared-memory tiled GEMM（`shaders/buffer/matmul.comp`）只服务 MatMul。要有收益只能：把 tiling 移植进 `shaders/image/conv2d.comp`，或 im2col + MatMul（激活膨胀 ~9x + 需要 buffer 后端）。参考：本机 fp16 相对 fp32 仅 5~6%（A2000 上 ~24%），说明 image conv kernel 没吃到 fp16 算力红利。
