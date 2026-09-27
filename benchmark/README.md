## benchmark

不会使用 CPU 做推理，CPU利用率低于2%，batch cmd 提交占用
measure 1000 rounds avgerage

download models from using script:
```
python3 model/download_models.py
```

convert models to vkopbin
```
python3 -m onnx2vkop.cli -i onnx_models/xxxx.onnx
```
quantize model
```
python3 -m onnx2vkop.cli -i onnx_models/xxxx.onnx -q int8
```


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

Apple M5 Max 一行于 2026-09-27 在 HEAD `fb16fc4` 重测（3 次 `avg time` 取中位，top-1 均为 `sports car`，概率与 onnxruntime 一致到 3 位小数：r18 0.691 / r34 0.437 / r50 0.280）。

## 性能开关（Apple M5 Max 实测）

resnet50 fp16。估计量用**每轮最小值**（100 轮去掉首尾各 3 轮后取 min）：GPU 上的干扰只会让个别轮次变慢（单边），min 比截尾中位数稳健得多；配置之间**轮换顺序、按 rep 配对**，n=5（graph/segment 关闭项 n=3）。
测于 2026-09-27 08:27，HEAD `fb16fc4`（kxk 权重折叠已是默认 + kx innermost 循环序），load average 3.0~3.4，每格 top-1 都是 `sports car (0.280)` 与 onnxruntime 一致。
两条口径提醒：① 上面那张跨 GPU 表用的是 vkbench 打印的 `avg time`（100 轮均值，含冷启动首轮），和本节不是同一口径；② **绝对值跨批次会漂**：本批 base 截尾中位数 7.288，上一批（HEAD `67e6b10`，折叠未默认）7.578，这 3.8% 里混着代码变更和批次漂移，不能拿跨批次的绝对值当结论，只有同批次内的相对值可信。

| 开关 | min (ms) | 相对基线 | n / 变快 | t | 判定 |
| --- | --- | --- | --- | --- | --- |
| 默认（graph submit ON, segment=64, fold ON, 70 levels） | 7.141 | — | — | — | 基线（5 个 rep 波动 1.1%） |
| `VKOP_GRAPH_SUBMIT=0` | 15.890 | **+123%** | 0/3，比值 2.22~2.23x | 483 | 最大项；已是默认行为 |
| 转换期 `VKOP_MERGE_LEVELS=8` | 6.865 | **-3.86%** | 5/5 | -13.1 | 正向，最有效 |
| `VKOP_GRAPH_NO_BARRIER=1` | 6.938 | **-2.84%** | 5/5 | -9.1 | 正向 |
| `VKOP_MERGE_LEVELS=8` + `VKOP_GRAPH_NO_BARRIER=1` | 6.859 | -3.95% | 5/5 | -12.8 | **≈单独开 ML8 → 不叠加** |
| `VKOP_GRAPH_SEGMENT=0`（整轮一段） | 7.103 | -0.47% | 2/3 | -0.6 | 不显著；70 levels 本就 ≤2 段 |
| `VKOP_CONV_WEIGHT_FOLD=0`（退回 array 布局） | 7.210 | **+0.97%** | 0/5 | 5.0 | 折叠现在是默认且是净收益 |

- 提交/屏障路径在 resnet 这种小图上总共只值约 4%，已经吃完了：`VKOP_GRAPH_NO_BARRIER`（运行时跳过 segment 内 pipeline barrier）和转换期 `VKOP_MERGE_LEVELS`（把 Kahn level 合并，70 层→个位数层）花的是同一份预算，后者吃得更干净；`VKOP_GRAPH_SUBMIT` 本身才是 2.2x 的量级，而它已经默认打开——所以基线剩下的优化空间在 kernel 里，不在提交路径。
- `VKOP_CONV_WEIGHT_FOLD` 的账要分两头看：延迟上它依赖循环序（旧序折叠 +0.8%，kx innermost 折叠 −0.9%，故 fb16fc4 把两者一起默认打开，本批实测关掉它贵 0.97%）；显存上它省的是 VkImage 的 array pitch 而非 payload 字节（只折 conv 权重：resnet18 fp32 −38.1%、resnet18 fp16 −33.4%、resnet50 fp32 −19.9%、resnet50 fp16 −15.4%，激活贡献不足 3%）。
- `VKOP_REPLAY` 在 graph 模式下被显式关闭，CNN 上无从生效；`VKOP_NO_ALIAS_BARRIER` 实测 ±0（收益在 LLM decode 的 buffer 回收路径）。
- 开关生效自检（每格都查过）：`VKOP_CONV_WEIGHT_FOLD=0` 的 stdout 里有未折叠权重 `3x1536x1 layers 128`、`3x192x1 layers 16`，默认格子里这些变成 `112x21x1 layers 1`、`48x192x1 layers 1`；`VKOP_LEVEL_STAT=1` 下基线每个 level 只有 1 个节点，ml8 模型的 level 里是 6~17 个节点（合并确实发生了）。
- 时间几乎全在 GPU：`VKOP_RUN_PROFILE=1` 下 7.4ms 里 gpu+reset 6.9ms、record 0.3ms、submit 0.2ms。别用 `VKOP_OPPROF` 判断占比——它的 per-op 时间只是 host 端 record 开销，整个 resnet50 加起来才 0.23ms。`gpu+reset` 只有 0.1ms 分辨率，不能拿来判 <1%。

### 启动方式：`DYLD_*` 要在启动命令上显式写出，不能靠继承

`/usr/bin/env`、`/bin/bash`、`/bin/zsh` 都是 SIP 保护的 shim，dyld 在 exec **它们**时会剥掉传进来的 `DYLD_*`，于是脚本里 `export` 之后 `bash script.sh`（或 `bash -c`）里的 vkbench 拿不到变量，`VulkanLib` 的 dlopen 失败（`Failed to load vulkan library` 然后 SIGSEGV）。只要把变量**写在这次启动的命令上**（赋值前缀、或 `env VAR=… ./vkbench` 这种显式形式）就能传到非保护的 vkbench 上——实测同一台机器：继承 export 失败，显式前缀正常。

```
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json \
VKOP_GRAPH_NO_BARRIER=1 ./build/benchmark/vkbench onnx_models/resnet50_fp16.vkopbin <image> benchmark/imagenet_classes.txt
```

另外：vkbench 的 INFO（`create image WxHxD layers L`、`avg time`）打到 **stdout**，只有 `[graph] …` 之类的行在 stderr；仓库根目录的 `log` 文件是 tests/example 里 `enableFileOutput("log")` 写的，vkbench 不写它，布局自检要去 grep stdout。

### 与 GEMM 优化的关系（重要）

resnet 在这份基线里**跑不到 tiled GEMM**：

- `Runtime::set_backend_buffer(true)` 只被 `llm_chat` / `llm_driver` / `visual_probe` / tests 调用，`vkbench` 从不切换，`backend_buffer_` 默认 false，所以 53 个 Conv 全部走 image 后端。
- image 后端的 conv 是 `shaders/image/conv2d.comp` 的 `grouped_conv2d()`：16x16 workgroup、**没有任何 shared memory 分块**（全文 0 处 `shared`）。fb16fc4 只改了权重布局（kxk 折叠）和转置族的循环序（kx innermost），分块状况不变。
- 新写的 shared-memory 64x64x16 tiled fp16 GEMM 在 `shaders/buffer/matmul.comp`，只有 SSBO buffer 后端的 **MatMul** 会用到。但 buffer 后端的 conv（`shaders/buffer/conv2d.comp`）同样是 naive direct conv（每线程一个输出标量、无 shared memory，fp16 还要 scratch+pack 两趟），所以**单纯给 vkbench 打开 buffer backend 并不会让 resnet 吃到 tiled GEMM**。

结论：今天没有任何一条路径把 resnet 的 53 个 conv 送进分块 GEMM。要有收益只能二选一——把 shared-memory tiling 移植进 `shaders/image/conv2d.comp`（真正被执行的那份），或者走 im2col + MatMul 让 conv 复用 `buffer/matmul.comp` 的 tiled GEMM（代价是激活膨胀约 9 倍 + 需要 buffer 后端）。参考成本：fp16 相对 fp32 在本机只有 5~6%（r18 3.80→3.60、r34 6.82→6.41、r50 7.86→7.43），而 A2000 上是 ~24%，说明当前 image conv kernel 没吃到 fp16 的算力红利。


## 模型转换
```
python3 -m onnx2vkop.cli <model.onnx>
```

## run benchmark
```
benchmark/vkbench <model.vkopbin> <image.jpg>
```