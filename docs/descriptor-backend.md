# 描述符后端：descriptor indexing 与 descriptor buffer

2026-11-09。本文记录两件事：设备能力如何探测，以及探测通过后引擎怎么用它们。

## 探测

```bash
VKOP_DEVCAPS=1 ./build/tests/all_tests --gtest_filter='AddTest.*'
```

`[devcaps]` 新增三行：

| 名称 | 含义 |
|---|---|
| `descriptor_indexing` | `descriptorBindingStorageBufferUpdateAfterBind`（Vulkan 1.2 core 或 `VK_EXT_descriptor_indexing`） |
| `descriptor_partially_bound` | `descriptorBindingPartiallyBound` |
| `descriptor_buffer` | `VK_EXT_descriptor_buffer`，并且需要 BDA 可用 |

`driver=` 是驱动报告的位，`device=` 是真正交给 `vkCreateDevice` 的位。三者都有
`available / supported / enabled` 三态（`not queried` 表示扩展没枚举到）。

设备初始化还会打印一行 `descriptor buffer: offsetAlign ... sb ... img ...`，
是 `descriptorBufferOffsetAlignment` 和各类型描述符字节数。

## descriptor indexing 的用法

引擎的 descriptor 路径本来就在用 update-after-bind，但旧实现只询问并启用
`descriptorBindingStorageBufferUpdateAfterBind`，而 `SplitImage` / `Conv2dImage`
等管线把 `VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT` 加在 image binding 上。
现在按 binding 的 `VkDescriptorType` 逐个查询并启用：

- `descriptorBindingStorageBufferUpdateAfterBind`
- `descriptorBindingStorageImageUpdateAfterBind`
- `descriptorBindingSampledImageUpdateAfterBind`
- `descriptorBindingUniformBufferUpdateAfterBind`

`VulkanBasePipeline::createDescriptorSetLayout` 只给设备真正启用了对应位的类型打
binding flag；只要有一个 binding 带 UAB，layout/pool 才加
`UPDATE_AFTER_BIND_POOL`。同时，如果 `descriptorBindingPartiallyBound` 可用，
所有 binding 都会带 `PARTIALLY_BOUND`：buffer 算子会声明比实际绑定更多的可选
SSBO（例如 Binary 的 shape-meta 通道），这样这些没写的 binding 是合法的。

## descriptor buffer 的用法

支持且 BDA 可用时，`VulkanPipeline` 在创建 layout 前调用
`VulkanBasePipeline::enable_descriptor_buffer()`。之后该管线：

- descriptor set layout 加 `VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT`，
  compute pipeline 加 `VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT`；
- 不再创建 descriptor pool / set。整台设备共享**一块**
  `VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT` 缓冲区
  （`DescriptorBufferPool`，默认 32 MiB），每个管线从里面切一块
  `kInflight * 16` 槽位的 region，槽位步长是
  `align_up(layoutSize, descriptorBufferOffsetAlignment)`；
- `allocDescriptorSets()` 返回槽位 token（`index + 1`），不再调用
  `vkAllocateDescriptorSets`；`updateDescriptorSets()` 变成对每个 binding 调
  `vkGetDescriptorEXT` 写入映射内存，不再调用 `vkUpdateDescriptorSets`；
- 记录命令时 `vkCmdBindDescriptorBuffersEXT` 每条 command buffer 只需一次，
  随后每个 dispatch 只记录 `vkCmdSetDescriptorBufferOffsetsEXT`（offset =
  region 基址 + 槽位偏移）。共享一块缓冲区是 b3 上消除每 op 重新绑定的关键。

descriptor 缓冲区优先放在 `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT`
（iGPU 的 UMA 类型、独显的 BAR）：CPU 写、GPU 读同一块本地内存。没有这种
内存类型时退回 `HOST_VISIBLE|HOST_COHERENT`（system RAM），此时独显默认
不启用 descriptor buffer（GPU 走 PCIe 取 descriptor 会明显变慢，见下文 b3
实测）。

SPIR-V 不需要任何改动：set/binding 声明不变，只是 descriptor 的来源换了。

支持的 descriptor 类型：`STORAGE_BUFFER`、`UNIFORM_BUFFER`、`STORAGE_IMAGE`、
`SAMPLED_IMAGE`、`COMBINED_IMAGE_SAMPLER`。声明了 texel buffer
（`UNIFORM_TEXEL_BUFFER` / `STORAGE_TEXEL_BUFFER`）的管线仍走 descriptor set：
descriptor buffer 用 address+format 表示 texel buffer，而
`VkWriteDescriptorSet` 只带 `VkBufferView`，这里拿不到所需信息。

任一步失败（entry point 缺失、描述符大小为 0、内存不可映射）都会销毁带
descriptor-buffer flag 的 layout 并重建，回落到原来的 descriptor set 路径。

## 开关

| 变量 | 作用 |
|---|---|
| `VKOP_NO_DESCRIPTOR_BUFFER=1` | 特性完全不出现在 `vkCreateDevice` 里，强制 descriptor set |
| `VKOP_DESCRIPTOR_BUFFER=1` | 强制启用（即使没有 device-local host-visible 内存） |
| `VKOP_DESCRIPTOR_BUFFER_POOL_MB=<n>` | 共享 descriptor buffer 大小，默认 32 MiB |
| `VKOP_DESCRIPTOR_BUFFER_DEBUG=1` | 每建一条 descriptor-buffer 管线打一行 `[descbuf]` |

默认策略：UMA 设备，或存在 `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT`
（独显 BAR）时默认启用；只有 system RAM 可用的独显默认走 descriptor set
（`VKOP_DESCRIPTOR_BUFFER=1` 可强制）。

## 本机实测（Intel ARL / Mesa 25.2.8 / Qwen2.5-0.5B decode dump）

`VK_DS_WRITE_PROF=1 ./build/llm_driver llm/exporter/llm.vkopbin llm/exporter/decode_dump`：

| | descriptor set | descriptor buffer |
|---|---|---|
| 加载阶段 set 分配 | 2040 次，0.9 ms（占 1.9 s 加载 0.05%） | 0 |
| 每轮 round update | 0.26–0.29 ms | 0.22–0.25 ms |
| 每轮 bind_ds | 0.12–0.13 ms | 0.14–0.16 ms |
| 结果 | 6/6 MATCH | 6/6 MATCH（`VKOP_BDA=1` 也 6/6） |

decode 一轮 ~110 ms 且 GPU 执行占满，descriptor 管理本来就只有 0.3%，
所以 round 时间无可测差异；descriptor buffer 的收益是去掉加载期 2040 次
descriptor set 分配，host 端 update 略降、bind 略升，基本抵消。

`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` 下，上述测试（含 Conv2d 图像
路径、Softmax 双 pass、ScatterND 双 pass）没有 descriptor buffer / descriptor
indexing 相关的 VUID。llvmpipe 也通过。

## Qwen3-VL-2B decode A/B（Intel ARL，2026-11-09）

模型：`llm/exporter/llm.onnx` 用当前 `build/vkop_shaders.bin` 重转
（`--target-backend buffer --extra-shaders TopK`，内嵌 137 条 shader）。
旧 `llm/exporter/llm.vkopbin`（10-29）的内嵌归档缺当前 kernel plan 的
`buffer_matmul_coop_*` / `buffer_topk`，默认配置跑不起来，必须先重转。

提示词 `What is the capital of France?`（`VKOP_RAW_PROMPT=1`），
`VKOP_GPU_ARGMAX=1`，`max_new=8 --max-kv 128`。每臂 6 次进程，A/B 按
A B B A 交替；A = 默认（descriptor buffer），B = `VKOP_NO_DESCRIPTOR_BUFFER=1`。

| | A: descriptor buffer | B: descriptor set |
|---|---|---|
| 7 步 per-step min 之和 / 7 | 66.47 ms/token | 63.31 ms/token |
| per-step 中位（全部 42 步） | 70.75 ms/token | 69.05 ms/token |
| `[submitprof]` decode TOTAL 中位 | 58.6 ms | 59.3 ms |
| `[submitprof]` MatMul 中位 | 54.3 ms | 55.0 ms |
| 稳态 `update` 中位 | 0.23 ms/步 | 0.31 ms/步 |
| 稳态 `bind_ds` 中位 | 0.17 ms/步 | 0.14 ms/步 |
| 加载期 set 分配 | 0 | 2040 次，~0.9 ms |

结论：

- 两臂 GPU 执行时间（submitprof TOTAL / MatMul）在噪声内相同，说明
  descriptor buffer 没有改变 kernel 执行；host 端 per-step 净变化约
  −0.05 ms（update −0.08，bind +0.03），远小于 iGPU 单轮 2–3x 噪声。
- 加速收益仍是**加载期 2040 次 descriptor set 分配归零**；decode 是 GPU-bound，
  拿不到 round 级收益。
- 与 `docs/down-act-fold-report.md` 记录的 ARL Qwen3-VL 基线对照：该文 fused
  67.56 ms/token / MatMul 54.96（凉机），unfused 62.09 / 50.90。本次重转日志为
  `fuse_swiglu_down_proj match (28 fused)`，即 fused 形态；A 臂 66.47 / 54.3、
  B 臂 63.31 / 55.0，与记录中的 fused 基线一致，没有回归。

## b3（RTX 4000 Ada）decode A/B（2026-11-09，修复前）

代码用 `git ls-files` 同步到 `b3:~/vkop_b6a23352`（b3 无 git），新建
`build_llm`：`g++-13`、`-O0 -g`（与仓库硬编码一致）、
`-DENABLE_TESTS=OFF -DENABLE_LLM_CHAT=ON -DENABLE_EXAMPLE=OFF
-DENABLE_BENCHMARK=OFF -DUSE_PLATFORM_XCB=OFF -DUSE_PLATFORM_XLIB=OFF`，
glslc 用 `/tmp/shaderc_b3/bin/glslc`（需 `LD_LIBRARY_PATH=/tmp/shaderc_b3/lib`）。
b3 没有 libutf8proc-dev，从开发机拷 `utf8proc.h` + `libutf8proc.a`，
`-DUTF8PROC_ROOT=~/vkop_run/utf8proc -DCMAKE_EXE_LINKER_FLAGS=-L.../utf8proc/lib`。

同一份重转模型、同一 prompt/参数（`max_new=14 --max-kv 128`，RAW prompt，
GPU argmax），每臂 6 次进程 A B B A 交替。A = 默认（独显策略：descriptor
set），B = `VKOP_DESCRIPTOR_BUFFER=1`。b3 的 NVIDIA EGL atexit 会在 main
返回后 SIGSEGV，按输出判读。

| | A: descriptor set（默认） | B: descriptor buffer（强制） |
|---|---|---|
| 稳态 per-step 中位（r≥6） | **27.95 ms** | 30.35 ms（+8.6%） |
| 稳态 per-step min / mean | 26.90 / 28.61 | 29.20 / 31.22 |
| `[submitprof]` decode TOTAL 中位 | 13.32 ms | 13.32 ms |
| `[submitprof]` MatMul 中位 | 11.89 ms | 11.88 ms |
| 稳态 `update` 中位 | 0.150 ms/步 | 0.170 ms/步 |
| 稳态 `bind_ds` 中位 | 0.150 ms/步 | 0.170 ms/步 |
| 加载期 set 分配 | 2040 次 | 0 |

结论：

- GPU kernel 时间完全相同（TOTAL/MatMul 到 0.01 ms），但 b3 上强制
  descriptor buffer 的 per-step wall 稳定慢 ~2.4 ms（8.6%）。这是独显 +
  host-visible descriptor 内存的额外取描述符/命令处理开销，且
  `vkGetDescriptorEXT` 在 NVIDIA 驱动上比 Mesa 贵；这就是默认只在 UMA
  启用、独显要 `VKOP_DESCRIPTOR_BUFFER=1` 才开的原因。b3 默认臂不受影响。
- 跨机器对照（同代码、同 -O0、同模型）：ARL 稳态 ~70.75 ms/token、
  GPU TOTAL 58.6 ms、MatMul 54.3 ms；b3 默认臂 27.95 / 13.32 / 11.89，
  即 wall ~2.5x、GPU ~4.4x、MatMul ~4.6x。b3 上 wall 与 GPU 的差距来自
  600 个 op 的 CPU 录制开销（GPU 太快，CPU 不再被隐藏）。

### 修复前那一版说明

上面 b3 表的 A 臂是默认 descriptor set，B 臂是强制 descriptor buffer；B 慢
~2.4 ms/步。下面追查这 2.4 ms 从哪来。

## b3 追查：不是 `vkGetDescriptorEXT`，是 descriptor 内存位置

新增 `tests/DescriptorBufferProbe`（独立探针，非 gtest；无扩展时 SKIP），
直接测 host 侧 API：

| 项目 | ARL (Mesa) | b3 (NVIDIA 595.99) |
|---|---|---|
| `vkGetDescriptorEXT` | 17.7 ns/call | **4.0 ns/call** |
| `vkUpdateDescriptorSets`（7 writes） | 115–131 ns/call | 49.7 ns/call |
| `vkCmdBindDescriptorSets`（录制） | 5.4 ns/cmd | 46.5 ns/cmd |
| `vkCmdSetDescriptorBufferOffsetsEXT`（buffer 已绑） | 5.1 ns/cmd | 10.9 ns/cmd |
| bind buffer + set offsets（录制） | 5.2 ns/cmd | 13.7 ns/cmd |

结论：**`vkGetDescriptorEXT` 在 b3 上不但没有额外开销，反而比
`vkUpdateDescriptorSets` 还便宜**；descriptor-buffer 命令的录制也比
`vkCmdBindDescriptorSets` 便宜。host 侧不是那 2.4 ms 的来源。

真正的原因：descriptor buffer 原来分配在 `HOST_VISIBLE|HOST_COHERENT`
（b3 上是 system RAM，heap 1），GPU 每次 dispatch 都要通过 PCIe 取描述符。
`[submitprof]` 是 shader 内的时间戳，看不到命令处理器取 descriptor 的开销，
所以表现为「GPU 时间相同、wall 变慢」。

b3 的内存类型里有：

```
[4] heap=2 246 MiB flags=0x7 DEVICE_LOCAL HOST_VISIBLE HOST_COHERENT  (BAR1)
```

## 规避 / 修复

分两步：

1. **bar 内存**：descriptor buffer 优先选
   `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT`（iGPU 的 UMA 类型、独显的 BAR），
   没有这种类型时回落到 `HOST_VISIBLE|HOST_COHERENT`。ARL 的 type 0 本来
   就是前者，行为不变。b3 上从 30.35 → 27.65 ms（相对 set 臂 27.10，
   +2.0%；修复前 +8.6%）。
2. **共享 descriptor buffer**：新增 `DescriptorBufferPool`，整台设备一块
   32 MiB 缓冲区，所有管线切 region。默认 `VKOP_GRAPH_SUBMIT` 是开的，
   一段的 op 在同一条 command buffer 里，所以原来「每 op 重新
   `vkCmdBindDescriptorBuffersEXT`」变成「每段一次」，每个 dispatch 只剩
   `vkCmdSetDescriptorBufferOffsetsEXT`。

最终 b3 A/B（新默认 = descriptor buffer，对比 `VKOP_NO_DESCRIPTOR_BUFFER=1`，
A B B A × 2，r≥6 稳态）：

| | A: descriptor buffer（默认） | B: descriptor set |
|---|---|---|
| per-step 中位 | **26.90 ms** | 27.10 ms |
| per-step mean / min | 27.07 / 26.60 | 27.25 / 26.80 |
| GPU decode TOTAL 中位 | 13.05 ms | 13.34 ms |
| MatMul 中位 | 11.80 ms | 11.91 ms |

两臂已在噪声内，descriptor buffer 略快。默认策略相应改为：

- UMA，或存在 `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT`（BAR）→ 默认开；
- 只有 system RAM 的独显 → 默认关（GPU 走 PCIe 取 descriptor，仍会亏；
  `VKOP_DESCRIPTOR_BUFFER=1` 可强制）；
- `VKOP_NO_DESCRIPTOR_BUFFER=1` 全局关闭。
