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
- 不再创建 descriptor pool / set。每个管线拥有一块
  `HOST_VISIBLE|HOST_COHERENT` 的 `VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT`
  缓冲区，按 `kInflight * 16` 个槽位划分，槽位步长是
  `align_up(layoutSize, descriptorBufferOffsetAlignment)`；
- `allocDescriptorSets()` 返回槽位 token（`index + 1`），不再调用
  `vkAllocateDescriptorSets`；`updateDescriptorSets()` 变成对每个 binding 调
  `vkGetDescriptorEXT` 写入映射内存，不再调用 `vkUpdateDescriptorSets`；
- 记录命令时只在切换管线时 `vkCmdBindDescriptorBuffersEXT` 一次，随后每个
  dispatch 记录 `vkCmdSetDescriptorBufferOffsetsEXT`。

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
| `VKOP_DESCRIPTOR_BUFFER=1` | 非 UMA（独显）也启用 descriptor buffer；默认只在 UMA 启用 |
| `VKOP_DESCRIPTOR_BUFFER_DEBUG=1` | 每建一条 descriptor-buffer 管线打一行 `[descbuf]` |

默认只在 UMA 设备启用，是因为 descriptor 缓冲区是 host-visible 内存：iGPU /
llvmpipe 上 CPU 写的就是 GPU 读的那块；独显上它位于 PCIe 后面，GPU 取
descriptor 可能比原来的 set 更慢。独显要用需显式 `VKOP_DESCRIPTOR_BUFFER=1`。

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
