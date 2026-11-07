# 双 GPU“类零拷贝”数据通路调研（iMac 2019 / macOS / MoltenVK）

日期：2026-11-06（2026-11-07 补充 4.7 节带宽实测）　环境：iMac 2019，i9-9980HK，AMD Radeon Pro 560X（4GB，独显）+ Intel UHD Graphics 630（核显），macOS，MoltenVK 1.4.2（`VkPhysicalDeviceProperties.driverName = MoltenVK`，driverVersion `0.2.2210`）。

证据标记：

- **[V]** vulkaninfo 实测原始输出
- **[P]** 本次自写 Vulkan 探针实测（源码与命令见第 6 节）
- **[C]** 引擎代码
- **[S]** MoltenVK 源码（v1.4.2 tag）
- **[I]** 文档/源码推断，未直接测到

---

## 0. 结论（先给判断）

1. **在两块 GPU 之间做真正的零拷贝：不可行。** 两块设备各自是一个独立的 `VkPhysicalDeviceGroup`（每组 1 个设备，`subsetAllocation=0`），没有 device group / peer memory 可用；`VK_EXT_external_memory_metal` 虽然报告 `importable|exportable` 且导入调用能成功，但把导入的 `MTLBuffer` 交给另一块 GPU 使用时，Intel→AMD 直接触发 AMD Metal 驱动断言崩溃，AMD→Intel 静默读到全 0。**这条路不仅不可用，而且危险。**
2. **单块卡内部“真零拷贝”只有 Intel 成立。** Intel 是单堆 UMA，`type[1] = DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT|HOST_CACHED`，MoltenVK 把它映射为 `MTLStorageModeShared`。主机写进去的字节 GPU 直接能读，实测无 staging 也 MATCH。AMD 没有这种类型：`type[2] = DEVICE_LOCAL|HOST_VISIBLE|HOST_CACHED` 没有 COHERENT，MoltenVK 在非 Apple GPU 的 macOS 上把它映射为 `MTLStorageModeManaged`，主机写的是 CPU 影子内存，`vkFlushMappedMemoryRanges` 会触发 `didModifyRange` 同步到 VRAM。**它减少拷贝“步骤”，但没有消除 PCIe 传输，严格意义上不是零拷贝。**
3. **真正能把“零拷贝”用在主数据通路上的，是 `VK_EXT_external_memory_host` + `type[1]`。** 把页对齐的主机内存直接 import 成 `VkDeviceMemory`，GPU 直接读同一份字节（AMD 上经 PCIe，Intel 上就是同一块 UMA 内存），实测两块卡都 MATCH。代价是 AMD 侧带宽受 PCIe 限制，只能用于小块、时延敏感的数据（token 输入、小标量、动态元数据），不能用于权重。
4. **引擎现在一块卡都没用满这套能力，更没用上第二块卡。** 默认只创建一个 `VulkanDevice`（排序后 AMD 是 0 号），Intel 只作为 `VKOP_DEVICE=1` 的候选存在。VMA 实测两块卡的设备侧 buffer 都落在 `type[0]`（纯 private），**Intel 的 UMA 共享类型从未被使用**（Intel 的 staging buffer 反而落在 type1，即设备本地共享内存）。
5. **退而求其次能拿到的东西：**（a）Intel 单卡真零拷贝；（b）主机内存直接 import，省掉全部上传拷贝但受 PCIe 带宽约束；（c）AMD 上把上传从“staging + vkCmdCopyBuffer”改成“managed 映射 + flush”，省掉 staging 分配和一次 dispatch/submit，但 PCIe 字节数不变。

---

## 1. 实测能力（vulkaninfo + 探针）

### 1.1 设备、类型、API 版本

`vulkaninfo --text` 原始片段（`/tmp/vk_full.txt`）：

```
	apiVersion        = 1.4.357 (4211045)
	driverVersion     = 0.2.2210 (10402)
	deviceType        = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
	deviceName        = AMD Radeon Pro 560X
...
	apiVersion        = 1.4.357 (4211045)
	driverVersion     = 0.2.2210 (10402)
	deviceType        = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
	deviceName        = Intel(R) UHD Graphics 630
```

说明：实例版本 1.4.357。探针在创建 1.3 实例时，两块设备回报的 `apiVersion` 是 `1.3.357`（`[P]`），vulkaninfo 用 1.4 实例时回报 `1.4.357`（`[V]`）。这是 MoltenVK 按实例版本协商的结果，不影响本调研结论。

### 1.2 是否同一个 `VkPhysicalDeviceGroup`

vulkaninfo 原始片段：

```
Device Groups:
==============
Group 0:
	Properties:
		physicalDevices: count = 1
			AMD Radeon Pro 560X (ID: 0)
		subsetAllocation = 0
...
Group 1:
	Properties:
		physicalDevices: count = 1
			Intel(R) UHD Graphics 630 (ID: 0)
		subsetAllocation = 0
```

探针 `vkEnumeratePhysicalDeviceGroups` 复核（`[P]`）：

```
== vkEnumeratePhysicalDeviceGroups: result=0 groupCount=2
   group[0]: deviceCount=1 subsetAllocation=0
      dev AMD Radeon Pro 560X
   group[1]: deviceCount=1 subsetAllocation=0
      dev Intel(R) UHD Graphics 630
```

**结论：没有跨设备的 device group，也没有 subset allocation。** `VK_KHR_device_group` / peer memory 路径在本机不成立。

### 1.3 内存堆与 memory type

> 说明：为可读性，下面把 vulkaninfo 中 `memoryTypes[i]` 的多行 `heapIndex/propertyFlags` 合并成一行，枚举值内容未改动；完整原文见 `/tmp/vk_full.txt`。

AMD（vulkaninfo 原始）：

```
memoryHeaps: count = 2
	memoryHeaps[0]:
		size   = 4294967296 (0x100000000) (4.00 GiB)
		budget = 4294967296 (0x100000000) (4.00 GiB)
		usage  = 0 (0x00000000) (0.00 B)
		flags: count = 1
			MEMORY_HEAP_DEVICE_LOCAL_BIT
	memoryHeaps[1]:
		size   = 34359738368 (0x800000000) (32.00 GiB)
		budget = 2995564544 (0xb28cb000) (2.79 GiB)
		usage  = 0 (0x00000000) (0.00 B)
		flags:
			None
memoryTypes: count = 3
	memoryTypes[0]: heapIndex = 0  flags = DEVICE_LOCAL
	memoryTypes[1]: heapIndex = 1  flags = HOST_VISIBLE|HOST_COHERENT|HOST_CACHED
	memoryTypes[2]: heapIndex = 0  flags = DEVICE_LOCAL|HOST_VISIBLE|HOST_CACHED
```

Intel（vulkaninfo 原始）：

```
memoryHeaps: count = 1
	memoryHeaps[0]:
		size   = 34359738368 (0x800000000) (32.00 GiB)
		budget = 1610612736 (0x60000000) (1.50 GiB)
		usage  = 8699904 (0x0084c000) (8.30 MiB)
		flags: count = 1
			MEMORY_HEAP_DEVICE_LOCAL_BIT
memoryTypes: count = 3
	memoryTypes[0]: heapIndex = 0  flags = DEVICE_LOCAL
	memoryTypes[1]: heapIndex = 0  flags = DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT|HOST_CACHED
	memoryTypes[2]: heapIndex = 0  flags = DEVICE_LOCAL|HOST_VISIBLE|HOST_CACHED
```

要点：

- **没有 `MULTI_INSTANCE` 堆**。
- AMD：`type[2]` 是唯一 `DEVICE_LOCAL+HOST_VISIBLE` 的类型，但**没有 HOST_COHERENT**；`type[1]` 在系统堆（heap 1），COHERENT+CACHED。
- Intel：单堆，`type[1]` 同时 DEVICE_LOCAL+HOST_VISIBLE+COHERENT+CACHED —— 这是真正的 UMA 共享类型。
- 预算快照：AMD VRAM 4.00 GiB；Intel 名义 32 GiB，但 `budget` 只有 **1.50 GiB**（Metal recommendedMaxWorkingSetSize）。当前 int8 权重约 1.72 GB，**超过 Intel 的推荐工作集**（不是不可分配，而是不保证常驻），这是“把整套模型放 Intel 跑”的硬约束之一。

### 1.4 扩展支持

两块设备各有 122 项设备扩展，**与本调研相关的内存/共享类扩展在两块卡上完全一致**（差异只在 fragment shader barycentric/interlock 之类的图形扩展）。相关项：

| 扩展 | 是否支持 |
|---|---|
| `VK_KHR_external_memory` / `..._capabilities` | 支持（实例级 + 设备级） |
| `VK_EXT_external_memory_host` | **支持**，`minImportedHostPointerAlignment = 0x1000` |
| `VK_EXT_external_memory_metal` | **支持**（rev 1） |
| `VK_EXT_metal_objects` | 支持（rev 2） |
| `VK_KHR_buffer_device_address` | 支持 |
| `VK_KHR_device_group` / `..._creation` | 支持（但没有跨设备 group） |
| `VK_KHR_maintenance1..9` | 全部支持 |
| `VK_EXT_memory_budget` | 支持 |
| `VK_KHR_external_memory_fd` | **不支持** |
| `VK_EXT_external_memory_dma_buf` | **不支持** |
| `VK_KHR_external_memory_win32` | **不支持** |

即：**Linux/Windows 那套 FD / dma_buf / win32 handle 共享在 macOS 上不存在。** 可用的只有 Metal 对象句柄和主机指针。

### 1.5 external buffer 能力（`vkGetPhysicalDeviceExternalBufferProperties`）

vulkaninfo 不打印这一项，用探针查询（`[P]`，两块卡结果相同）：

```
== external buffer properties for GPU0 (AMD Radeon Pro 560X)
   OPAQUE_FD  compatibleHandleTypes=0x0 features=0x0 exportFromImported=0x0
   DMA_BUF    compatibleHandleTypes=0x0 features=0x0 exportFromImported=0x0
   MTLBUFFER  compatibleHandleTypes=0x10000 features=0x6 exportFromImported=0x10000
   MTLTEXTURE compatibleHandleTypes=0x0 features=0x0 exportFromImported=0x0

== external buffer properties for GPU1 (Intel(R) UHD Graphics 630)
   （与 GPU0 完全相同）
```

`features=0x6` = `EXPORTABLE(0x2) | IMPORTABLE(0x4)`。注意这是 **VkBuffer** 查询；`MTLTEXTURE` 对 buffer 为 0。IOSurface 只对 image 有意义，而引擎的张量全是 SSBO buffer，不适用。

### 1.6 关键假设实验矩阵（全部为探针实测）

| 实验 | AMD 560X | Intel UHD 630 |
|---|---|---|
| `type[0]`（private）`vkMapMemory` | **FAIL** | **FAIL** |
| `type[1]` `vkMapMemory` | OK（系统堆，coherent） | OK（UMA，coherent） |
| `type[2]`（DEVICE_LOCAL+HOST_VISIBLE）`vkMapMemory` | OK（无 coherent，需 flush） | OK（需 flush） |
| 主机直接写映射的 `type[2]` + flush → GPU `vkCmdCopyBuffer` 读回 | **MATCH** | **MATCH**（用的是 type1） |
| 主机指针 import（`VK_EXT_external_memory_host`）到 `type[1]` → GPU 读 | **MATCH** | **MATCH** |
| 同上 import 到 `type[2]` | 分配成功、绑定成功，**GPU 读到 MISMATCH** | MATCH（UMA 下 Managed 行为退化） |
| 同上 import 到 `type[0]` | `VK_ERROR_INVALID_EXTERNAL_HANDLE (-1000072003)` | 同上 |
| 跨设备 Metal `MTLBuffer` 导入后使用（Intel→AMD） | **AMD Metal 驱动断言，进程 abort**：`cmdBufInfo->device == rsrc->_res.device` | — |
| 跨设备 Metal `MTLBuffer` 导入后使用（AMD→Intel） | — | 导入/绑定均返回成功，**GPU 读到 0x0（MISMATCH）** |

崩溃原始输出（`[P]`）：

```
Assertion failed: (cmdBufInfo->device == rsrc->_res.device),
function amdMtlBronzeCmdBufUpdateRsrc, file amd_mtl_bronze_hw_cmd_buffer.c, line 1126.
```

**这张表是本报告最重要的实测结果：** API 层“可导入/可导出”不等于可用；MoltenVK 不校验 Metal 设备归属，AMD 驱动会在命令缓冲引用跨设备资源时直接断言。

### 1.7 limits（跨设备传输相关）

> 说明：这些值分散在 `VkPhysicalDeviceLimits` / `VkPhysicalDeviceMaintenance3Properties` 等处，下面按行筛选汇总，值本身来自 vulkaninfo 原文。

vulkaninfo 原始值：

```
AMD:
	maxStorageBufferRange                           = 3221225472   (3 GiB)
	maxMemoryAllocationCount                        = 1073741824
	bufferImageGranularity                          = 0x00000100   (256 B)
	minImportedHostPointerAlignment = 0x00001000                   (4 KiB)
	maxMemoryAllocationSize           = 0xc0000000                 (3 GiB)

Intel:
	maxStorageBufferRange                           = 2147483648   (2 GiB)
	maxMemoryAllocationCount                        = 1073741824
	bufferImageGranularity                          = 0x00000100   (256 B)
	minImportedHostPointerAlignment = 0x00001000                   (4 KiB)
	maxMemoryAllocationSize           = 0x80000000                 (2 GiB)
```

对方案的约束：单次主机指针 import 必须 4 KiB 对齐；单块 buffer/分配上限 3 GiB（AMD）/2 GiB（Intel）；引擎现有的 1 GB staging 池上限与 64 MB 分块逻辑不受这两个上限限制，但“把整个 1.72 GB 权重做成一个 managed buffer”在 AMD 上仍可行（<3 GiB），在 Intel 上会超过 `maxMemoryAllocationSize` 和预算。

---

## 2. MoltenVK 在 Intel Mac + 独显上的真实内存模型

依据 MoltenVK v1.4.2 源码（`MVKDevice.mm` / `MVKDeviceMemory.mm`）。

### 2.1 映射规则（源码 `MVKPhysicalDevice::getMTLStorageModeFromVkMemoryPropertyFlags`）

```objc
// If not visible to the host, use Private, or Memoryless if available and lazily allocated.
if ( !mvkAreAllFlagsEnabled(vkFlags, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ) {
    ...
    return MTLStorageModePrivate;
}
// If visible to the host and coherent: Shared
if (mvkAreAllFlagsEnabled(vkFlags, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
    return MTLStorageModeShared;
}
// If visible to the host, but not coherent: Shared on Apple Silicon, Managed on other GPUs.
#if MVK_MACOS
return _gpuCapabilities.isAppleGPU ? MTLStorageModeShared : MTLStorageModeManaged;
#else
return MTLStorageModeShared;
#endif
```

对应到本机（两块卡都不是 Apple GPU）：

| Vulkan 类型 | MoltenVK `MTLStorageMode` | 实际含义 |
|---|---|---|
| AMD `type[0]`、Intel `type[0]` | `Private` | 纯 GPU 私有；`vkMapMemory` 直接报 `VK_ERROR_MEMORY_MAP_FAILED`（实测一致） |
| AMD `type[1]`（系统堆） | `Shared` | 系统内存，CPU/GPU 同一份；coherent |
| Intel `type[1]`（单堆） | `Shared` | 系统内存即“显存”；coherent，真 UMA |
| AMD `type[2]` | **`Managed`** | VRAM + CPU 影子内存；非 coherent |
| Intel `type[2]` | **`Managed`** | 同上（Intel GPU 也走 managed 分支） |

### 2.2 `vkMapMemory` 与 flush/invalidate 的真实行为

源码（`MVKDeviceMemory.mm`）：

```objc
// map()
_pMemory = isMemoryHostAccessible() ? buf.contents : nullptr;
...
// flushToDevice()
if ( !isUnifiedMemoryGPU() && _mtlBuffer && _mtlStorageMode == MTLStorageModeManaged ) {
    [_mtlBuffer didModifyRange: NSMakeRange(offset, memSize)];
}
...
// pullFromDevice()
if ( !isUnifiedMemoryGPU() && pBlitEnc && _mtlBuffer && _mtlStorageMode == MTLStorageModeManaged ) {
    [pBlitEnc->mtlBlitEncoder synchronizeResource: _mtlBuffer];
}
```

因此：

- AMD `type[2]` 的 `vkMapMemory` **是真映射**，返回 managed buffer 的 CPU 影子地址，不是“暂存区”的模拟。
- 主机写完必须 `vkFlushMappedMemoryRanges` → Metal `didModifyRange`；缺省不 flush 不可见。
- GPU→CPU 读回走 `vkInvalidateMappedMemoryRanges` → `synchronizeResource` **blit 编码器 + 提交**，不是免费操作。
- `isUnifiedMemoryGPU()` 为真时（Intel 单堆），flush/invalidate 是空操作。
- **[I]** `didModifyRange` 内部由 Metal 把 CPU 影子同步到 VRAM，是否 PCIe DMA 由驱动决定。本次只实测“flush 后 GPU 能读到”，没有测机制/带宽。**因此不能说 AMD 的 type2 是零拷贝，它是“少一次 Vulkan 层拷贝/提交”。**

### 2.3 为什么 `type[2]` 的主机指针 import 会坏

源码里 imported host memory 走：

```objc
buf = [getMTLDevice() newBufferWithBytesNoCopy: _pHostMemory length: memLen
                                       options: rezOpts deallocator: nil];
```

`rezOpts` 由上面的 storage mode 决定。AMD 选 `type[2]` → `Managed`，`bytesNoCopy` 包装的主机内存与 managed 的 VRAM 副本不是同一份；GPU 读到的是未同步的 VRAM 内容。实测就是 `MISMATCH`。这是一个**静默错误**，比崩溃更危险。

---

## 3. 引擎现状盘点

### 3.1 设备选择：只选一块，默认 AMD

- [VulkanInstance.cpp#L312-L346](file:///Users/uqland/github/vkop/vulkan/VulkanInstance.cpp#L312-L346)：`vkEnumeratePhysicalDevices` 后按 `DISCRETE_GPU` → `INTEGRATED_GPU` → `VIRTUAL_GPU` → `CPU` 排序，所以 `phydevs[0]` 是 AMD 560X。
- [llm_chat.cpp#L579-L591](file:///Users/uqland/github/vkop/llm/exporter/driver/llm_chat.cpp#L579-L591)：`device_index = 0`，可由 `VKOP_DEVICE` 覆盖；只 `std::make_shared<VulkanDevice>(phydevs[device_index])` 一个逻辑设备。
- 全仓库 `VKOP_DEVICE` 只出现这两行；没有第二个 `VulkanDevice`、没有第二个 `VkDevice`、没有跨设备队列/信号量/传输。

**所以：推理时 Intel 是“已枚举、未实例化”的闲置设备；两块卡从来不同时工作。**

### 3.2 分配策略：设备 buffer 全部 private

- [Tensor.hpp#L1541-L1583](file:///Users/uqland/github/vkop/core/Tensor.hpp#L1541-L1583)：`make_vkbuff` 对 SSBO 固定要求 `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`。
- [VulkanBuffer.cpp#L70-L83](file:///Users/uqland/github/vkop/vulkan/VulkanBuffer.cpp#L70-L83)：VMA `createBuffer`，`usage=AUTO_PREFER_DEVICE`，`requiredFlags=DEVICE_LOCAL`。
- [VMA.cpp#L144-L174](file:///Users/uqland/github/vkop/vulkan/VMA.cpp#L144-L174)：同上；staging 走 `AUTO + HOST_ACCESS_RANDOM + MAPPED`，要求 `HOST_VISIBLE|HOST_COHERENT`。

用同一份 VMA 参数复现实测（`[P]`，`/tmp/vkop_vma_probe.cpp`）：

```
GPU0 AMD Radeon Pro 560X    engine-style alloc: memoryType=0  DEVICE_LOCAL=1 HOST_VISIBLE=0
                            staging-style alloc: memoryType=1  DEVICE_LOCAL=0 HOST_VISIBLE=1 HOST_COHERENT=1
GPU1 Intel(R) UHD Graphics 630  engine-style alloc: memoryType=0  DEVICE_LOCAL=1 HOST_VISIBLE=0
                                staging-style alloc: memoryType=1  DEVICE_LOCAL=1 HOST_VISIBLE=1 HOST_COHERENT=1
```

关键结论：

- **两块卡的设备侧 buffer 都拿到 `type[0]`（private），即使 Intel 有现成的 `type[1]` UMA 共享类型也没用上。**
- Intel 的 staging buffer 落在 `type[1]`（设备本地共享内存）。也就是说 Intel 上的“上传”是：主机 memcpy 到 staging（已经在同一块内存）→ `vkCmdCopyBuffer` 到另一个 buffer（还是同一块内存）→ 提交+等待。**纯属多余的本机拷贝。**
- AMD 的 staging 落在 `type[1]`（系统堆），上传是真正的 staging→VRAM 拷贝。

### 3.3 上传/回读：经典 staging + 阻塞提交

- [Tensor.hpp#L1621-L1761](file:///Users/uqland/github/vkop/core/Tensor.hpp#L1621-L1761)：`copyToGPUBuffer` = 在 staging 池 `allocate` → `memcpy` 主机数据 → `flush` → 记录 `vkCmdCopyBuffer(staging → device)` → `submit` → `wait` → `stpool->reset()`。批量模式只在最后等一次。
- [VulkanStagingBufferPool.cpp#L40-L50](file:///Users/uqland/github/vkop/vulkan/VulkanStagingBufferPool.cpp#L40-L50)：staging 是持久映射的（`VMA_ALLOCATION_CREATE_MAPPED_BIT`），但每批数据仍要重新分配游标、flush、复位。
- 回读 `copyBufferToCPU` 走对称的 device→staging→`memcpy` 路径。
- 大张量（>1 GB staging 上限）走 [Tensor.hpp#L1704](file:///Users/uqland/github/vkop/core/Tensor.hpp#L1704) 的 64 MB 分块上传，每块都 submit+wait。

### 3.4 UMA 检测存在，但只喂给 kernel 选择

[VulkanDevice.cpp#L1805-L1830](file:///Users/uqland/github/vkop/vulkan/VulkanDevice.cpp#L1805-L1830) 用“堆数量==1”判定 UMA，注释明确知道 dGPU 的 BAR 也会呈 `DEVICE_LOCAL|HOST_VISIBLE`、因此不能用该标志判 UMA。判定结果只进入 `device_traits_` 供 MatMul 选路；`is_unified_memory` 没有参与任何内存分配。

### 3.5 `VKOP_NO_COOP=1`

[Matmul.hpp#L308-L320](file:///Users/uqland/github/vkop/ops/Matmul.hpp#L308-L320)：MoltenVK 的 SPIR-V→MSL 只lower 8x8 coopmat，8x16/16x8 建管线失败，所以用环境变量跳过 coop 路径，回落 tiled/ksplit。**与内存/拷贝路径无关**，本调研不需要改它。

---

## 4. “类零拷贝”方案候选（按可行性排序）

### 方案 1（推荐，真零拷贝）：Intel UMA 直接分配（单卡）

- **Vulkan 路径**：Intel `type[1]`（`DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT|HOST_CACHED`，`MTLStorageModeShared`），buffer 常驻映射。
- **在本机是否成立**：**成立，已实测**。映射 OK，主机写入后 GPU `vkCmdCopyBuffer` 读回 MATCH，无需 staging、无需 flush。
- **能省掉**：整个 staging 分配/复位、主机→staging 的 memcpy、`vkCmdCopyBuffer`、submit+wait。对 Intel 来说这是**真正的零拷贝**。
- **风险**：
  - Intel 当前 `budget=1.50 GiB`（recommended working set，不是硬上限），小于 int8 权重 1.72 GB；**超过后不保证常驻，会发生换页/驱逐**，因此只适合小模型、部分张量或小数据通路（单次分配上限 2 GiB 不是瓶颈）。
  - iGPU 算力/带宽低于 AMD 560X（项目历史基线：Intel 墙钟约 173ms/token，AMD 约 87.5ms）。
  - 需要给 VMA 引入“显式目标内存类型/显式 pool”的能力；当前 `VulkanBuffer` 只传 `requiredFlags`。
- **适合先落地的范围**：模型加载期一次性的小张量、每步动态 shape/元数据、输入输出等“主机产生、GPU 消费”的 buffer。

### 方案 2（推荐，零拷贝但带宽受限）：`VK_EXT_external_memory_host` + `type[1]`

- **Vulkan 路径**：`VkImportMemoryHostPointerInfoEXT`（`HOST_ALLOCATION`），memory type 必须是 `type[1]`（coherent 系统内存 / Intel 的 UMA）。
- **是否成立**：**两块卡都成立，已实测 MATCH**；`type[0]` 报 `VK_ERROR_INVALID_EXTERNAL_HANDLE`，`type[2]` 绑定成功但 GPU 读错。
- **能省掉**：全部上传拷贝与 staging，GPU 直接通过 PCIe 读主机内存（AMD）或直接读同一份 UMA 内存（Intel）。
- **实测带宽（2026-11-07，见 4.7）**：AMD 上持续读 **~5.6 GB/s**（PCIe 受限），只有纯 VRAM 读（61.5 GB/s）的 **1/11**；但一次性"上传+消费"比 staging 中转快 **~1.7×**（256MB：48ms vs 81ms）。Intel 上与原生速度相同（26 GB/s），staging 反而慢 8×。
- **风险**：AMD 上每次访问都过 PCIe，带宽远低于 VRAM（权重 `lm_head` 目前 38 GB/s 是纯 VRAM 读）；**绝不能用于权重或热激活**。4 KiB 对齐、生命周期必须覆盖整个 GPU 使用期。
- **适用**：token id、小 shape 元数据、少量每步输入、以及“主机需要频繁读回的小结果”。小块（≤64KB）场景的真正收益是**省掉独立的一次上传 submit**（本机 AMD 每次 submit+wait 固定开销 ~465µs），而不是拷贝本身。

### 方案 3（可行，减少拷贝步骤）：AMD managed 常驻映射上传

- **Vulkan 路径**：AMD `type[2]`（`DEVICE_LOCAL|HOST_VISIBLE|HOST_CACHED`，`MTLStorageModeManaged`），`vkMapMemory` 常驻，写入后 `vkFlushMappedMemoryRanges`。
- **是否成立**：**地址映射与正确性已实测**（flush 后 GPU 读到 MATCH）。
- **能省掉**：staging buffer 的分配/轮转、`vkCmdCopyBuffer`、一次 submit+wait。**不能省掉 PCIe 字节数**（`didModifyRange` 仍然同步影子→VRAM）。
- **风险**：
  - 非 coherent，任何漏 flush 都会读到旧数据。
  - 每个 managed buffer 都有 CPU 影子，粗放地用会让 4 GB VRAM 之外再多一份主机内存开销。
  - 回读触发 `synchronizeResource` blit + 提交，可能比 staging 回读更重；**只用于 H2D 上传，不用于回读**。
  - 需要绕过 VMA 的默认选型（见方案 1 的实现约束）。
- **预期收益**：主要是**时延与 Vulkan 调用开销**（少一次 dispatch/submit/wait），不是带宽。对 1.7 GB 权重一次性加载意义有限；对每步小上传有意义。**实测补充（4.7）**：若每步都重写+flush，managed 的 e2e 与 host-import 持平（256MB 均 ~5.6–5.8 GB/s）——Metal 在 GPU 使用前仍要把影子同步过 PCIe，带宽没省；flush 的 CPU 计时（256MB 仅 2.6µs）是下界，重复轮数据未变时 `didModifyRange` 可能被驱动短路。

### 方案 4（可行但收益小）：持久映射 staging ring

- 引擎的 staging 已经是 `VMA_ALLOCATION_CREATE_MAPPED_BIT` 持久映射，且按 16 MB→1 GB 增长。进一步做成多槽 ring、避免 `reset()` 阻塞，只能省游标管理和少量等待，**不改变拷贝次数**。优先级最低。

### 方案 5（不可行）：`VK_KHR_device_group` / peer memory

- 实测 2 个 group、每组 1 设备、`subsetAllocation=0`；没有 peer memory。MoltenVK 也没有实现跨 `MTLDevice` 的 peer 访问。**直接排除。**

### 方案 6（不可行且危险）：跨设备 `VK_EXT_external_memory_metal` `MTLBuffer`

- 能力查询说 `importable|exportable`，`vkAllocateMemory` 导入也返回成功，但：
  - Intel→AMD：AMD Metal 驱动断言 `cmdBufInfo->device == rsrc->_res.device`，**进程 abort**。
  - AMD→Intel：不崩，但 GPU 读到全 0（**静默错误**）。
- 根因：Metal buffer 绑定创建它的 `MTLDevice`，MoltenVK 不校验归属。**结论：不能用于跨设备，且必须防呆。**
- `MTLTEXTURE`/IOSurface 只对 image 有意义，本引擎张量是 SSBO buffer，`compatibleHandleTypes=0`；且没有 dma_buf/OPAQUE_FD。

### 4.7 实测带宽对比：方案二 vs staging 中转（2026-11-07，[P]）

探针：`tools/perf/zero_copy_bw_probe/`（构建/运行见该目录 README；原始日志 `tools/perf/logs/zero_copy_bw_probe_20261107.log`）。四种路径用同一个 read-sum kernel（真实读满每个字节，20 组校验和全部 MATCH）：

- `vram`：数据预先上传到 private VRAM，kernel 直读（基线）
- `staging`：每轮 CPU memcpy → 持久映射 staging(type1) → `vkCmdCopyBuffer` → kernel 读（引擎现状）
- `host-import`：方案二，页对齐主机内存 import 到 type1，kernel 直读
- `managed`：方案三，type2 常驻映射，每轮 flush → kernel 读

口径：尺寸 4KB/64KB/1MB/16MB/256MB；每轮内四模式交错（A/B 同块），3 轮取 min；reps=200（≤1MB）/60（16MB）/20（256MB）。计时为**墙钟**（record+submit+wait）：本机 MoltenVK/AMD 栈上 timestamp query 永不 available（`vkGetQueryPoolResults` 恒 `VK_NOT_READY`），故不用 GPU 时间戳；大块下提交开销占比 <10%。

AMD Radeon Pro 560X（e2e GB/s；小块为 µs/op）：

| 块大小 | vram 基线 | staging 中转 | host-import（方案二） | managed（方案三） |
|---|---|---|---|---|
| 4 KB | 464 µs | 467 µs | 466 µs | 470 µs |
| 64 KB | 467 µs | 493 µs | 472 µs | 473 µs |
| 1 MB | 2.24 | 1.33 | **1.77** | 1.72 |
| 16 MB | 35.5 | 2.94 | **5.10** | 5.45 |
| 256 MB | 61.5 | 3.32 | **5.58** | 5.81 |

Intel UHD 630：

| 块大小 | vram 基线 | staging 中转 | host-import（方案二） | managed（方案三） |
|---|---|---|---|---|
| 4 KB | 161 µs | 201 µs | 161 µs | 159 µs |
| 64 KB | 160 µs | 208 µs | 163 µs | 162 µs |
| 1 MB | 6.08 | 2.89 | **6.04** | 6.07 |
| 16 MB | 21.5 | 3.59 | **22.2** | 22.1 |
| 256 MB | 26.0 | 3.25 | **26.3** | 26.6 |

解读：

1. **AMD 一次性“上传+消费”：方案二比 staging 快 ~1.7×**（256MB：48ms vs 81ms）。staging 的 81ms 分解：CPU memcpy ~36ms（主机 memcpy 仅 ~7 GB/s）+ GPU copy ~40ms（~6.6 GB/s）+ VRAM 读 ~4ms；方案二只剩一次 PCIe 读 ~5.6 GB/s。
2. **AMD 反复读同一份数据：staging 完胜**——首次 copy 后数据驻留 VRAM，61.5 GB/s vs 方案二每次都过 PCIe 的 5.6 GB/s，**11× 差距**。权重/热激活必须走 staging 一次性上传。
3. **Intel：方案二 ≈ 原生速度**（UMA 同一块内存），staging 是多余的本机拷贝，慢 2–8×，印证 3.2 节结论。
4. **小块（≤64KB）各路径差异只有几 µs**，被提交固定开销淹没：**AMD 每次 submit+wait ~465µs，Intel ~160µs**。方案二对小数据的真正收益是省掉独立的一次上传 submit（本探针每模式都 dispatch 一次，测不到这部分）；同时说明引擎每步小上传都在付这 465µs，值得单独优化。
5. **managed 每步重写场景没有带宽收益**：e2e 与 host-import 持平，Metal 在 GPU 使用前仍把影子同步过 PCIe（见方案 3 的实测补充）。

### 方案总表

| 方案 | 路径 | 跨设备？ | 真零拷贝？ | 省拷贝 | 省 PCIe | 本机成立 | 实测 e2e 256MB（AMD / Intel） | 风险 |
|---|---|---|---|---|---|---|---|---|
| 1 Intel UMA | type1 Shared | 否（仅 Intel） | **是** | 全部 | 是（同内存） | 实测成立 | —（未单测，参照 host-import ≈26 GB/s） | Intel 预算 1.5 GiB，算力低 |
| 2 主机指针 import | `external_memory_host` | 否（单卡内） | **是** | 全部上传 | 否（PCIe 读） | 实测成立 | 5.58 / 26.3 GB/s | 带宽差，仅小块或一次性消费 |
| 3 AMD managed 映射 | type2 Managed | 否 | 否 | staging+copy+submit | 否 | 实测成立 | 5.81 / 26.6 GB/s（每步重写时） | 非 coherent，回读贵 |
| 4 staging ring | 持久映射 | 否 | 否 | 少量调用开销 | 否 | 现有基础 | ≈staging：3.32 / 3.25 GB/s | 收益极小 |
| 5 device group | peer memory | 是 | — | — | — | **不成立** | — | 无 |
| 6 Metal 句柄 | MTLBUFFER | 是 | — | — | — | **不成立** | — | 崩溃/静默错误 |

---

## 5. 如果继续做，建议的最小落地顺序

1. **先把内存分配策略参数化**（不改行为）：给 `VulkanBuffer` 增加显式的“目标类型/用途”枚举（`Private` / `HostVisibleManaged` / `HostImported`），VMA 侧支持显式 memory type 或专用 pool。现在是 `AUTO_PREFER_DEVICE` 隐式选中 `type[0]`，无法表达方案 1/3。
2. **Intel 单卡真零拷贝（方案 1）**：只对 `host-written` 类张量启用；用 `VKOP_DEVICE=1` 跑 e2e，验证逐 token 输出与 AMD 基线一致，并测 upload/readback 时延。
3. **主机指针 import（方案 2）**：先在“小输入/小结果”上用；严格校验 4 KiB 对齐与生命周期；类型必须锁 `type[1]`，`type[0]` 直接拒绝，`type[2]` 禁止（实测坏）。
4. **AMD managed 上传（方案 3）**：仅用于 H2D，不允许回读；所有 map 生命周期内 `flush` 必须覆盖写入范围。用 normalize 比值（不是绝对墙钟）判断是否有收益。
5. **不做**：device group / peer memory / 跨设备 MTLBuffer。若未来要防呆，在 `VulkanDevice` 初始化时检测“同一 `VkInstance` 下多个 `MTLDevice`”并拒绝启用 `VK_EXT_external_memory_metal` 的跨设备导入。

**验收口径**（沿用项目现有约定）：所有改动要过 17 项 MatMul 单测 + 12 token e2e 逐 token 一致；性能测量 reps≥200、A/B 同块交错、min-of-3，不能取单次或平均。

---

## 6. 复现与原始产物

```bash
# 1) 完整能力
vulkaninfo --text > /tmp/vk_full.txt
vulkaninfo --summary

# 2) 能力探针（device group / external props / map / import / 跨设备）
clang++ -std=c++17 -I/usr/local/include /tmp/vkop_zero_copy_probe.cpp \
  -L/usr/local/lib -Wl,-rpath,/usr/local/lib -lvulkan -o /tmp/vkop_zero_copy_probe
/tmp/vkop_zero_copy_probe              # Intel→AMD 会在最后触发 AMD 驱动断言
PROBE_REVERSE=1 /tmp/vkop_zero_copy_probe   # AMD→Intel，静默 MISMATCH

# 3) VMA 选型复现
clang++ -std=c++17 -Wno-nullability-completeness -I/usr/local/include \
  /tmp/vkop_vma_probe.cpp -L/usr/local/lib -Wl,-rpath,/usr/local/lib -lvulkan \
  -o /tmp/vkop_vma_probe && /tmp/vkop_vma_probe

# 4) 带宽对比探针（4.7 节，已固化进仓库；原 /tmp 探针在 2026-11-07 重启后丢失）
# 构建步骤见 tools/perf/zero_copy_bw_probe/README.md
tools/perf/zero_copy_bw_probe/probe --rounds 3
```

原始输出：

- `/tmp/vk_full.txt`（vulkaninfo 全文，约 2900 行）
- `/tmp/vk_probe_out.txt`（Intel→AMD，含断言）
- `/tmp/vk_probe_reverse2.txt`（AMD→Intel，含 host import 结果）
- `/tmp/vma_probe_out.txt`（VMA 选型）
- `tools/perf/logs/zero_copy_bw_probe_20261107.log`（4.7 节带宽对比，重启后重测）

注意：`/tmp` 下的能力探针源码（`vkop_zero_copy_probe.cpp`、`vkop_vma_probe.cpp`）与输出在 2026-11-07 机器重启后已丢失，Trae 历史中无备份；能力结论以本报告记录为准。带宽探针已重写并固化在 `tools/perf/zero_copy_bw_probe/`。

---

## 7. 证据分级声明

| 结论 | 来源 |
|---|---|
| 两块设备、类型、堆大小、budget、扩展、limits | **[V]** vulkaninfo 原始输出 |
| 2 个 group / 每 group 1 设备 / subsetAllocation=0 | **[V]** + **[P]** 双重确认 |
| `type[1]`/`type[2]` 可映射、flush 后 GPU 可见、host import 可用 | **[P]** 自写探针 |
| 跨设备 MTLBuffer 崩溃/静默读 0 | **[P]** 自写探针（崩溃日志已保留） |
| 引擎只选一块卡、全部 DEVICE_LOCAL、staging 上传 | **[C]** 代码引用 |
| 两块卡实际选到 type0、Intel staging 选到 type1 | **[P]** VMA 复现 |
| MoltenVK storage mode 映射与 flush/invalidate 语义 | **[S]** MoltenVK v1.4.2 源码 |
| staging / host-import / managed / VRAM 的 e2e 带宽与提交固定开销（4.7 节） | **[P]** 仓库内探针 `tools/perf/zero_copy_bw_probe/`，2026-11-07 实测 |
| `didModifyRange` 内部是否 PCIe DMA、managed 同步的具体机制 | **[I]** 文档推断；e2e 行为（每步重写时 managed ≈ host-import 带宽）已实测，机制未直接观测 |
