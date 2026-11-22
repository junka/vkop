// Copyright 2025 @junka
#include <cstdint>
#include <iostream>
#ifdef USE_VMA

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"

#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include "vulkan/vk_mem_alloc.h"

#pragma GCC diagnostic pop

#include "vulkan/VMA.hpp"

#include <atomic>
#include <stdexcept>

namespace vkop {

namespace {
// RenderDoc reconstructs mapped-memory contents only from map/unmap/flush
// events. VMA's own flush/invalidate are no-ops on HOST_COHERENT types, so
// under a capture session (VKOP_RDOC) or VKOP_STRICT_FLUSH the explicit
// range calls below are what make direct host writes visible to replay.
bool strict_flush_enabled() {
    static const bool on = std::getenv("VKOP_RDOC") != nullptr ||
                           std::getenv("VKOP_STRICT_FLUSH") != nullptr;
    return on;
}
std::atomic<VkDevice> s_vma_device{VK_NULL_HANDLE};
} // namespace

VMA::VMA(VkPhysicalDevice physicalDevice, VkDevice device,
         bool useBufferDeviceAddress, bool useMemoryBudget) {
    VmaVulkanFunctions vulkan_functions = {};
    vulkan_functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vulkan_functions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo allocator_info = {};
    allocator_info.physicalDevice = physicalDevice;
    allocator_info.device = device;
    allocator_info.instance = VulkanInstance::getVulkanInstance().getInstance();
    allocator_info.vulkanApiVersion = VK_API_VERSION_1_2;
    // Dedicated allocation / bind_memory2 are core in 1.1+. DEVICE_COHERENT
    // only lets VMA consider the AMD device-coherent types, which exist only
    // when VulkanDevice enabled VK_AMD_device_coherent_memory.
    allocator_info.flags = VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT |
                           VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT |
                           VMA_ALLOCATOR_CREATE_AMD_DEVICE_COHERENT_MEMORY_BIT;
    // Only legal once VK_EXT_memory_budget is enabled on the device.
    if (useMemoryBudget) {
        allocator_info.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
    }
    if (useBufferDeviceAddress) {
        allocator_info.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    }
    // VMA's default block on a large heap is 256 MB: a partly used block (plus
    // the one empty block VMA keeps cached) is resident slack. Inference
    // tensors are either tiny (sub-allocated well by 64 MB blocks) or big
    // (dedicated, see kDedicatedThreshold), so a smaller block costs nothing.
    allocator_info.preferredLargeHeapBlockSize = kBlockSize;
    allocator_info.pVulkanFunctions = &vulkan_functions;

    if (vmaCreateAllocator(&allocator_info, &allocator_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create VMA allocator");
    }
    s_vma_device.store(device);
}

VMA::~VMA() { vmaDestroyAllocator(allocator_); }

VkResult VMA::createBuffer(VkBufferCreateInfo *bufferInfo,
                           struct VmaBuffer *buf, bool local,
                           VkMemoryPropertyFlags requiredFlags) {
    VmaAllocationCreateInfo alloc_info = {};
    // The caller's memory properties are a REQUIREMENT, not a preference:
    // HOST_VISIBLE|HOST_COHERENT buffers are read and written directly
    // through the mapped pointer (scratch headers, staging), so a type
    // without coherence silently loses host<->device stores.
    alloc_info.requiredFlags = requiredFlags;
    // A big buffer gets its own VkDeviceMemory: it would otherwise pin a
    // whole block that stays resident after it is freed, and a weight or KV
    // cache of this size gains nothing from sub-allocation.
    if (bufferInfo->size >= kDedicatedThreshold) {
        alloc_info.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    }
    if (local) {
        alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        // DEVICE_LOCAL|HOST_VISIBLE (UMA weights written in place): VMA
        // must hand back a persistently mapped allocation.
        if (requiredFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
            alloc_info.flags |=
                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
        }
    } else {
        // Staging pool serves BOTH directions: H2D upload writes and D2H
        // readback reads (memcpy out of the mapped pointer in
        // Tensor::copyBufferToCPU). HOST_ACCESS_RANDOM makes VMA prefer
        // HOST_CACHED system memory. The previous SEQUENTIAL_WRITE +
        // ALLOW_TRANSFER_INSTEAD combo steered VMA to uncached,
        // write-combined DEVICE_LOCAL memory (BAR on dGPUs) — good for pure
        // writes, disastrous for every readback memcpy. ALLOW_TRANSFER_INSTEAD
        // was also invalid here: VMA may hand back non-host-visible memory,
        // which this pool maps and dereferences directly.
        alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
        alloc_info.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }
    buf->allocator = allocator_;
    return vmaCreateBuffer(allocator_, bufferInfo, &alloc_info, &buf->buffer,
                           &buf->allocation, nullptr);
}

VkResult VMA::createImage(VkImageCreateInfo *imageInfo, struct VmaImage *img) {
    VmaAllocationCreateInfo alloc_info = {};
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    // No explicit dedicated threshold: VMA already gives an image above half
    // a block (32 MB) or one the driver prefers dedicated its own memory.
    img->allocator = allocator_;
    return vmaCreateImage(allocator_, imageInfo, &alloc_info, &img->image,
                          &img->allocation, nullptr);
}

uint64_t VMA::getAllocatedSize(const struct VmaImage *img) {
    VmaAllocationInfo info;
    vmaGetAllocationInfo(img->allocator, img->allocation, &info);
    return static_cast<uint64_t>(info.size);
}

void *VMA::getMappedMemory(struct VmaBuffer *buf) {
    VmaAllocationInfo info;
    vmaGetAllocationInfo(buf->allocator, buf->allocation, &info);
    return info.pMappedData;
}

void VMA::flushAllocation(struct VmaBuffer *buf, VkDeviceSize offset,
                          VkDeviceSize size) {
    // vmaFlushAllocation is a no-op on HOST_COHERENT memory types; the
    // explicit call only costs a branch there.
    vmaFlushAllocation(buf->allocator, buf->allocation, offset, size);
    if (!strict_flush_enabled())
        return;
    VmaAllocationInfo info{};
    vmaGetAllocationInfo(buf->allocator, buf->allocation, &info);
    if (info.deviceMemory == VK_NULL_HANDLE)
        return;
    VkMappedMemoryRange range = {};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = info.deviceMemory;
    range.offset = info.offset + offset;
    range.size = size;
    vkFlushMappedMemoryRanges(s_vma_device.load(), 1, &range);
}

void VMA::invalidateAllocation(struct VmaBuffer *buf, VkDeviceSize offset,
                               VkDeviceSize size) {
    vmaInvalidateAllocation(buf->allocator, buf->allocation, offset, size);
    if (!strict_flush_enabled())
        return;
    VmaAllocationInfo info{};
    vmaGetAllocationInfo(buf->allocator, buf->allocation, &info);
    if (info.deviceMemory == VK_NULL_HANDLE)
        return;
    VkMappedMemoryRange range = {};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = info.deviceMemory;
    range.offset = info.offset + offset;
    range.size = size;
    vkInvalidateMappedMemoryRanges(s_vma_device.load(), 1, &range);
}

void VMA::destroyBuffer(struct VmaBuffer *buf) {
    vmaDestroyBuffer(allocator_, buf->buffer, buf->allocation);
}

void VMA::destroyImage(struct VmaImage *img) {
    vmaDestroyImage(allocator_, img->image, img->allocation);
}

void VMA::getStats() {
    // vmaGetHeapBudgets 只填实际存在的 heap 个数，尾部元素保持调用前的值——
    // 不清零就会打印出一堆越界的垃圾统计。
    VmaBudget budgets[VK_MAX_MEMORY_HEAPS] = {};
    vmaGetHeapBudgets(allocator_, budgets);
    char *stats_string = nullptr;
    vmaBuildStatsString(allocator_, &stats_string, 1U);
    printf("%s\n", stats_string);
    for (auto &budget : budgets) {
        if (budget.budget == 0 && budget.usage == 0 &&
            budget.statistics.allocationCount == 0)
            continue; // 不存在（或已空）的 heap 不必刷屏
        std::cout << "heap currently has " << budget.statistics.allocationCount
                  << " allocations taking " << budget.statistics.allocationBytes
                  << " B," << std::endl;
        std::cout << "allocated out of " << budget.statistics.blockCount
                  << " Vulkan device memory blocks taking "
                  << budget.statistics.blockBytes << " B," << std::endl;
        std::cout << "Vulkan reports total usage " << budget.usage
                  << " B with budget " << budget.budget << " B." << std::endl;
    }
}

} // namespace vkop
#endif // USE_VMA
