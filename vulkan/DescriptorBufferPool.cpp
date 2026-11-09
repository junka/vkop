// Copyright 2026 @junka
#include "vulkan/DescriptorBufferPool.hpp"

#include <algorithm>
#include <stdexcept>

#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanDevice.hpp"

namespace vkop {

namespace {
VkDeviceSize round_up(VkDeviceSize value, VkDeviceSize alignment) {
    if (alignment <= 1) {
        return value;
    }
    return (value + alignment - 1) / alignment * alignment;
}
} // namespace

DescriptorBufferPool::DescriptorBufferPool(
    const std::shared_ptr<VulkanDevice> &device, VkDeviceSize alignment)
    : device_(device), alignment_(std::max<VkDeviceSize>(1, alignment)) {}

DescriptorBufferPool::~DescriptorBufferPool() = default;

std::shared_ptr<DescriptorBufferPool>
DescriptorBufferPool::create(const std::shared_ptr<VulkanDevice> &device,
                             const DescriptorBufferInfo *info,
                             VkMemoryPropertyFlags memory_flags,
                             VkDeviceSize capacity) {
    if (!device || !info) {
        return nullptr;
    }
    const VkDeviceSize alignment = std::max<VkDeviceSize>(
        1, info->offsetAlignment); // descriptorBufferOffsetAlignment
    auto pool = std::shared_ptr<DescriptorBufferPool>(
        new DescriptorBufferPool(device, alignment));
    try {
        // VulkanBuffer's ctor takes a non-const shared_ptr reference.
        std::shared_ptr<VulkanDevice> dev = device;
        pool->capacity_ = round_up(capacity, alignment);
        pool->storage_ = std::make_unique<VulkanBuffer>(
            dev, pool->capacity_,
            VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT, memory_flags);
        pool->map_ = pool->storage_->getMappedMemory();
        pool->address_ = pool->storage_->getDeviceAddress();
    } catch (const std::exception &) {
        return nullptr;
    }
    if (!pool->map_ || pool->address_ == 0) {
        return nullptr;
    }
    return pool;
}

bool DescriptorBufferPool::alloc(VkDeviceSize size, VkDeviceSize alignment,
                                 VkDeviceSize &offset) {
    std::lock_guard<std::mutex> lock(mutex_);
    const VkDeviceSize align = std::max(alignment_, alignment);
    const VkDeviceSize need = round_up(std::max<VkDeviceSize>(size, 1), align);

    // First fit. Every block was aligned when the pool was built and sizes are
    // rounded to `align`, so a block's offset is already legal.
    for (auto it = free_blocks_.begin(); it != free_blocks_.end(); ++it) {
        if (it->second < need) {
            continue;
        }
        offset = it->first;
        if (it->second == need) {
            free_blocks_.erase(it);
        } else {
            const VkDeviceSize left = it->second - need;
            const VkDeviceSize start = it->first + need;
            free_blocks_.erase(it);
            free_blocks_[start] = left;
        }
        return true;
    }
    if (bump_ + need > capacity_) {
        return false;
    }
    offset = bump_;
    bump_ += need;
    return true;
}

void DescriptorBufferPool::free(VkDeviceSize offset, VkDeviceSize size) {
    if (size == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const VkDeviceSize need =
        round_up(std::max<VkDeviceSize>(size, 1), alignment_);
    free_blocks_[offset] = need;
    // Coalesce with the next block.
    auto it = free_blocks_.find(offset);
    auto next = std::next(it);
    if (next != free_blocks_.end() && it->first + it->second == next->first) {
        it->second += next->second;
        free_blocks_.erase(next);
    }
    // Coalesce with the previous block.
    if (it != free_blocks_.begin()) {
        auto prev = std::prev(it);
        if (prev->first + prev->second == it->first) {
            prev->second += it->second;
            free_blocks_.erase(it);
        }
    }
}

VkBuffer DescriptorBufferPool::buffer() const {
    return storage_ ? storage_->getBuffer() : VK_NULL_HANDLE;
}

} // namespace vkop
