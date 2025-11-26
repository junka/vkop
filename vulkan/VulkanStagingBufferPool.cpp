// Copyright 2025 @junka
#include "vulkan/VulkanStagingBufferPool.hpp"
#include "ops/Operator.hpp"

namespace vkop {
#define ALIGN_UP(x, y) (((x) + ((y)-1)) & ~((y)-1))
namespace {
constexpr int kStagingBufferSize =
    1024 * 1024 * 8; // 8M,  greater than 1024 * 1024 * 3 for one image
}

VulkanStagingBufferPool::VulkanStagingBufferPool(
    std::shared_ptr<VulkanDevice> &vdev)
    : m_vdev_(vdev) {
    m_buffer_ = std::make_unique<VulkanBuffer>(
        m_vdev_, kStagingBufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (!m_buffer_) {
        throw std::runtime_error("Failed to create staging buffer");
    }
    mapped_memory_ = m_buffer_->getMappedMemory();
    m_poolSize_ = kStagingBufferSize;
}

VulkanStagingBufferPool::~VulkanStagingBufferPool() {
    m_buffer_->unmapMemory();
}

std::optional<StagingAllocation>
VulkanStagingBufferPool::allocate(size_t size, size_t alignment) {

    assert(size != 0);

    // 对齐 writePos
    VkDeviceSize aligned_write = ALIGN_UP(m_writePos_, alignment);
    VkDeviceSize next_write = aligned_write + size;

    // 检查是否撞上 readPos（考虑 wrap-around）
    // 逻辑：可用空间 = (m_readPos + m_poolSize) - m_writePos （在环形空间中）
    // 但简化处理：如果 pending 提交太多，等 GPU 完成
    // printf("writePos %lu, readPos %lu, poolSize %lu, aligned_write %lu, size
    // "
    //        "%lu, next_write %lu\n",
    //        m_writePos_, m_readPos_, m_poolSize_, aligned_write, size,
    //        next_write);
    if (next_write - m_readPos_ > m_poolSize_) {
        // 池满，无法分配
        return std::nullopt;
    }

    VkDeviceSize physical_offset = aligned_write % m_poolSize_;

    StagingAllocation alloc;
    alloc.ptr = static_cast<char *>(mapped_memory_) + physical_offset;
    alloc.offset = physical_offset;
    alloc.size = size;
    m_writePos_ = next_write;
    return alloc;
}

void VulkanStagingBufferPool::markSubmit(uint64_t timelineValue) {
    m_submittedRanges_.push_back({timelineValue, m_writePos_});
}

void VulkanStagingBufferPool::reclaimCompleted(
    uint64_t completedTimelineValue) {
    while (!m_submittedRanges_.empty() &&
           m_submittedRanges_.front().timelineValue <= completedTimelineValue) {
        m_readPos_ = m_submittedRanges_.front().endPos;
        m_submittedRanges_.pop_front();
    }
}

} // namespace vkop