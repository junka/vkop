// Copyright 2025 @junka
#include "vulkan/VulkanStagingBufferPool.hpp"
#include "ops/Operator.hpp"

namespace vkop {
#define ALIGN_UP(x, y) (((x) + ((y)-1)) & ~((y)-1))
namespace {
constexpr int kStagingBufferSize =
    1024 * 1024 * 16; // 32M,  greater than 1024 * 1024 * 3 for one image
}

VulkanStagingBufferPool::VulkanStagingBufferPool(
    std::shared_ptr<VulkanDevice> &vdev)
    : m_vdev_(vdev) {
    m_buffer_ = std::make_unique<VulkanBuffer>(
        m_vdev_, kStagingBufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (!m_buffer_) {
        throw std::runtime_error("Failed to create staging buffer");
    }
    mapped_memory_ = m_buffer_->getMappedMemory();
    assert(mapped_memory_ != nullptr);
    m_poolSize_ = kStagingBufferSize;
}

VulkanStagingBufferPool::~VulkanStagingBufferPool() {
    m_buffer_->unmapMemory();
}

std::optional<StagingAllocation>
VulkanStagingBufferPool::allocate(size_t size, size_t alignment) {
    assert(size != 0);

    VkDeviceSize aligned_write = ALIGN_UP(m_writePos_, alignment);
    VkDeviceSize next_write = aligned_write + size;

    // Check if allocation would go beyond buffer end
    if (next_write > m_poolSize_) {
        // Try wrapping to the beginning if there's space
        VkDeviceSize wrapped_start = ALIGN_UP(0, alignment);
        VkDeviceSize wrapped_end = wrapped_start + size;

        // Only wrap if:
        // 1. It fits at the beginning
        // 2. It doesn't overlap with existing unread data
        if (wrapped_end <= m_readPos_ || m_readPos_ == m_writePos_) {
            aligned_write = wrapped_start;
            next_write = wrapped_end;
        } else {
            // Cannot wrap due to overlapping with unread data
            return std::nullopt;
        }
    }

    VkDeviceSize physical_offset = aligned_write % m_poolSize_;

    StagingAllocation alloc;
    alloc.ptr = static_cast<char *>(mapped_memory_) + physical_offset;
    alloc.offset = physical_offset;
    alloc.size = size;
    m_writePos_ = next_write;
    // printf("alloc size %ld, offset %ld\n", size, physical_offset);
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