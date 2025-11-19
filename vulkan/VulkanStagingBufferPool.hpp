// Copyright 2025 @junka
#ifndef VULKAN_VULKANSTAGINGBUFFERPOOL_HPP_
#define VULKAN_VULKANSTAGINGBUFFERPOOL_HPP_

#include <deque>
#include <memory>
#include <optional>

#include <vulkan/vulkan.hpp>

#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanDevice.hpp"

namespace vkop {
struct StagingAllocation {
    void *ptr = nullptr;
    VkDeviceSize offset = 0;
    size_t size = 0;
};

class VulkanStagingBufferPool {
  public:
    explicit VulkanStagingBufferPool(std::shared_ptr<VulkanDevice> &vdev);

    ~VulkanStagingBufferPool() = default;

    std::optional<StagingAllocation> allocate(size_t size,
                                              size_t alignment = 256);

    void markSubmit(uint64_t timelineValue);
    void reclaimCompleted(uint64_t completedTimelineValue);

    VkBuffer getBuffer() const { return m_buffer_->getBuffer(); }

  private:
    std::shared_ptr<VulkanDevice> m_vdev_;
    std::unique_ptr<VulkanBuffer> m_buffer_;

    void *mapped_memory_ = nullptr;

    struct SubmittedRange {
        uint64_t timelineValue;
        VkDeviceSize endPos;
    };

    VkDeviceSize m_writePos_ = 0;
    VkDeviceSize m_readPos_ = 0;
    VkDeviceSize m_poolSize_;

    std::deque<SubmittedRange> m_submittedRanges_;
};

} // namespace vkop

#endif // VULKAN_VULKANSTAGINGBUFFERPOOL_HPP_
