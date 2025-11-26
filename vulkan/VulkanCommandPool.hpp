// Copyright 2025 @junka
#ifndef SRC_VULKANCOMMANDPOOL_HPP_
#define SRC_VULKANCOMMANDPOOL_HPP_

#include <atomic>
#include <vector>
#include <vulkan/vulkan.hpp>

#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanStagingBufferPool.hpp"

namespace vkop {
class VulkanCommandPool {
  public:
    explicit VulkanCommandPool(std::shared_ptr<VulkanDevice> &vdev);
    ~VulkanCommandPool();

    void reset(VkCommandPoolResetFlags flags = 0);

    VkCommandPool getCommandPool() const { return m_commandPool_; }

    uint64_t getCompletedTimelineValue();

    uint64_t getNextSubmitValue() {
        return m_timelineValue_.fetch_add(1, std::memory_order_relaxed);
    }
    VkSemaphore getSemaphore() const { return m_semaphore_; }

    std::shared_ptr<VulkanStagingBufferPool> getStagingBufferPool() const {
        return stagingbuffer_pool_;
    }

  private:
    std::shared_ptr<VulkanDevice> m_vdev_;
    VkCommandPool m_commandPool_ = VK_NULL_HANDLE;
    VkSemaphore m_semaphore_ = VK_NULL_HANDLE;
    std::atomic<uint64_t> m_timelineValue_{0};
    std::shared_ptr<VulkanStagingBufferPool> stagingbuffer_pool_;

    void createCommandPool(uint32_t queueFamilyIndex);

    void createTimelineSemaphore();
};

} // namespace vkop
#endif // SRC_VULKANCOMMANDPOOL_HPP_
