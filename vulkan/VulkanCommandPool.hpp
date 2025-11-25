// Copyright 2025 @junka
#ifndef SRC_VULKANCOMMANDPOOL_HPP_
#define SRC_VULKANCOMMANDPOOL_HPP_

#include <vector>
#include <vulkan/vulkan.hpp>

#include "vulkan/VulkanCommandBuffer.hpp"
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

    uint64_t getNextSubmitValue() { return ++m_timelineValue_; }
    VkSemaphore getSemaphore() const { return m_semaphore_; }
    VkFence getFence() const { return m_fence_; }

    std::shared_ptr<VulkanStagingBufferPool> getStagingBufferPool() const {
        return stagingbuffer_pool_;
    }

  private:
    // VkDevice m_device_;
    std::shared_ptr<VulkanDevice> m_vdev_;
    VkCommandPool m_commandPool_ = VK_NULL_HANDLE;
    std::vector<VulkanCommandBuffer> buffers_;
    VkSemaphore m_semaphore_ = VK_NULL_HANDLE;
    VkFence m_fence_ = VK_NULL_HANDLE;
    uint64_t m_timelineValue_ = 0;
    std::shared_ptr<VulkanStagingBufferPool> stagingbuffer_pool_;

    void createCommandPool(uint32_t queueFamilyIndex);

    void createTimelineSemaphore();
    void createFence();
};

} // namespace vkop
#endif // SRC_VULKANCOMMANDPOOL_HPP_
