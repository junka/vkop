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

    std::shared_ptr<VulkanDevice> getVulkanDevice() const { return m_vdev_; }
    VkCommandPool getCommandPool(int idx = 0) const {
        return m_commandPool_[idx % m_commandPool_.size()];
    }

    uint64_t getNextSubmitValue() {
        return m_timelineValue_.fetch_add(1, std::memory_order_relaxed);
    }

    std::shared_ptr<VulkanStagingBufferPool> getStagingBufferPool() const {
        return stagingbuffer_pool_;
    }

  private:
    std::shared_ptr<VulkanDevice> m_vdev_;
    std::vector<VkCommandPool> m_commandPool_;
    std::atomic<uint64_t> m_timelineValue_{1};
    std::shared_ptr<VulkanStagingBufferPool> stagingbuffer_pool_;

    void createCommandPool(uint32_t queueFamilyIndex);
};

} // namespace vkop
#endif // SRC_VULKANCOMMANDPOOL_HPP_
