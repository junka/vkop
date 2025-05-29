// Copyright 2025 @junka
#ifndef SRC_VULKANCOMMANDPOOL_HPP_
#define SRC_VULKANCOMMANDPOOL_HPP_

#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>

#include "VulkanCommandBuffer.hpp"

namespace vkop {
class VulkanCommandPool {
  public:
    VulkanCommandPool(VkDevice device, uint32_t queueFamilyIndex);
    ~VulkanCommandPool();

    void reset(VkCommandPoolResetFlags flags = 0);

    VkCommandPool getCommandPool() const { return m_commandPool_; }

  private:
    VkDevice m_device_;
    VkCommandPool m_commandPool_ = VK_NULL_HANDLE;
    std::vector<VulkanCommandBuffer> buffers_;

    void createCommandPool(uint32_t queueFamilyIndex);
    void destroyCommandPool();
};

} // namespace vkop
#endif // SRC_VULKANCOMMANDPOOL_HPP_
