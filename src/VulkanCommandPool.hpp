#ifndef VULKAN_COMMAND_POOL_HPP
#define VULKAN_COMMAND_POOL_HPP

#include <vulkan/vulkan.hpp>
#include <vector>

#include "VulkanCommandBuffer.hpp"

namespace vkop {
class VulkanCommandPool {
public:
    VulkanCommandPool(VkDevice device, uint32_t queueFamilyIndex);
    ~VulkanCommandPool();

    void reset(VkCommandPoolResetFlags flags = 0);

    VkCommandPool getCommandPool() const { return m_commandPool; }

private:
    VkDevice m_device;
    VkCommandPool m_commandPool;
    std::vector<VulkanCommandBuffer> buffers;

    void createCommandPool(uint32_t queueFamilyIndex);
    void destroyCommandPool();
};

}
#endif // VULKAN_COMMAND_POOL_HPP