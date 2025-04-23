#include "VulkanLib.hpp"
#include "VulkanCommandPool.hpp"
#include <stdexcept>

namespace vkop {

VulkanCommandPool::VulkanCommandPool(VkDevice device, uint32_t queueFamilyIndex)
    : m_device(device), m_commandPool(VK_NULL_HANDLE) {
    createCommandPool(queueFamilyIndex);
}

VulkanCommandPool::~VulkanCommandPool() {
    destroyCommandPool();
}

void VulkanCommandPool::createCommandPool(uint32_t queueFamilyIndex) {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamilyIndex;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_commandPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool");
    }
}

void VulkanCommandPool::destroyCommandPool() {
    if (m_commandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(m_device, m_commandPool, nullptr);
        m_commandPool = VK_NULL_HANDLE;
    }
}

void VulkanCommandPool::reset(VkCommandPoolResetFlags flags) {
    if (vkResetCommandPool(m_device, m_commandPool, flags) != VK_SUCCESS) {
        throw std::runtime_error("Failed to reset command pool");
    }
}

}