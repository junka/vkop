// Copyright 2025 @junka
#include "vulkan/VulkanCommandPool.hpp"
#include "vulkan/VulkanLib.hpp"
#include <stdexcept>

namespace vkop {

VulkanCommandPool::VulkanCommandPool(VkDevice device, uint32_t queueFamilyIndex)
    : m_device_(device) {
    createCommandPool(queueFamilyIndex);
}

VulkanCommandPool::~VulkanCommandPool() {
    if (m_commandPool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(m_device_, m_commandPool_, nullptr);
        m_commandPool_ = VK_NULL_HANDLE;
    }
}

void VulkanCommandPool::createCommandPool(uint32_t queueFamilyIndex) {
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = queueFamilyIndex;
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                      VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(m_device_, &pool_info, nullptr, &m_commandPool_) !=
        VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool");
    }
}

void VulkanCommandPool::reset(VkCommandPoolResetFlags flags) {
    if (vkResetCommandPool(m_device_, m_commandPool_, flags) != VK_SUCCESS) {
        throw std::runtime_error("Failed to reset command pool");
    }
}

} // namespace vkop
