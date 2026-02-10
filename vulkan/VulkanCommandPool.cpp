// Copyright 2025 @junka
#include "vulkan/VulkanCommandPool.hpp"
#include "vulkan/VulkanLib.hpp"
#include <stdexcept>

namespace vkop {

VulkanCommandPool::VulkanCommandPool(std::shared_ptr<VulkanDevice> &vdev)
    : m_vdev_(vdev) {
    for (auto [qfidx, qcnt, qflags] : vdev->getComputeQueueFamilyIndex()) {
        printf("create command pool for queue family index %d\n", qfidx);
        createCommandPool(qfidx);
    }
    stagingbuffer_pool_ = std::make_shared<VulkanStagingBufferPool>(m_vdev_);
}

VulkanCommandPool::~VulkanCommandPool() {
    for (auto *cmdpool : m_commandPool_) {
        if (cmdpool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(m_vdev_->getLogicalDevice(), cmdpool, nullptr);
        }
    }
}

void VulkanCommandPool::createCommandPool(uint32_t queueFamilyIndex) {
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = queueFamilyIndex;
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                      VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool cmdpool;
    auto ret = vkCreateCommandPool(m_vdev_->getLogicalDevice(), &pool_info,
                                   nullptr, &cmdpool);
    if (ret != VK_SUCCESS) {
        printf("Failed to create command pool %d\n", ret);
        throw std::runtime_error("Failed to create command pool");
    }
    m_commandPool_.emplace_back(cmdpool);
}

void VulkanCommandPool::reset(VkCommandPoolResetFlags flags) {
    for (auto *cmdpool : m_commandPool_) {
        if (cmdpool != VK_NULL_HANDLE) {
            if (vkResetCommandPool(m_vdev_->getLogicalDevice(), cmdpool,
                                   flags) != VK_SUCCESS) {
                throw std::runtime_error("Failed to reset command pool");
            }
        }
    }
}

} // namespace vkop
