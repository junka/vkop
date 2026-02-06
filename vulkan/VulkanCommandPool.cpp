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
    if (vkCreateCommandPool(m_vdev_->getLogicalDevice(), &pool_info, nullptr,
                            &cmdpool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool");
    }
    m_commandPool_.emplace_back(cmdpool);
}

uint64_t VulkanCommandPool::getCompletedTimelineValue(
    const std::shared_ptr<VulkanQueue> &queue) {
    if (m_vdev_->is_support_timeline_semaphore()) {
        uint64_t completed_value = 0;
        if (VK_SUCCESS != vkGetSemaphoreCounterValue(
                              m_vdev_->getLogicalDevice(),
                              queue->getSemaphore(), &completed_value)) {
            throw std::runtime_error("Failed to get semaphore counter value");
        }
        return completed_value;
    }
    return m_timelineValue_.load();
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
