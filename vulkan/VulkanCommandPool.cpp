// Copyright 2025 @junka
#include "vulkan/VulkanCommandPool.hpp"
#include "vulkan/VulkanLib.hpp"
#include <stdexcept>

namespace vkop {

VulkanCommandPool::VulkanCommandPool(std::shared_ptr<VulkanDevice> &vdev)
    : m_vdev_(vdev) {
    uint32_t queue_family_index = m_vdev_->getComputeQueueFamilyIndex();
    createCommandPool(queue_family_index);
    createTimelineSemaphore();
    stagingbuffer_pool_ = std::make_shared<VulkanStagingBufferPool>(m_vdev_);
}

VulkanCommandPool::~VulkanCommandPool() {
    if (m_commandPool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(m_vdev_->getLogicalDevice(), m_commandPool_,
                             nullptr);
        m_commandPool_ = VK_NULL_HANDLE;
    }
    if (m_semaphore_ != VK_NULL_HANDLE) {
        vkDestroySemaphore(m_vdev_->getLogicalDevice(), m_semaphore_, nullptr);
        m_semaphore_ = VK_NULL_HANDLE;
    }
}

void VulkanCommandPool::createCommandPool(uint32_t queueFamilyIndex) {
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = queueFamilyIndex;
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                      VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(m_vdev_->getLogicalDevice(), &pool_info, nullptr,
                            &m_commandPool_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool");
    }
}

void VulkanCommandPool::createTimelineSemaphore() {
    VkSemaphoreTypeCreateInfo timeline_info{};
    timeline_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    timeline_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timeline_info.initialValue = 0;

    VkSemaphoreCreateInfo sem_info{};
    sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    sem_info.pNext = &timeline_info;
    if (vkCreateSemaphore(m_vdev_->getLogicalDevice(), &sem_info, nullptr,
                          &m_semaphore_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create timeline semaphore");
    }
}

uint64_t VulkanCommandPool::getCompletedTimelineValue() {
    uint64_t completed_value = 0;
    if (VK_SUCCESS != vkGetSemaphoreCounterValue(m_vdev_->getLogicalDevice(),
                                                 m_semaphore_,
                                                 &completed_value)) {
        throw std::runtime_error("Failed to get semaphore counter value");
    }
    return completed_value;
}

void VulkanCommandPool::reset(VkCommandPoolResetFlags flags) {
    if (vkResetCommandPool(m_vdev_->getLogicalDevice(), m_commandPool_,
                           flags) != VK_SUCCESS) {
        throw std::runtime_error("Failed to reset command pool");
    }
}

} // namespace vkop
