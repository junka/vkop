#include "vulkan/VulkanSemaphore.hpp"

namespace vkop {
VulkanSemaphore::VulkanSemaphore(VkDevice vdev, bool is_support_timeline,
                                 uint64_t initvalue)
    : m_dev_(vdev) {
    VkSemaphoreTypeCreateInfo timeline_info{};
    timeline_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    if (is_support_timeline) {
        timeline_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    } else {
        timeline_info.semaphoreType = VK_SEMAPHORE_TYPE_BINARY;
    }
    timeline_info.initialValue = initvalue;

    VkSemaphoreCreateInfo sem_info{};
    sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    sem_info.pNext = &timeline_info;
    if (vkCreateSemaphore(m_dev_, &sem_info, nullptr, &m_sem_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create timeline semaphore");
    }
}

VulkanSemaphore::~VulkanSemaphore() {
    if (m_sem_) {
        vkDestroySemaphore(m_dev_, m_sem_, nullptr);
    }
}

} // namespace vkop