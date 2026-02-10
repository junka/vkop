// Copyright 2025 - 2026 @junka
#ifndef SRC_VULKANSEMAPHORE_HPP_
#define SRC_VULKANSEMAPHORE_HPP_

#include "vulkan/VulkanLib.hpp"

namespace vkop {

class VulkanSemaphore {
  public:
    explicit VulkanSemaphore(VkDevice vdev, bool is_support_timeline = false,
                             uint64_t initvalue = 0);

    ~VulkanSemaphore();

    VkSemaphore getSemaphore() { return m_sem_; }

    VulkanSemaphore(const VulkanSemaphore &) = delete;
    VulkanSemaphore &operator=(const VulkanSemaphore &) = delete;
    VulkanSemaphore &operator=(VulkanSemaphore &&other) = delete;

  private:
    VkDevice m_dev_;
    VkSemaphore m_sem_ = VK_NULL_HANDLE;
};

} // namespace vkop

#endif // SRC_VULKANSEMAPHORE_HPP_
