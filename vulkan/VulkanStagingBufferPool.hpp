// Copyright 2025 @junka
#ifndef VULKAN_VULKANSTAGINGBUFFERPOOL_HPP_
#define VULKAN_VULKANSTAGINGBUFFERPOOL_HPP_

#include <memory>
#include <optional>

#include <vulkan/vulkan.hpp>

#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanDevice.hpp"

namespace vkop {
struct StagingAllocation {
    VkBuffer buffer = VK_NULL_HANDLE;
    void *ptr = nullptr;
    VkDeviceSize offset = 0;
    size_t size = 0;
};

class VulkanStagingBufferPool {
  public:
    explicit VulkanStagingBufferPool(std::shared_ptr<VulkanDevice> &vdev);

    ~VulkanStagingBufferPool();

    std::optional<StagingAllocation> allocate(size_t size,
                                              size_t alignment = 256);

    // Host writes to `alloc` (upload staging) must be flushed before the GPU
    // copy that consumes it is submitted; device writes into `alloc` (D2H
    // readback) must be invalidated after the copy's wait and before host
    // reads of alloc.ptr. No-ops when the pool's memory is HOST_COHERENT.
    void flush(const StagingAllocation &alloc);
    void invalidate(const StagingAllocation &alloc);

    void reset();

  private:
    std::shared_ptr<VulkanDevice> m_vdev_;
    std::unique_ptr<VulkanBuffer> m_buffer_;

    VkDeviceSize m_writePos_ = 0;
    VkDeviceSize m_readPos_ = 0;
    VkDeviceSize m_poolSize_;

    bool resizeBuffer(size_t newSize);
};

} // namespace vkop

#endif // VULKAN_VULKANSTAGINGBUFFERPOOL_HPP_