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
    // Biggest single allocation this pool will ever hand out. A staging buffer
    // cannot grow without bound (it is host memory that stays pinned for the
    // whole upload, and every Vulkan allocation has to fit the device's
    // maxMemoryAllocationSize), so a transfer larger than this must be split
    // into several allocate/copy rounds by the caller. Readback of a wide
    // lm_head logits plane (q * vocab * 2 bytes) crosses this past q≈3500,
    // which is what the Tensor-side chunked paths exist for.
    static constexpr size_t kMaxAllocation = static_cast<size_t>(1024) << 20;

    explicit VulkanStagingBufferPool(std::shared_ptr<VulkanDevice> &vdev);

    ~VulkanStagingBufferPool();

    // A transfer larger than this returns nullopt from allocate() and must be
    // chunked by the caller.
    static constexpr size_t maxAllocationSize() { return kMaxAllocation; }

    std::optional<StagingAllocation> allocate(size_t size,
                                              size_t alignment = 256);

    // Host writes to `alloc` (upload staging) must be flushed before the GPU
    // copy that consumes it is submitted; device writes into `alloc` (D2H
    // readback) must be invalidated after the copy's wait and before host
    // reads of alloc.ptr. No-ops when the pool's memory is HOST_COHERENT.
    void flush(const StagingAllocation &alloc);
    void invalidate(const StagingAllocation &alloc);

    void reset();

    // Rewind the allocator without dropping a grown buffer back to the initial
    // size. A chunked transfer reuses one size for every chunk, and reset()
    // would otherwise re-grow (and re-zero) the buffer per chunk.
    void reset_keep_size() {
        m_writePos_ = 0;
        m_readPos_ = 0;
    }

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