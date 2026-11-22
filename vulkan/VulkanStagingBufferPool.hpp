// Copyright 2025 @junka
#ifndef VULKAN_VULKANSTAGINGBUFFERPOOL_HPP_
#define VULKAN_VULKANSTAGINGBUFFERPOOL_HPP_

#include <memory>
#include <optional>
#include <vector>

#include <vulkan/vulkan.hpp>

#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanDevice.hpp"

namespace vkop {
struct StagingAllocation {
    VkBuffer buffer = VK_NULL_HANDLE;
    void *ptr = nullptr;
    VkDeviceSize offset = 0;
    size_t size = 0;
    // The VulkanBuffer behind `buffer`, for flush/invalidate. Owned by the
    // pool and valid until the next reset()/trim().
    VulkanBuffer *owner = nullptr;
};

// Linear (bump) allocator over host-visible staging memory.
//
// Contract: every allocation stays valid -- its bytes untouched and its
// VkBuffer alive -- until the caller's next reset() or trim(). The caller
// calls those only after the GPU work that consumes the allocations has
// completed (submit + wait). Nothing is ever recycled behind its back: when
// the current buffer is full, the pool moves it to a retired list (still
// alive, still referenced by recorded copies) and continues in a new one.
//
// Memory: the resident buffer never grows past kMaxPooledSize (64 MB). A
// single request above that (an image upload/readback, which cannot be
// split) gets a one-shot buffer released at the next reset(). Buffer
// transfers above chunkSize() are expected to be chunked by the caller, so
// the staging footprint of a large weight is one chunk, not the weight.
class VulkanStagingBufferPool {
  public:
    // Resident pool size cap, and the chunk size large buffer transfers use.
    static constexpr size_t kMaxPooledSize = static_cast<size_t>(64) << 20;
    // Hard cap on a single allocation (one-shot buffers included); every
    // Vulkan allocation also has to fit maxMemoryAllocationSize.
    static constexpr size_t kMaxAllocation = static_cast<size_t>(1024) << 20;

    explicit VulkanStagingBufferPool(std::shared_ptr<VulkanDevice> &vdev);
    ~VulkanStagingBufferPool();

    // Largest single allocation; nullopt above it.
    static constexpr size_t maxAllocationSize() { return kMaxAllocation; }
    // What a chunked buffer transfer should use per round: it is served
    // from the resident buffer without a one-shot allocation.
    static constexpr size_t chunkSize() { return kMaxPooledSize; }

    std::optional<StagingAllocation> allocate(size_t size,
                                              size_t alignment = 256);

    // Host writes to `alloc` (upload staging) must be flushed before the GPU
    // copy that consumes it is submitted; device writes into `alloc` (D2H
    // readback) must be invalidated after the copy's wait and before host
    // reads of alloc.ptr. Driver no-ops on HOST_COHERENT memory.
    void flush(const StagingAllocation &alloc);
    void invalidate(const StagingAllocation &alloc);

    // Rewind and drop retired / one-shot buffers. Keeps the resident buffer
    // at its current size, so a chunked transfer that resets per chunk does
    // not reallocate it.
    void reset();

    // reset(), then release the resident buffer as well if it grew past the
    // initial size. Call when a burst of large transfers is over (end of
    // model load): the pool would otherwise keep up to kMaxPooledSize pinned
    // for the rest of the process.
    void trim();

    // Bytes currently held (resident + retired + one-shot), for diagnostics.
    size_t residentBytes() const;

  private:
    std::shared_ptr<VulkanDevice> m_vdev_;
    std::unique_ptr<VulkanBuffer> m_buffer_;
    // Full buffers whose allocations may still be referenced by recorded,
    // not yet completed copies; and one-shot oversize buffers.
    std::vector<std::unique_ptr<VulkanBuffer>> m_retired_;

    VkDeviceSize m_writePos_ = 0;
    VkDeviceSize m_poolSize_ = 0;

    std::unique_ptr<VulkanBuffer> createBuffer(size_t size);
};

} // namespace vkop

#endif // VULKAN_VULKANSTAGINGBUFFERPOOL_HPP_
