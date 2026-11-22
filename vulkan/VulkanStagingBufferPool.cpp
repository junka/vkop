// Copyright 2025 @junka
#include "vulkan/VulkanStagingBufferPool.hpp"

#include <algorithm>
#include <cassert>

#include "include/logger.hpp"

namespace vkop {
#define ALIGN_UP(x, y) (((x) + ((y) - 1)) & ~((y) - 1))

namespace {
constexpr size_t kInitialStagingBufferSize = static_cast<size_t>(16) << 20;
// Growth granularity: a 40 MB request gets 48 MB, not the 64 MB a doubling
// policy would pick (and a 300 MB one used to get 512 MB).
constexpr size_t kGrowthStep = static_cast<size_t>(16) << 20;
} // namespace

VulkanStagingBufferPool::VulkanStagingBufferPool(
    std::shared_ptr<VulkanDevice> &vdev)
    : m_vdev_(vdev) {
    // Allocated lazily on first use: a run that never stages (UMA weights are
    // written in place) never pays for the buffer.
}

VulkanStagingBufferPool::~VulkanStagingBufferPool() = default;

std::unique_ptr<VulkanBuffer>
VulkanStagingBufferPool::createBuffer(size_t size) {
    // COHERENT stays a hard requirement: on devices where HOST_CACHED comes
    // without it (Intel exposes exactly such a split), a required-CACHED mask
    // would match no type at all. The cached preference is made inside
    // VMA::createBuffer (HOST_ACCESS_RANDOM) or the non-VMA type scorer.
    return std::make_unique<VulkanBuffer>(
        m_vdev_, size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

std::optional<StagingAllocation>
VulkanStagingBufferPool::allocate(size_t size, size_t alignment) {
    assert(size != 0);
    assert(alignment != 0 && (alignment & (alignment - 1)) == 0);

    if (size > kMaxAllocation) {
        LOG_ERROR("staging request %zu exceeds the %zu-byte per-transfer "
                  "limit; caller must chunk\n",
                  size, kMaxAllocation);
        return std::nullopt;
    }

    auto make = [](VulkanBuffer *buf, VkDeviceSize offset, size_t size) {
        StagingAllocation alloc;
        alloc.buffer = buf->getBuffer();
        alloc.ptr = static_cast<char *>(buf->getMappedMemory()) + offset;
        alloc.offset = offset;
        alloc.size = size;
        alloc.owner = buf;
        return alloc;
    };

    // Oversize: a one-shot buffer, released at the next reset(). Never
    // becomes the resident buffer, so the pool does not stay at this size.
    if (size > kMaxPooledSize) {
        m_retired_.push_back(createBuffer(size));
        return make(m_retired_.back().get(), 0, size);
    }

    VkDeviceSize aligned_write = ALIGN_UP(m_writePos_, alignment);
    if (m_buffer_ && aligned_write + size <= m_poolSize_) {
        m_writePos_ = aligned_write + size;
        return make(m_buffer_.get(), aligned_write, size);
    }

    // Does not fit. Earlier allocations from the current buffer may still be
    // referenced by recorded copies, so it is retired (kept alive until
    // reset()) rather than reused or resized in place.
    size_t new_size =
        std::max(static_cast<size_t>(m_poolSize_), kInitialStagingBufferSize);
    if (new_size < size) {
        new_size = std::min(ALIGN_UP(size, kGrowthStep), kMaxPooledSize);
    }
    if (m_buffer_ && m_writePos_ > 0) {
        m_retired_.push_back(std::move(m_buffer_));
    }
    m_buffer_.reset();
    m_buffer_ = createBuffer(new_size);
    m_poolSize_ = new_size;
    m_writePos_ = size;
    return make(m_buffer_.get(), 0, size);
}

void VulkanStagingBufferPool::flush(const StagingAllocation &alloc) {
    alloc.owner->flushMappedRange(alloc.offset, alloc.size);
}

void VulkanStagingBufferPool::invalidate(const StagingAllocation &alloc) {
    alloc.owner->invalidateMappedRange(alloc.offset, alloc.size);
}

void VulkanStagingBufferPool::reset() {
    m_writePos_ = 0;
    m_retired_.clear();
}

void VulkanStagingBufferPool::trim() {
    reset();
    if (m_poolSize_ > kInitialStagingBufferSize) {
        m_buffer_.reset();
        m_poolSize_ = 0;
    }
}

size_t VulkanStagingBufferPool::residentBytes() const {
    size_t total = m_buffer_ ? static_cast<size_t>(m_poolSize_) : 0;
    for (const auto &b : m_retired_) {
        total += static_cast<size_t>(b->getSize());
    }
    return total;
}

} // namespace vkop
