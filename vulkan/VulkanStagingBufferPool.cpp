// Copyright 2025 @junka
#include "vulkan/VulkanStagingBufferPool.hpp"
#include <cmath>

#include "include/logger.hpp"

namespace vkop {
#define ALIGN_UP(x, y) (((x) + ((y) - 1)) & ~((y) - 1))

namespace {
constexpr int kInitialStagingBufferSize = 1024 * 1024 * 16; // 16MB initial
constexpr int kMaxStagingBufferSize = 1024 * 1024 * 512;    // 512MB max
constexpr int kGrowthFactor = 2; // Double when needed
} // namespace

VulkanStagingBufferPool::VulkanStagingBufferPool(
    std::shared_ptr<VulkanDevice> &vdev)
    : m_vdev_(vdev), m_poolSize_(0) {

    resizeBuffer(kInitialStagingBufferSize);

    if (!m_buffer_) {
        throw std::runtime_error("Failed to create staging buffer");
    }
}

VulkanStagingBufferPool::~VulkanStagingBufferPool() {
    if (m_buffer_) {
        m_buffer_->unmapMemory();
    }
}

bool VulkanStagingBufferPool::resizeBuffer(size_t newSize) {
    if (newSize <= m_poolSize_) {
        return true;
    }

    auto new_buffer = std::make_unique<VulkanBuffer>(
        m_vdev_, newSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (!new_buffer) {
        return false;
    }

    if (m_buffer_ && m_readPos_ != m_writePos_) {
        auto *mapped_memory = m_buffer_->getMappedMemory();
        void *new_mapped = new_buffer->getMappedMemory();

        size_t data_size = (m_writePos_ >= m_readPos_)
                               ? (m_writePos_ - m_readPos_)
                               : (m_poolSize_ - m_readPos_ + m_writePos_);

        if (m_writePos_ >= m_readPos_) {
            memcpy(static_cast<char *>(new_mapped),
                   static_cast<char *>(mapped_memory) + m_readPos_, data_size);
        } else {
            size_t first_part_size = m_poolSize_ - m_readPos_;
            memcpy(static_cast<char *>(new_mapped),
                   static_cast<char *>(mapped_memory) + m_readPos_,
                   first_part_size);
            memcpy(static_cast<char *>(new_mapped) + first_part_size,
                   static_cast<char *>(mapped_memory), m_writePos_);
        }

        m_writePos_ = data_size;
        m_readPos_ = 0;
    } else {
        m_writePos_ = 0;
        m_readPos_ = 0;
    }

    if (m_buffer_) {
        m_buffer_->unmapMemory();
        m_buffer_.reset();
    }
    new_buffer->unmapMemory();

    m_buffer_ = std::move(new_buffer);
    m_poolSize_ = newSize;

    return true;
}

std::optional<StagingAllocation>
VulkanStagingBufferPool::allocate(size_t size, size_t alignment) {
    assert(size != 0);

    if (size > m_poolSize_) {
        size_t new_size = kInitialStagingBufferSize;
        while (new_size < size) {
            new_size *= kGrowthFactor;
        }
        if (new_size > kMaxStagingBufferSize) {
            LOG_ERROR("size too large %lu\n", size);
            return std::nullopt;
        }

        if (!resizeBuffer(new_size)) {
            LOG_ERROR("fail to resize\n");
            return std::nullopt;
        }
    }

    VkDeviceSize aligned_write = ALIGN_UP(m_writePos_, alignment);
    VkDeviceSize next_write = aligned_write + size;
    auto *mapped_memory = m_buffer_->getMappedMemory();

    if (next_write <= m_poolSize_) {
        StagingAllocation alloc;
        alloc.ptr = static_cast<char *>(mapped_memory) + aligned_write;
        alloc.offset = aligned_write;
        alloc.size = size;
        alloc.buffer = m_buffer_->getBuffer();
        m_writePos_ = next_write;
        return alloc;
    }

    VkDeviceSize wrapped_start = ALIGN_UP(0, alignment);
    VkDeviceSize wrapped_end = wrapped_start + size;

    if (wrapped_end <= m_readPos_ || m_readPos_ == 0) {
        m_writePos_ = 0;
        aligned_write = wrapped_start;
        next_write = wrapped_end;

        StagingAllocation alloc;
        alloc.ptr = static_cast<char *>(mapped_memory) + aligned_write;
        alloc.offset = aligned_write;
        alloc.size = size;
        alloc.buffer = m_buffer_->getBuffer();
        m_writePos_ = next_write;

        return alloc;
    }

    if (m_readPos_ > 0 && m_readPos_ < m_writePos_) {
        size_t remaining_datasize = m_writePos_ - m_readPos_;

        if (remaining_datasize + size <= m_poolSize_) {
            std::memmove(mapped_memory,
                         static_cast<char *>(mapped_memory) + m_readPos_,
                         remaining_datasize);
            m_writePos_ = remaining_datasize;
            m_readPos_ = 0;

            VkDeviceSize final_aligned_write = ALIGN_UP(m_writePos_, alignment);
            VkDeviceSize final_next_write = final_aligned_write + size;

            if (final_next_write <= m_poolSize_) {
                StagingAllocation alloc;
                alloc.ptr =
                    static_cast<char *>(mapped_memory) + final_aligned_write;
                alloc.offset = final_aligned_write;
                alloc.size = size;
                alloc.buffer = m_buffer_->getBuffer();
                m_writePos_ = final_next_write;

                return alloc;
            }
        }
    }

    return std::nullopt;
}

void VulkanStagingBufferPool::reset() {
    m_writePos_ = 0;
    m_readPos_ = 0;

    if (m_poolSize_ > kInitialStagingBufferSize * 2) {
        resizeBuffer(kInitialStagingBufferSize);
    }
}

} // namespace vkop