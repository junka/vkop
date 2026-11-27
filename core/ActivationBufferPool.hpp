// Copyright 2026
#ifndef CORE_ACTIVATION_BUFFER_POOL_HPP_
#define CORE_ACTIVATION_BUFFER_POOL_HPP_

#include <algorithm>
#include <cstdio>
#include <memory>
#include <unordered_map>
#include <vector>

#include "vulkan/VulkanBuffer.hpp"

namespace vkop {
namespace core {

// A Runtime-owned pool, active only while that Runtime records a synchronous
// inference round. The caller retires buffers only after their last GPU use.
class ActivationBufferPool {
  public:
    class Scope {
      public:
        explicit Scope(ActivationBufferPool *pool) : previous_(current()) {
            current() = pool;
        }
        ~Scope() { current() = previous_; }
        Scope(const Scope &) = delete;
        Scope &operator=(const Scope &) = delete;

      private:
        ActivationBufferPool *previous_;
    };

    static ActivationBufferPool *active() { return current(); }

    std::shared_ptr<VulkanBuffer> acquire(
        std::shared_ptr<VulkanDevice> &dev, size_t bytes,
        VkBufferUsageFlags flags, VkMemoryPropertyFlags properties) {
        for (auto it = free_.begin(); it != free_.end(); ++it) {
            if (it->bytes == bytes && it->flags == flags &&
                it->properties == properties && it->device == dev.get()) {
                auto buffer = std::move(it->buffer);
                free_.erase(it);
                ++hits_;
                return buffer;
            }
        }
        auto buffer = std::make_shared<VulkanBuffer>(dev, bytes, flags,
                                                     properties);
        records_[buffer.get()] = {buffer, dev.get(), bytes, flags, properties};
        ++allocations_;
        return buffer;
    }

    // The pool accepts only allocations it created, and only when no tensor,
    // view, operator, or external caller still owns the buffer.
    void recycle(std::shared_ptr<VulkanBuffer> buffer) {
        if (!buffer || buffer->is_view_range() || buffer->is_host_imported() ||
            buffer->view_aliased() || buffer.use_count() != 1) {
            ++rejected_;
            return;
        }
        auto it = records_.find(buffer.get());
        if (it == records_.end() || it->second.buffer.expired()) {
            ++rejected_;
            return;
        }
        const auto &record = it->second;
        free_.push_back({std::move(buffer), record.device, record.bytes,
                         record.flags, record.properties});
        ++retired_;
    }

    // Scratch leases may be shared by different operators, but never within
    // the same level. The Runtime clears leases only after waiting that level.
    std::shared_ptr<VulkanBuffer> acquire_scratch(
        std::shared_ptr<VulkanDevice> &dev, size_t bytes) {
        Scratch *best = nullptr;
        for (auto &entry : scratch_) {
            if (!entry.busy && entry.device == dev.get() &&
                entry.buffer->getSize() >= bytes &&
                (!best || entry.buffer->getSize() < best->buffer->getSize())) {
                best = &entry;
            }
        }
        if (best) {
            best->busy = true;
            ++scratch_hits_;
            return best->buffer;
        }
        auto buffer = std::make_shared<VulkanBuffer>(
            dev, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        scratch_.push_back({buffer, dev.get(), true});
        ++scratch_allocations_;
        return buffer;
    }

    void finish_level() {
        for (auto &entry : scratch_) entry.busy = false;
    }

    size_t available_count() const { return free_.size(); }
    size_t hit_count() const { return hits_; }

    void report() const {
        size_t free_bytes = 0, scratch_bytes = 0;
        for (const auto &entry : free_) free_bytes += entry.bytes;
        for (const auto &entry : scratch_) scratch_bytes += entry.buffer->getSize();
        std::fprintf(stderr,
                     "[activation-reuse] alloc=%zu hit=%zu retired=%zu "
                     "rejected=%zu free=%.2f MiB scratch_alloc=%zu "
                     "scratch_hit=%zu scratch=%.2f MiB\n",
                     allocations_, hits_, retired_, rejected_,
                     free_bytes / 1048576.0, scratch_allocations_,
                     scratch_hits_, scratch_bytes / 1048576.0);
    }

  private:
    struct Record {
        std::weak_ptr<VulkanBuffer> buffer;
        const VulkanDevice *device;
        size_t bytes;
        VkBufferUsageFlags flags;
        VkMemoryPropertyFlags properties;
    };
    struct FreeBuffer {
        std::shared_ptr<VulkanBuffer> buffer;
        const VulkanDevice *device;
        size_t bytes;
        VkBufferUsageFlags flags;
        VkMemoryPropertyFlags properties;
    };
    struct Scratch {
        std::shared_ptr<VulkanBuffer> buffer;
        const VulkanDevice *device;
        bool busy;
    };

    static ActivationBufferPool *&current() {
        static thread_local ActivationBufferPool *pool = nullptr;
        return pool;
    }

    std::unordered_map<const VulkanBuffer *, Record> records_;
    std::vector<FreeBuffer> free_;
    std::vector<Scratch> scratch_;
    size_t allocations_ = 0, hits_ = 0, retired_ = 0, rejected_ = 0;
    size_t scratch_allocations_ = 0, scratch_hits_ = 0;
};

} // namespace core
} // namespace vkop
#endif
