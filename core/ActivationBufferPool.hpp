// Copyright 2026
#ifndef CORE_ACTIVATION_BUFFER_POOL_HPP_
#define CORE_ACTIVATION_BUFFER_POOL_HPP_

#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "vulkan/VulkanBuffer.hpp"

namespace vkop {
namespace core {

// Runtime-owned activation buffer pool, active (on this thread) only while
// that Runtime records a round with VKOP_ACTIVATION_REUSE.
//
// Ownership. The pool keeps a strong reference to every buffer it allocates,
// so no pool buffer is ever destroyed in the middle of a round: dropping the
// last tensor/view/binding reference only makes it "unowned" (use_count 1),
// never frees a VkBuffer that already-recorded, not-yet-executed commands
// still access. Unowned buffers return to the free list when the Runtime
// collects them (per level for the buffers it just released, and once per
// round for everything), and only buffers idle for kTrimRounds completed
// rounds are destroyed -- after the round's final wait.
//
// Ordering. A buffer freed at level L is handed out while recording a later
// level; the Runtime separates levels with a barrier that orders shader and
// transfer reads/writes, so the new writer runs after every earlier access.
// Such a buffer is marked reuse_pending until the next device-wide sync,
// because a host write or a separate queue submit into it is NOT ordered by
// that barrier; Tensor::copyToGPUBuffer calls sync_for_host() first.
class ActivationBufferPool {
  public:
    static constexpr uint32_t kTrimRounds = 2;

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

    ActivationBufferPool() = default;
    ActivationBufferPool(const ActivationBufferPool &) = delete;
    ActivationBufferPool &operator=(const ActivationBufferPool &) = delete;

    std::shared_ptr<VulkanBuffer> acquire(
        std::shared_ptr<VulkanDevice> &dev, size_t bytes,
        VkBufferUsageFlags flags, VkMemoryPropertyFlags properties) {
        const Key key{dev.get(), bytes, flags, properties};
        auto list = free_.find(key);
        if (list != free_.end() && !list->second.empty()) {
            Entry *entry = list->second.back();
            list->second.pop_back();
            entry->free = false;
            entry->idle_rounds = 0;
            if (entry->pending) {
                entry->buffer->set_reuse_pending(true);
                pending_out_.push_back(entry->buffer.get());
            }
            ++hits_;
            return entry->buffer;
        }
        auto buffer =
            std::make_shared<VulkanBuffer>(dev, bytes, flags, properties);
        auto entry = std::make_unique<Entry>();
        entry->buffer = buffer;
        entry->key = key;
        entries_.emplace(buffer.get(), std::move(entry));
        ++allocations_;
        return buffer;
    }

    bool owns(const VulkanBuffer *buffer) const {
        return buffer && entries_.count(buffer) != 0;
    }

    // The Runtime dropped its references to `buffer` (a pool buffer) at a
    // level end. It is free once nothing else owns it; otherwise it is
    // retried at later level ends (a view of it may be the next to die).
    void release(const VulkanBuffer *buffer) {
        auto it = entries_.find(buffer);
        Entry &entry = *it->second;
        if (entry.free || entry.held) return;
        if (!try_free(entry, /*pending=*/true)) {
            entry.held = true;
            held_.push_back(&entry);
        }
    }

    // Level end: retry the buffers still owned by someone at their release.
    void collect_held() {
        size_t keep = 0;
        for (Entry *entry : held_) {
            if (!entry->free && !try_free(*entry, /*pending=*/true)) {
                held_[keep++] = entry;
            } else {
                entry->held = false;
            }
        }
        held_.resize(keep);
    }

    // Scratch leases may be shared by operators of different levels, never
    // within one level; finish_level() ends this level's leases.
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
            dev, bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
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

    // The Runtime's "submit everything recorded so far and wait for the
    // device" for the current round; empty outside a round.
    void set_host_sync(std::function<void()> sync) {
        host_sync_ = std::move(sync);
    }
    // A host write / separate submit is about to touch a reuse_pending buffer.
    void sync_for_host() {
        ++host_syncs_;
        if (host_sync_) host_sync_();
        mark_device_idle();
    }

    // Everything recorded so far has executed: nothing handed out or free is
    // pending any more.
    void mark_device_idle() {
        for (VulkanBuffer *buffer : pending_out_) {
            buffer->set_reuse_pending(false);
        }
        pending_out_.clear();
        for (auto &item : entries_) item.second->pending = false;
    }

    // After the round's final device wait: free every unowned buffer, then
    // destroy the ones that stayed free for kTrimRounds rounds (shapes that
    // stopped occurring) so the cache cannot grow without bound.
    void finish_round() {
        mark_device_idle();
        for (Entry *entry : held_) entry->held = false;
        held_.clear();
        for (auto it = entries_.begin(); it != entries_.end();) {
            Entry &entry = *it->second;
            if (!entry.free) {
                try_free(entry, /*pending=*/false);
            } else if (++entry.idle_rounds >= kTrimRounds &&
                       entry.buffer.use_count() == 1) {
                auto &list = free_[entry.key];
                for (size_t i = 0; i < list.size(); ++i) {
                    if (list[i] == &entry) {
                        list[i] = list.back();
                        list.pop_back();
                        break;
                    }
                }
                ++trimmed_;
                it = entries_.erase(it);
                continue;
            }
            ++it;
        }
    }

    size_t available_count() const {
        size_t n = 0;
        for (const auto &item : free_) n += item.second.size();
        return n;
    }
    size_t hit_count() const { return hits_; }
    size_t allocation_count() const { return allocations_; }
    size_t owned_count() const { return entries_.size(); }

    void report() const {
        size_t free_bytes = 0, owned_bytes = 0, scratch_bytes = 0;
        for (const auto &item : entries_) {
            owned_bytes += item.second->key.bytes;
            if (item.second->free) free_bytes += item.second->key.bytes;
        }
        for (const auto &entry : scratch_)
            scratch_bytes += entry.buffer->getSize();
        std::fprintf(stderr,
                     "[activation-reuse] alloc=%zu hit=%zu freed=%zu "
                     "trimmed=%zu host_sync=%zu owned=%zu (%.2f MiB) "
                     "free=%.2f MiB scratch_alloc=%zu scratch_hit=%zu "
                     "scratch=%.2f MiB\n",
                     allocations_, hits_, freed_, trimmed_, host_syncs_,
                     entries_.size(), owned_bytes / 1048576.0,
                     free_bytes / 1048576.0, scratch_allocations_,
                     scratch_hits_, scratch_bytes / 1048576.0);
    }

  private:
    struct Key {
        const VulkanDevice *device;
        size_t bytes;
        VkBufferUsageFlags flags;
        VkMemoryPropertyFlags properties;
        bool operator==(const Key &o) const {
            return device == o.device && bytes == o.bytes &&
                   flags == o.flags && properties == o.properties;
        }
    };
    struct KeyHash {
        size_t operator()(const Key &k) const {
            size_t h = std::hash<const void *>()(k.device);
            h ^= std::hash<size_t>()(k.bytes) + 0x9e3779b97f4a7c15ULL +
                 (h << 6) + (h >> 2);
            h ^= (static_cast<size_t>(k.flags) << 32) ^ k.properties;
            return h;
        }
    };
    struct Entry {
        std::shared_ptr<VulkanBuffer> buffer;
        Key key{};
        bool free = false;
        // Freed during the current round (see the class comment).
        bool pending = false;
        // Waiting in held_ for its other owners to let go.
        bool held = false;
        uint32_t idle_rounds = 0;
    };
    struct Scratch {
        std::shared_ptr<VulkanBuffer> buffer;
        const VulkanDevice *device;
        bool busy;
    };

    bool try_free(Entry &entry, bool pending) {
        if (entry.buffer.use_count() != 1) return false;
        entry.free = true;
        entry.pending = pending;
        entry.idle_rounds = 0;
        entry.buffer->reset_view_tracking();
        free_[entry.key].push_back(&entry);
        ++freed_;
        return true;
    }

    static ActivationBufferPool *&current() {
        static thread_local ActivationBufferPool *pool = nullptr;
        return pool;
    }

    std::unordered_map<const VulkanBuffer *, std::unique_ptr<Entry>> entries_;
    std::unordered_map<Key, std::vector<Entry *>, KeyHash> free_;
    std::vector<Entry *> held_;
    std::vector<VulkanBuffer *> pending_out_;
    std::vector<Scratch> scratch_;
    std::function<void()> host_sync_;
    size_t allocations_ = 0, hits_ = 0, freed_ = 0, trimmed_ = 0;
    size_t host_syncs_ = 0;
    size_t scratch_allocations_ = 0, scratch_hits_ = 0;
};

} // namespace core
} // namespace vkop
#endif
