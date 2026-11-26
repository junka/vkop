// Copyright 2025 @junka
//
// Page a resident weight's bytes in from an external file, in place.
//
// Why this exists: on a card too small for the whole model, some weights have
// to live outside device memory. The one requirement for swapping them back in
// is that the swap be numerically identical to having loaded them at startup --
// otherwise nothing downstream can be trusted. That property is measured, not
// assumed: tools/probes/weight_paging_probe.cpp rewrites a live weight buffer
// and requires bit-exact agreement with a fresh LoadModel, including under hot
// segment replay and the BDA path. This class is that probe's mechanism, made
// usable by a driver.
//
// What it deliberately does NOT do, and why:
//
//   * It never allocates or replaces a VkBuffer. The resident buffer keeps its
//     identity for the life of the process, because three separate things bake
//     that identity in: the BDA address table (an address, rewritten per
//     dispatch, so the bytes under it may change freely), segment replay's
//     (handle, offset, size) key, and every descriptor a segment recorded.
//     Reallocating "works" numerically and destroys all three.
//
//   * It never submits inside Runtime::Run(). A copy submitted from here runs
//     between rounds, which is why a replayed segment is safe: replay skips
//     re-recording, not the memory it reads, and the memory has already been
//     updated. A pager that injected its copy into the level loop would be
//     skipped by the very segments it needs to affect.
//
//   * It holds ONCE-ALLOCATED, pinned staging and reuses it. Measured on a
//     170 MiB band: asking a staging pool per page-in costs 3x end to end,
//     because a band above the pool's 64 MiB cap becomes a fresh
//     vkCreateBuffer + vkAllocateMemory every time. The pinned path measures
//     22-26 GB/s for the device copy (UMA and PCIe gen4 x16 alike).
//
// Bandwidth, so a caller can budget: a 170 MiB band is ~8 ms of device copy
// plus ~14-22 ms of host copy on the machines measured. The host half is the
// larger term, so overlapping it is the only way to make paging cheap; see
// page_in_async().
//
// A note on where the bytes come from: `read_at` is a pread into staging, not a
// memcpy out of the model's mmap. Both measure within noise once the page cache
// is warm, but a pager has no reason to fault a multi-GB blob in 16 KiB pages
// when it can ask for exact ranges (core/runtime.cpp's loader makes the same
// choice for the same reason).

#ifndef CORE_WEIGHT_PAGER_HPP_
#define CORE_WEIGHT_PAGER_HPP_

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/Tensor.hpp"
#include "core/runtime.hpp"
#include "model/load.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanCommandBuffer.hpp"
#include "vulkan/VulkanCommandPool.hpp"
#include "vulkan/VulkanDevice.hpp"

namespace vkop {
namespace core {

// Where one weight's bytes live outside device memory.
struct PagedSource {
    std::string name;        // the resident initializer this fills
    std::string path;        // file holding the bytes
    uint64_t offset = 0;     // byte offset into `path`
    uint64_t size = 0;       // payload bytes; must equal the resident payload
    std::string group;       // band this weight belongs to ("3", "layer:7", ...)
};

// A band is the unit a schedule pages: every weight whose `group` matches is
// moved by one page_in_group call, in one command buffer and one submit.
struct PageStats {
    size_t weights = 0;
    size_t bytes = 0;
    double host_ms = 0.0;    // pread into staging + flush
    double device_ms = 0.0;  // submit + wait
    double total_ms() const { return host_ms + device_ms; }
    double device_gbps() const {
        return device_ms > 0 ? (static_cast<double>(bytes) / 1e9) /
                                   (device_ms / 1e3)
                             : 0.0;
    }
};

class WeightPager {
  public:
    // `rt` must outlive this object: the pager reads its resident tensors on
    // every page-in rather than caching handles, so a tensor the runtime
    // replaces is picked up instead of paged into a dead buffer.
    //
    // `pool` may be null: the declaration half (add / add_from_model / groups)
    // reads only the model file and needs no device, which is what lets a
    // caller inspect and validate a paging plan before opening a GPU. Every
    // page-in_* method requires a real pool and throws if it was not given one.
    WeightPager(std::shared_ptr<Runtime> &rt,
                const std::shared_ptr<VulkanCommandPool> &pool)
        : rt_(rt), pool_(pool), dev_(pool ? pool->getVulkanDevice() : nullptr) {}

    WeightPager(const WeightPager &) = delete;
    WeightPager &operator=(const WeightPager &) = delete;

    // Declare one weight's external home. Repeats for the same name replace the
    // earlier entry (and keep its position), so a caller may re-declare after a
    // model swap without reordering the bands.
    void add(PagedSource src) {
        if (src.name.empty() || src.path.empty()) {
            throw std::runtime_error(
                "WeightPager::add: a paged source needs a name and a path");
        }
        const std::string key = src.name;
        auto it = by_name_.find(key);
        if (it != by_name_.end()) {
            it->second = src;
            for (auto &kv : order_) {
                if (kv.first == key) {
                    kv.second = std::move(src);
                    break;
                }
            }
            return;
        }
        order_.emplace_back(key, src);
        by_name_.emplace(key, std::move(src));
    }

    // Declare every initializer of `model` that carries a moe_source (the
    // converter's paging declaration: a weight whose bytes live in an external
    // checkpoint rather than the trailer). Paths are resolved relative to the
    // vkopbin's directory, matching how the loader records them.
    //
    // This is the bridge from what onnx2vkop already writes to what a pager
    // needs: no schema change, no new converter field.
    size_t add_from_model(const load::VkModel &model,
                          const std::string &vkopbin_path) {
        std::string dir;
        const auto slash = vkopbin_path.find_last_of('/');
        if (slash != std::string::npos) dir = vkopbin_path.substr(0, slash + 1);
        // Sorted by name before declaring: `model.initializers` is an
        // unordered_map, so iterating it put L0.up_proj ahead of L0.gate_proj
        // and made the band order arbitrary. Declaration order becomes
        // groups() order, which a schedule walks, so it has to be stable.
        std::vector<std::string> names;
        names.reserve(model.initializers.size());
        for (const auto &kv : model.initializers) {
            if (!kv.second.moe_source.empty()) names.push_back(kv.first);
        }
        std::sort(names.begin(), names.end());
        size_t added = 0;
        for (const auto &name : names) {
            const auto &in = model.initializers.at(name);
            PagedSource s;
            s.name = in.name;
            s.path = dir + in.moe_source;
            s.offset = in.moe_source_offset;
            s.size = in.moe_source_size;
            // A slot's group is the place it lives, which is what a schedule
            // orders by. Falling back to the name keeps a slot pageable even
            // when the exporter left the group empty.
            s.group = in.moe_group;
            add(std::move(s));
            ++added;
        }
        return added;
    }

    size_t size() const { return by_name_.size(); }
    bool empty() const { return by_name_.empty(); }

    // Derive bands from the graph instead of from converter declarations.
    //
    // Measured on llm.vkopbin: 312 of 313 weights are consumed by exactly ONE
    // concurrent level, so "the band a weight belongs to" is the level that
    // reads it -- a fact, not a heuristic. That makes a dense-LLM paging plan
    // need no converter change at all: a band is single-shot within a round,
    // so the pager never needs a double buffer, only prefetch.
    //
    // `min_bytes` skips small weights (norms, biases, shape constants), which
    // are not worth a transfer of their own and stay resident. Returns the
    // number of bands declared (one per level that owns a skipped weight).
    //
    // `scope` names the band group: "level:<N>". A weight read by more than one
    // level is left alone -- it must stay resident, since paging it would have
    // to be re-done mid-round, and that is exactly the case this derivation
    // refuses to guess about.
    size_t add_bands_from_graph(const load::VkModel &model,
                                const std::string &vkopbin_path,
                                size_t min_bytes = 1u << 20) {
        std::string dir;
        const auto slash = vkopbin_path.find_last_of('/');
        if (slash != std::string::npos) dir = vkopbin_path.substr(0, slash + 1);

        // node name -> level index
        std::unordered_map<std::string, size_t> level_of;
        for (size_t li = 0; li < model.concurrent_execution_levels.size(); ++li) {
            for (const auto &n : model.concurrent_execution_levels[li]) {
                level_of[n] = li;
            }
        }
        // weight name -> the set of levels that read it
        std::unordered_map<std::string, std::vector<size_t>> readers;
        for (const auto &n : model.nodes) {
            auto it = level_of.find(n.name);
            if (it == level_of.end()) continue;
            for (const auto &in : n.inputs) {
                if (model.initializers.count(in.name) == 0) continue;
                auto &v = readers[in.name];
                if (std::find(v.begin(), v.end(), it->second) == v.end()) {
                    v.push_back(it->second);
                }
            }
        }

        size_t bands = 0;
        for (const auto &kv : readers) {
            if (kv.second.size() != 1) continue; // shared: keep resident
            const auto &init = model.initializers.at(kv.first);
            if (init.size < min_bytes) continue;
            // A declared external source (MoE slot) already has a home; leave
            // it to add_from_model so the two paths cannot disagree.
            if (!init.moe_source.empty()) continue;
            PagedSource s;
            s.name = kv.first;
            s.path = vkopbin_path;
            s.offset = model.initializer_file_offset +
                       model.initializer_offsets.at(kv.first);
            s.size = init.size;
            s.group = "level:" + std::to_string(kv.second.front());
            add(std::move(s));
            ++bands;
        }
        // Report bands, which is how a schedule walks them, not weights.
        return groups().size();
    }

    // Per-band byte totals, in declaration order. A schedule sizes its arena
    // from this: the largest band is what one slot must hold, and bands differ
    // by ~8x on a real LLM (lm_head vs a per-layer FFN), so a single arena
    // sized to the max wastes the difference on every other slot.
    std::vector<std::pair<std::string, size_t>> band_bytes() const {
        std::vector<std::pair<std::string, size_t>> out;
        for (const auto &kv : order_) {
            auto it = std::find_if(out.begin(), out.end(),
                                   [&](const auto &p) {
                                       return p.first == kv.second.group;
                                   });
            if (it == out.end()) {
                out.emplace_back(kv.second.group,
                                 static_cast<size_t>(kv.second.size));
            } else {
                it->second += static_cast<size_t>(kv.second.size);
            }
        }
        return out;
    }

    // Attach to a Runtime's per-level seam, in the one shape that works:
    // page the band named "level:<N>" when level N starts, and do nothing for
    // a level that owns no band.
    //
    // This is the whole driver-side integration when bands came from
    // add_bands_from_graph(), which names them by level. Bands declared any
    // other way (add_from_model's MoE groups, say) need a schedule that knows
    // which level reads them, so this helper refuses rather than guessing.
    //
    // `on_page` is optional and called with each page-in's stats, so a driver
    // can log or account for the transfer without wrapping the hook.
    void attach_to(std::shared_ptr<Runtime> &rt,
                   std::function<void(const PageStats &)> on_page = {}) {
        if (!pool_) {
            throw std::runtime_error(
                "WeightPager::attach_to needs a command pool");
        }
        // Adopt `rt` as the runtime this pager drives. The declaration half
        // (add_bands_from_graph etc.) needs no runtime, so a driver usually
        // builds the pager with a null one and supplies the real Runtime only
        // here -- but resolve_() dereferences rt_ on every page-in, so attaching
        // without rebinding segfaults on the FIRST band (measured: llm_serve's
        // --page-weights path died inside resolve_ on level 1's page-in).
        if (!rt) {
            throw std::runtime_error(
                "WeightPager::attach_to given a null Runtime");
        }
        rt_ = rt;
        for (const auto &g : groups()) {
            if (g.rfind("level:", 0) != 0) {
                throw std::runtime_error(
                    "WeightPager::attach_to: band '" + g +
                    "' is not named level:<N>; attach_to only knows the "
                    "graph-derived schedule (use set_level_hook directly for a "
                    "custom one)");
            }
        }
        // Precompute the band set once: the hook runs once per level, and
        // groups() walks the whole declaration order, so rebuilding it inside
        // the hook would be O(levels x weights).
        auto bands = groups();
        auto band_set = std::unordered_map<std::string, char>();
        band_set.reserve(bands.size() * 2);
        for (const auto &g : bands) band_set.emplace(g, 1);
        rt->set_level_hook([this, band_set = std::move(band_set),
                            on_page](size_t level) {
            const std::string g = "level:" + std::to_string(level);
            if (band_set.count(g) == 0) {
                return; // this level owns no band
            }
            auto st = page_in_group(g);
            if (on_page) on_page(st);
        });
    }

    // Every declared weight's name, in declaration order. Exactly the set a
    // driver hands to Runtime::set_paged_initializers, so the plan and the
    // pager cannot disagree about which weights the arena is responsible for.
    std::vector<std::string> names() const {
        std::vector<std::string> out;
        out.reserve(order_.size());
        for (const auto &kv : order_) out.push_back(kv.first);
        return out;
    }

    // band_bytes() under the name a caller would look for: per-band totals, so
    // a driver can pick which bands stay resident. Largest first is the rule
    // that matters on a real artifact -- a 27B lm_head band is 1212.5 MiB
    // against 170 MiB for a per-layer band, so keeping the head resident shrinks
    // the shared arena 7x on its own.
    std::vector<std::pair<std::string, size_t>> group_bytes() const {
        return band_bytes();
    }

    // Drop a band from the paging plan. A driver that keeps a band resident
    // calls this so it is neither page-in'd nor passed to
    // set_paged_initializers, which is what leaves its own buffer uploaded.
    void drop_group(const std::string &group) {
        order_.erase(std::remove_if(order_.begin(), order_.end(),
                                    [&](const auto &kv) {
                                        return kv.second.group == group;
                                    }),
                     order_.end());
        for (auto it = by_name_.begin(); it != by_name_.end();) {
            if (it->second.group == group)
                it = by_name_.erase(it);
            else
                ++it;
        }
    }

    // weight name -> band name, for PagedArenaPlan::slot_of.
    //
    // A band is paged in by one call and read by one level, so its members are
    // live together and must get DISJOINT arena offsets; separate bands can
    // reuse the same region. Without this map the plan sees a flat list and
    // gives every weight the same offset -- see the corruption note in
    // Runtime's arena layout: a k/q/v band then keeps only v_proj's bytes.
    std::unordered_map<std::string, std::string> slot_groups() const {
        std::unordered_map<std::string, std::string> out;
        out.reserve(order_.size());
        for (const auto &kv : order_) out.emplace(kv.first, kv.second.group);
        return out;
    }

    // Distinct band names, in declaration order.
    //
    // Declaration order, not map order: a schedule that walks bands in the
    // order the file lists them gets the file's locality (consecutive bands are
    // usually consecutive in the checkpoint), while unordered_map iteration
    // measured as arbitrary -- it returned up_proj before gate_proj.
    std::vector<std::string> groups() const {
        std::vector<std::string> out;
        for (const auto &kv : order_) {
            if (std::find(out.begin(), out.end(), kv.second.group) ==
                out.end())
                out.push_back(kv.second.group);
        }
        return out;
    }

    // Move every weight of `group` into its resident buffer. One staging copy
    // per weight, one command buffer, one submit+wait.
    PageStats page_in_group(const std::string &group) {
        std::vector<const PagedSource *> picked;
        // Walk declaration order (not by_name_) so a band's weights load in
        // file order, which keeps the reads sequential on the source file.
        for (const auto &kv : order_) {
            if (kv.second.group == group) picked.push_back(&kv.second);
        }
        return page_in_all(picked);
    }

    PageStats page_in(const std::string &name) {
        auto it = by_name_.find(name);
        if (it == by_name_.end()) {
            throw std::runtime_error("WeightPager: '" + name +
                                     "' is not a declared paged weight");
        }
        return page_in_all({&it->second});
    }

    // The whole set, in declaration order. This is what a "cold start" or a
    // model with no free VRAM above its working set uses.
    PageStats page_in_all() {
        std::vector<const PagedSource *> all;
        all.reserve(by_name_.size());
        for (const auto &kv : order_) all.push_back(&kv.second);
        return page_in_all(all);
    }

  private:
    struct Destination {
        const PagedSource *src = nullptr;
        std::shared_ptr<VulkanBuffer> buf;
    };

    // Resolve every source to its live destination and check the payload fits.
    // Fail-closed: a source larger than its buffer, or a name the runtime no
    // longer holds, aborts the whole page-in rather than moving a partial band
    // (a half-written band is a silently wrong model, not a slow one).
    std::vector<Destination>
    resolve_(const std::vector<const PagedSource *> &sources) const {
        std::vector<Destination> out;
        out.reserve(sources.size());
        for (const auto *s : sources) {
            auto t = rt_->GetInitializer(s->name);
            if (!t) {
                throw std::runtime_error(
                    "WeightPager: runtime has no resident initializer '" +
                    s->name + "' (was it paged out of the graph?)");
            }
            auto vb = std::dynamic_pointer_cast<VulkanBuffer>(t->gpu_buffer());
            if (!vb) {
                throw std::runtime_error(
                    "WeightPager: '" + s->name +
                    "' has no Vulkan buffer (the image backend is unsupported)");
            }
            if (s->size > vb->getSize()) {
                throw std::runtime_error(
                    "WeightPager: '" + s->name + "' source is " +
                    std::to_string(s->size) + " bytes but its buffer holds " +
                    std::to_string(vb->getSize()));
            }
            out.push_back({s, std::move(vb)});
        }
        return out;
    }

    // One pinned staging buffer, grown on demand and kept. Not a pool: a band
    // larger than the pool's cap would otherwise allocate a fresh VkBuffer per
    // page-in, measured at 3x the end-to-end cost.
    VulkanBuffer &staging_for_(size_t bytes) {
        if (!staging_buf_ || staging_buf_->getSize() < bytes) {
            auto &dev_mut =
                const_cast<std::shared_ptr<VulkanDevice> &>(dev_);
            staging_buf_ = std::make_shared<VulkanBuffer>(
                dev_mut, bytes,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        }
        return *staging_buf_;
    }

    PageStats page_in_all(const std::vector<const PagedSource *> &sources) {
        PageStats st;
        if (sources.empty()) return st;
        auto dst = resolve_(sources);

        size_t total = 0;
        for (const auto &d : dst) total += static_cast<size_t>(d.src->size);
        st.weights = dst.size();
        st.bytes = total;
        if (total == 0) return st;

        if (!pool_ || !dev_) {
            throw std::runtime_error(
                "WeightPager: page_in needs a VulkanCommandPool; this pager "
                "was constructed without one (declaration-only use)");
        }

        const auto t0 = std::chrono::steady_clock::now();

        // ---- host half: exact-range reads into one pinned staging buffer ----
        // Grouped by path so each file is opened once, in the order the
        // arguments arrive; a band is one file in every case measured.
        VulkanBuffer &stage = staging_for_(total);
        auto *base = static_cast<char *>(stage.getMappedMemory());
        struct Copy {
            size_t dst_off;
            const PagedSource *src;
        };
        std::vector<Copy> copies;
        std::unordered_map<std::string, std::unique_ptr<load::FileMapping>>
            files;
        size_t off = 0;
        for (const auto &d : dst) {
            auto &fm = files[d.src->path];
            if (!fm) {
                fm = std::make_unique<load::FileMapping>();
                if (!fm->map_file(d.src->path)) {
                    throw std::runtime_error(
                        "WeightPager: cannot open '" + d.src->path + "'");
                }
            }
            fm->read_at(base + off, static_cast<size_t>(d.src->size),
                        static_cast<size_t>(d.src->offset));
            copies.push_back({off, d.src});
            off += static_cast<size_t>(d.src->size);
        }
        stage.flushMappedRange(0, total);

        const auto t1 = std::chrono::steady_clock::now();
        st.host_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        // ---- device half: one command buffer, one submit, one wait ----
        VulkanCommandBuffer cmd(pool_);
        cmd.begin();
        for (size_t i = 0; i < copies.size(); ++i) {
            auto &buf = dst[i].buf;
            const size_t n = static_cast<size_t>(copies[i].src->size);
            // Set the tracked access to SHADER_READ before the copy so the
            // helper's closing barrier hands the buffer back readable instead
            // of restoring access 0 (the trap ops/Operator.cpp:287 records for
            // the BDA address table: a fresh table's access is 0, and a
            // restored 0 leaves the write with no visibility edge).
            //
            // Destination offset stays 0 on purpose: `buf` is the weight's
            // sub-range of the paged arena, and VulkanBuffer forwards a copy
            // on a parented buffer to the parent with `offset + range_offset_`
            // added. That forwarding is what routes each band member to its own
            // slot -- see the group layout in Runtime::LoadModel. Passing the
            // offset explicitly here would apply it twice.
            buf->readBarrier(cmd.get(), n, 0);
            buf->copyStageBufferToBuffer(cmd.get(), stage.getBuffer(),
                                         copies[i].dst_off, n, 0);
        }
        cmd.end();
        cmd.submit(dev_->getComputeQueue());
        cmd.wait();

        const auto t2 = std::chrono::steady_clock::now();
        st.device_ms =
            std::chrono::duration<double, std::milli>(t2 - t1).count();
        return st;
    }

  public:
    // Ask the GPU to finish before the staging buffer is next overwritten.
    // page_in_all()/page_in_group() wait already; this exists for a caller that
    // pipelines several page-ins into one submit.
    void wait_idle() { dev_->wait_all_done(); }

  private:
    std::shared_ptr<Runtime> rt_;
    std::shared_ptr<VulkanCommandPool> pool_;
    std::shared_ptr<VulkanDevice> dev_;
    std::unordered_map<std::string, PagedSource> by_name_;
    // Declaration order, for groups() and page_in_all(). Holds copies so a
    // re-declare does not reorder an existing band.
    std::vector<std::pair<std::string, PagedSource>> order_;
    std::shared_ptr<VulkanBuffer> staging_buf_;
};

} // namespace core
} // namespace vkop

#endif // CORE_WEIGHT_PAGER_HPP_
