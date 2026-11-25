// Proof that a resident weight can be REPLACED IN PLACE, mid-session, and that
// the model then computes exactly what it would have computed had that weight
// been loaded at startup.
//
// This is the load-bearing assumption behind layer paging on a small-VRAM card:
// if swapping a weight's bytes into the SAME VkBuffer is numerically identical
// to a fresh LoadModel, then paging needs no new Vulkan capability -- only a
// scheduler. If it is not identical, no amount of scheduling helps.
//
// The test builds two artifacts from the SAME graph whose only difference is
// the MatMul weight's bytes (model_a / model_b, converted separately). It then:
//
//   1. runs model_a and model_b in their own Runtimes, recording each output
//      as ground truth (load-time residency);
//   2. loads model_a, runs it, and checks the output matches ground truth A;
//   3. stages model_b's weight bytes and vkCmdCopyBuffers them over model_a's
//      live weight buffer, WITHOUT touching the tensor object, the VkBuffer
//      handle, or any descriptor;
//   4. runs again and requires a bit-exact match with ground truth B.
//
// Step 4 is the proof. Step 3's constraints (same handle, same offset) are what
// segment replay and the BDA address table require, so the check also asserts
// the handle is unchanged -- a "correct" result produced by reallocating would
// pass the numbers and fail the thing paging actually needs.
//
// It also times the transfer so the PCIe/DRAM floor is measured rather than
// assumed: decode pays this per token for every spilled byte.
//
//   probes weight-paging <model_a.vkopbin> <model_b.vkopbin> [name] [reps]
//
// `name` is the weight initializer (default "w").

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <string>
#include <vector>

#include "core/Tensor.hpp"
#include "core/runtime.hpp"
#include "include/logger.hpp"
#include "model/load.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanCommandBuffer.hpp"
#include "vulkan/VulkanCommandPool.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "vulkan/VulkanStagingBufferPool.hpp"

#if __has_include("probe_common.hpp")
#include "probe_common.hpp"
#define WP_HAS_PROBE_COMMON 1
#endif

using vkop::VulkanCommandBuffer;
using vkop::VulkanCommandPool;
using vkop::VulkanDevice;
using vkop::VulkanInstance;
using vkop::VulkanBuffer;
using vkop::core::ITensor;
using vkop::core::Runtime;

namespace vkop {
namespace probes {
namespace weight_paging {

namespace {

double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
               clock::now().time_since_epoch())
        .count();
}

// The first graph input is fed a fixed ramp so both arms see the same
// activation; only the weight differs.
void fill_input(const std::shared_ptr<Runtime> &rt,
                const std::shared_ptr<VulkanCommandPool> &pool,
                uint32_t seed) {
    auto t = rt->GetInput("x");
    if (!t) {
        throw std::runtime_error("graph has no input named 'x'");
    }
    const auto dims = t->getShape();
    size_t n = 1;
    for (int d : dims) n *= static_cast<size_t>(d);
    std::uint32_t s = seed;
    auto rnd = [&s]() {
        s = s * 1664525u + 1013904223u;
        return (static_cast<float>((s >> 8) & 0xFFFF) / 65535.0f) - 0.5f;
    };
    if (auto f = vkop::core::as_tensor<float>(t)) {
        std::vector<float> v(n);
        for (auto &x : v) x = rnd();
        f->fillToCPU(v);
        f->copyToGPU(pool);
    } else if (auto h = vkop::core::as_tensor<uint16_t>(t)) {
        std::vector<uint16_t> v(n);
        for (auto &x : v) x = ITensor::fp32_to_fp16(rnd());
        h->fillToCPU(v);
        h->copyToGPU(pool);
    } else {
        throw std::runtime_error("input 'x' is neither fp32 nor fp16");
    }
}

std::vector<uint16_t> read_output_fp16(
    const std::shared_ptr<Runtime> &rt,
    const std::shared_ptr<VulkanCommandPool> &pool) {
    auto out = rt->GetOutput();
    if (!out) throw std::runtime_error("model has no output");
    auto h = vkop::core::as_tensor<uint16_t>(out);
    if (!h) throw std::runtime_error("output is not fp16");
    h->copyToCPU(pool);
    return h->data();
}

struct Arm {
    std::vector<uint16_t> out;
    bool ok = false;
};

Arm run_artefact(const std::shared_ptr<VulkanDevice> &dev,
                 const std::shared_ptr<VulkanCommandPool> &pool,
                 const std::string &path) {
    Arm a;
    auto rt = std::make_shared<Runtime>(pool, path, /*precision=*/0);
    rt->set_backend_buffer(true); // deterministic SSBO path; no image layout
    rt->LoadModel();
    fill_input(rt, pool, 0xC0FFEEu);
    rt->Run();
    a.out = read_output_fp16(rt, pool);
    a.ok = true;
    return a;
}

int mismatches(const std::vector<uint16_t> &a, const std::vector<uint16_t> &b) {
    if (a.size() != b.size()) return -1;
    int bad = 0;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) ++bad;
    return bad;
}

// Max |a-b| in the fp16 domain, so a "close but not identical" result is still
// reported usefully rather than as a bare count.
double max_abs_diff(const std::vector<uint16_t> &a,
                    const std::vector<uint16_t> &b) {
    double m = 0.0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs(static_cast<double>(ITensor::fp16_to_fp32(a[i])) -
                                   static_cast<double>(ITensor::fp16_to_fp32(b[i])));
        if (d > m) m = d;
    }
    return m;
}

int run(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: probes weight-paging <model_a.vkopbin> "
                     "<model_b.vkopbin> [weight_name] [reps]\n");
        return 2;
    }
    const std::string path_a = argv[1];
    const std::string path_b = argv[2];
    const std::string wname = argc >= 4 ? argv[3] : "w";
    const int reps = argc >= 5 ? std::atoi(argv[4]) : 5;
    // VKOP_PAGING_MMAP=1 switches the source read from pread to a memcpy out of
    // the model's mmap, so the two can be compared on the same machine.
    const char *mmap_env = std::getenv("VKOP_PAGING_MMAP");
    const bool use_pread = !(mmap_env && mmap_env[0] == '1');

    Logger::getInstance().setLevel(LOG_WARNING);
    const auto &phydevs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
    auto dev = std::make_shared<VulkanDevice>(phydevs[0]);
    if (dev->getDeviceName().find("llvmpipe") != std::string::npos) {
        std::printf("no valid vulkan device\n");
        return 1;
    }
    auto pool = std::make_shared<VulkanCommandPool>(dev);
    std::printf("[wp] device: %s\n", dev->getDeviceName().c_str());

    // ---- 1. ground truth: each artifact with its own weight resident ----
    const Arm gt_a = run_artefact(dev, pool, path_a);
    const Arm gt_b = run_artefact(dev, pool, path_b);
    if (mismatches(gt_a.out, gt_b.out) <= 0) {
        std::printf("[wp] FAIL: the two artifacts compute the same output; the "
                    "weights did not differ\n");
        return 1;
    }
    std::printf("[wp] ground truth: A and B differ as expected (%d of %zu "
                "elements)\n",
                mismatches(gt_a.out, gt_b.out), gt_a.out.size());

    // ---- 2. the paging arm: model_a, then overwrite its weight bytes ----
    auto rt = std::make_shared<Runtime>(pool, path_a, /*precision=*/0);
    rt->set_backend_buffer(true);
    rt->LoadModel();
    fill_input(rt, pool, 0xC0FFEEu);
    rt->Run();
    auto before = read_output_fp16(rt, pool);
    if (mismatches(before, gt_a.out) != 0) {
        std::printf("[wp] FAIL: baseline run of A does not reproduce its own "
                    "ground truth (%d mismatches)\n",
                    mismatches(before, gt_a.out));
        return 1;
    }
    std::printf("[wp] baseline: model_a reproduces ground truth A exactly\n");

    // Warm-up rounds: with VKOP_SEGMENT_REPLAY=1 this is what drives the
    // segment cache past its warm-up so segments genuinely HIT and skip their
    // recording. Without them a replay run would only ever be in the
    // miss/record phase and could not expose a stale-weight replay.
    if (const char *w = std::getenv("VKOP_PAGING_WARMUP")) {
        const int n = std::atoi(w);
        for (int i = 0; i < n; ++i) rt->Run();
        auto warm = read_output_fp16(rt, pool);
        if (mismatches(warm, gt_a.out) != 0) {
            std::printf("[wp] FAIL: warm-up rounds diverged from ground truth A "
                        "(%d mismatches)\n", mismatches(warm, gt_a.out));
            return 1;
        }
        std::printf("[wp] warm-up: %d round(s), still exactly A\n", n);
    }

    // Every weight B replaces: `w`, or `w0..wL-1` for the chain graph. Paging
    // is per-layer, so the multi-weight case is the one that matters -- a
    // single-weight pass would not show a stale second layer.
    vkop::load::VkModel mb(path_b);
    std::vector<std::string> names;
    if (mb.initializers.count(wname)) {
        names.push_back(wname);
    } else if (mb.initializers.count(wname + "0")) {
        for (int i = 0;; ++i) {
            const std::string n = wname + std::to_string(i);
            if (!mb.initializers.count(n)) break;
            names.push_back(n);
        }
    }
    if (names.empty()) {
        std::printf("[wp] FAIL: %s has no initializer '%s' (or '%s0..')\n",
                    path_b.c_str(), wname.c_str(), wname.c_str());
        return 1;
    }
    std::printf("[wp] %zu weight(s) to replace in place\n", names.size());

    // Resolve every source -> destination pair up front, and pin the handle of
    // each destination: an in-place rewrite must not change it.
    struct Pair {
        std::string name;
        const uint8_t *src;
        size_t src_file_offset;  // absolute, for the pread arm
        size_t bytes;
        std::shared_ptr<VulkanBuffer> dst;
        VkBuffer handle;
    };
    std::vector<Pair> pairs;
    size_t total_bytes = 0;
    for (const auto &n : names) {
        const auto &ib = mb.initializers.at(n);
        const size_t nb = static_cast<size_t>(ib.size);
        auto wt = rt->GetInitializer(n);
        if (!wt) {
            std::printf("[wp] FAIL: runtime has no resident initializer '%s'\n",
                        n.c_str());
            return 1;
        }
        auto vb = std::dynamic_pointer_cast<VulkanBuffer>(wt->gpu_buffer());
        if (!vb) {
            std::printf("[wp] FAIL: '%s' GPU resource is not a VulkanBuffer\n",
                        n.c_str());
            return 1;
        }
        if (nb > vb->getSize()) {
            std::printf("[wp] FAIL: '%s' source payload %zu > buffer %llu\n",
                        n.c_str(), nb,
                        static_cast<unsigned long long>(vb->getSize()));
            return 1;
        }
        pairs.push_back({n, mb.initializer_memory + mb.initializer_offsets.at(n),
                         mb.initializer_file_offset +
                             mb.initializer_offsets.at(n),
                         nb, vb, vb->getBuffer()});
        total_bytes += nb;
    }
    std::printf("[wp] %zu bytes across %zu buffer(s)\n", total_bytes,
                pairs.size());

    // Stage and copy -- this is the whole "page a weight in" primitive:
    // transfer dst usage (already set by as_storage_buffer), an offset copy,
    // and the visibility barrier the copy helper emits. All copies for a round
    // go into ONE command buffer and ONE submit+wait, which is the shape a
    // layer-band prefetch has to take.
    //
    // Two staging strategies, because they measure very different things:
    //   pool  -- ask the staging pool per round. Above its 64 MiB cap that is a
    //            one-shot vkCreateBuffer + vkAllocateMemory EVERY round, which
    //            is a cost of this probe's loop, not of paging.
    //   pinned-- one long-lived staging buffer reused for every round, i.e.
    //            what a real pager does (and what double-buffering requires).
    // Default is pinned; VKOP_PAGING_POOL=1 selects the pool arm.
    auto stpool = pool->getStagingBufferPool();
    const char *pool_env = std::getenv("VKOP_PAGING_POOL");
#ifdef WP_STAGING_HAS_OWNER
    const bool use_pool = pool_env && pool_env[0] == '1';
#else
    // Older trees' StagingAllocation has no `owner`, so the pool arm cannot
    // flush a host-written staging range. Pinned staging is the arm that
    // matters (it is what a pager uses); refuse the pool arm loudly.
    const bool use_pool = false;
    if (pool_env && pool_env[0] == '1')
        std::printf("[wp] note: this tree has no StagingAllocation::owner; "
                    "VKOP_PAGING_POOL ignored\n");
#endif
    std::shared_ptr<VulkanBuffer> pinned;
    if (!use_pool) {
        auto &dev_mut = const_cast<std::shared_ptr<VulkanDevice> &>(dev);
        pinned = std::make_shared<VulkanBuffer>(
            dev_mut, total_bytes,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        std::printf("[wp] staging: one pinned %zu-byte buffer, reused\n",
                    total_bytes);
    } else {
        std::printf("[wp] staging: staging-pool allocation per round\n");
    }
    double total_ms = 0.0, host_ms = 0.0, gpu_ms = 0.0;
    for (int r = 0; r < reps; ++r) {
        const double t0 = now_ms();
        std::vector<std::optional<vkop::StagingAllocation>> allocs;
        allocs.reserve(pairs.size());
        std::vector<std::pair<VkBuffer, VkDeviceSize>> staged;
        staged.reserve(pairs.size());
        // Host half: get staging, move the bytes in, flush.
        // Split out because a real pager streams file bytes straight into the
        // staging buffer (copyToGPU's StagingFillFn) and never pays this copy;
        // charging it to "paging bandwidth" would understate the GPU side.
        size_t off = 0;
        for (const auto &p : pairs) {
            void *dst = nullptr;
            VkBuffer sbuf = VK_NULL_HANDLE;
            VkDeviceSize soff = 0;
            if (use_pool) {
                auto a = stpool->allocate(p.bytes);
                if (!a) {
                    std::printf("[wp] FAIL: staging allocation of %zu bytes "
                                "refused\n", p.bytes);
                    return 1;
                }
                dst = a->ptr;
                sbuf = a->buffer;
                soff = a->offset;
                allocs.push_back(a);
            } else {
                dst = static_cast<char *>(pinned->getMappedMemory()) + off;
                sbuf = pinned->getBuffer();
                soff = off;
            }
            // pread the weight bytes straight into the staging buffer, the way
            // copyToGPU's StagingFillFn does. This is the difference between a
            // pager (one copy, file -> staging) and a memcpy out of the mmap
            // (page fault, then copy). The mmap arm is kept for the A/B.
            if (use_pread) {
                mb.fileMapping().read_at(dst, p.bytes, p.src_file_offset);
            } else {
                std::memcpy(dst, p.src, p.bytes);
            }
            if (use_pool) {
#ifdef WP_STAGING_HAS_OWNER
                allocs.back()->owner->flushMappedRange(soff, p.bytes);
#endif
            } else {
                pinned->flushMappedRange(soff, p.bytes);
            }
            staged.emplace_back(sbuf, soff);
            off += p.bytes;
        }
        const double t1 = now_ms();
        {
            VulkanCommandBuffer cmd(pool);
            cmd.begin();
            for (size_t i = 0; i < pairs.size(); ++i) {
                const auto &p = pairs[i];
                // Set the tracked access to SHADER_READ *before* the copy so
                // the helper's closing barrier hands the buffer back as
                // readable instead of restoring access 0 (the BDA table's
                // documented trap).
                p.dst->readBarrier(cmd.get(), p.bytes, 0);
                p.dst->copyStageBufferToBuffer(cmd.get(), staged[i].first,
                                               staged[i].second, p.bytes, 0);
            }
            cmd.end();
            cmd.submit(dev->getComputeQueue());
            cmd.wait();
        }
        const double t2 = now_ms();
        host_ms += t1 - t0;
        gpu_ms += t2 - t1;
        total_ms += t2 - t0;
        std::printf("[wp] rewrite %d/%d: %.3f ms total (host %.3f + gpu %.3f) "
                    "-> %.1f GB/s gpu-only\n",
                    r + 1, reps, t2 - t0, t1 - t0, t2 - t1,
                    (t2 - t1) > 0 ? (static_cast<double>(total_bytes) / 1e9) /
                                        ((t2 - t1) / 1e3)
                                  : 0.0);
        stpool->reset();
    }

    std::printf("[wp] mean: %.3f ms total = %.3f host + %.3f gpu; "
                "%.1f GB/s end-to-end, %.1f GB/s gpu-only\n",
                total_ms / reps, host_ms / reps, gpu_ms / reps,
                (static_cast<double>(total_bytes) / 1e9) /
                    ((total_ms / reps) / 1e3),
                (static_cast<double>(total_bytes) / 1e9) /
                    ((gpu_ms / reps) / 1e3));

    // ---- 3. the proof ----
    for (const auto &p : pairs) {
        if (p.dst->getBuffer() != p.handle) {
            std::printf("[wp] FAIL: '%s' buffer handle changed; this is not an "
                        "in-place rewrite\n", p.name.c_str());
            return 1;
        }
    }
    rt->Run();
    auto after = read_output_fp16(rt, pool);
    const int bad_b = mismatches(after, gt_b.out);
    const int bad_a = mismatches(after, gt_a.out);
    std::printf("[wp] after rewrite: vs ground truth B -> %d mismatch(es), "
                "max|d| %.6g; vs A -> %d\n",
                bad_b, max_abs_diff(after, gt_b.out), bad_a);

    if (bad_a == 0) {
        std::printf("[wp] FAIL: output still matches A -- the rewrite had no "
                    "effect (stale binding or invisible write)\n");
        return 1;
    }
    if (bad_b != 0) {
        std::printf("[wp] FAIL: rewritten model does NOT reproduce the "
                    "load-time result for B\n");
        return 1;
    }
    std::printf("[wp] PASS: in-place weight replacement is numerically "
                "identical to a fresh LoadModel, same VkBuffer handle\n");

    // ---- 4. round trip: page A's bytes back and require an exact return ----
    // A pager swaps a band out and a different band in, round after round. If
    // the swap were not a pure byte overwrite (a partial copy, a drifting
    // offset, a barrier that lets a stale read through) this is where it shows
    // up: the second visit to A must be bit-identical to the first.
    {
        vkop::load::VkModel ma(path_a);
        size_t off = 0;
        for (const auto &p : pairs) {
            ma.fileMapping().read_at(
                static_cast<char *>(pinned->getMappedMemory()) + off, p.bytes,
                ma.initializer_file_offset + ma.initializer_offsets.at(p.name));
            pinned->flushMappedRange(off, p.bytes);
            off += p.bytes;
        }
        VulkanCommandBuffer cmd(pool);
        cmd.begin();
        size_t so = 0;
        for (const auto &p : pairs) {
            p.dst->readBarrier(cmd.get(), p.bytes, 0);
            p.dst->copyStageBufferToBuffer(cmd.get(), pinned->getBuffer(), so,
                                           p.bytes, 0);
            so += p.bytes;
        }
        cmd.end();
        cmd.submit(dev->getComputeQueue());
        cmd.wait();
        rt->Run();
        auto back = read_output_fp16(rt, pool);
        const int bad = mismatches(back, gt_a.out);
        if (bad != 0) {
            std::printf("[wp] FAIL: swap-back to A is not exact (%d mismatch(es)"
                        ") -- paging accumulates drift\n", bad);
            return 1;
        }
        std::printf("[wp] PASS: swap-back to A reproduces it exactly; repeated "
                    "page-in is a pure byte overwrite\n");
    }

    // ---- 5. double-buffered overlap: the pager's actual value claim ----
    // A real pager holds TWO bands: while band N's GPU copy is in flight, the
    // host prefetches band N+1 into the other staging buffer. Whether that
    // actually hides the host half depends on the machine:
    //   * UMA (this ARL box): host and device share DRAM, so "overlap" just
    //     splits one bandwidth budget -- expect NO win.
    //   * discrete + PCIe: host DRAM writes and PCIe reads are different
    //     links, so the overlap should be close to free.
    // This measures the serial arm and a 2-thread pipelined arm on the same
    // bytes so the difference is attributable to the machine, not the code.
    {
        auto &dev_mut = const_cast<std::shared_ptr<VulkanDevice> &>(dev);
        auto make_band = [&]() {
            return std::make_shared<VulkanBuffer>(
                dev_mut, total_bytes,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        };
        auto band0 = make_band();
        auto band1 = make_band();
        vkop::load::VkModel ma2(path_a);
        vkop::load::VkModel mb2(path_b);

        // Prefetch `m`'s bytes into `band`; returns the band's total offset.
        auto prefetch = [&](VulkanBuffer &band, vkop::load::VkModel &m) {
            size_t off = 0;
            for (const auto &p : pairs) {
                m.fileMapping().read_at(
                    static_cast<char *>(band.getMappedMemory()) + off, p.bytes,
                    m.initializer_file_offset +
                        m.initializer_offsets.at(p.name));
                off += p.bytes;
            }
            band.flushMappedRange(0, off);
            return off;
        };
        // Copy `band` over the live weights, with per-pair staging offsets.
        auto copy_band = [&](VulkanBuffer &band) {
            VulkanCommandBuffer cmd(pool);
            cmd.begin();
            size_t off = 0;
            for (const auto &p : pairs) {
                p.dst->readBarrier(cmd.get(), p.bytes, 0);
                p.dst->copyStageBufferToBuffer(cmd.get(), band.getBuffer(), off,
                                               p.bytes, 0);
                off += p.bytes;
            }
            cmd.end();
            cmd.submit(dev->getComputeQueue());
            cmd.wait();
        };

        // Serial: prefetch, copy, repeat. This is the number the main loop
        // above already reports; repeated here so both arms share one clock.
        const int dbl_reps = std::max(2, reps);
        double serial_ms = 0.0;
        for (int r = 0; r < dbl_reps; ++r) {
            const double t0 = now_ms();
            prefetch(*band0, (r % 2 == 0) ? mb2 : ma2);
            copy_band(*band0);
            serial_ms += now_ms() - t0;
        }

        // Pipelined: two threads, two bands. The producer prefetches band N+1
        // while the consumer thread is blocked in band N's submit+wait. That
        // is the exact shape of a 2-deep paging ring, and it is the only way to
        // tell whether the host half is actually hideable on this machine.
        double pipe_ms = 0.0;
        {
            // Counting semaphore, C++17 has none.
            struct Sem {
                std::mutex m;
                std::condition_variable cv;
                int n = 0;
                void post() {
                    { std::lock_guard<std::mutex> g(m); ++n; }
                    cv.notify_one();
                }
                void wait() {
                    std::unique_lock<std::mutex> lk(m);
                    cv.wait(lk, [&] { return n > 0; });
                    --n;
                }
            };
            Sem slot_free[2];
            Sem slot_ready[2];
            slot_free[0].post();
            slot_free[1].post();
            std::atomic<bool> stop{false};

            std::thread producer([&] {
                for (int i = 0; i < dbl_reps && !stop.load(); ++i) {
                    const int s = i % 2;
                    slot_free[s].wait();
                    if (stop.load()) break;
                    prefetch(s == 0 ? *band0 : *band1, i % 2 == 0 ? mb2 : ma2);
                    slot_ready[s].post();
                }
            });
            for (int i = 0; i < dbl_reps; ++i) {
                const int s = i % 2;
                const double t0 = now_ms();
                slot_ready[s].wait();
                copy_band(s == 0 ? *band0 : *band1);
                pipe_ms += now_ms() - t0;
                slot_free[s].post();
            }
            stop.store(true);
            // Release the producer if it is parked on a slot_free wait.
            slot_free[0].post();
            slot_free[1].post();
            producer.join();
        }
        std::printf("[wp] overlap A/B (%d bands): serial %.3f ms/band, "
                    "2-thread pipelined %.3f ms/band -> %.2fx\n",
                    dbl_reps, serial_ms / dbl_reps, pipe_ms / dbl_reps,
                    pipe_ms > 0 ? (serial_ms / dbl_reps) / (pipe_ms / dbl_reps)
                                : 0.0);
    }

    std::fflush(stdout);
    _exit(0); // b3's NVIDIA EGL atexit handler segfaults; probes _exit
}

#ifdef WP_HAS_PROBE_COMMON
REGISTER_PROBE("weight-paging",
               "in-place weight replacement == fresh LoadModel (paging proof)",
               nullptr, run);
#endif

} // namespace
} // namespace weight_paging
} // namespace probes
} // namespace vkop

// Standalone build (an older tree with no tools/probes dispatcher):
//   g++ -x c++ tools/probes/weight_paging_probe.cpp -o weight-paging
//       -I<src> -I<src>/model -I<src>/model/SYSTEM -I<src>/model/generated
//       <build>/libvkop.a <build>/model/libvload.a -ldl -lpthread
#ifndef WP_HAS_PROBE_COMMON
int main(int argc, char **argv) {
    return vkop::probes::weight_paging::run(argc, argv);
}
#endif
