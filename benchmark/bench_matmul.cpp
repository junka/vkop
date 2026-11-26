// Standalone MatMul GEMV microbenchmark (decode-shape fp16, buffer backend).
//
// Drives the real MatMul op through ops::create_from_type, records once via
// the replay cache (two onExecute rounds -> CACHED) with a GPU timestamp pair
// inside the recording, then re-submits the kept SIMULTANEOUS_USE cmd buffer R
// times and waits once. Two rows come out of that loop: `gpu` (the last replay's
// own timestamp span = one dispatch's kernel time) and `wall` (the same submits
// as seen by the host, so it also contains R vkQueueSubmit calls). Only the gpu
// row is a bandwidth measurement: on a 560X behind MoltenVK the 4 MB decode
// shapes are submit-bound, with wall 1.7-2.4x the kernel span.
//
// Build (mirrors the tests link line: libvkop + libvload; libvkop carries no
// SPIR-V, so -DVKOP_DEV_SHADER_ARCHIVE must point at build/vkop_shaders.bin):
//   /usr/local/bin/g++-14 -nostdinc++
//     -isystem /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1
//     -isystem /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include
//     -O2 -std=gnu++17 -mf16c -DUSE_VMA -DVK_ENABLE_BETA_EXTENSIONS
//     -DVKOP_DEV_SHADER_ARCHIVE=\"/Users/uqland/github/vkop/build/vkop_shaders.bin\"
//     -I/usr/local/include -I/Users/uqland/github/vkop -I/Users/uqland/github/vkop/model
//     -I/Users/uqland/github/vkop/model/SYSTEM -I/Users/uqland/github/vkop/model/generated
//     benchmark/bench_matmul.cpp -o build/bench_matmul
//     build/libvkop.a build/model/libvload.a -lc++
//
// Run (x86_64 binary -> the /usr/local Vulkan SDK stack, not /opt/homebrew):
//   DYLD_FALLBACK_LIBRARY_PATH=/usr/local/lib
//   VK_ICD_FILENAMES=/usr/local/etc/vulkan/icd.d/MoltenVK_icd.json
//   VKOP_NO_COOP=1 ./build/bench_matmul
//
// Env: VKOP_NO_COOP=1 on MoltenVK (16x16 coopmat tiles cannot compile to MSL).
//      BENCH_DEV=<n> picks the physical device (default 0).
//      BENCH_INT8=1 swaps B for a byte int8 weight + a per-column fp32 scale,
//      which is the only way to exercise the byte WIDE (16 cols/lane) GEMV
//      without a whole quantized model on disk.
//      BENCH_BF16=1 marks the same [K, N] weight as bfloat16 (raw bf16 bits,
//      elem_kind kBFloat16): same bytes and routes as fp16, only the B unpack
//      differs, so the pair is the A/B for the bf16 read.
//      BENCH_BF16_WIDE=1 (with BENCH_BF16) sets the bf16_wide attribute, which
//      keeps the bf16 weight off the half-staging tile routes.
//      BENCH_W4=int4|nf4|uint4|nvfp4 swaps B for a nibble-packed 4-bit
//      weight (group 64 fp32 scales; nvfp4: 16-value blocks of e4m3 scales
//      plus a global fp32 scale). GB/s counts the nibble payload only.
//      BENCH_W4_GROUP=<g> overrides the int4/nf4/uint4 group (default 64).
//      BENCH_GPU_MED=1 adds a median-of-reps per-replay GPU span row.
//      BENCH_PATTERN=1 times the column-ramp weight instead of refilling it
//      with incompressible bytes after the correctness probe. The A/B against
//      the default; on a part that does not compress storage-buffer reads the
//      two agree (measured on the 560X).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "vulkan/VulkanInstance.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanCommandPool.hpp"
#include "vulkan/VulkanCommandBuffer.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanQueryPool.hpp"
#include "core/Tensor.hpp"
#include "ops/OperatorFactory.hpp"
#include "include/dev_shaders.hpp"
#include "include/logger.hpp"

using namespace vkop;

// GCC-14 at -O2 emits calls to __cxa_call_terminate from noexcept cleanup
// paths inside libc++ headers. The system libc++.1.dylib (1900.x) predates
// that ABI symbol (llm_chat never hits it because CMake builds everything at
// -O0). Provide it: the real one just routes to the terminate handler, and no
// noexcept function here ever throws in this harness.
extern "C" void __cxa_call_terminate(void *) { abort(); }

namespace {

struct Shape {
    int K, N;
    const char *name;
};

void fill_ones(const std::shared_ptr<core::Tensor<uint16_t>> &t) {
    t->reserveOnCPU();
    std::fill(t->data().begin(), t->data().end(), 0x3C00); // fp16 1.0
}

// Column-only weight ramp: B[k][j] = wpat(j), so with A all ones every output
// element is exactly K * wpat(j) -- independent of k, and cheap enough to
// check against a one-line reference at any size. Unlike an all-ones B this is
// sensitive to the COLUMN indexing, which is the thing a ksplit_q change can
// break: the fold remaps (column quad, K slice) through oct/slice, and the
// byte kernel additionally applies a per-column scale at a c+4/c+8/c+12
// offset. A remap or a scale phase error makes out[j] come out as K*wpat(j')
// for the wrong j', which the all-ones probe cannot see.
inline int wpat(int j) { return (j % 7) - 3; }

// 4-bit probe: the column-only nibble code and its decoded value (scale 1).
const float kNf4[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f,
    0.33791524171829224f, 0.44070982933044434f, 0.5626170039176941f,
    0.7229568362236023f, 1.0f};
const float kE2m1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
inline int w4code(int j) { return j % 16; }
float w4dec(const char *fmt, int c) {
    if (!strcmp(fmt, "nf4")) return kNf4[c];
    if (!strcmp(fmt, "uint4")) return static_cast<float>(c);
    if (!strcmp(fmt, "nvfp4")) return (c & 8) ? -kE2m1[c & 7] : kE2m1[c & 7];
    return static_cast<float>(c >= 8 ? c - 16 : c);
}

// The timed weight is incompressible by default. The ramp is seven repeated
// values per column, which is not a bandwidth stimulus on a part that compresses
// memory: it does not cost its size in DRAM traffic. So the probe runs on the
// ramp (where K*wpat(j) is an exact reference) and the weight is then refilled
// from an LCG and re-uploaded into the SAME buffer the recorded cmd binds,
// before anything is timed.
//
// Measured on the 560X: the refill made NO difference outside run-to-run
// variance (4 MB int8: 0.230/0.217 ms random vs 0.237/0.232 ms ramp; 33.5 MB:
// 0.825/0.510 vs 0.923/0.843). GCN's DCC compresses images, not storage-buffer
// reads, so this card was never compressing the ramp. The earlier "104 GB/s
// with the ramp vs 27.5 GB/s random" reading was NOT a compression effect -- it
// came from the broken bracketing timestamp pair described at the timing loop.
// The default stands because it removes the assumption, not because a local
// artifact was measured.
//
// Variance caveat: incompressible bytes did not remove the run-to-run spread at
// 500 MB-1 GB either, so take the min over several runs rather than attributing
// all of it to data content.
void fill_random_i8(const std::shared_ptr<core::Tensor<int8_t>> &t) {
    // copyToGPU clears AND shrinks the host vector once the upload is done, so
    // the probe's upload left data_ empty: without this resize the refill writes
    // nothing and the second upload memcpys from a null pointer.
    t->data().resize(static_cast<size_t>(t->num_elements()));
    uint32_t st = 0x12345678u;
    int8_t *p = t->data().data();
    for (size_t i = 0; i < t->data().size(); ++i) {
        st = st * 1664525u + 1013904223u;
        p[i] = static_cast<int8_t>((st >> 16) & 0xFF);
    }
}

// fp16 gets random mantissas under a FIXED normal exponent, not random bits:
// random halfwords are mostly NaN/Inf/denormal, and a denormal or NaN operand
// can cost different ALU time than the normal values a real weight holds.
// 0x3C00 is exponent 15 (bias 15), so every value is normal in [1,2) or [-2,-1).
void fill_random_f16(const std::shared_ptr<core::Tensor<uint16_t>> &t) {
    t->data().resize(static_cast<size_t>(t->num_elements())); // see fill_random_i8
    uint32_t st = 0x12345678u;
    uint16_t *p = t->data().data();
    for (size_t i = 0; i < t->data().size(); ++i) {
        st = st * 1664525u + 1013904223u;
        p[i] = static_cast<uint16_t>(0x3C00 | ((st >> 16) & 0x03FF) |
                                     ((st >> 31) << 15));
    }
}

} // namespace

int main(int argc, char **argv) {
    Logger::getInstance().setLevel(LOG_ERROR);
    vkop::dev::seed_shader_archive_from_build_tree();
    const auto &phydevs =
        VulkanInstance::getVulkanInstance().getPhysicalDevices();
    if (phydevs.empty()) {
        printf("no vulkan device\n");
        return 1;
    }
    // BENCH_DEV picks the physical device (0 = first enumerated). This machine
    // lists the AMD dGPU first and the Intel iGPU second.
    const char *dev_env = std::getenv("BENCH_DEV");
    const int dev_idx = dev_env ? std::atoi(dev_env) : 0;
    if (dev_idx < 0 || dev_idx >= static_cast<int>(phydevs.size())) {
        printf("no vulkan device %d\n", dev_idx);
        return 1;
    }
    auto dev = std::make_shared<VulkanDevice>(phydevs[dev_idx]);
    printf("GPU: %s\n", dev->getDeviceName().c_str());
    auto cmdpool = std::make_shared<VulkanCommandPool>(dev);
    auto queue = dev->getComputeQueue(0);

    // Qwen3-2B decode shapes: A is one hidden row, B is the projection weight
    // [K, N] (transB == 0). All must hit the ksplit GEMV gate: n%4==0, k%2==0,
    // batch*m<=8, k>=256.
    std::vector<Shape> shapes = {
        {2048, 2048, "q/k/v/o"},
        {2048, 6144, "gate/up"},
        {6144, 2048, "down"},
        {2048, 151936, "lm_head"},
    };
    // BENCH_SWEEP=1 replaces the table with a working-set ladder that crosses
    // the L2 boundary. This matters more than it looks: the default shapes and
    // any single mid-size run measure a weight that fits (or nearly fits) in
    // L2 and is replayed reps times, so they report L2 bandwidth and not DRAM.
    // On a 48 MB-L2 Ada part the same K=5120 GEMV measured 250 GB/s at N=10240
    // (50 MB) and ~50 GB/s at N=102400 (500 MB) -- a 5x difference that is
    // entirely working-set, and that made a model running at its real DRAM
    // rate look 3x slower than "the kernel could do".
    //
    // Sizes bracket a 48 MB L2 from half of it to 42x at int8 (fp16 doubles
    // them): 25 MB, 50 MB, 200 MB, 500 MB, 1 GB, 2 GB. Every entry keeps
    // N % 16 == 0 so the byte WIDE route stays eligible; the tag names the
    // working set in L2s, which is the variable that actually changes. Note
    // the measurements are NOT monotonic in size -- 500 MB and 1 GB have both
    // come out slower than 2 GB in separate runs -- so read the ladder as
    // "where does the number stop being L2", not as a curve.
    if (std::getenv("BENCH_SWEEP") != nullptr) {
        shapes.clear();
        const int k = 5120;
        // int8 weight bytes = K*N; the tag says how many 48 MB L2s that is.
        static const char *kSweepNames[] = {"L2x0.5", "L2x1", "L2x4",
                                            "L2x10", "L2x20", "L2x40"};
        const int kN[] = {5120, 10240, 40960, 102400, 204800, 409600};
        for (int i = 0; i < 6; ++i) {
            shapes.push_back({k, kN[i], kSweepNames[i]});
        }
    }
    if (argc > 1) {
        // Override: bench_matmul K N [reps]
        shapes.clear();
        shapes.push_back({std::atoi(argv[1]), std::atoi(argv[2]), "custom"});
    }

    const int reps = (argc > 3) ? std::atoi(argv[3]) : 30;
    const bool int8_mode = std::getenv("BENCH_INT8") != nullptr;
    const bool bf16_mode = std::getenv("BENCH_BF16") != nullptr;
    const char *w4_fmt = std::getenv("BENCH_W4");
    const bool w4_mode = w4_fmt != nullptr && *w4_fmt;
    const int w4_group = [] {
        const char *v = std::getenv("BENCH_W4_GROUP");
        return (v && *v) ? std::atoi(v) : 64;
    }();
    if (w4_mode) {
        printf("weight: %s (4-bit, group %d)\n", w4_fmt,
               strcmp(w4_fmt, "nvfp4") ? w4_group : 16);
    }
    if (int8_mode) {
        printf("weight: int8 (byte WIDE path)\n");
    }
    printf("weight fill: %s\n",
           std::getenv("BENCH_PATTERN") != nullptr
               ? "column ramp (compressible; probe reference)"
               : "LCG random (incompressible; refilled after the probe)");

    // DRAM ceiling probe: vkCmdCopyBuffer of a 256MB region, same submit
    // batch as the GEMV measurements. Read+write so effective bandwidth is
    // bytes_moved/2/s per direction.
    //
    // Skip with BENCH_SKIP_MEMPROBE=1 when only the GEMV rows matter: moving
    // 256MB x reps immediately before the measured window puts the iGPU under
    // a sustained memory load that its power management reacts to, and a run
    // that inherits a partial throttle measures the matmul ~2x slow at exactly
    // the sizes where the window is longest. The min-of-N is then bimodal
    // (one un-throttled round). Nothing in the probe is needed for the GEMV
    // A/B, so drop it.
    if (std::getenv("BENCH_SKIP_MEMPROBE") == nullptr) {
        // 256 MB by default, which is 5x a 48 MB L2 -- enough on this class of
        // part, but BENCH_MEMPROBE_MB raises it for a clean DRAM number (a
        // 1 GB copy leaves no doubt that L2 cannot hold the working set).
        size_t mb = 256;
        if (const char *e = std::getenv("BENCH_MEMPROBE_MB")) {
            const int v = std::atoi(e);
            if (v > 0) {
                mb = static_cast<size_t>(v);
            }
        }
        const size_t bytes = mb << 20;
        auto src = std::make_shared<VulkanBuffer>(
            dev, bytes,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        auto dst = std::make_shared<VulkanBuffer>(
            dev, bytes,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        auto cmd = std::make_shared<VulkanCommandBuffer>(cmdpool, 0);
        cmd->set_replayable(true);
        cmd->begin();
        VkBufferCopy region{0, 0, static_cast<VkDeviceSize>(bytes)};
        vkop::vkCmdCopyBuffer(cmd->get(), src->getBuffer(), dst->getBuffer(),
                              1, &region);
        cmd->end();
        cmd->submit(queue);
        dev->wait_all_done();
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) {
            cmd->submit(queue);
        }
        dev->wait_all_done();
        auto t1 = std::chrono::steady_clock::now();
        double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
        printf("%-9s %zu MB copy  : %8.3f ms  %7.1f GB/s (read+write)\n",
               "memcpy", bytes >> 20, ms, 2.0 * bytes / ms / 1e6);

        // Per-submit CPU overhead: an empty (replayable) command buffer.
        auto empty = std::make_shared<VulkanCommandBuffer>(cmdpool, 0);
        empty->set_replayable(true);
        empty->begin();
        empty->end();
        empty->submit(queue);
        dev->wait_all_done();
        t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < 200; ++r) {
            empty->submit(queue);
        }
        dev->wait_all_done();
        t1 = std::chrono::steady_clock::now();
        printf("%-9s empty submit : %8.4f ms\n", "overhead",
               std::chrono::duration<double, std::milli>(t1 - t0).count() /
                   200);
    }

    // BENCH_M: rows in A. Default 1 (true decode). M > 1 is the batched-decode
    // / small-prefill regime, where the split-K GEMV's one-workgroup-per-row
    // layout reads the whole weight M times.
    const int bench_m = [] {
        const char *v = std::getenv("BENCH_M");
        return (v && *v) ? std::atoi(v) : 1;
    }();
    for (const auto &s : shapes) {
        auto A = std::make_shared<core::Tensor<uint16_t>>(
            std::vector<int>{bench_m, s.K});
        auto O = std::make_shared<core::Tensor<uint16_t>>(
            std::vector<int>{bench_m, s.N});
        fill_ones(A);
        // Allocate + upload BEFORE binding: as_storage_buffer only allocates
        // (make_vkbuff), the upload is copyToGPU's staging copy. Binding an
        // unuploaded tensor silently computes on zeros.
        A->as_storage_buffer(dev);
        A->copyToGPU(cmdpool);

        // B is either an fp16 weight [K, N] or, in int8 mode, an int8 weight
        // [K, N] plus the per-column fp32 dequant scale the byte kernels read
        // (A fp16 x int8 weight x scale -> fp16 out is the W8A16 path).
        std::shared_ptr<core::ITensor> b_keep, s_keep, keep_g;
        std::vector<std::shared_ptr<core::ITensor>> in;
        if (w4_mode) {
            const bool nv = !strcmp(w4_fmt, "nvfp4");
            const int grp = nv ? 16 : w4_group;
            const int ngroups = s.K / grp;
            auto B4 = std::make_shared<core::Tensor<int8_t>>(
                std::vector<int>{s.K, s.N});
            B4->set_elem_kind(!strcmp(w4_fmt, "nf4")     ? core::ElemKind::kNF4
                              : !strcmp(w4_fmt, "uint4") ? core::ElemKind::kUint4
                              : nv ? core::ElemKind::kFloat4E2M1
                                   : core::ElemKind::kInt4);
            const size_t nbytes = static_cast<size_t>(s.K) * s.N / 2;
            B4->set_payload_bytes(static_cast<int>(nbytes));
            std::vector<int8_t> wb(nbytes);
            for (size_t i = 0; i < nbytes; ++i) {
                const int j0 = static_cast<int>((2 * i) % s.N);
                wb[i] = static_cast<int8_t>((w4code(j0) & 15) |
                                            ((w4code(j0 + 1) & 15) << 4));
            }
            B4->fillToCPU(wb);
            B4->as_storage_buffer(dev);
            B4->copyToGPU(cmdpool);
            b_keep = B4;
            if (nv) {
                auto S8 = std::make_shared<core::Tensor<int8_t>>(
                    std::vector<int>{ngroups * s.N});
                S8->set_elem_kind(core::ElemKind::kFloat8E4M3FN);
                S8->fillToCPU(std::vector<int8_t>(
                    static_cast<size_t>(ngroups) * s.N, 0x38)); // e4m3 1.0
                auto G = std::make_shared<core::Tensor<float>>(
                    std::vector<int>{1});
                G->fillToCPU(std::vector<float>{1.0f});
                S8->as_storage_buffer(dev);
                S8->copyToGPU(cmdpool);
                G->as_storage_buffer(dev);
                G->copyToGPU(cmdpool);
                s_keep = S8;
                in = {A, B4, S8, G};
                keep_g = G;
            } else {
                auto S = std::make_shared<core::Tensor<float>>(
                    std::vector<int>{ngroups * s.N});
                S->fillToCPU(std::vector<float>(
                    static_cast<size_t>(ngroups) * s.N, 1.0f));
                S->as_storage_buffer(dev);
                S->copyToGPU(cmdpool);
                s_keep = S;
                in = {A, B4, S};
            }
        } else if (int8_mode) {
            auto B8 = std::make_shared<core::Tensor<int8_t>>(
                std::vector<int>{s.K, s.N});
            B8->reserveOnCPU();
            // wpat(j) is column-only, so every row of B is identical and the
            // whole weight cycles through seven byte values. That is exactly
            // right for the correctness probe it was written for -- it pins
            // the column indexing, and a remap or a scale-phase error shows up
            // as K*wpat(j'). It is NOT what gets timed: fill_random_i8 replaces
            // it after the probe (see the refill below the probe).
            for (int j = 0; j < s.N; ++j) {
                for (int k = 0; k < s.K; ++k) {
                    B8->data()[k * s.N + j] = static_cast<int8_t>(wpat(j));
                }
            }
            auto S = std::make_shared<core::Tensor<float>>(
                std::vector<int>{s.N});
            S->reserveOnCPU();
            std::fill(S->data().begin(), S->data().end(), 1.0f);
            B8->as_storage_buffer(dev);
            S->as_storage_buffer(dev);
            B8->copyToGPU(cmdpool);
            S->copyToGPU(cmdpool);
            b_keep = B8;
            s_keep = S;
            in = {A, B8, S};
        } else {
            auto B = std::make_shared<core::Tensor<uint16_t>>(
                std::vector<int>{s.K, s.N});
            B->reserveOnCPU();
            for (int j = 0; j < s.N; ++j) {
                const float wv = static_cast<float>(wpat(j));
                uint16_t v = bf16_mode ? core::float_to_bf16(wv)
                                       : core::ITensor::fp32_to_fp16(wv);
                for (int k = 0; k < s.K; ++k) {
                    B->data()[k * s.N + j] = v;
                }
            }
            if (bf16_mode) {
                B->set_elem_kind(core::ElemKind::kBFloat16);
            }
            B->as_storage_buffer(dev);
            B->copyToGPU(cmdpool);
            b_keep = B;
            in = {A, B};
        }
        std::vector<std::shared_ptr<core::ITensor>> out = {O};

        auto op = ops::create_from_type(ops::OpType::MATMUL, /*fp16=*/1,
                                        /*use_tensorcore=*/0,
                                        /*backend_buffer=*/true);
        op->set_name(s.name);
        op->set_runtime_device(dev, cmdpool);
        // BENCH_BF16_WIDE=1 (with BENCH_BF16): the converter's bf16_wide=1,
        // i.e. the weight holds |w| > 65504, so the half-staging tile routes
        // are off. The A/B for what that costs at a prefill BENCH_M.
        if (bf16_mode && std::getenv("BENCH_BF16_WIDE") != nullptr) {
            op->setAttribute({{"bf16_wide", "1"}});
        }
        op->enable_replay(true);
        // GPU time comes from a timestamp pair recorded INSIDE the op's own
        // command buffer (Operator::set_prof_query -- the VKOP_SUBMIT_PROF
        // mechanism), set before recording so the CACHED replay carries it.
        // A pair of empty command buffers bracketing the submit loop does NOT
        // work here: on MoltenVK/560X it returned 1-10 us spans for a 4 MB
        // weight read (7654 GB/s on a card whose copy ceiling is 72 GB/s) and
        // varied 34x between identical runs, because the samples never land on
        // the work being timed.
        VulkanQueryPool ts(dev->getLogicalDevice(), 2);
        op->set_prof_query(ts.get(), 0);
        op->onExecute(in, out, 0); // round 0: record + store fingerprint
        op->onExecute(in, out, 0); // round 1: match -> CACHED (cmd kept)
        auto cmd = op->get_record();
        if (!cmd) {
            printf("%-9s K=%-6d N=%-6d : no recorded cmd\n", s.name, s.K, s.N);
            continue;
        }
        // VKOP_MATMUL_ROUTE=1 prints the route the op picked for this shape,
        // the same readout matmul_m_sweep prints.
        if (std::getenv("VKOP_MATMUL_ROUTE") != nullptr &&
            !op->route_info().empty()) {
            printf("[route] %-9s %s\n", s.name, op->route_info().c_str());
        }

        // warmup submit + sync
        cmd->submit(queue);
        dev->wait_all_done();

        // Correctness probe: A is all ones and B[k][j] = wpat(j), so every
        // output element must equal K * wpat(j) exactly. Reported as the first
        // mismatching column, since the interesting failure is a column remap
        // (right value, wrong column) rather than a bad sum.
        cmd->submit(queue);
        dev->wait_all_done();
        O->reserveOnCPU();
        O->copyToCPU(cmdpool);
        printf("  probe: O[0]=%f O[1]=%f O[N-1]=%f\n",
               core::ITensor::fp16_to_fp32(O->data()[0]),
               core::ITensor::fp16_to_fp32(O->data()[1]),
               core::ITensor::fp16_to_fp32(O->data()[s.N - 1]));
        bool ok = true;
        for (int j = 0; j < s.N; ++j) {
            uint16_t v = O->data()[j];
            float f = vkop::core::ITensor::fp16_to_fp32(v);
            float want =
                w4_mode ? core::ITensor::fp16_to_fp32(core::ITensor::fp32_to_fp16(
                              static_cast<float>(s.K) *
                              w4dec(w4_fmt, w4code(j))))
                        : static_cast<float>(s.K) * static_cast<float>(wpat(j));
            if (w4_mode ? std::fabs(f - want) > 2e-3f * std::fabs(want) + 1e-2f
                        : f != want) {
                ok = false;
                printf("  MISMATCH at col %d: got %f, want %f (K*wpat(%d))\n",
                       j, f, want, j);
                break;
            }
        }
        if (!ok) {
            printf("%-9s K=%-6d N=%-6d : WRONG OUTPUT, skip timing\n", s.name,
                   s.K, s.N);
            continue;
        }

        // The probe is done, so the weight can stop being compressible. Refill
        // the host copy and re-upload into the buffer the recorded cmd already
        // binds -- toCPU() clears the on-GPU flag, without which copyToGPU is a
        // no-op and the timed loop would still read the ramp.
        if (std::getenv("BENCH_PATTERN") == nullptr) {
            if (int8_mode || w4_mode) {
                auto b8 = std::dynamic_pointer_cast<core::Tensor<int8_t>>(b_keep);
                fill_random_i8(b8);
                b8->toCPU();
                b8->copyToGPU(cmdpool);
            } else {
                auto b16 =
                    std::dynamic_pointer_cast<core::Tensor<uint16_t>>(b_keep);
                fill_random_f16(b16);
                b16->toCPU();
                b16->copyToGPU(cmdpool);
            }
            dev->wait_all_done();
        }

        // Every replay re-emits the same timestamp pair, so after the loop the
        // queries hold the LAST replay's span: TOP_OF_PIPE (this cmd reaches the
        // front of the queue) to BOTTOM_OF_PIPE (its work is done), which is the
        // dispatch's own execution and excludes any idle gap before it. Whether
        // the queue was actually saturated is what wall/gpu below answers -- the
        // 4 MB decode shapes on this 560X are submit-bound, so the GPU idles
        // between replays and wall is 1.3-2.5x the kernel span. wall GB/s is
        // therefore labelled wall, not presented as bandwidth. A device that
        // returns no usable timestamps gets the wall row alone, labelled.
        // BENCH_GPU_MED=1: submit+wait each replay on its own and report the
        // median of the per-replay timestamp spans. A single replay's span is
        // noisy at the 0.05-0.2 ms decode sizes (2x between identical runs).
        if (std::getenv("BENCH_GPU_MED") != nullptr) {
            std::vector<double> spans;
            for (int r = 0; r < reps; ++r) {
                ts.reset();
                cmd->submit(queue);
                dev->wait_all_done();
                auto tk = ts.getResults();
                if (tk[1] > tk[0]) {
                    spans.push_back(static_cast<double>(tk[1] - tk[0]) *
                                    static_cast<double>(
                                        dev->getTimestampPeriod()) *
                                    1e-6);
                }
            }
            if (!spans.empty()) {
                std::sort(spans.begin(), spans.end());
                const double med = spans[spans.size() / 2];
                const double wb = static_cast<double>(s.K) * s.N *
                                  (w4_mode ? 0.5 : int8_mode ? 1 : 2);
                printf("%-9s K=%-6d N=%-6d : %8.4f ms  %7.1f GB/s  (gpu-med, "
                       "min %.4f)\n",
                       s.name, s.K, s.N, med, wb / med / 1e6, spans.front());
            }
        }
        ts.reset();
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) {
            cmd->submit(queue);
        }
        dev->wait_all_done();
        auto t1 = std::chrono::steady_clock::now();
        double wall_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
        auto ticks = ts.getResults();
        double gpu_ms = static_cast<double>(ticks[1] - ticks[0]) *
                        static_cast<double>(dev->getTimestampPeriod()) * 1e-6;
        double bytes = static_cast<double>(s.K) * s.N *
                       (w4_mode ? 0.5 : int8_mode ? 1 : 2); // weight bytes
        // ticks[1] > ticks[0] is part of the test, not decoration: the unsigned
        // subtraction wraps to ~1.8e19 if a driver ever writes them out of
        // order, which would print a near-zero GB/s as if it were a measurement.
        if (gpu_ms > 0.0 && ticks[1] > ticks[0]) {
            printf("%-9s K=%-6d N=%-6d : %8.3f ms  %7.1f GB/s  (gpu)\n",
                   s.name, s.K, s.N, gpu_ms, bytes / gpu_ms / 1e6);
            // wall/gpu, not wall-gpu: gpu is ONE replay's span while wall is the
            // average over all reps, so the difference goes negative on a
            // gpu-bound shape (measured: lm_head wall 5.606 vs gpu 5.719). The
            // ratio says which side is the bottleneck -- above 1 the host cannot
            // submit fast enough to keep the queue full, at 1 the kernel is the
            // limit and wall GB/s is finally a bandwidth number.
            printf("%-9s wall      : %8.3f ms  %7.1f GB/s  (%.2fx gpu, %s)\n",
                   s.name, wall_ms, bytes / wall_ms / 1e6, wall_ms / gpu_ms,
                   wall_ms > gpu_ms * 1.05 ? "submit-bound" : "gpu-bound");
        } else {
            printf("%-9s K=%-6d N=%-6d : %8.3f ms  %7.1f GB/s  "
                   "(wall, no GPU timestamps)\n",
                   s.name, s.K, s.N, wall_ms, bytes / wall_ms / 1e6);
        }

        // Staging D2H readback probe: the same copyToCPU path the runtime
        // uses (GPU->staging copy, submit+wait, then a host memcpy out of the
        // MAPPED STAGING POINTER). The GEMV number above never touches host
        // memory, so this row is the one that moves when the staging pool's
        // memory type changes (HOST_CACHED vs write-combined BAR). Min over
        // reps; O holds an fp16 row of N elements.
        double rb_ms = 1e9;
        for (int r = 0; r < reps; ++r) {
            auto rb0 = std::chrono::steady_clock::now();
            O->copyToCPU(cmdpool);
            auto rb1 = std::chrono::steady_clock::now();
            rb_ms = std::min(rb_ms, std::chrono::duration<double, std::milli>(
                                        rb1 - rb0)
                                        .count());
        }
        double rb_bytes = static_cast<double>(s.N) * 2; // fp16 out row
        printf("%-9s readback %6.0f kB: %8.4f ms  %7.2f GB/s\n", s.name,
               rb_bytes / 1e3, rb_ms, rb_bytes / rb_ms / 1e6);
    }
    return 0;
}
