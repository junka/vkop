// Standalone MatMul GEMV microbenchmark (decode-shape fp16, buffer backend).
//
// Drives the real MatMul op through ops::create_from_type, records once via
// the replay cache (two onExecute rounds -> CACHED), then re-submits the kept
// SIMULTANEOUS_USE cmd buffer R times and waits once, so the measurement is
// pure GPU time without per-call sync overhead.
//
// Build (mirrors the tests link line: libvkop + libvload; libvkop carries no
// SPIR-V, so -DVKOP_DEV_SHADER_ARCHIVE must point at build/vkop_shaders.bin):
//   /usr/local/bin/g++-14 -nostdinc++ \
//     -isystem /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1 \
//     -isystem /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include \
//     -O2 -std=gnu++17 -mf16c -DUSE_VMA -DVK_ENABLE_BETA_EXTENSIONS \
//     -DVKOP_DEV_SHADER_ARCHIVE=\"/Users/uqland/github/vkop/build/vkop_shaders.bin\" \
//     -I/usr/local/include -I/Users/uqland/github/vkop -I/Users/uqland/github/vkop/model \
//     -I/Users/uqland/github/vkop/model/SYSTEM -I/Users/uqland/github/vkop/model/generated \
//     benchmark/bench_matmul.cpp -o build/bench_matmul \
//     build/libvkop.a build/model/libvload.a -lc++
//
// Run (x86_64 binary -> the /usr/local Vulkan SDK stack, not /opt/homebrew):
//   DYLD_FALLBACK_LIBRARY_PATH=/usr/local/lib \
//   VK_ICD_FILENAMES=/usr/local/etc/vulkan/icd.d/MoltenVK_icd.json \
//   VKOP_NO_COOP=1 ./build/bench_matmul
//
// Env: VKOP_NO_COOP=1 on MoltenVK (16x16 coopmat tiles cannot compile to MSL).
//      BENCH_DEV=<n> picks the physical device (default 0).
//      BENCH_INT8=1 swaps B for a byte int8 weight + a per-column fp32 scale,
//      which is the only way to exercise the byte WIDE (16 cols/lane) GEMV
//      without a whole quantized model on disk.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "vulkan/VulkanInstance.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanCommandPool.hpp"
#include "vulkan/VulkanBuffer.hpp"
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
    if (argc > 1) {
        // Override: bench_matmul K N [reps]
        shapes.clear();
        shapes.push_back({std::atoi(argv[1]), std::atoi(argv[2]), "custom"});
    }

    const int reps = (argc > 3) ? std::atoi(argv[3]) : 30;
    const bool int8_mode = std::getenv("BENCH_INT8") != nullptr;
    if (int8_mode) {
        printf("weight: int8 (byte WIDE path)\n");
    }

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
        const size_t bytes = 256u << 20;
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
        std::shared_ptr<core::ITensor> b_keep, s_keep;
        std::vector<std::shared_ptr<core::ITensor>> in;
        if (int8_mode) {
            auto B8 = std::make_shared<core::Tensor<int8_t>>(
                std::vector<int>{s.K, s.N});
            B8->reserveOnCPU();
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
                uint16_t v = core::ITensor::fp32_to_fp16(static_cast<float>(wpat(j)));
                for (int k = 0; k < s.K; ++k) {
                    B->data()[k * s.N + j] = v;
                }
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
        op->enable_replay(true);
        op->onExecute(in, out, 0); // round 0: record + store fingerprint
        op->onExecute(in, out, 0); // round 1: match -> CACHED (cmd kept)
        auto cmd = op->get_record();
        if (!cmd) {
            printf("%-9s K=%-6d N=%-6d : no recorded cmd\n", s.name, s.K, s.N);
            continue;
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
            float want = static_cast<float>(s.K) * static_cast<float>(wpat(j));
            if (f != want) {
                ok = false;
                printf("  MISMATCH at col %d: got %f, want %f (K*wpat(%d))\n", j,
                       f, want, j);
                break;
            }
        }
        if (!ok) {
            printf("%-9s K=%-6d N=%-6d : WRONG OUTPUT, skip timing\n", s.name,
                   s.K, s.N);
            continue;
        }

        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) {
            cmd->submit(queue);
        }
        dev->wait_all_done();
        auto t1 = std::chrono::steady_clock::now();
        double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
        double bytes = static_cast<double>(s.K) * s.N *
                       (int8_mode ? 1 : 2); // weight bytes
        printf("%-9s K=%-6d N=%-6d : %8.3f ms  %7.1f GB/s\n", s.name, s.K, s.N,
               ms, bytes / ms / 1e6);

        // Staging D2H readback probe: the same copyToCPU path the runtime
        // uses (GPU->staging copy, submit+wait, then a host memcpy out of the
        // MAPPED STAGING POINTER). The GEMV number above never touches host
        // memory, so this row is the one that moves when the staging pool's
        // memory type changes (HOST_CACHED vs write-combined BAR). Min over
        // reps; O holds a verified fp16 row of N elements.
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
