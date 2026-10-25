// Standalone int8 weight-only MatMul microbenchmark (buffer backend).
//
// Sibling of bench_matmul.cpp, which is fp16-GEMV only. This one drives the
// real MatMul op with an int8 weight and a parametrizable M, so the two int8
// decode/prefill kernels can be A/B'd directly:
//   - Below tiled_min_m the host routes a byte weight to the split-K /
//     byte-WIDE GEMV (weight-bandwidth-bound): m < 2, or m < 12 when the
//     weight row stride is a power of two >= 64 KiB.
//   - At or above it the shared-memory tiled kernel takes the shape. The
//     crossover is device-dependent; benchmark/README.md has the 560X profile
//     where byte m = 2..8 is still GEMV territory and the tile starts at 9.
//   - VKOP_COOP_I8=1 forces the W8A8 SINT8 cooperative-matrix path, which is
//     otherwise gated off (ops/Matmul.hpp).
//
// Env: VKOP_COOP_I8=1 to enable the W8A8 coopmat arm. All other VKOP_* knobs
// (VKOP_KSPLIT_Q, VKOP_KSPLIT_WIDE[_N]) apply as in the real op.
// BENCH_DEV=<n> picks the physical device (default 0).
//
// Build (mirrors bench_matmul.cpp; libvkop carries no SPIR-V, so
// -DVKOP_DEV_SHADER_ARCHIVE must point at build/vkop_shaders.bin):
//   g++ -O2 -std=gnu++17 -mf16c -DUSE_VMA -DVK_ENABLE_BETA_EXTENSIONS \
//     -DVKOP_DEV_SHADER_ARCHIVE='"/home/gwm/code/github/vkop/build/vkop_shaders.bin"' \
//     -I. -Imodel -Imodel/SYSTEM -Imodel/generated \
//     benchmark/bench_matmul_i8.cpp -o build/bench_matmul_i8 \
//     build/libvkop.a build/model/libvload.a -lpthread
//
// Run:
//   build/bench_matmul_i8                 # default sweep at M=1
//   build/bench_matmul_i8 M K N [reps]
//   VKOP_COOP_I8=1 build/bench_matmul_i8 M K N
#include <algorithm>
#include <chrono>
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

extern "C" void __cxa_call_terminate(void *) { abort(); }

namespace {

struct Shape {
    int K, N;
    const char *name;
};

void fill_fp16_ones(const std::shared_ptr<core::Tensor<uint16_t>> &t) {
    t->reserveOnCPU();
    std::fill(t->data().begin(), t->data().end(), 0x3C00); // fp16 1.0
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

    // Qwen2.5-0.5B projection shapes (hidden 896, intermediate 4864,
    // vocab 151936). transB == 0, weight is [K, N] int8, one scale per column.
    std::vector<Shape> shapes = {
        {896, 896, "q/k/v/o"},
        {896, 4864, "gate/up"},
        {4864, 896, "down"},
        {896, 151936, "lm_head"},
    };
    int M = 1;
    if (argc > 3) {
        M = std::atoi(argv[1]);
        shapes.clear();
        shapes.push_back({std::atoi(argv[2]), std::atoi(argv[3]), "custom"});
    }
    const int reps = (argc > 4) ? std::atoi(argv[4]) : 30;
    printf("M=%d  coop_i8=%s\n", M, std::getenv("VKOP_COOP_I8") ? "ON" : "off");

    for (const auto &s : shapes) {
        auto A = std::make_shared<core::Tensor<uint16_t>>(
            std::vector<int>{M, s.K});
        auto B = std::make_shared<core::Tensor<int8_t>>(
            std::vector<int>{s.K, s.N});
        auto S = std::make_shared<core::Tensor<float>>(std::vector<int>{s.N});
        auto O = std::make_shared<core::Tensor<uint16_t>>(
            std::vector<int>{M, s.N});

        fill_fp16_ones(A);
        // B[k][j] = k % 7 (independent of j): with A all ones the result is
        // sum_k (k%7) in EVERY column, which an all-ones weight could not
        // distinguish from a k-indexing error. sum over a full period of 7 is
        // 21, so the reference is 21 * (K/7) -- with the tail handled.
        B->reserveOnCPU();
        for (int k = 0; k < s.K; ++k) {
            for (int j = 0; j < s.N; ++j) {
                B->data()[k * s.N + j] = static_cast<int8_t>(k % 7);
            }
        }
        const int64_t expect_k = [&] {
            int64_t t = 0;
            for (int k = 0; k < s.K; ++k)
                t += k % 7;
            return t;
        }();
        S->reserveOnCPU();
        std::fill(S->data().begin(), S->data().end(), 1.0F);

        A->as_storage_buffer(dev);
        B->as_storage_buffer(dev);
        S->as_storage_buffer(dev);
        A->copyToGPU(cmdpool);
        B->copyToGPU(cmdpool);
        S->copyToGPU(cmdpool);

        auto op = ops::create_from_type(ops::OpType::MATMUL, /*fp16=*/1,
                                        /*use_tensorcore=*/0,
                                        /*backend_buffer=*/true);
        op->set_name(s.name);
        op->set_runtime_device(dev, cmdpool);
        op->enable_replay(true);
        op->setAttribute({{"transB", "0"}});
        std::vector<std::shared_ptr<core::ITensor>> in = {A, B, S}, out = {O};
        op->onExecute(in, out, 0); // round 0: record + store fingerprint
        op->onExecute(in, out, 0); // round 1: match -> CACHED (cmd kept)
        auto cmd = op->get_record();
        if (!cmd) {
            printf("%-9s K=%-6d N=%-6d : no recorded cmd\n", s.name, s.K, s.N);
            continue;
        }

        cmd->submit(queue);
        dev->wait_all_done();

        // Correctness probe: A=B=1, scale=1 -> every output element == K.
        // BENCH_NOCHECK=1 skips it, for kernel ablations whose whole point is
        // to change the numerics (timing stays valid: same instruction and
        // memory path, only the scale factor differs).
        bool ok = true;
        if (!std::getenv("BENCH_NOCHECK")) {
            O->reserveOnCPU();
            O->copyToCPU(cmdpool);
            for (int j = 0; j < s.N; ++j) {
                float f = vkop::core::ITensor::fp16_to_fp32(O->data()[j]);
                // fp16 output cannot hold these sums exactly once they pass
                // 2048 (spacing doubles each octave), so compare on a relative
                // tolerance far tighter than any indexing error would produce.
                const double want = static_cast<double>(expect_k);
                const double rel =
                    std::fabs(static_cast<double>(f) - want) /
                    (want != 0.0 ? std::fabs(want) : 1.0);
                if (rel > 1e-3) {
                    ok = false;
                    printf("  MISMATCH at %d: %f != %lld\n", j, f,
                           (long long)expect_k);
                    break;
                }
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
        double bytes = static_cast<double>(s.K) * s.N; // int8 weights
        printf("%-9s K=%-6d N=%-6d : %8.3f ms  %7.1f GB/s\n", s.name, s.K, s.N,
               ms, bytes / ms / 1e6);
    }
    return 0;
}
