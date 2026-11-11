// Standalone fp8 (E4M3) weight-only MatMul microbenchmark (buffer backend).
//
// Sibling of bench_matmul_i8.cpp: same shapes, same op, same routes -- only the
// weight type changes (int8 -> float8e4m3fn). Running both on one machine
// isolates the byte-to-float step, because everything else about the dispatch
// is identical.
//
// It exists to answer one question: is the fp8 decode worth a separate shader
// module? Two builds of shaders/buffer/matmul.comp answer it between them --
// the default reassembles the exponent/mantissa by hand, the -DFP8_HW build
// uses VK_EXT_shader_float8's conversion. Point this benchmark at an archive
// containing the fp8 keys and a device that enables the extension, and the
// route picks the hardware module; without either it silently keeps the
// software decode, which is what makes the A/B a one-variable comparison.
//
// Measured on Ada (896x4864, one byte per weight, min-of-3): int8 87 GB/s,
// fp8 software 38 GB/s, fp8 hardware 111 GB/s (E4M3) / 128 GB/s (E5M2). See
// benchmark/README.md and docs/vulkan-1.4-side-by-side.md.
//
// Env: all the usual VKOP_* knobs apply. VKOP_KSPLIT512=1 selects the 512-lane
// route, VKOP_BDA=1 the descriptor-buffer addressing; each has its own fp8
// module, so both matter for this benchmark. VKOP_FP8_E5M2=1 exercises the
// E5M2 grid instead of E4M3 (the format, like a real model's, comes from the
// weight tensor's elem_kind). VKOP_ADA_ARCHIVE overrides the baked archive path
// (needed on a host where the build tree is unreadable). BENCH_DEV=<n> picks
// the physical device (default 0).
//
// Run:
//   bench_matmul_fp8                 # default sweep at M=1, E4M3
//   bench_matmul_fp8 M K N [reps]
//   VKOP_BDA=1 bench_matmul_fp8 1 896 4864
//   VKOP_FP8_E5M2=1 bench_matmul_fp8 1 896 4864
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "core/ShaderArchive.hpp"
#include "core/Tensor.hpp"
#include "include/dev_shaders.hpp"
#include "include/logger.hpp"
#include "ops/OperatorFactory.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanCommandPool.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"

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
    // b3 cannot read the CMake-baked local path (/home/gwm is unreadable by
    // the wjj account), so allow an explicit archive like the probes do.
    if (const char *archive = std::getenv("VKOP_ADA_ARCHIVE")) {
        vkop::core::ShaderArchive::set_process_default(archive);
    } else {
        vkop::dev::seed_shader_archive_from_build_tree();
    }
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
    // VKOP_FP8_E5M2=1 benchmarks the E5M2 grid instead of E4M3. The op picks
    // the format from the weight tensor's elem_kind alone, so this is the same
    // switch a model carries in its .vkopbin -- nothing about the route or the
    // kernel changes, and both grids run through the same module.
    const bool e5m2 = std::getenv("VKOP_FP8_E5M2") != nullptr;
    printf("M=%d  coop_i8=%s  fp8=%s\n", M,
           std::getenv("VKOP_COOP_I8") ? "ON" : "off", e5m2 ? "E5M2" : "E4M3");

    for (const auto &s : shapes) {
        auto A =
            std::make_shared<core::Tensor<uint16_t>>(std::vector<int>{M, s.K});
        auto B =
            std::make_shared<core::Tensor<int8_t>>(std::vector<int>{s.K, s.N});
        // The bytes are an fp8 payload, not two's complement: elem_kind is what
        // routes the op to the fp8 decode and which of the two grids it reads
        // (see DType.cpp on why an fp8 value rides an int8_t container).
        B->set_elem_kind(e5m2 ? core::ElemKind::kFloat8E5M2
                              : core::ElemKind::kFloat8E4M3FN);
        auto S = std::make_shared<core::Tensor<float>>(std::vector<int>{s.N});
        auto O =
            std::make_shared<core::Tensor<uint16_t>>(std::vector<int>{M, s.N});

        fill_fp16_ones(A);
        // B[k][j] = the fp8 encoding of (k % 7) (independent of j): with A all
        // ones the result is sum_k (k%7) in EVERY column. The codes for the
        // integers 0..6 are built here rather than computed, so the bench does
        // not depend on a host-side fp8 encoder.
        //
        //   E4M3 (bias 7, 3 mantissa bits): 1.0 = 0x38, then +0x04 per unit
        //      (up to 2.0), so 1..6 = 38 40 44 48 4A 4C
        //   E5M2 (bias 15, 2 mantissa bits): 1.0 = 0x3C, +0x04 to 2.0 then
        //      +0x02 per unit, so 1..6 = 3C 40 42 44 45 46
        //
        // Both are exactly representable, which is what lets the reference sum
        // be the integer (k % 7) with no rounding anywhere.
        static const uint8_t e4m3_code[7] = {0x00, 0x38, 0x40, 0x44,
                                             0x48, 0x4A, 0x4C};
        static const uint8_t e5m2_code[7] = {0x00, 0x3C, 0x40, 0x42,
                                             0x44, 0x45, 0x46};
        const uint8_t *code = e5m2 ? e5m2_code : e4m3_code;
        B->reserveOnCPU();
        for (int k = 0; k < s.K; ++k) {
            for (int j = 0; j < s.N; ++j) {
                B->data()[k * s.N + j] = static_cast<int8_t>(code[k % 7]);
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
        const bool do_check = std::getenv("BENCH_NOCHECK") == nullptr;
        if (do_check) {
            O->reserveOnCPU();
            O->copyToCPU(cmdpool);
            for (int j = 0; j < s.N; ++j) {
                float f = vkop::core::ITensor::fp16_to_fp32(O->data()[j]);
                // fp16 output cannot hold these sums exactly once they pass
                // 2048 (spacing doubles each octave), so compare on a relative
                // tolerance far tighter than any indexing error would produce.
                const double want = static_cast<double>(expect_k);
                const double rel = std::fabs(static_cast<double>(f) - want) /
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
        printf("  CHECKED ok (want %.1f)\n", (double)expect_k);

        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) {
            cmd->submit(queue);
        }
        dev->wait_all_done();
        auto t1 = std::chrono::steady_clock::now();
        double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
        double bytes = static_cast<double>(s.K) * s.N; // one byte per weight
        printf("%-9s K=%-6d N=%-6d : %8.3f ms  %7.1f GB/s\n", s.name, s.K, s.N,
               ms, bytes / ms / 1e6);
    }
    return 0;
}
