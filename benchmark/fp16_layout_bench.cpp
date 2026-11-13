// Copyright 2026 @junka
// Native fp16 storage vs packed-half2 storage: is the pack/unpack pair
// actually costing anything?
//
// Three qwen3_5 ops ship both builds from one source (see
// docs/fp16-storage-layouts.md):
//   <key>_fp16         float16_t data[]      one 16-bit load/store per element
//   <key>_fp16_packed  uint data[]           unpackHalf2x16 / packHalf2x16 per
//                                            element, plus the thread-per-word
//                                            discipline that pair requires
//
// This probes the three shapes that matter on a Qwen3.5-27B decode round:
//   Conv1d           [C=3840, S=1]         x 48 layers/round
//   LinearAttention  Hv=48, S=1, 128x128   x 48 layers/round
//   CumSum           [48, 1, 128]          the decay prefix in each linear layer
//
// Method: record once through the replay cache, then submit the SAME kept
// command buffer R times and wait once, so the number is replayed GPU execution
// with no per-call host sync or re-record. A/B is interleaved (native, packed,
// native, ... per repetition) so clock drift hits both arms.
//
// Usage: fp16_layout_bench [reps]      (default 200)
//        VKOP_FP16_PACKED=<op,...> selects the packed arm, as the runtime does.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "core/Tensor.hpp"
#include "include/dev_shaders.hpp"
#include "include/logger.hpp"
#include "ops/OperatorFactory.hpp"
#include "ops/Ops.hpp"
#include "vulkan/VulkanCommandBuffer.hpp"
#include "vulkan/VulkanCommandPool.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"

using vkop::VulkanCommandBuffer;
using vkop::VulkanCommandPool;
using vkop::VulkanDevice;
using vkop::VulkanInstance;
using vkop::core::Tensor;
using vkop::ops::OpType;

namespace {

template <typename T>
std::shared_ptr<Tensor<T>> one_shot(std::shared_ptr<VulkanDevice> &dev,
                                    std::shared_ptr<VulkanCommandPool> &pool,
                                    const std::vector<int> &shape, float v) {
    auto t = std::make_shared<Tensor<T>>(shape);
    std::vector<T> data(t->num_elements());
    for (auto &e : data) {
        if constexpr (std::is_same_v<T, uint16_t>) {
            e = vkop::core::ITensor::fp32_to_fp16(v);
        } else {
            e = static_cast<T>(v);
        }
    }
    t->fillToCPU(data);
    t->as_storage_buffer(dev);
    t->copyToGPU(pool);
    return t;
}

struct Timing {
    double min_ms = 1e30;
    double med_ms = 0;
};

Timing time_op(std::shared_ptr<VulkanDevice> &dev,
               vkop::ops::Operator *op,
               const std::vector<std::shared_ptr<vkop::core::ITensor>> &ins,
               const std::vector<std::shared_ptr<vkop::core::ITensor>> &outs,
               int reps) {
    op->onExecute(ins, outs, 0);
    auto cmd = op->get_record();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    VulkanCommandBuffer::submit(dev->getComputeQueue(), info);
    cmd->wait();
    dev->wait_all_done();

    std::vector<double> samples;
    samples.reserve(reps);
    for (int i = 0; i < reps; i++) {
        auto t0 = std::chrono::steady_clock::now();
        VulkanCommandBuffer::submit(dev->getComputeQueue(), info);
        cmd->wait();
        auto t1 = std::chrono::steady_clock::now();
        samples.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(samples.begin(), samples.end());
    Timing t;
    t.min_ms = samples.front();
    t.med_ms = samples[samples.size() / 2];
    return t;
}

// One arm of the A/B for a given op, at 48 layers' worth of repeats.
struct Arm {
    const char *name;
    Timing (*run)(std::shared_ptr<VulkanDevice> &, std::shared_ptr<
                  VulkanCommandPool> &, int);
};

Timing bench_conv1d_impl(std::shared_ptr<VulkanDevice> &dev,
                         std::shared_ptr<VulkanCommandPool> &pool,
                         int reps) {
    // Qwen3.5-27B decode: conv_dim = 2*16*128 + 48*128 = 10240. That is a big
    // tensor and the kernel is a 4-tap window, so this is the bandwidth case.
    const int C = 10240, S = 1, K = 4;
    auto x = one_shot<uint16_t>(dev, pool, {C, S}, 0.5F);
    auto w = one_shot<uint16_t>(dev, pool, {C, K}, 0.25F);
    auto st = one_shot<float>(dev, pool, {C, K - 1}, 0.0F);
    auto y = one_shot<uint16_t>(dev, pool, {C, S}, 0.0F);
    auto op = vkop::ops::create_from_type(OpType::CONV1D, 1, 0, true);
    op->setAttribute({{"kernel", std::to_string(K)}});
    op->set_runtime_device(dev, pool);
    return time_op(dev, op.get(), {x, w, st}, {y, st}, reps);
}

Timing bench_linear_attention_impl(std::shared_ptr<VulkanDevice> &dev,
                                   std::shared_ptr<VulkanCommandPool> &pool,
                                   int reps) {
    const int Hk = 16, Hv = 48, S = 1, Dk = 128, Dv = 128;
    auto q = one_shot<uint16_t>(dev, pool, {Hk, S, Dk}, 0.1F);
    auto k = one_shot<uint16_t>(dev, pool, {Hk, S, Dk}, 0.1F);
    auto v = one_shot<uint16_t>(dev, pool, {Hv, S, Dv}, 0.1F);
    auto g = one_shot<uint16_t>(dev, pool, {Hv, S}, -0.1F);
    auto b = one_shot<uint16_t>(dev, pool, {Hv, S}, 0.5F);
    auto st = one_shot<float>(dev, pool, {Hv, Dk, Dv}, 0.0F);
    auto out = one_shot<uint16_t>(dev, pool, {Hv, S, Dv}, 0.0F);
    auto op = vkop::ops::create_from_type(OpType::LINEAR_ATTENTION, 1, 0, true);
    op->set_runtime_device(dev, pool);
    return time_op(dev, op.get(), {q, k, v, g, b, st}, {out, st}, reps);
}

Timing bench_cumsum_impl(std::shared_ptr<VulkanDevice> &dev,
                         std::shared_ptr<VulkanCommandPool> &pool, int reps) {
    // The decay prefix each linear layer runs: [Hv, S, Dv] summed over S.
    const std::vector<int> shape{48, 4, 128};
    auto x = one_shot<uint16_t>(dev, pool, shape, 0.5F);
    auto y = one_shot<uint16_t>(dev, pool, shape, 0.0F);
    auto op = vkop::ops::create_from_type(OpType::CUMSUM, 1, 0, true);
    op->setAttribute({{"axis", "1"}});
    op->set_runtime_device(dev, pool);
    return time_op(dev, op.get(), {x}, {y}, reps);
}

// The three benchmarks, selected by argv[1].
using BenchFn = Timing (*)(std::shared_ptr<VulkanDevice> &,
                          std::shared_ptr<VulkanCommandPool> &, int);

// One arm's sample set: the per-submit times of a single record.
std::vector<double> samples_for(const char *label, BenchFn fn, int rounds,
                                bool packed_arm) {
    if (packed_arm) {
        setenv("VKOP_FP16_PACKED", label, 1);
    } else {
        unsetenv("VKOP_FP16_PACKED");
    }
    Logger::getInstance().setLevel(LOG_ERROR);
    vkop::dev::seed_shader_archive_from_build_tree();
    const auto &phys = VulkanInstance::getVulkanInstance().getPhysicalDevices();
    if (phys.empty()) {
        return {};
    }
    auto dev = std::make_shared<VulkanDevice>(phys[0]);
    auto pool = std::make_shared<VulkanCommandPool>(dev);
    Timing t = fn(dev, pool, rounds);
    return {t.min_ms, t.med_ms};
}

int run_one(const char *label, BenchFn fn, int rounds, int outer) {
    // Interleave the arms: an iGPU downclocks under sustained load (measured:
    // the same arm drifting 10x between back-to-back runs), so a sequential
    // A-then-B measures the clock, not the layout. Each outer rep takes a full
    // fresh sample from both arms, in alternating order.
    std::vector<double> nat_min, nat_med, pk_min, pk_med;
    for (int r = 0; r < outer; r++) {
        const bool flipped = (r % 2) == 1;
        auto a = samples_for(label, fn, rounds, /*packed_arm=*/flipped);
        auto b = samples_for(label, fn, rounds, /*packed_arm=*/!flipped);
        if (a.empty() || b.empty()) {
            std::printf("SKIP: no Vulkan device\n");
            return 0;
        }
        auto &n = flipped ? b : a;
        auto &p = flipped ? a : b;
        nat_min.push_back(n[0]);
        nat_med.push_back(n[1]);
        pk_min.push_back(p[0]);
        pk_med.push_back(p[1]);
    }
    unsetenv("VKOP_FP16_PACKED");
    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    const double nm = median(nat_min), pm = median(pk_min);
    const double nmed = median(nat_med), pmed = median(pk_med);
    std::printf("%-18s native min=%8.4f med=%8.4f | packed min=%8.4f "
                "med=%8.4f | packed/native min=%.3f med=%.3f\n",
                label, nm, nmed, pm, pmed, nm > 0 ? pm / nm : 0.0,
                nmed > 0 ? pmed / nmed : 0.0);
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    const int rounds = argc > 1 ? std::atoi(argv[1]) : 200;
    const int outer = argc > 2 ? std::atoi(argv[2]) : 7;
    std::printf("[bench] %d replayed submits per sample, %d interleaved "
                "samples per arm\n", rounds, outer);
    run_one("Conv1d", bench_conv1d_impl, rounds, outer);
    run_one("LinearAttention", bench_linear_attention_impl, rounds, outer);
    run_one("CumSum", bench_cumsum_impl, rounds, outer);
    std::fflush(nullptr);
    return 0;
}
