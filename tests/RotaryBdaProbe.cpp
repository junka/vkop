// Copyright 2026 @junka
// BDA probe for RotaryEmbedding (half-split RoPE). Four data buffers fit in
// the 128 B push-constant limit: RotaryPC is 40 B of scalars + 4 pointers
// (72 B total).
//
// ROTARY_BDA_DESC_ONLY=1 unsets VKOP_BDA: same values through descriptors.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <type_traits>
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
using vkop::core::ITensor;
using vkop::core::Tensor;
using vkop::ops::OpType;

namespace {

int failures = 0;

void check(bool condition, const char *what) {
    if (!condition) {
        std::printf("FAIL: %s\n", what);
        failures++;
    }
}

template <typename T> T to_storage(float v) {
    if constexpr (std::is_same_v<T, uint16_t>) {
        return ITensor::fp32_to_fp16(v);
    } else {
        return static_cast<T>(v);
    }
}

template <typename T> float from_storage(T v) {
    if constexpr (std::is_same_v<T, uint16_t>) {
        return ITensor::fp16_to_fp32(v);
    } else {
        return static_cast<float>(v);
    }
}

template <typename T>
std::shared_ptr<Tensor<T>> upload(std::shared_ptr<VulkanDevice> &dev,
                                  std::shared_ptr<VulkanCommandPool> &pool,
                                  const std::vector<int> &shape,
                                  const std::vector<float> &vals) {
    auto t = std::make_shared<Tensor<T>>(shape);
    std::vector<T> bytes(vals.size());
    for (size_t i = 0; i < vals.size(); ++i) {
        bytes[i] = to_storage<T>(vals[i]);
    }
    t->fillToCPU(bytes);
    t->as_storage_buffer(dev);
    t->copyToGPU(pool);
    return t;
}

template <typename T>
void run_case(std::shared_ptr<VulkanDevice> &dev,
              std::shared_ptr<VulkanCommandPool> &pool, int fp16,
              bool desc_only) {
    const std::vector<int> x_shape = {1, 2, 3, 4};
    const std::vector<int> cs_shape = {1, 1, 3, 4};
    const int heads = 2, seq = 3, head_dim = 4, rd = 4, half = 2;
    const size_t n = 24;
    std::vector<float> xv(n), cv(12), sv(12), want(n);
    for (size_t i = 0; i < n; ++i) {
        xv[i] = 0.1F + 0.09F * static_cast<float>(i);
    }
    for (size_t i = 0; i < 12; ++i) {
        cv[i] = 0.9F - 0.01F * static_cast<float>(i);
        sv[i] = 0.1F + 0.01F * static_cast<float>(i);
    }
    for (int b = 0; b < 1; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (int s = 0; s < seq; ++s) {
                for (int d = 0; d < head_dim; ++d) {
                    const int idx = ((b * heads + h) * seq + s) * head_dim + d;
                    const int cs = (b * seq + s) * rd + d;
                    want[idx] =
                        (d < half) ? xv[idx] * cv[cs] - xv[idx + half] * sv[cs]
                                   : xv[idx - half] * sv[cs] + xv[idx] * cv[cs];
                }
            }
        }
    }

    auto x = upload<T>(dev, pool, x_shape, xv);
    auto cs = upload<T>(dev, pool, cs_shape, cv);
    auto sn = upload<T>(dev, pool, cs_shape, sv);
    auto out = std::make_shared<Tensor<T>>(x_shape);
    out->as_storage_buffer(dev);
    out->toGPU();

    auto op = vkop::ops::create_from_type(OpType::ROTARY_EMBEDDING, fp16, 0,
                                          /*backend_buffer=*/true);
    op->set_runtime_device(dev, pool);
    op->onExecute({x, cs, sn}, {out}, 0);
    auto cmd = op->get_record();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    VulkanCommandBuffer::submit(dev->getComputeQueue(), info);
    cmd->wait();
    dev->wait_all_done();
    out->copyToCPU(pool);

    for (size_t i = 0; i < n; ++i) {
        const float got = from_storage<T>((*out)[i]);
        if (std::fabs(got - want[i]) >
            0.02F * std::max(1.0F, std::fabs(want[i]))) {
            std::printf("FAIL: Rotary %s[%zu] got %f want %f\n",
                        fp16 ? "fp16" : "fp32", i, got, want[i]);
            failures++;
            return;
        }
    }
    if (!desc_only) {
        check(op->bda_pipeline_ready(), "Rotary BDA pipeline");
        check(op->bda_dispatched(), "Rotary BDA dispatch");
    }
    std::printf("PASS: Rotary %s (bda=%d)\n", fp16 ? "fp16" : "fp32",
                op->bda_dispatched() ? 1 : 0);
}

} // namespace

int main() {
    const bool desc_only = std::getenv("ROTARY_BDA_DESC_ONLY") != nullptr;
    if (desc_only) {
        unsetenv("VKOP_BDA");
    } else {
        setenv("VKOP_BDA", "1", 1);
    }
    Logger::getInstance().setLevel(LOG_ERROR);
    vkop::dev::seed_shader_archive_from_build_tree();
    const auto &phydevs =
        VulkanInstance::getVulkanInstance().getPhysicalDevices();
    if (phydevs.empty()) {
        std::printf("SKIP: no Vulkan device\n");
        return 0;
    }
    auto dev = std::make_shared<VulkanDevice>(phydevs[0]);
    auto pool = std::make_shared<VulkanCommandPool>(dev);
    std::printf("GPU: %s, BDA=%d, variant=%s\n", dev->getDeviceName().c_str(),
                dev->is_support_buffer_device_address() ? 1 : 0,
                desc_only ? "descriptor (gate)" : "bda");
    if (!dev->is_support_buffer_device_address()) {
        std::printf("SKIP: buffer device address unsupported\n");
        return 0;
    }

    run_case<float>(dev, pool, 0, desc_only);
    run_case<uint16_t>(dev, pool, 1, desc_only);

    std::printf(failures ? "RESULT: %d FAILURE(S)\n" : "RESULT: all pass\n",
                failures);
    return failures == 0 ? 0 : 1;
}
