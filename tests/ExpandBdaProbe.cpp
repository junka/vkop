// Copyright 2026 @junka
// BDA probe for Expand (broadcast; 2 direct addresses, 80->96 B PC).
// EXPAND_BDA_DESC_ONLY=1 unsets VKOP_BDA: same values through descriptors.

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
void run_case(std::shared_ptr<VulkanDevice> &dev,
              std::shared_ptr<VulkanCommandPool> &pool, int fp16,
              bool desc_only) {
    const std::vector<int> in_shape = {1, 2, 1, 3};
    const std::vector<int> out_shape = {4, 2, 5, 3};
    const std::vector<int> target = {4, 1, 5, 1};
    const int b = 4, c = 2, y = 5, x = 3;
    std::vector<float> in_vals(6);
    for (size_t i = 0; i < in_vals.size(); ++i) {
        in_vals[i] = 0.5F + 0.1F * static_cast<float>(i);
    }
    std::vector<float> want(b * c * y * x);
    for (int ob = 0; ob < b; ++ob) {
        for (int oc = 0; oc < c; ++oc) {
            for (int oy = 0; oy < y; ++oy) {
                for (int ox = 0; ox < x; ++ox) {
                    // inDims = {1,2,1,3}: only c and x are broadcast-relevant.
                    want[((ob * c + oc) * y + oy) * x + ox] =
                        in_vals[(0 * 2 + oc) * 3 + ox];
                }
            }
        }
    }

    auto input = std::make_shared<Tensor<T>>(in_shape);
    std::vector<T> bytes(in_vals.size());
    for (size_t i = 0; i < in_vals.size(); ++i) {
        bytes[i] = to_storage<T>(in_vals[i]);
    }
    input->fillToCPU(bytes);
    input->as_storage_buffer(dev);
    input->copyToGPU(pool);

    auto tgt = std::make_shared<Tensor<int64_t>>(
        std::vector<int>{static_cast<int>(target.size())});
    std::vector<int64_t> tv(target.begin(), target.end());
    tgt->fillToCPU(tv);

    auto out = std::make_shared<Tensor<T>>(out_shape);
    out->as_storage_buffer(dev);
    out->toGPU();

    auto op = vkop::ops::create_from_type(vkop::ops::OpType::EXPAND, fp16, 0,
                                          /*backend_buffer=*/true);
    op->set_runtime_device(dev, pool);
    op->onExecute({input, tgt}, {out}, 0);
    auto cmd = op->get_record();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    VulkanCommandBuffer::submit(dev->getComputeQueue(), info);
    cmd->wait();
    dev->wait_all_done();
    out->copyToCPU(pool);

    for (size_t i = 0; i < want.size(); ++i) {
        const float got = from_storage<T>((*out)[i]);
        if (std::fabs(got - want[i]) >
            0.02F * std::max(1.0F, std::fabs(want[i]))) {
            std::printf("FAIL: Expand %s[%zu] got %f want %f\n",
                        fp16 ? "fp16" : "fp32", i, got, want[i]);
            failures++;
            return;
        }
    }
    if (!desc_only) {
        check(op->bda_pipeline_ready(), "Expand BDA pipeline");
        check(op->bda_dispatched(), "Expand BDA dispatch");
    }
    std::printf("PASS: Expand %s (bda=%d)\n", fp16 ? "fp16" : "fp32",
                op->bda_dispatched() ? 1 : 0);
}

} // namespace

int main() {
    const bool desc_only = std::getenv("EXPAND_BDA_DESC_ONLY") != nullptr;
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
