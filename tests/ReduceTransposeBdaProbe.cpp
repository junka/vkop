// Copyright 2026 @junka
// BDA probe for Reduce and Transpose: both have two data buffers and enough
// push-constant room for direct addresses (Reduce 96->112 B, Transpose
// 112->128 B), so neither needs an address table.
//
// MISC_BDA_DESC_ONLY=1 unsets VKOP_BDA: same values through descriptors.

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
std::shared_ptr<Tensor<T>>
make_tensor(const std::vector<int> &shape, const std::vector<float> &vals,
            std::shared_ptr<VulkanDevice> &dev,
            std::shared_ptr<VulkanCommandPool> &pool, bool host_import) {
    auto t = std::make_shared<Tensor<T>>(shape);
    if (host_import) {
        t->as_host_imported_buffer(dev, vals.size());
        auto *p = static_cast<T *>(t->host_import_ptr());
        for (size_t i = 0; i < vals.size(); ++i) {
            p[i] = to_storage<T>(vals[i]);
        }
    } else {
        std::vector<T> bytes(vals.size());
        for (size_t i = 0; i < vals.size(); ++i) {
            bytes[i] = to_storage<T>(vals[i]);
        }
        t->fillToCPU(bytes);
        t->as_storage_buffer(dev);
        t->copyToGPU(pool);
    }
    return t;
}

template <typename T>
void run_reduce(std::shared_ptr<VulkanDevice> &dev,
                std::shared_ptr<VulkanCommandPool> &pool, int fp16,
                bool host_import, bool desc_only) {
    const std::vector<int> in_shape = {4, 5};
    std::vector<float> in_vals(20);
    for (size_t i = 0; i < in_vals.size(); ++i) {
        in_vals[i] = 0.5F + 0.1F * static_cast<float>(i);
    }
    std::vector<float> want(4, 0.0F);
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 5; ++c) {
            want[r] += in_vals[r * 5 + c];
        }
    }

    auto input = make_tensor<T>(in_shape, in_vals, dev, pool, host_import);
    auto out = std::make_shared<Tensor<T>>(std::vector<int>{4, 1});
    if (host_import) {
        out->as_host_imported_buffer(dev, 4);
    } else {
        out->as_storage_buffer(dev);
        out->toGPU();
    }

    auto op = vkop::ops::create_from_type(OpType::REDUCE, fp16, 0,
                                          /*backend_buffer=*/true);
    op->setAttribute(
        {{"reduce_op", "sum"}, {"axes", "[1]"}, {"keepdims", "1"}});
    op->set_runtime_device(dev, pool);
    op->onExecute({input}, {out}, 0);
    auto cmd = op->get_record();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    VulkanCommandBuffer::submit(dev->getComputeQueue(), info);
    cmd->wait();
    dev->wait_all_done();

    if (!host_import) {
        out->copyToCPU(pool);
    }
    const T *p = host_import ? static_cast<const T *>(out->host_import_ptr())
                             : out->data().data();
    for (int r = 0; r < 4; ++r) {
        const float got = from_storage<T>(p[r]);
        if (std::fabs(got - want[r]) >
            0.02F * std::max(1.0F, std::fabs(want[r]))) {
            std::printf("FAIL: Reduce %s%s[%d] got %f want %f\n",
                        fp16 ? "fp16" : "fp32",
                        host_import ? " host-import" : "", r, got, want[r]);
            failures++;
            return;
        }
    }
    if (!desc_only) {
        check(op->bda_pipeline_ready(), "Reduce BDA pipeline");
        if (host_import) {
            check(!op->bda_dispatched(),
                  "Reduce host-import must take the descriptor path");
        } else {
            check(op->bda_dispatched(), "Reduce BDA dispatch");
        }
    }
    std::printf("PASS: Reduce %s%s (bda=%d)\n", fp16 ? "fp16" : "fp32",
                host_import ? " host-import fallback" : "",
                op->bda_dispatched() ? 1 : 0);
}

template <typename T>
void run_transpose(std::shared_ptr<VulkanDevice> &dev,
                   std::shared_ptr<VulkanCommandPool> &pool, int fp16,
                   bool desc_only) {
    const std::vector<int> in_shape = {2, 3, 4};
    const std::vector<int> out_shape = {4, 2, 3};
    std::vector<float> in_vals(24);
    for (size_t i = 0; i < in_vals.size(); ++i) {
        in_vals[i] = 0.25F + 0.1F * static_cast<float>(i);
    }
    std::vector<float> want(24);
    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 3; ++b) {
            for (int c = 0; c < 4; ++c) {
                want[(c * 2 + a) * 3 + b] = in_vals[(a * 3 + b) * 4 + c];
            }
        }
    }

    auto input = make_tensor<T>(in_shape, in_vals, dev, pool, false);
    auto out = std::make_shared<Tensor<T>>(out_shape);
    out->as_storage_buffer(dev);
    out->toGPU();

    auto op = vkop::ops::create_from_type(OpType::TRANSPOSE, fp16, 0,
                                          /*backend_buffer=*/true);
    op->setAttribute({{"perm", "[2,0,1]"}});
    op->set_runtime_device(dev, pool);
    op->onExecute({input}, {out}, 0);
    auto cmd = op->get_record();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    VulkanCommandBuffer::submit(dev->getComputeQueue(), info);
    cmd->wait();
    dev->wait_all_done();
    out->copyToCPU(pool);
    for (int i = 0; i < 24; ++i) {
        const float got = from_storage<T>((*out)[i]);
        if (std::fabs(got - want[i]) >
            0.02F * std::max(1.0F, std::fabs(want[i]))) {
            std::printf("FAIL: Transpose %s[%d] got %f want %f\n",
                        fp16 ? "fp16" : "fp32", i, got, want[i]);
            failures++;
            return;
        }
    }
    if (!desc_only) {
        check(op->bda_pipeline_ready(), "Transpose BDA pipeline");
        check(op->bda_dispatched(), "Transpose BDA dispatch");
    }
    std::printf("PASS: Transpose %s (bda=%d)\n", fp16 ? "fp16" : "fp32",
                op->bda_dispatched() ? 1 : 0);
}

} // namespace

int main() {
    const bool desc_only = std::getenv("MISC_BDA_DESC_ONLY") != nullptr;
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

    run_reduce<float>(dev, pool, 0, false, desc_only);
    run_reduce<uint16_t>(dev, pool, 1, false, desc_only);
    run_reduce<float>(dev, pool, 0, true, desc_only);
    run_transpose<float>(dev, pool, 0, desc_only);
    run_transpose<uint16_t>(dev, pool, 1, desc_only);

    std::printf(failures ? "RESULT: %d FAILURE(S)\n" : "RESULT: all pass\n",
                failures);
    return failures == 0 ? 0 : 1;
}
