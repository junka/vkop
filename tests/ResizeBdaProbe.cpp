// Copyright 2026 @junka
// BDA probe for Resize (nearest/asymmetric; 2 direct addresses, 80->96 B PC).
// RESIZE_BDA_DESC_ONLY=1 unsets VKOP_BDA: same values through descriptors.

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
              bool host_import, bool desc_only) {
    const std::vector<int> in_shape = {1, 2, 4, 4};
    const std::vector<int> out_shape = {1, 2, 8, 8};
    const int n = 1, c = 2, ih = 4, iw = 4, oh = 8, ow = 8;
    std::vector<float> in_vals(n * c * ih * iw);
    for (size_t i = 0; i < in_vals.size(); ++i) {
        in_vals[i] = 0.25F + 0.07F * static_cast<float>(i);
    }
    std::vector<float> want(n * c * oh * ow);
    for (int b = 0; b < n; ++b) {
        for (int ch = 0; ch < c; ++ch) {
            for (int y = 0; y < oh; ++y) {
                for (int x = 0; x < ow; ++x) {
                    const int sy = y * ih / oh;
                    const int sx = x * iw / ow;
                    want[((b * c + ch) * oh + y) * ow + x] =
                        in_vals[((b * c + ch) * ih + sy) * iw + sx];
                }
            }
        }
    }

    auto input = std::make_shared<Tensor<T>>(in_shape);
    auto out = std::make_shared<Tensor<T>>(out_shape);
    if (host_import) {
        if (!dev->is_support_host_import()) {
            std::printf("SKIP: host import unsupported; fallback untested\n");
            return;
        }
        input->as_host_imported_buffer(dev, in_vals.size());
        out->as_host_imported_buffer(dev, want.size());
        auto *p = static_cast<T *>(input->host_import_ptr());
        for (size_t i = 0; i < in_vals.size(); ++i) {
            p[i] = to_storage<T>(in_vals[i]);
        }
    } else {
        std::vector<T> bytes(in_vals.size());
        for (size_t i = 0; i < in_vals.size(); ++i) {
            bytes[i] = to_storage<T>(in_vals[i]);
        }
        input->fillToCPU(bytes);
        input->as_storage_buffer(dev);
        input->copyToGPU(pool);
        out->as_storage_buffer(dev);
        out->toGPU();
    }

    auto op = vkop::ops::create_from_type(vkop::ops::OpType::RESIZE, fp16, 0,
                                          /*backend_buffer=*/true);
    op->setAttribute({{"mode", "nearest"},
                      {"coordinate_transformation_mode", "asymmetric"},
                      {"sizes", "[8,8]"}});
    op->set_runtime_device(dev, pool);
    op->onExecute({input}, {out}, 0);
    auto cmd = op->get_record();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    VulkanCommandBuffer::submit(dev->getComputeQueue(), info);
    cmd->wait();
    dev->wait_all_done();

    const T *p = nullptr;
    if (host_import) {
        p = static_cast<const T *>(out->host_import_ptr());
    } else {
        out->copyToCPU(pool);
        p = out->data().data();
    }
    for (size_t i = 0; i < want.size(); ++i) {
        const float got = from_storage<T>(p[i]);
        if (std::fabs(got - want[i]) >
            0.02F * std::max(1.0F, std::fabs(want[i]))) {
            std::printf("FAIL: Resize %s%s[%zu] got %f want %f\n",
                        fp16 ? "fp16" : "fp32",
                        host_import ? " host-import" : "", i, got, want[i]);
            failures++;
            return;
        }
    }
    if (!desc_only) {
        check(op->bda_pipeline_ready(), "Resize BDA pipeline");
        if (host_import) {
            check(!op->bda_dispatched(),
                  "Resize host-import must take the descriptor path");
        } else {
            check(op->bda_dispatched(), "Resize BDA dispatch");
        }
    }
    std::printf("PASS: Resize %s%s (bda=%d)\n", fp16 ? "fp16" : "fp32",
                host_import ? " host-import fallback" : "",
                op->bda_dispatched() ? 1 : 0);
}

} // namespace

int main() {
    const bool desc_only = std::getenv("RESIZE_BDA_DESC_ONLY") != nullptr;
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

    run_case<float>(dev, pool, 0, false, desc_only);
    run_case<uint16_t>(dev, pool, 1, false, desc_only);
    run_case<float>(dev, pool, 0, true, desc_only);

    std::printf(failures ? "RESULT: %d FAILURE(S)\n" : "RESULT: all pass\n",
                failures);
    return failures == 0 ? 0 : 1;
}
