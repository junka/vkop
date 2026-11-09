// Copyright 2026 @junka
// End-to-end probe for the RMSNorm device-address variant.
//
// RMSNorm is the first NON-unary op with a -DBDA build: three data buffers
// (out/x/weight) and a 40-byte push-constant block, versus the unary family's
// two buffers. Checks:
//   1. BDA pipeline built when VKOP_BDA=1 and buffer_rmsnorm_bda[_fp16] is in
//      the dev archive; RMSNORM_BDA_DESC_ONLY pins the descriptor path;
//   2. fp32/fp16 values match a host reference;
//   3. a host-imported input/output still runs through descriptors.

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
int run_case(std::shared_ptr<VulkanDevice> &dev,
             std::shared_ptr<VulkanCommandPool> &pool, int fp16,
             bool host_import) {
    const std::vector<int> shape = {4, 6};
    const int outer = 4;
    const int inner = 6;
    const size_t n = 24;
    const float eps = 1e-5F;

    std::vector<float> xv(n);
    std::vector<float> wv(inner);
    for (size_t i = 0; i < n; ++i) {
        xv[i] = -1.0F + 0.13F * static_cast<float>(i);
    }
    for (int j = 0; j < inner; ++j) {
        wv[j] = 0.4F + 0.1F * static_cast<float>(j);
    }
    std::vector<float> want(n);
    for (int o = 0; o < outer; ++o) {
        float sumsq = 0.0F;
        for (int j = 0; j < inner; ++j) {
            const float x = xv[o * inner + j];
            sumsq += x * x;
        }
        const float inv = 1.0F / std::sqrt(sumsq / float(inner) + eps);
        for (int j = 0; j < inner; ++j) {
            want[o * inner + j] = xv[o * inner + j] * inv * wv[j];
        }
    }

    auto input = std::make_shared<Tensor<T>>(shape);
    auto weight = std::make_shared<Tensor<T>>(std::vector<int>{inner});
    auto output = std::make_shared<Tensor<T>>(shape);

    if (host_import) {
        if (!dev->is_support_host_import()) {
            std::printf("SKIP: host import unsupported; fallback untested\n");
            return 0;
        }
        input->as_host_imported_buffer(dev, n);
        output->as_host_imported_buffer(dev, n);
        auto *in_ptr = static_cast<T *>(input->host_import_ptr());
        for (size_t i = 0; i < n; ++i) {
            in_ptr[i] = to_storage<T>(xv[i]);
        }
        std::vector<T> wbytes(inner);
        for (int j = 0; j < inner; ++j) {
            wbytes[j] = to_storage<T>(wv[j]);
        }
        weight->fillToCPU(wbytes);
        weight->as_storage_buffer(dev);
        weight->copyToGPU(pool);
    } else {
        std::vector<T> xbytes(n);
        std::vector<T> wbytes(inner);
        for (size_t i = 0; i < n; ++i) {
            xbytes[i] = to_storage<T>(xv[i]);
        }
        for (int j = 0; j < inner; ++j) {
            wbytes[j] = to_storage<T>(wv[j]);
        }
        input->fillToCPU(xbytes);
        weight->fillToCPU(wbytes);
        input->as_storage_buffer(dev);
        input->copyToGPU(pool);
        weight->as_storage_buffer(dev);
        weight->copyToGPU(pool);
        output->as_storage_buffer(dev);
        output->toGPU();
    }

    auto op = vkop::ops::create_from_type(OpType::RMSNORM, fp16, 0,
                                          /*backend_buffer=*/true);
    check(op != nullptr, "create_from_type(RMSNorm)");
    if (!op) {
        return 1;
    }
    op->setAttribute({{"eps", std::to_string(eps)}});
    op->set_runtime_device(dev, pool);
    const bool desc_only = std::getenv("RMSNORM_BDA_DESC_ONLY") != nullptr;
    if (!host_import) {
        check(op->bda_pipeline_ready() == !desc_only, "RMSNorm BDA pipeline");
    }

    op->onExecute({input, weight}, {output}, 0);
    auto cmd = op->get_record();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    VulkanCommandBuffer::submit(dev->getComputeQueue(), info);
    cmd->wait();
    dev->wait_all_done();

    const T *out_ptr = nullptr;
    if (host_import) {
        out_ptr = static_cast<const T *>(output->host_import_ptr());
    } else {
        output->copyToCPU(pool);
        out_ptr = output->data().data();
    }
    for (size_t i = 0; i < n; ++i) {
        const float got = from_storage<T>(out_ptr[i]);
        const float err =
            std::fabs(got - want[i]) / std::max(1.0F, std::fabs(want[i]));
        if (err > 0.02F) {
            std::printf("FAIL: RMSNorm %s%s[%zu] got %f want %f\n",
                        fp16 ? "fp16" : "fp32",
                        host_import ? " host-import" : "", i, got, want[i]);
            failures++;
            return 1;
        }
    }
    std::printf("PASS: RMSNorm %s%s (bda_pipeline=%d)\n",
                fp16 ? "fp16" : "fp32",
                host_import ? " host-import fallback" : "",
                op->bda_pipeline_ready() ? 1 : 0);
    return 0;
}

} // namespace

int main() {
    const bool desc_only = std::getenv("RMSNORM_BDA_DESC_ONLY") != nullptr;
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

    run_case<float>(dev, pool, /*fp16=*/0, /*host_import=*/false);
    run_case<uint16_t>(dev, pool, /*fp16=*/1, /*host_import=*/false);
    run_case<float>(dev, pool, /*fp16=*/0, /*host_import=*/true);

    std::printf(failures ? "RESULT: %d FAILURE(S)\n" : "RESULT: all pass\n",
                failures);
    return failures == 0 ? 0 : 1;
}
