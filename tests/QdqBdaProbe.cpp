// Copyright 2026 @junka
// End-to-end probe for the QuantizeLinear/DequantizeLinear device-address
// variant. One format-polymorphic module covers quantize and dequantize, so
// the only BDA key is buffer_quantize_linear_bda.
//
// Uses exact multiples of the scale so roundEven is unambiguous:
//   q = k - 10,  scale = 0.5,  zp = 3
//   dequant expected = (q - zp) * scale
//   quant input     = (k - 10) * scale, expected q = k - 7

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

const float kScale = 0.5F;
const int8_t kZp = 3;
const std::vector<int> kShape = {4, 5};

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

std::shared_ptr<Tensor<float>> make_scale() {
    auto t = std::make_shared<Tensor<float>>(std::vector<int>{1});
    t->fillToCPU(std::vector<float>{kScale});
    return t;
}

std::shared_ptr<Tensor<int8_t>> make_zp() {
    auto t = std::make_shared<Tensor<int8_t>>(std::vector<int>{1});
    t->fillToCPU(std::vector<int8_t>{kZp});
    return t;
}

template <typename FloatT>
int run_dequant(std::shared_ptr<VulkanDevice> &dev,
                std::shared_ptr<VulkanCommandPool> &pool, bool host_import) {
    const size_t n = 20;
    std::vector<int8_t> q(n);
    for (size_t i = 0; i < n; ++i) {
        q[i] = static_cast<int8_t>(static_cast<int>(i) - 10);
    }

    auto input = std::make_shared<Tensor<int8_t>>(kShape);
    auto output = std::make_shared<Tensor<FloatT>>(kShape);
    if (host_import) {
        if (!dev->is_support_host_import()) {
            std::printf("SKIP: host import unsupported; fallback untested\n");
            return 0;
        }
        input->as_host_imported_buffer(dev, n);
        output->as_host_imported_buffer(dev, n);
        auto *in_ptr = static_cast<int8_t *>(input->host_import_ptr());
        for (size_t i = 0; i < n; ++i) {
            in_ptr[i] = q[i];
        }
    } else {
        input->fillToCPU(q);
        input->as_storage_buffer(dev);
        input->copyToGPU(pool);
        output->as_storage_buffer(dev);
        output->toGPU();
    }
    auto scale = make_scale();
    auto zp = make_zp();

    auto op = vkop::ops::create_from_type(OpType::QUANTIZE_LINEAR, 0, 0,
                                          /*backend_buffer=*/true);
    check(op != nullptr, "create_from_type(DequantizeLinear)");
    if (!op) {
        return 1;
    }
    op->set_runtime_device(dev, pool);
    const bool desc_only = std::getenv("QDQ_BDA_DESC_ONLY") != nullptr;
    if (!host_import) {
        check(op->bda_pipeline_ready() == !desc_only, "QDQ BDA pipeline");
    }

    op->onExecute({input, scale, zp}, {output}, 0);
    auto cmd = op->get_record();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    VulkanCommandBuffer::submit(dev->getComputeQueue(), info);
    cmd->wait();
    dev->wait_all_done();

    const FloatT *out_ptr = nullptr;
    if (host_import) {
        out_ptr = static_cast<const FloatT *>(output->host_import_ptr());
    } else {
        output->copyToCPU(pool);
        out_ptr = output->data().data();
    }
    for (size_t i = 0; i < n; ++i) {
        const float got = from_storage<FloatT>(out_ptr[i]);
        const float want =
            (static_cast<float>(q[i]) - static_cast<float>(kZp)) * kScale;
        const float err =
            std::fabs(got - want) / std::max(1.0F, std::fabs(want));
        if (err > 0.02F) {
            std::printf("FAIL: Dequant %s%s[%zu] got %f want %f\n",
                        std::is_same_v<FloatT, uint16_t> ? "fp16" : "fp32",
                        host_import ? " host-import" : "", i, got, want);
            failures++;
            return 1;
        }
    }
    std::printf("PASS: Dequant %s%s (bda_pipeline=%d)\n",
                std::is_same_v<FloatT, uint16_t> ? "fp16" : "fp32",
                host_import ? " host-import fallback" : "",
                op->bda_pipeline_ready() ? 1 : 0);
    return 0;
}

template <typename FloatT>
int run_quant(std::shared_ptr<VulkanDevice> &dev,
              std::shared_ptr<VulkanCommandPool> &pool) {
    const size_t n = 20;
    std::vector<FloatT> x(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = to_storage<FloatT>((static_cast<float>(i) - 10.0F) * kScale);
    }

    auto input = std::make_shared<Tensor<FloatT>>(kShape);
    input->fillToCPU(x);
    input->as_storage_buffer(dev);
    input->copyToGPU(pool);
    auto output = std::make_shared<Tensor<int8_t>>(kShape);
    output->as_storage_buffer(dev);
    output->toGPU();
    auto scale = make_scale();
    auto zp = make_zp();

    auto op = vkop::ops::create_from_type(OpType::QUANTIZE_LINEAR, 0, 0,
                                          /*backend_buffer=*/true);
    check(op != nullptr, "create_from_type(QuantizeLinear)");
    if (!op) {
        return 1;
    }
    op->set_runtime_device(dev, pool);
    const bool desc_only = std::getenv("QDQ_BDA_DESC_ONLY") != nullptr;
    check(op->bda_pipeline_ready() == !desc_only, "QDQ BDA pipeline");

    op->onExecute({input, scale, zp}, {output}, 0);
    auto cmd = op->get_record();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    VulkanCommandBuffer::submit(dev->getComputeQueue(), info);
    cmd->wait();
    dev->wait_all_done();
    output->copyToCPU(pool);

    for (size_t i = 0; i < n; ++i) {
        const int got = static_cast<int>((*output)[i]);
        const int want = static_cast<int>(i) - 7;
        if (got != want) {
            std::printf("FAIL: Quant %s[%zu] got %d want %d\n",
                        std::is_same_v<FloatT, uint16_t> ? "fp16" : "fp32", i,
                        got, want);
            failures++;
            return 1;
        }
    }
    std::printf("PASS: Quant %s (bda_pipeline=%d)\n",
                std::is_same_v<FloatT, uint16_t> ? "fp16" : "fp32",
                op->bda_pipeline_ready() ? 1 : 0);
    return 0;
}

} // namespace

int main() {
    const bool desc_only = std::getenv("QDQ_BDA_DESC_ONLY") != nullptr;
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

    run_dequant<float>(dev, pool, /*host_import=*/false);
    run_dequant<uint16_t>(dev, pool, /*host_import=*/false);
    run_dequant<float>(dev, pool, /*host_import=*/true);
    run_quant<float>(dev, pool);
    run_quant<uint16_t>(dev, pool);

    std::printf(failures ? "RESULT: %d FAILURE(S)\n" : "RESULT: all pass\n",
                failures);
    return failures == 0 ? 0 : 1;
}
