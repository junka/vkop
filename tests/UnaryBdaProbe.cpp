// Copyright 2026 @junka
// End-to-end probe for the buffer-device-address unary variants.
//
// Every unary op that ships <key>_bda[_fp16] is driven through the real op
// class. This checks:
//   1. the BDA pipeline is built when VKOP_BDA=1 and the dev archive carries
//      the key (UNARY_BDA_DESC_ONLY pins the opposite: no BDA pipeline);
//   2. fp32 and fp16 values are correct through the address path;
//   3. host-imported buffers (hasDeviceAddress()==false) still compute
//      correctly through the descriptor pipeline of the same op.
//
// Skips cleanly without BDA; descriptor-only devices are covered by
// BufferBackendTest.

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

struct Case {
    OpType type;
    const char *name;
    float (*ref)(float);
};

const Case kCases[] = {
    {OpType::NEG, "Neg", [](float x) { return -x; }},
    {OpType::SIGMOID, "Sigmoid",
     [](float x) { return 1.0F / (1.0F + std::exp(-x)); }},
    {OpType::TANH, "Tanh", [](float x) { return std::tanh(x); }},
    {OpType::ERF, "Erf", [](float x) { return std::erf(x); }},
    {OpType::SQRT, "Sqrt", [](float x) { return std::sqrt(x); }},
    {OpType::ATAN, "Atan", [](float x) { return std::atan(x); }},
    {OpType::COS, "Cos", [](float x) { return std::cos(x); }},
    {OpType::SIN, "Sin", [](float x) { return std::sin(x); }},
    {OpType::SOFTPLUS, "Softplus",
     [](float x) { return std::log(std::exp(x) + 1.0F); }},
    {OpType::RELU, "Relu", [](float x) { return std::max(0.0F, x); }},
    {OpType::FLOOR, "Floor", [](float x) { return std::floor(x); }},
};

float ratio_error(float got, float want) {
    return std::fabs(got - want) / std::max(1.0F, std::fabs(want));
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
             std::shared_ptr<VulkanCommandPool> &pool, const Case &c, int fp16,
             bool host_import) {
    const std::vector<int> shape = {4, 5};
    const size_t n = 20;
    auto input = std::make_shared<Tensor<T>>(shape);
    auto output = std::make_shared<Tensor<T>>(shape);
    std::vector<float> in_vals(n);
    std::vector<float> want(n);
    for (size_t i = 0; i < n; ++i) {
        // Positive range keeps Sqrt well-defined for every case.
        in_vals[i] = 0.5F + 0.25F * static_cast<float>(i);
        want[i] = c.ref(in_vals[i]);
    }

    if (host_import) {
        if (!dev->is_support_host_import()) {
            std::printf("SKIP: host import unsupported; fallback untested\n");
            return 0;
        }
        input->as_host_imported_buffer(dev, n);
        output->as_host_imported_buffer(dev, n);
        auto *in_ptr = static_cast<T *>(input->host_import_ptr());
        for (size_t i = 0; i < n; ++i) {
            in_ptr[i] = to_storage<T>(in_vals[i]);
        }
    } else {
        std::vector<T> bytes(n);
        for (size_t i = 0; i < n; ++i) {
            bytes[i] = to_storage<T>(in_vals[i]);
        }
        input->fillToCPU(bytes);
        input->as_storage_buffer(dev);
        input->copyToGPU(pool);
        output->as_storage_buffer(dev);
        output->toGPU();
    }

    auto op = vkop::ops::create_from_type(c.type, fp16, 0,
                                          /*backend_buffer=*/true);
    check(op != nullptr, c.name);
    if (!op) {
        return 1;
    }
    op->set_runtime_device(dev, pool);
    const bool desc_only = std::getenv("UNARY_BDA_DESC_ONLY") != nullptr;
    if (!host_import) {
        check(op->bda_pipeline_ready() == !desc_only, c.name);
    }

    op->onExecute({input}, {output}, 0);
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
        if (ratio_error(got, want[i]) > 0.02F) {
            std::printf("FAIL: %s %s%s[%zu] got %f want %f\n", c.name,
                        fp16 ? "fp16" : "fp32",
                        host_import ? " host-import" : "", i, got, want[i]);
            failures++;
            return 1;
        }
    }
    std::printf("PASS: %s %s%s (bda_pipeline=%d)\n", c.name,
                fp16 ? "fp16" : "fp32",
                host_import ? " host-import fallback" : "",
                op->bda_pipeline_ready() ? 1 : 0);
    return 0;
}

} // namespace

int main() {
    // Shader variants are opt-in at this stage (see try_build_bda_pipeline).
    const bool desc_only = std::getenv("UNARY_BDA_DESC_ONLY") != nullptr;
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

    for (const Case &c : kCases) {
        run_case<float>(dev, pool, c, /*fp16=*/0, /*host_import=*/false);
        run_case<uint16_t>(dev, pool, c, /*fp16=*/1, /*host_import=*/false);
    }
    // One fallback case is enough: all unary ops share the same submit path.
    run_case<float>(dev, pool, kCases[0], /*fp16=*/0, /*host_import=*/true);

    std::printf(failures ? "RESULT: %d FAILURE(S)\n" : "RESULT: all pass\n",
                failures);
    return failures == 0 ? 0 : 1;
}
