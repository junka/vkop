// Copyright 2026 @junka
// End-to-end probe for the binary device-address variant.
//
// BinaryElemPC is already 128 B, so the BDA build pushes a 24 B scalar block
// plus the address of a 120 B args buffer (three broadcast shapes + three data
// pointers). Covers broadcast and non-broadcast, fp32/fp16, the activation
// path (Add) and the no-activation path (Min/Max), Pow's special negative-base
// handling, and the host-import descriptor fallback.

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

struct Case {
    OpType type;
    const char *name;
    float (*ref)(float, float);
    bool pow_exponent;
};

const Case kCases[] = {
    {OpType::ADD, "Add", [](float a, float b) { return a + b; }, false},
    {OpType::SUB, "Sub", [](float a, float b) { return a - b; }, false},
    {OpType::MUL, "Mul", [](float a, float b) { return a * b; }, false},
    {OpType::DIV, "Div", [](float a, float b) { return a / b; }, false},
    {OpType::MIN, "Min", [](float a, float b) { return std::min(a, b); },
     false},
    {OpType::MAX, "Max", [](float a, float b) { return std::max(a, b); },
     false},
    {OpType::MOD, "Mod",
     [](float a, float b) { return a - b * std::floor(a / b); }, false},
    {OpType::POW, "Pow", [](float a, float b) { return std::pow(a, b); }, true},
};

template <typename T>
int run_case(std::shared_ptr<VulkanDevice> &dev,
             std::shared_ptr<VulkanCommandPool> &pool, const Case &c, int fp16,
             bool broadcast, bool host_import) {
    const std::vector<int> a_shape = {4, 5};
    const std::vector<int> b_shape =
        broadcast ? std::vector<int>{5} : std::vector<int>{4, 5};
    const size_t n = 20;
    std::vector<float> av(n), bv(broadcast ? 5 : 20), want(n);
    for (size_t i = 0; i < n; ++i) {
        av[i] = 0.25F + 0.13F * static_cast<float>(i);
    }
    for (size_t j = 0; j < bv.size(); ++j) {
        bv[j] = c.pow_exponent ? 2.0F : 0.5F + 0.07F * static_cast<float>(j);
    }
    for (size_t i = 0; i < n; ++i) {
        want[i] = c.ref(av[i], bv[broadcast ? (i % 5) : i]);
    }

    auto a = std::make_shared<Tensor<T>>(a_shape);
    auto b = std::make_shared<Tensor<T>>(b_shape);
    auto out = std::make_shared<Tensor<T>>(a_shape);

    auto upload = [&](const std::shared_ptr<Tensor<T>> &t,
                      const std::vector<float> &vals) {
        std::vector<T> bytes(vals.size());
        for (size_t i = 0; i < vals.size(); ++i) {
            bytes[i] = to_storage<T>(vals[i]);
        }
        t->fillToCPU(bytes);
        t->as_storage_buffer(dev);
        t->copyToGPU(pool);
    };

    if (host_import) {
        if (!dev->is_support_host_import()) {
            std::printf("SKIP: host import unsupported; fallback untested\n");
            return 0;
        }
        a->as_host_imported_buffer(dev, n);
        out->as_host_imported_buffer(dev, n);
        auto *a_ptr = static_cast<T *>(a->host_import_ptr());
        for (size_t i = 0; i < n; ++i) {
            a_ptr[i] = to_storage<T>(av[i]);
        }
        upload(b, bv);
    } else {
        upload(a, av);
        upload(b, bv);
        out->as_storage_buffer(dev);
        out->toGPU();
    }

    auto op = vkop::ops::create_from_type(c.type, fp16, 0,
                                          /*backend_buffer=*/true);
    check(op != nullptr, c.name);
    if (!op) {
        return 1;
    }
    op->set_runtime_device(dev, pool);
    const bool desc_only = std::getenv("BINARY_BDA_DESC_ONLY") != nullptr;
    if (!host_import) {
        check(op->bda_pipeline_ready() == !desc_only, c.name);
    }

    op->onExecute({a, b}, {out}, 0);
    auto cmd = op->get_record();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    VulkanCommandBuffer::submit(dev->getComputeQueue(), info);
    cmd->wait();
    dev->wait_all_done();

    const T *out_ptr = nullptr;
    if (host_import) {
        out_ptr = static_cast<const T *>(out->host_import_ptr());
    } else {
        out->copyToCPU(pool);
        out_ptr = out->data().data();
    }
    for (size_t i = 0; i < n; ++i) {
        const float got = from_storage<T>(out_ptr[i]);
        const float err =
            std::fabs(got - want[i]) / std::max(1.0F, std::fabs(want[i]));
        if (err > 0.02F) {
            std::printf("FAIL: %s %s%s%s[%zu] got %f want %f\n", c.name,
                        fp16 ? "fp16" : "fp32", broadcast ? " broadcast" : "",
                        host_import ? " host-import" : "", i, got, want[i]);
            failures++;
            return 1;
        }
    }
    std::printf("PASS: %s %s%s%s (bda_pipeline=%d)\n", c.name,
                fp16 ? "fp16" : "fp32", broadcast ? " broadcast" : "",
                host_import ? " host-import fallback" : "",
                op->bda_pipeline_ready() ? 1 : 0);
    return 0;
}

} // namespace

int main() {
    const bool desc_only = std::getenv("BINARY_BDA_DESC_ONLY") != nullptr;
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
        run_case<float>(dev, pool, c, /*fp16=*/0, /*broadcast=*/true,
                        /*host_import=*/false);
    }
    for (const Case &c : kCases) {
        run_case<uint16_t>(dev, pool, c, /*fp16=*/1, /*broadcast=*/true,
                           /*host_import=*/false);
    }
    run_case<float>(dev, pool, kCases[0], /*fp16=*/0, /*broadcast=*/false,
                    /*host_import=*/false);
    run_case<float>(dev, pool, kCases[0], /*fp16=*/0, /*broadcast=*/true,
                    /*host_import=*/true);

    std::printf(failures ? "RESULT: %d FAILURE(S)\n" : "RESULT: all pass\n",
                failures);
    return failures == 0 ? 0 : 1;
}
