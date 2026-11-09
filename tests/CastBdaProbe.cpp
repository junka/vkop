// Copyright 2026 @junka
// End-to-end probe for the Cast device-address variant.
//
// Cast is the first BDA op whose ONE shader module handles both directions
// (mode selects the word layout), so its only BDA key is buffer_cast_bda.
// Checks fp32->fp16, fp16->fp32, identity fp16->fp16, and the host-import
// descriptor fallback. Skips cleanly without BDA.

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

const std::vector<float> &values() {
    static const std::vector<float> v = [] {
        std::vector<float> r(20);
        for (size_t i = 0; i < r.size(); ++i) {
            r[i] = -1.5F + 0.25F * static_cast<float>(i);
        }
        return r;
    }();
    return v;
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

template <typename InT, typename OutT>
int run_case(std::shared_ptr<VulkanDevice> &dev,
             std::shared_ptr<VulkanCommandPool> &pool, int to, const char *name,
             bool host_import) {
    const std::vector<int> shape = {4, 5};
    const size_t n = 20;
    const auto &vals = values();

    auto input = std::make_shared<Tensor<InT>>(shape);
    auto output = std::make_shared<Tensor<OutT>>(shape);

    if (host_import) {
        if (!dev->is_support_host_import()) {
            std::printf("SKIP: host import unsupported; fallback untested\n");
            return 0;
        }
        input->as_host_imported_buffer(dev, n);
        output->as_host_imported_buffer(dev, n);
        auto *in_ptr = static_cast<InT *>(input->host_import_ptr());
        for (size_t i = 0; i < n; ++i) {
            in_ptr[i] = to_storage<InT>(vals[i]);
        }
    } else {
        std::vector<InT> bytes(n);
        for (size_t i = 0; i < n; ++i) {
            bytes[i] = to_storage<InT>(vals[i]);
        }
        input->fillToCPU(bytes);
        input->as_storage_buffer(dev);
        input->copyToGPU(pool);
        output->as_storage_buffer(dev);
        output->toGPU();
    }

    auto op = vkop::ops::create_from_type(OpType::CAST, 0, 0,
                                          /*backend_buffer=*/true);
    check(op != nullptr, "create_from_type(Cast)");
    if (!op) {
        return 1;
    }
    op->setAttribute({{"to", std::to_string(to)}});
    op->set_runtime_device(dev, pool);
    const bool desc_only = std::getenv("CAST_BDA_DESC_ONLY") != nullptr;
    if (!host_import) {
        check(op->bda_pipeline_ready() == !desc_only, "Cast BDA pipeline");
    }

    op->onExecute({input}, {output}, 0);
    auto cmd = op->get_record();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    VulkanCommandBuffer::submit(dev->getComputeQueue(), info);
    cmd->wait();
    dev->wait_all_done();

    const OutT *out_ptr = nullptr;
    if (host_import) {
        out_ptr = static_cast<const OutT *>(output->host_import_ptr());
    } else {
        output->copyToCPU(pool);
        out_ptr = output->data().data();
    }
    for (size_t i = 0; i < n; ++i) {
        const float got = from_storage<OutT>(out_ptr[i]);
        const float want = vals[i];
        const float err =
            std::fabs(got - want) / std::max(1.0F, std::fabs(want));
        if (err > 0.02F) {
            std::printf("FAIL: Cast %s%s[%zu] got %f want %f\n", name,
                        host_import ? " host-import" : "", i, got, want);
            failures++;
            return 1;
        }
    }
    std::printf("PASS: Cast %s%s (bda_pipeline=%d)\n", name,
                host_import ? " host-import fallback" : "",
                op->bda_pipeline_ready() ? 1 : 0);
    return 0;
}

} // namespace

int main() {
    const bool desc_only = std::getenv("CAST_BDA_DESC_ONLY") != nullptr;
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

    run_case<float, uint16_t>(dev, pool, /*to=*/10, "fp32->fp16",
                              /*host_import=*/false);
    run_case<uint16_t, float>(dev, pool, /*to=*/1, "fp16->fp32",
                              /*host_import=*/false);
    run_case<uint16_t, uint16_t>(dev, pool, /*to=*/10, "fp16->fp16",
                                 /*host_import=*/false);
    run_case<float, uint16_t>(dev, pool, /*to=*/10, "fp32->fp16",
                              /*host_import=*/true);

    std::printf(failures ? "RESULT: %d FAILURE(S)\n" : "RESULT: all pass\n",
                failures);
    return failures == 0 ? 0 : 1;
}
