// Copyright 2026 @junka
// End-to-end probe for the Softmax device-address variant.
//
// Softmax exercises the two-dispatch BDA pattern: the fp16 build reduces into
// a scratch buffer and then runs a pack pass from the SAME BDA pipeline with a
// different push constant. Shape {2,3,4}, axis=1 -> axis_size=3, inner=4,
// outer=8, so the reduce's inner stride is non-trivial.

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
    const std::vector<int> shape = {2, 3, 4};
    const int outer = 2;
    const int axis = 3;
    const int inner = 4;
    const size_t n = 24;

    std::vector<float> xv(n);
    for (size_t i = 0; i < n; ++i) {
        xv[i] = -1.0F + 0.17F * static_cast<float>(i);
    }
    std::vector<float> want(n);
    for (int o = 0; o < outer; ++o) {
        for (int in = 0; in < inner; ++in) {
            float mx = -1e30F;
            for (int a = 0; a < axis; ++a) {
                mx = std::max(mx, xv[o * axis * inner + a * inner + in]);
            }
            float sum = 0.0F;
            for (int a = 0; a < axis; ++a) {
                sum += std::exp(xv[o * axis * inner + a * inner + in] - mx);
            }
            for (int a = 0; a < axis; ++a) {
                const size_t i = o * axis * inner + a * inner + in;
                want[i] = std::exp(xv[i] - mx) / sum;
            }
        }
    }

    auto input = std::make_shared<Tensor<T>>(shape);
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
    } else {
        std::vector<T> bytes(n);
        for (size_t i = 0; i < n; ++i) {
            bytes[i] = to_storage<T>(xv[i]);
        }
        input->fillToCPU(bytes);
        input->as_storage_buffer(dev);
        input->copyToGPU(pool);
        output->as_storage_buffer(dev);
        output->toGPU();
    }

    auto op = vkop::ops::create_from_type(OpType::SOFTMAX, fp16, 0,
                                          /*backend_buffer=*/true);
    check(op != nullptr, "create_from_type(Softmax)");
    if (!op) {
        return 1;
    }
    op->setAttribute({{"axis", "1"}});
    op->set_runtime_device(dev, pool);
    const bool desc_only = std::getenv("SOFTMAX_BDA_DESC_ONLY") != nullptr;
    if (!host_import) {
        check(op->bda_pipeline_ready() == !desc_only, "Softmax BDA pipeline");
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
        if (std::fabs(got - want[i]) > 0.02F) {
            std::printf("FAIL: Softmax %s%s[%zu] got %f want %f\n",
                        fp16 ? "fp16" : "fp32",
                        host_import ? " host-import" : "", i, got, want[i]);
            failures++;
            return 1;
        }
    }
    std::printf("PASS: Softmax %s%s (bda_pipeline=%d)\n",
                fp16 ? "fp16" : "fp32",
                host_import ? " host-import fallback" : "",
                op->bda_pipeline_ready() ? 1 : 0);
    return 0;
}

} // namespace

int main() {
    const bool desc_only = std::getenv("SOFTMAX_BDA_DESC_ONLY") != nullptr;
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
