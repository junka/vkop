// Copyright 2026 @junka
// End-to-end probe for the MatMul address-table BDA variant.
//
// MatMul's push constant is already 128 B, so its BDA build binds ONE address
// table as a storage descriptor and reads all data pointers from it. This probe
// forces the non-coop routes (VKOP_NO_COOP=1):
//   M=1 -> ksplit, base submit() kernel, must dispatch through BDA;
//   M=4 -> tiled_ks, a split/fold pair that stays on descriptors, included to
//          pin the ordinary path after the shader source refactor.
//
// MATMUL_BDA_DESC_ONLY=1 unsets VKOP_BDA: same values through descriptors.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
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

void run_case(std::shared_ptr<VulkanDevice> &dev,
              std::shared_ptr<VulkanCommandPool> &pool, int m, int n, int k,
              bool expect_bda, bool desc_only) {
    auto a = std::make_shared<Tensor<uint16_t>>(std::vector<int>{m, k});
    auto b = std::make_shared<Tensor<uint16_t>>(std::vector<int>{k, n});
    auto out = std::make_shared<Tensor<uint16_t>>(std::vector<int>{m, n});

    std::vector<uint16_t> av(m * k), bv(k * n);
    std::vector<float> af(m * k), bf(k * n);
    for (int i = 0; i < m * k; ++i) {
        af[i] = 0.01F * static_cast<float>(i % 7);
        av[i] = ITensor::fp32_to_fp16(af[i]);
    }
    for (int i = 0; i < k * n; ++i) {
        bf[i] = 0.02F * static_cast<float>(i % 5);
        bv[i] = ITensor::fp32_to_fp16(bf[i]);
    }
    a->fillToCPU(av);
    b->fillToCPU(bv);
    a->as_storage_buffer(dev);
    a->copyToGPU(pool);
    b->as_storage_buffer(dev);
    b->copyToGPU(pool);
    out->as_storage_buffer(dev);
    out->toGPU();

    auto op = vkop::ops::create_from_type(vkop::ops::OpType::MATMUL, 1, 0,
                                          /*backend_buffer=*/true);
    op->set_runtime_device(dev, pool);
    op->onExecute({a, b}, {out}, 0);
    auto cmd = op->get_record();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    VulkanCommandBuffer::submit(dev->getComputeQueue(), info);
    cmd->wait();
    dev->wait_all_done();
    std::printf("M=%d N=%d K=%d route=%s bda_pipeline=%d bda_dispatched=%d\n",
                m, n, k, op->route_info().c_str(),
                op->bda_pipeline_ready() ? 1 : 0, op->bda_dispatched() ? 1 : 0);

    if (!desc_only && expect_bda && !op->bda_pipeline_ready()) {
        std::printf("FAIL: MatMul did not build its BDA pipeline\n");
        failures++;
    }
    if (!desc_only && expect_bda && !op->bda_dispatched()) {
        std::printf("FAIL: MatMul did not dispatch through BDA\n");
        failures++;
    }
    if (desc_only || !expect_bda) {
        if (!desc_only && op->bda_dispatched()) {
            std::printf("FAIL: route %s unexpectedly used BDA\n",
                        op->route_info().c_str());
            failures++;
        }
    }

    out->copyToCPU(pool);
    for (int r = 0; r < m; ++r) {
        for (int j = 0; j < n; ++j) {
            float want = 0.0F;
            for (int t = 0; t < k; ++t) {
                want += af[r * k + t] * bf[t * n + j];
            }
            const float got = ITensor::fp16_to_fp32((*out)[r * n + j]);
            const float err =
                std::fabs(got - want) / std::max(1.0F, std::fabs(want));
            if (err > 0.02F) {
                std::printf("FAIL: out[%d,%d] got %f want %f (route %s)\n", r,
                            j, got, want, op->route_info().c_str());
                failures++;
                return;
            }
        }
    }
    std::printf("PASS: MatMul %dx%dx%d\n", m, n, k);
}

} // namespace

int main() {
    const bool desc_only = std::getenv("MATMUL_BDA_DESC_ONLY") != nullptr;
    if (desc_only) {
        unsetenv("VKOP_BDA");
    } else {
        setenv("VKOP_BDA", "1", 1);
    }
    // Force a non-coop route unless the caller wants to exercise the coop
    // kernel (whose pack pass is also BDA).
    if (std::getenv("MATMUL_BDA_ALLOW_COOP") == nullptr) {
        setenv("VKOP_NO_COOP", "1", 1);
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

    run_case(dev, pool, 1, 256, 256, /*expect_bda=*/true, desc_only);
    run_case(dev, pool, 4, 256, 256, /*expect_bda=*/true, desc_only);
    // Odd N: the reduce fallback writes fp32 scratch, then the pack pass runs.
    run_case(dev, pool, 1, 255, 256, /*expect_bda=*/true, desc_only);
    // M=16: coopmat when allowed (its pack pass is BDA), tiled_ks otherwise.
    run_case(dev, pool, 16, 256, 256, /*expect_bda=*/true, desc_only);
    // Coop-sized shape (forced primary coop in the test harness).
    run_case(dev, pool, 64, 1024, 1024, /*expect_bda=*/true, desc_only);

    std::printf(failures ? "RESULT: %d FAILURE(S)\n" : "RESULT: all pass\n",
                failures);
    return failures == 0 ? 0 : 1;
}
