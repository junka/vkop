// Copyright 2026 @junka
// BDA probe for the int8-weight cooperative-matrix MatMul route.
// Uses VKOP_COOP_I8=1 so the selector can pick coopmat_i8; checks the values
// (A is quantized per row inside the kernel, so tolerance is looser) and that
// the BDA pipeline dispatched. MATMUL_I8_BDA_DESC_ONLY=1 pins descriptors.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "core/ShaderArchive.hpp"
#include "core/Tensor.hpp"
#include "include/dev_shaders.hpp"
#include "include/logger.hpp"
#include "ops/OperatorFactory.hpp"
#include "ops/Ops.hpp"
#include "vulkan/VulkanCommandBuffer.hpp"
#include "vulkan/VulkanCommandPool.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"

using vkop::core::ITensor;
using vkop::core::Tensor;
using vkop::VulkanCommandBuffer;
using vkop::VulkanCommandPool;
using vkop::VulkanDevice;
using vkop::VulkanInstance;

int main() {
    const bool desc_only =
        std::getenv("MATMUL_I8_BDA_DESC_ONLY") != nullptr;
    // Force the UINT8→UINT32 MMA module (it is the same kernel templated on
    // the signedness; a device may list one and not the other).
    const bool uint8_mode = [] {
        const char *e = std::getenv("VKOP_FORCE_I8_UINT8");
        return e != nullptr && e[0] == '1';
    }();
    if (desc_only) {
        unsetenv("VKOP_BDA");
    } else {
        setenv("VKOP_BDA", "1", 1);
    }
    setenv("VKOP_COOP_I8", "1", 1);
    Logger::getInstance().setLevel(LOG_ERROR);
    // Override for hosts where the CMake-baked build-tree path does not
    // exist (the b3 Ada box, where the archive is copied next to the binary).
    if (const char *archive = std::getenv("VKOP_ADA_ARCHIVE")) {
        vkop::core::ShaderArchive::set_process_default(archive);
    } else {
        vkop::dev::seed_shader_archive_from_build_tree();
    }
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

    const int m = 32, n = 64, k = 64;
    auto a = std::make_shared<Tensor<uint16_t>>(std::vector<int>{m, k});
    auto b = std::make_shared<Tensor<int8_t>>(std::vector<int>{k, n});
    auto scale_t = std::make_shared<Tensor<float>>(std::vector<int>{n});
    auto out = std::make_shared<Tensor<uint16_t>>(std::vector<int>{m, n});

    std::vector<uint16_t> av(m * k);
    std::vector<int8_t> bv(k * n);
    std::vector<float> sv(n);
    std::vector<float> af(m * k);
    for (int i = 0; i < m * k; ++i) {
        af[i] = 0.1F + 0.01F * static_cast<float>(i % 7);
        av[i] = ITensor::fp32_to_fp16(af[i]);
    }
    for (int i = 0; i < k * n; ++i) {
        bv[i] = static_cast<int8_t>((i % 17) - 8);
    }
    for (int j = 0; j < n; ++j) {
        sv[j] = 0.01F + 0.001F * static_cast<float>(j % 5);
    }
    a->fillToCPU(av);
    b->fillToCPU(bv);
    scale_t->fillToCPU(sv);
    a->as_storage_buffer(dev);
    a->copyToGPU(pool);
    b->as_storage_buffer(dev);
    b->copyToGPU(pool);
    scale_t->as_storage_buffer(dev);
    scale_t->copyToGPU(pool);
    out->as_storage_buffer(dev);
    out->toGPU();

    auto op = vkop::ops::create_from_type(vkop::ops::OpType::MATMUL, 1, 0,
                                          /*backend_buffer=*/true);
    op->set_runtime_device(dev, pool);
    op->onExecute({a, b, scale_t}, {out}, 0);
    auto cmd = op->get_record();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    VulkanCommandBuffer::submit(dev->getComputeQueue(), info);
    cmd->wait();
    dev->wait_all_done();
    std::printf("route=%s bda_pipeline=%d bda_dispatched=%d\n",
                op->route_info().c_str(), op->bda_pipeline_ready() ? 1 : 0,
                op->bda_dispatched() ? 1 : 0);

    out->copyToCPU(pool);
    int failures = 0;
    if (uint8_mode &&
        op->route_info().find("coopmat_i8") == std::string::npos) {
        // The device lists no UINT8 cooperative-matrix tile, so the forced
        // module was never built and the selector fell back. Skip rather than
        // claim coverage that did not happen.
        std::printf("SKIP: device lists no UINT8 coopmat tile\n");
        return 0;
    }
    if (!desc_only) {
        if (!op->bda_pipeline_ready()) {
            std::printf("FAIL: MatMul i8 BDA pipeline\n");
            failures++;
        }
        if (!op->bda_dispatched()) {
            std::printf("FAIL: MatMul i8 BDA dispatch\n");
            failures++;
        }
    }
    for (int r = 0; r < m; ++r) {
        for (int j = 0; j < n; ++j) {
            float want = 0.0F;
            for (int t = 0; t < k; ++t) {
                const float w = uint8_mode
                                    ? static_cast<float>(
                                          static_cast<uint8_t>(bv[t * n + j]))
                                    : static_cast<float>(bv[t * n + j]);
                want += af[r * k + t] * (w * sv[j]);
            }
            const float got = ITensor::fp16_to_fp32((*out)[r * n + j]);
            // A is quantized to int8 per row inside the kernel.
            if (std::fabs(got - want) >
                0.05F * std::max(1.0F, std::fabs(want))) {
                std::printf("FAIL: out[%d,%d] got %f want %f (route %s)\n", r,
                            j, got, want, op->route_info().c_str());
                failures++;
                r = m;
                break;
            }
        }
    }
    std::printf(failures ? "RESULT: %d FAILURE(S)\n" : "RESULT: all pass\n",
                failures);
    return failures == 0 ? 0 : 1;
}
