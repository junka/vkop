// Copyright 2026 @junka
// BDA probe for the SwiGLU down-projection fold (MatMul with down_act=1).
// The fold reads nine data buffers -- several through uvec4/uvec2 aliases --
// so it is the widest BDA address table in the MatMul family. Checks the
// values against a CPU reference and that the BDA pipeline dispatched.
// DOWNACT_BDA_DESC_ONLY=1 pins the descriptor modules.
//
// M==1 selects the trimmed (_DA_M1_ONLY) module, M>1 the multi-row one; both
// have their own BDA build, and both are exercised below.

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

namespace {

// (i*37) % 29 - 14 scaled to +-1.75: 37 and 29 are coprime, so the residues
// sweep the range pseudo-randomly and every 29-term cycle sums to zero. That
// keeps each partial sum a few residuals wide, far inside fp16 range -- an
// arange ramp would overflow the N_up-wide recombination. Same pattern the
// BufferRankTest fold cases use.
float pattern(int i, float scale) {
    return static_cast<float>(((i * 37) % 29) - 14) * scale;
}

float sigmoid(float v) { return 1.0F / (1.0F + std::exp(-v)); }

bool run_case(std::shared_ptr<VulkanDevice> dev,
              const std::shared_ptr<VulkanCommandPool> &pool, int m, int k_out,
              int n_up, int n, bool desc_only) {
    const float scale = 0.125F;
    std::vector<uint16_t> xv(m * k_out);
    std::vector<uint16_t> wdv(n_up * n);
    std::vector<uint16_t> wgv(k_out * n_up);
    std::vector<uint16_t> wuv(k_out * n_up);
    for (auto &v : xv)
        v = ITensor::fp32_to_fp16(1.0F);
    for (int i = 0; i < n_up * n; ++i)
        wdv[i] = ITensor::fp32_to_fp16(pattern(i, scale));
    for (int i = 0; i < k_out * n_up; ++i) {
        wgv[i] = ITensor::fp32_to_fp16(pattern(i, scale));
        wuv[i] = wgv[i];
    }

    auto x = std::make_shared<Tensor<uint16_t>>(std::vector<int>{m, k_out});
    auto wd = std::make_shared<Tensor<uint16_t>>(std::vector<int>{n_up, n});
    auto wg = std::make_shared<Tensor<uint16_t>>(std::vector<int>{k_out, n_up});
    auto wu = std::make_shared<Tensor<uint16_t>>(std::vector<int>{k_out, n_up});
    auto out = std::make_shared<Tensor<uint16_t>>(std::vector<int>{m, n});
    x->fillToCPU(xv);
    wd->fillToCPU(wdv);
    wg->fillToCPU(wgv);
    wu->fillToCPU(wuv);
    for (auto &t : {x, wd, wg, wu}) {
        t->as_storage_buffer(dev);
        t->copyToGPU(pool);
    }
    out->as_storage_buffer(dev);
    out->toGPU();

    auto op = vkop::ops::create_from_type(vkop::ops::OpType::MATMUL, 1, 0,
                                          /*backend_buffer=*/true);
    op->set_runtime_device(dev, pool);
    op->setAttribute({{"down_act", "1"}});
    // The fused node's operand order: [x, Wd, Wg, Wu].
    op->onExecute({x, wd, wg, wu}, {out}, 0);
    auto cmd = op->get_record();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    VulkanCommandBuffer::submit(dev->getComputeQueue(), info);
    cmd->wait();
    dev->wait_all_done();

    std::printf("m=%d k_out=%d n_up=%d n=%d route=%s bda_pipeline=%d "
                "bda_dispatched=%d\n",
                m, k_out, n_up, n, op->route_info().c_str(),
                op->bda_pipeline_ready() ? 1 : 0,
                op->bda_dispatched() ? 1 : 0);
    int failures = 0;
    if (!desc_only) {
        if (!op->bda_pipeline_ready()) {
            std::printf("FAIL: fold BDA pipeline (m=%d)\n", m);
            failures++;
        }
        if (!op->bda_dispatched()) {
            std::printf("FAIL: fold BDA dispatch (m=%d)\n", m);
            failures++;
        }
    }

    out->copyToCPU(pool);
    // CPU reference in fp32, from the same committed pattern: gate/up are
    // recomputed, silu applied to the full contraction, then Wd.
    std::vector<float> act(static_cast<size_t>(n_up));
    int bad = 0;
    for (int r = 0; r < m; ++r) {
        for (int j = 0; j < n_up; ++j) {
            float g = 0.0F, u = 0.0F;
            for (int t = 0; t < k_out; ++t) {
                // x is identically 1, so each projection sums its column.
                g += pattern(t * n_up + j, scale);
                u += pattern(t * n_up + j, scale);
            }
            act[static_cast<size_t>(j)] = g * sigmoid(g) * u;
        }
        for (int c = 0; c < n; ++c) {
            float want = 0.0F;
            for (int j = 0; j < n_up; ++j)
                want += act[static_cast<size_t>(j)] * pattern(j * n + c, scale);
            const float got = ITensor::fp16_to_fp32((*out)[r * n + c]);
            if (std::fabs(got - want) > std::max(std::fabs(want) * 0.03F, 0.05F)) {
                if (bad < 8)
                    std::printf("FAIL: out[%d,%d] got %f want %f\n", r, c, got,
                                want);
                ++bad;
            }
        }
    }
    if (bad) {
        std::printf("FAIL: m=%d shape has %d bad elements\n", m, bad);
        failures++;
    }
    return failures == 0;
}

} // namespace

int main() {
    const bool desc_only = std::getenv("DOWNACT_BDA_DESC_ONLY") != nullptr;
    if (desc_only) {
        unsetenv("VKOP_BDA");
    } else {
        setenv("VKOP_BDA", "1", 1);
    }
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

    int failures = 0;
    // M==1: the trimmed module (DA_M1_ONLY); two up-blocks (GX=2) moves the
    // deposit grid-x split.
    failures += !run_case(dev, pool, 1, 32, 64, 32, desc_only);
    failures += !run_case(dev, pool, 1, 32, 128, 64, desc_only);
    // M>1: the multi-row module.
    failures += !run_case(dev, pool, 4, 32, 64, 32, desc_only);
    std::printf(failures ? "RESULT: %d FAILURE(S)\n" : "RESULT: all pass\n",
                failures);
    return failures == 0 ? 0 : 1;
}
