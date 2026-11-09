// Copyright 2026 @junka
// BDA probe for the Ada-only MatMul kernels: coopmat_sklast and the two
// 16x16x16 ACC geometries (64x64 / 64x128). Both routes are unreachable on
// ARL (it lists only the 8x8x16 MMA), so this probe SKIPs there; it exists for
// the b3 (RTX 4000 Ada) run.
//
// One process per mode because KernelSelect's knobs are resolved once:
//   MATMUL_ADA_CASE=sklast   -> VKOP_COOP_SK=2 (kCoopmatSkLast, cs falls back
//                               to the baked SK_CS=1 module)
//   MATMUL_ADA_CASE=acc2x1   -> VKOP_COOP_ACC=2x1
//   MATMUL_ADA_CASE=acc2x2   -> VKOP_COOP_ACC=2x2
// MATMUL_ADA_BDA_DESC_ONLY=1 pins the descriptor modules.

#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
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

int failures = 0;

void run_case(std::shared_ptr<VulkanDevice> &dev,
              std::shared_ptr<VulkanCommandPool> &pool, int m, int n, int k,
              const char *want_route, bool desc_only) {
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

    const std::string route = op->route_info();
    std::printf("M=%d N=%d K=%d route=%s bda_pipeline=%d bda_dispatched=%d\n",
                m, n, k, route.c_str(), op->bda_pipeline_ready() ? 1 : 0,
                op->bda_dispatched() ? 1 : 0);

    if (route.find(want_route) == std::string::npos) {
        std::printf("FAIL: route does not contain '%s'\n", want_route);
        failures++;
    }
    if (!desc_only) {
        if (!op->bda_pipeline_ready()) {
            std::printf("FAIL: BDA pipeline not built\n");
            failures++;
        }
        if (!op->bda_dispatched()) {
            std::printf("FAIL: dispatch did not go through BDA\n");
            failures++;
        }
    } else if (op->bda_dispatched()) {
        std::printf("FAIL: descriptor-only run dispatched through BDA\n");
        failures++;
    }

    out->copyToCPU(pool);
    int bad = 0;
    for (int r = 0; r < m; ++r) {
        for (int j = 0; j < n; ++j) {
            float want = 0.0F;
            for (int t = 0; t < k; ++t)
                want += af[r * k + t] * bf[t * n + j];
            const float got = ITensor::fp16_to_fp32((*out)[r * n + j]);
            if (std::fabs(got - want) > 0.02F * std::max(1.0F, std::fabs(want)) &&
                bad < 8) {
                std::printf("FAIL: out[%d,%d] got %f want %f\n", r, j, got,
                            want);
                ++bad;
            }
        }
    }
    if (bad) {
        std::printf("FAIL: %dx%dx%d value mismatch\n", m, n, k);
        failures++;
    } else {
        std::printf("PASS: MatMul %dx%dx%d\n", m, n, k);
    }
}

} // namespace

int main() {
    const char *case_env = std::getenv("MATMUL_ADA_CASE");
    const std::string mode = case_env ? case_env : "sklast";
    const bool desc_only = std::getenv("MATMUL_ADA_BDA_DESC_ONLY") != nullptr;

    // Knobs are parsed once per process, so set them before any op exists.
    if (desc_only) {
        unsetenv("VKOP_BDA");
    } else {
        setenv("VKOP_BDA", "1", 1);
    }
    setenv("VKOP_FORCE_COOP", "1", 1);
    const char *want_route = nullptr;
    if (mode == "sklast") {
        setenv("VKOP_COOP_SK", "2", 1);
        want_route = "coopmat_sklast";
    } else if (mode == "acc2x1") {
        setenv("VKOP_COOP_ACC", "2x1", 1);
        want_route = "bm=64 bn=64";
    } else if (mode == "acc2x2") {
        setenv("VKOP_COOP_ACC", "2x2", 1);
        want_route = "bm=64 bn=128";
    } else {
        std::printf("FAIL: unknown MATMUL_ADA_CASE='%s'\n", mode.c_str());
        return 2;
    }

    Logger::getInstance().setLevel(LOG_ERROR);
    // The build-tree path is baked in by CMake and does not exist on the b3
    // host. VKOP_ADA_ARCHIVE lets that run point at the shipped archive copy
    // without rebuilding.
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
    std::printf("GPU: %s, BDA=%d, case=%s, variant=%s\n",
                dev->getDeviceName().c_str(),
                dev->is_support_buffer_device_address() ? 1 : 0, mode.c_str(),
                desc_only ? "descriptor (gate)" : "bda");
    if (!dev->is_support_buffer_device_address()) {
        std::printf("SKIP: buffer device address unsupported\n");
        return 0;
    }
    // Both routes need the 16x16x16 MMA module, which ARL does not list.
    if (!dev->supports_coopmat_fp16_tile(16, 16, 16)) {
        std::printf("SKIP: device lists no 16x16x16 fp16 coopmat tile\n");
        return 0;
    }

    if (mode == "sklast") {
        // The smart selector picks cs>1 only when the tile workgroup count is
        // below 2*SM (Ada: 96). Both shapes land there (cs=2 and cs=4); the
        // host falls back to the baked SK_CS=1 module, which is why the BDA
        // table is the only thing under test.
        run_case(dev, pool, 16, 4096, 512, want_route, desc_only);
        run_case(dev, pool, 16, 2048, 512, want_route, desc_only);
    } else {
        // Keep the smart selector on plain coopmat (cs=1) with pick.tile16
        // true: the 64x128 candidate's 128 workgroups clear the target and
        // n > kCoopAltMaxN keeps the 16x8x16 alt out. The ACC pin then
        // overrides the geometry.
        run_case(dev, pool, 256, 4096, 512, want_route, desc_only);
    }

    std::printf(failures ? "RESULT: %d FAILURE(S)\n" : "RESULT: all pass\n",
                failures);
    // The NVIDIA driver on b3 segfaults in its EGL atexit handler
    // (__cxa_finalize -> libnvidia-eglcore) for every probe, after main has
    // printed the result. Leave the process without running static
    // destructors so the exit status reports the test outcome, not the
    // driver's teardown bug.
    std::fflush(nullptr);
    _exit(failures == 0 ? 0 : 1);
}
