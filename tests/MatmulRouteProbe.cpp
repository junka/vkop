// Copyright 2026 @junka
// Prints the KernelSelect route for a simple fp16 MatMul, so a BDA/table
// experiment can pick a shape that lands on the base `submit()` kernel rather
// than coop/ksplit512/down_act. Not an assertion test: prints and exits 0.

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

int main() {
    if (std::getenv("MATMUL_ROUTE_ALLOW_COOP") == nullptr) {
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

    for (const auto &dims : std::vector<std::vector<int>>{{1, 256, 256},
                                                          {1, 896, 896},
                                                          {4, 256, 256},
                                                          {16, 256, 256},
                                                          {64, 256, 256},
                                                          {64, 1024, 1024}}) {
        const int m = dims[0], n = dims[1], k = dims[2];
        auto a = std::make_shared<Tensor<uint16_t>>(std::vector<int>{m, k});
        auto b = std::make_shared<Tensor<uint16_t>>(std::vector<int>{k, n});
        auto out = std::make_shared<Tensor<uint16_t>>(std::vector<int>{m, n});
        std::vector<uint16_t> av(m * k, ITensor::fp32_to_fp16(1.0F));
        std::vector<uint16_t> bv(k * n, ITensor::fp32_to_fp16(1.0F));
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
        std::printf("M=%d N=%d K=%d -> %s\n", m, n, k,
                    op->route_info().c_str());
    }
    return 0;
}
