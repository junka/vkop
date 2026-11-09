// Copyright 2026 @junka
// BDA probe for EmbeddingForward fixed-hotness sum (five pointers, 32->72 B
// PC). EMBEDDING_BDA_DESC_ONLY=1 unsets VKOP_BDA: same values through
// descriptors.

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
using vkop::core::Tensor;

int main() {
    const bool desc_only = std::getenv("EMBEDDING_BDA_DESC_ONLY") != nullptr;
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

    const int cats = 4, ew = 8, batch = 2, hots = 2;
    auto embed = std::make_shared<Tensor<float>>(std::vector<int>{cats, ew});
    std::vector<float> ev(cats * ew);
    for (size_t i = 0; i < ev.size(); ++i) {
        ev[i] = 0.1F + 0.01F * static_cast<float>(i);
    }
    embed->fillToCPU(ev);
    embed->as_storage_buffer(dev);
    embed->copyToGPU(pool);

    auto idx = std::make_shared<Tensor<int>>(std::vector<int>{batch, hots});
    idx->fillToCPU(std::vector<int>{0, 1, 2, 3});
    idx->as_storage_buffer(dev);
    idx->copyToGPU(pool);

    auto out = std::make_shared<Tensor<float>>(std::vector<int>{batch, ew});
    out->as_storage_buffer(dev);
    out->toGPU();

    std::vector<float> want(batch * ew);
    for (int b = 0; b < batch; ++b) {
        for (int k = 0; k < ew; ++k) {
            want[b * ew + k] =
                ev[(b * hots + 0) * ew + k] + ev[(b * hots + 1) * ew + k];
        }
    }

    auto op = vkop::ops::create_from_type(vkop::ops::OpType::EMBEDDING_FORWARD,
                                          0, 0, /*backend_buffer=*/true);
    op->setAttribute({{"num_hots", "2"},
                      {"embed_width", "8"},
                      {"is_csr", "0"},
                      {"is_weighted", "0"},
                      {"combine_mode", "0"}});
    op->set_runtime_device(dev, pool);
    op->onExecute({embed, idx}, {out}, 0);
    auto cmd = op->get_record();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    VulkanCommandBuffer::submit(dev->getComputeQueue(), info);
    cmd->wait();
    dev->wait_all_done();
    out->copyToCPU(pool);

    int failures = 0;
    for (int i = 0; i < batch * ew; ++i) {
        const float got = (*out)[i];
        if (std::fabs(got - want[i]) >
            0.02F * std::max(1.0F, std::fabs(want[i]))) {
            std::printf("FAIL: Embedding[%d] got %f want %f\n", i, got,
                        want[i]);
            failures++;
            break;
        }
    }
    if (!desc_only) {
        if (!op->bda_pipeline_ready()) {
            std::printf("FAIL: Embedding BDA pipeline\n");
            failures++;
        }
        if (!op->bda_dispatched()) {
            std::printf("FAIL: Embedding BDA dispatch\n");
            failures++;
        }
    }
    std::printf("PASS: Embedding fixed-sum (bda=%d)\n",
                op->bda_dispatched() ? 1 : 0);
    std::printf(failures ? "RESULT: %d FAILURE(S)\n" : "RESULT: all pass\n",
                failures);
    return failures == 0 ? 0 : 1;
}
