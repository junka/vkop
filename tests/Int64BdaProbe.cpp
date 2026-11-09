// Copyright 2026 @junka
// BDA probe for the int64 shape-meta twins (buffer_*_int64). These paths are
// only reached with VKOP_HOST_SHAPE=0, so the probe forces it before any op is
// created. Each op checks exact int64 results and that the BDA module
// dispatched. INT64_BDA_DESC_ONLY=1 pins the descriptor modules.

#include <unistd.h>

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

using vkop::core::Tensor;
using vkop::VulkanCommandBuffer;
using vkop::VulkanCommandPool;
using vkop::VulkanDevice;
using vkop::VulkanInstance;

namespace {

int failures = 0;
std::shared_ptr<VulkanDevice> g_dev;
std::shared_ptr<VulkanCommandPool> g_pool;

std::shared_ptr<Tensor<int64_t>> make_in(const std::vector<int> &shape,
                                         const std::vector<int64_t> &vals) {
    auto t = std::make_shared<Tensor<int64_t>>(shape);
    t->fillToCPU(vals);
    t->as_storage_buffer(g_dev);
    t->copyToGPU(g_pool);
    return t;
}

std::shared_ptr<Tensor<int64_t>> make_out(const std::vector<int> &shape) {
    auto t = std::make_shared<Tensor<int64_t>>(shape);
    t->as_storage_buffer(g_dev);
    t->toGPU();
    return t;
}

void check(const char *name, vkop::ops::Operator *op,
           const std::shared_ptr<Tensor<int64_t>> &out,
           const std::vector<int64_t> &want, bool desc_only) {
    auto cmd = op->get_record();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    VulkanCommandBuffer::submit(g_dev->getComputeQueue(), info);
    cmd->wait();
    g_dev->wait_all_done();
    std::printf("%s route=%s bda_pipeline=%d bda_dispatched=%d\n", name,
                op->route_info().c_str(), op->bda_pipeline_ready() ? 1 : 0,
                op->bda_dispatched() ? 1 : 0);
    if (!desc_only) {
        if (!op->bda_pipeline_ready() || !op->bda_dispatched()) {
            std::printf("FAIL: %s did not dispatch through BDA\n", name);
            failures++;
        }
    } else if (op->bda_dispatched()) {
        std::printf("FAIL: %s dispatched through BDA in descriptor mode\n",
                    name);
        failures++;
    }
    out->copyToCPU(g_pool);
    auto got = out->data();
    for (size_t i = 0; i < want.size(); ++i) {
        if (got[i] != want[i]) {
            std::printf("FAIL: %s [%zu] got %lld want %lld\n", name, i,
                        static_cast<long long>(got[i]),
                        static_cast<long long>(want[i]));
            failures++;
            return;
        }
    }
}

} // namespace

int main() {
    const bool desc_only = std::getenv("INT64_BDA_DESC_ONLY") != nullptr;
    if (desc_only) {
        unsetenv("VKOP_BDA");
    } else {
        setenv("VKOP_BDA", "1", 1);
    }
    // Force the GPU int64 twins (the default host path never builds them).
    setenv("VKOP_HOST_SHAPE", "0", 1);
    Logger::getInstance().setLevel(LOG_ERROR);
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
    g_dev = std::make_shared<VulkanDevice>(phydevs[0]);
    g_pool = std::make_shared<VulkanCommandPool>(g_dev);
    std::printf("GPU: %s, BDA=%d, variant=%s\n", g_dev->getDeviceName().c_str(),
                g_dev->is_support_buffer_device_address() ? 1 : 0,
                desc_only ? "descriptor (gate)" : "bda");
    if (!g_dev->is_support_buffer_device_address()) {
        std::printf("SKIP: buffer device address unsupported\n");
        return 0;
    }

    auto make_op = [](vkop::ops::OpType t,
                      const std::unordered_map<std::string, std::string> &a) {
        auto op = vkop::ops::create_from_type(t, 0, 0,
                                              /*backend_buffer=*/true);
        op->set_runtime_device(g_dev, g_pool);
        if (!a.empty())
            op->setAttribute(a);
        return op;
    };

    // Gather: int64 data, axis 0, indices {2,0}.
    {
        auto data = make_in({3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
        auto idx = make_in({2}, {2, 0});
        auto out = make_out({2, 4});
        auto op = make_op(vkop::ops::OpType::GATHER, {{"axis", "0"}});
        op->onExecute({data, idx}, {out}, 0);
        check("Gather", op.get(), out, {9, 10, 11, 12, 1, 2, 3, 4}, desc_only);
    }
    // Concat: two [2,3] along axis 0.
    {
        auto a = make_in({2, 3}, {1, 2, 3, 4, 5, 6});
        auto b = make_in({2, 3}, {7, 8, 9, 10, 11, 12});
        auto out = make_out({4, 3});
        auto op = make_op(vkop::ops::OpType::CONCAT, {{"axis", "0"}});
        op->onExecute({a, b}, {out}, 0);
        check("Concat", op.get(), out, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
              desc_only);
    }
    // Equal: [4] vs [4] -> int64 1/0.
    {
        auto a = make_in({4}, {1, 2, 3, 4});
        auto b = make_in({4}, {1, 9, 3, -4});
        auto out = make_out({4});
        auto op = make_op(vkop::ops::OpType::EQUAL, {});
        op->onExecute({a, b}, {out}, 0);
        check("Equal", op.get(), out, {1, 0, 1, 0}, desc_only);
    }
    // Where: cond {1,0,1,0} selects X/Y.
    {
        auto c = make_in({4}, {1, 0, 1, 0});
        auto x = make_in({4}, {11, 12, 13, 14});
        auto y = make_in({4}, {21, 22, 23, 24});
        auto out = make_out({4});
        auto op = make_op(vkop::ops::OpType::WHERE, {});
        op->onExecute({c, x, y}, {out}, 0);
        check("Where", op.get(), out, {11, 22, 13, 24}, desc_only);
    }
    // Expand: [1,3] -> [2,3] via an int64 target-shape tensor.
    {
        auto data = make_in({1, 3}, {7, 8, 9});
        auto target = make_in({2}, {2, 3});
        auto out = make_out({2, 3});
        auto op = make_op(vkop::ops::OpType::EXPAND, {});
        op->onExecute({data, target}, {out}, 0);
        check("Expand", op.get(), out, {7, 8, 9, 7, 8, 9}, desc_only);
    }

    std::printf(failures ? "RESULT: %d FAILURE(S)\n" : "RESULT: all pass\n",
                failures);
    // g_dev/g_pool are file-scope shared_ptrs; destroying them after the
    // Vulkan instance at static teardown aborts in vkDestroyCommandPool.
    // Skip static destructors -- the probe's outcome is what matters.
    std::fflush(nullptr);
    _exit(failures == 0 ? 0 : 1);
}
