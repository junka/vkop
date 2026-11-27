// GPU integration check for VKOP_ACTIVATION_REUSE.
// Build the ActivationReuseCheck target and pass a fixture produced by
// tests/make_activation_reuse_fixture.py.
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <vector>

#include "core/ActivationBufferPool.hpp"
#include "core/Tensor.hpp"
#include "core/runtime.hpp"
#include "vulkan/VulkanCommandPool.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"

using namespace vkop;
using namespace vkop::core;

namespace {
int failures = 0;

void check(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

void check_pool(std::shared_ptr<VulkanDevice> &dev) {
    ActivationBufferPool pool;
    const auto flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const auto props = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    auto first = pool.acquire(dev, 64, flags, props);
    const auto handle = first->getBuffer();
    pool.recycle(std::move(first));
    check(pool.available_count() == 1, "unaliased buffer enters pool");
    auto same = pool.acquire(dev, 64, flags, props);
    check(same->getBuffer() == handle && pool.hit_count() == 1,
          "exact-size allocation reuses the buffer");
    auto pinned = same;
    pool.recycle(std::move(same));
    check(pool.available_count() == 0, "external owner prevents reuse");
    auto aliased = pool.acquire(dev, 64, flags, props);
    aliased->mark_view_aliased();
    pool.recycle(std::move(aliased));
    check(pool.available_count() == 0, "view-aliased buffer is not recycled");
    auto parent = pool.acquire(dev, 64, flags, props);
    auto range = std::make_shared<VulkanBuffer>(parent, 0, 32);
    pool.recycle(std::move(range));
    check(pool.available_count() == 0, "range view is not recycled");
    pool.recycle(std::move(parent));
    check(pool.available_count() == 1, "unaliased parent can be recycled");
    auto scratch1 = pool.acquire_scratch(dev, 128);
    auto scratch2 = pool.acquire_scratch(dev, 128);
    check(scratch1->getBuffer() != scratch2->getBuffer(),
          "scratch leases in one level are distinct");
    pool.finish_level();
    auto scratch3 = pool.acquire_scratch(dev, 128);
    check(scratch3->getBuffer() == scratch1->getBuffer(),
          "completed scratch lease is reused in the next level");
}

std::vector<float> run(Runtime &runtime,
                       const std::shared_ptr<VulkanCommandPool> &cmdpool,
                       int rows, int round, int reuse) {
    if (reuse == 2) unsetenv("VKOP_ACTIVATION_REUSE");
    else setenv("VKOP_ACTIVATION_REUSE", reuse ? "1" : "0", 1);
    runtime.ResizeInput("x", {static_cast<uint32_t>(rows), 16});
    auto input = as_tensor<float>(runtime.GetInput("x"));
    std::vector<float> values(static_cast<size_t>(rows) * 16);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = (static_cast<int>(i % 17) - 8 + round) / 8.0f;
    }
    input->fillToCPU(values);
    input->copyToGPU(cmdpool);
    runtime.Run();
    runtime.ReadResult();
    auto output = as_tensor<float>(runtime.GetOutput("y"));
    return output->data();
}
} // namespace

int run_check(const char *fixture) {
    auto devices = VulkanInstance::getVulkanInstance().getPhysicalDevices();
    if (devices.empty()) return 2;
    auto dev = std::make_shared<VulkanDevice>(devices[0]);
    auto cmdpool = std::make_shared<VulkanCommandPool>(dev);
    check_pool(dev);
    Runtime reference(cmdpool, fixture, 0);
    Runtime optimized(cmdpool, fixture, 0);
    reference.set_backend_buffer(true);
    optimized.set_backend_buffer(true);
    reference.LoadModel();
    optimized.LoadModel();
    setenv("VKOP_GRAPH_SUBMIT", "0", 1);
    setenv("VKOP_LEVEL_SYNC", "1", 1);
    const int sizes[] = {1, 2, 4, 2, 1};
    for (int round = 0; round < 5; ++round) {
        auto expected = run(reference, cmdpool, sizes[round], round, false);
        auto actual = run(optimized, cmdpool, sizes[round], round,
                          round % 2 == 0 ? 2 : 1);
        check(actual.size() == static_cast<size_t>(sizes[round] * 16),
              "dynamic output size");
        check(actual == expected, "reuse matches non-reuse bit-for-bit");
        for (size_t i = 0; i < actual.size(); ++i) {
            const float x = (static_cast<int>(i % 17) - 8 + round) / 8.0f;
            check(actual[i] == x + std::max(2.0f * x, 0.0f),
                  "output matches the independent CPU reference");
        }
        auto retired = as_tensor<float>(optimized.GetTensor("b"));
        check(retired && !retired->gpu_resource(), "dead activation retired");
        if (retired) {
            bool threw = false;
            try {
                retired->copyToCPU(cmdpool);
            } catch (const std::runtime_error &) {
                threw = true;
            }
            check(threw, "reading a retired activation fails explicitly");
        }
    }
    for (int round = 5; round < 8; ++round) {
        const bool reuse = round != 5;
        const auto expected = run(reference, cmdpool, 2, round, false);
        const auto actual = run(optimized, cmdpool, 2, round, reuse);
        check(actual == expected, "same Runtime works when reuse is toggled");
    }
    const auto expected = run(reference, cmdpool, 2, 8, false);
    setenv("VKOP_DUMP_NODE", "no_matching_node", 1);
    const auto fallback = run(optimized, cmdpool, 2, 8, 2);
    check(fallback == expected, "default mode falls back for node dumps");
    auto kept = as_tensor<float>(optimized.GetTensor("b"));
    check(kept && kept->gpu_resource(),
          "fallback retains intermediate activation");
    unsetenv("VKOP_DUMP_NODE");
    setenv("VKOP_ACTIVATION_REUSE", "1", 1);
    setenv("VKOP_SEGMENT_REPLAY", "1", 1);
    bool rejected_replay = false;
    try {
        optimized.Run();
    } catch (const std::runtime_error &) {
        rejected_replay = true;
    }
    check(rejected_replay, "replay is rejected while reuse is enabled");
    unsetenv("VKOP_SEGMENT_REPLAY");
    setenv("VKOP_NODE_DUMP", "0", 1);
    bool rejected_dump = false;
    try {
        optimized.Run();
    } catch (const std::runtime_error &) {
        rejected_dump = true;
    }
    check(rejected_dump, "node dump is rejected even when its value is zero");
    unsetenv("VKOP_NODE_DUMP");
    unsetenv("VKOP_ACTIVATION_REUSE");
    std::printf("ActivationReuseCheck: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: ActivationReuseCheck fixture.vkopbin\n");
        return 2;
    }
    const int result = run_check(argv[1]);
    std::fflush(stdout);
    std::fflush(stderr);
    // b3's NVIDIA EGL atexit handler crashes after Vulkan teardown.
    if (std::getenv("VKOP_TEST_SKIP_ATEXIT")) std::_Exit(result);
    return result;
}
