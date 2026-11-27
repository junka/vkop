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
    const VulkanBuffer *first_ptr = first.get();
    first.reset();
    pool.release(first_ptr);
    check(pool.available_count() == 1, "unowned buffer enters the free list");
    auto same = pool.acquire(dev, 64, flags, props);
    check(same->getBuffer() == handle && pool.hit_count() == 1,
          "exact-size allocation reuses the buffer");
    check(same->reuse_pending(),
          "a buffer freed this round is handed out as reuse-pending");
    pool.mark_device_idle();
    check(!same->reuse_pending(), "device idle clears reuse-pending");

    // Another owner (a view tensor, a binding) defers the free until it lets
    // go; the free list never takes a buffer someone still references.
    auto other = same;
    const VulkanBuffer *same_ptr = same.get();
    same.reset();
    pool.release(same_ptr);
    check(pool.available_count() == 0, "a second owner keeps it out");
    other.reset();
    pool.collect_held();
    check(pool.available_count() == 1, "freed once the last owner lets go");

    auto aliased = pool.acquire(dev, 64, flags, props);
    aliased->mark_view_aliased();
    aliased->set_view_writer(7);
    const VulkanBuffer *aliased_ptr = aliased.get();
    aliased.reset();
    pool.release(aliased_ptr);
    auto fresh = pool.acquire(dev, 64, flags, props);
    check(fresh.get() == aliased_ptr && !fresh->view_aliased() &&
              fresh->view_writer() == -1,
          "alias bookkeeping does not follow a buffer to its next owner");

    // A sub-range view keeps its parent; the parent frees after the view.
    auto range = std::make_shared<VulkanBuffer>(fresh, 0, 32);
    const VulkanBuffer *parent_ptr = fresh.get();
    fresh.reset();
    pool.release(parent_ptr);
    check(pool.available_count() == 0, "a live range view keeps its parent");
    range.reset();
    pool.collect_held();
    check(pool.available_count() == 1, "parent frees after its range view");

    // Unused free buffers are trimmed after kTrimRounds rounds.
    const size_t owned = pool.owned_count();
    for (uint32_t r = 0; r < ActivationBufferPool::kTrimRounds; ++r)
        pool.finish_round();
    check(pool.owned_count() == owned - 1 && pool.available_count() == 0,
          "idle free buffers are trimmed");

    // A buffer whose owners all vanished without a release() call (e.g. a
    // resize dropped it) is still found by the round-end sweep.
    auto orphan = pool.acquire(dev, 128, flags, props);
    orphan.reset();
    pool.finish_round();
    check(pool.available_count() == 1, "round-end sweep frees orphans");

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
    // reuse == 2: the default, which is off.
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
    // The reference takes the per-op submit path; reuse forces graph mode and
    // must not depend on any per-level CPU wait.
    setenv("VKOP_GRAPH_SUBMIT", "0", 1);
    unsetenv("VKOP_LEVEL_SYNC");
    const int sizes[] = {1, 2, 4, 2, 1};
    for (int round = 0; round < 5; ++round) {
        auto expected = run(reference, cmdpool, sizes[round], round, false);
        auto actual = run(optimized, cmdpool, sizes[round], round, 1);
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
    const auto fallback = run(optimized, cmdpool, 2, 8, 2);
    check(fallback == expected, "reuse is off by default");
    auto kept = as_tensor<float>(optimized.GetTensor("b"));
    check(kept && kept->gpu_resource(),
          "the default retains intermediate activations");
    setenv("VKOP_DUMP_NODE", "no_matching_node", 1);
    bool rejected_dump_node = false;
    try {
        run(optimized, cmdpool, 2, 9, 1);
    } catch (const std::runtime_error &) {
        rejected_dump_node = true;
    }
    check(rejected_dump_node, "node dumps are rejected while reuse is on");
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
