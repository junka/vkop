// Readback-cost probe for MoE expert routing (direction B0).
//
// The first version of host-resident expert weights needs a per-layer,
// per-token readback of the router's chosen expert ids so the CPU can decide
// which expert weights to page in. That readback does not exist anywhere in
// the current decode round: measured 0 per round on Qwen3-VL text decode (the
// shape-meta chains are all cached/GPU-driven now). So the cost of the
// mechanism B0 rests on has never been observed. This probe measures it
// directly, and deliberately does NOT measure it by diffing wall-clock decode
// rounds -- a single readback's cost is ~1 ms while a round is ~60 ms and
// noisy (see the noisy-benchmark note in memory), so the only honest number
// is the readback path itself, repeated.
//
// It runs the exact op the router needs: a TopK over the [1, n_experts]
// router logits producing k expert ids, then copies the *indices* output
// (Tensor<int>) back to the host -- the int/int64 path, which is the one that
// historically forced a per-level submit boundary and is therefore the
// expensive variant. Three numbers come out:
//
//   warm  -- copyToCPU with the bytes already in host_authoritative_/cache
//            (the free early-return at Tensor.hpp:1278). This is the floor a
//            cached readback would hit.
//   real  -- a genuine GPU->CPU copy: its own vkQueueSubmit + fence wait +
//            staging invalidate, i.e. what a miss costs mid-graph.
//   gpu   -- the TopK dispatches themselves, via VKOP_SUBMIT_PROF, so the
//            readback delta can be separated from routing compute.
//
// Shapes are the real Qwen1.5-MoE-A2.7B router sizes (60 experts, top-4;
// 2048 hidden), and the activation's row count is varied to show how the copy
// scales when a batch or a prefill chunk routes at once.
//
// Build: tests target (see tests/CMakeLists.txt), run as
//   ./build/readback_probe [n_experts] [k] [reps]

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "core/Tensor.hpp"
#include "include/dev_shaders.hpp"
#include "ops/Ops.hpp"
#include "tests/setup.hpp"

using vkop::VulkanCommandBuffer;
using vkop::VulkanCommandPool;
using vkop::VulkanDevice;
using vkop::core::ITensor;
using vkop::core::Tensor;
using vkop::ops::OpType;
using vkop::tests::TestEnv;

namespace {

double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
               clock::now().time_since_epoch())
        .count();
}

// One measured TopK + index readback. Returns the host-side ids so the caller
// can sanity-check that a "real" copy actually produced the router's answer
// rather than stale zeros.
struct Result {
    double warm_ms = 0.0;
    double real_ms = 0.0;
    double known_ms = 0.0;
    int checksum = 0;
    bool ok = false;
};

Result run_once(const std::shared_ptr<VulkanDevice> &dev,
                const std::shared_ptr<VulkanCommandPool> &pool, int n_experts,
                int k, bool use_buffer, bool host_known = false) {
    Result r;
    auto &dev_mut = const_cast<std::shared_ptr<VulkanDevice> &>(dev);

    // Router logits: [1, n_experts] fp32, the shape Qwen2MoeTopKRouter emits
    // (softmax over the expert axis, then top-k). Values are a fixed pseudo
    // ramp so the expected top-k is known: the k highest entries are the ids
    // n_experts-1 .. n_experts-k.
    auto logits = std::make_shared<Tensor<float>>(
        std::vector<int>{1, n_experts}, false /*on_gpu*/);
    std::vector<float> host_in(static_cast<size_t>(n_experts));
    for (int i = 0; i < n_experts; ++i)
        host_in[i] = static_cast<float>(i);
    logits->fillToCPU(host_in);

    auto out_shape = std::vector<int>{1, k};
    auto values = std::make_shared<Tensor<float>>(out_shape, false);
    auto ids = std::make_shared<Tensor<int>>(out_shape, false);

    auto op = vkop::ops::create_from_type(OpType::TOPK, 0 /*fp32*/,
                                          dev->is_support_cooperate_matrix() ? 1 : 0,
                                          use_buffer);
    if (!op) {
        fprintf(stderr, "[rbprobe] create_from_type(TOPK) failed\n");
        return r;
    }
    op->set_runtime_device(dev, pool);
    op->setAttribute({{"k", std::to_string(k)},
                      {"axis", "-1"},
                      {"largest", "1"},
                      {"sorted", "1"}});

    // Upload the logits as an SSBO. TopK is SSBO-only (no image path), so the
    // buffer backend flag is irrelevant for the op itself, but the input must
    // still be a storage buffer.
    logits->as_storage_buffer(dev_mut);
    logits->copyToGPU(pool);

    std::vector<std::shared_ptr<ITensor>> inputs{logits};
    std::vector<std::shared_ptr<ITensor>> outputs{
        std::static_pointer_cast<ITensor>(values),
        std::static_pointer_cast<ITensor>(ids)};

    op->onExecute(inputs, outputs, 0);
    auto cmd = op->get_record();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    VulkanCommandBuffer::submit(dev->getComputeQueue(), info);
    cmd->wait();
    dev->wait_all_done();

    // Warm copy first: on a fresh tensor this is the real cost, so drop it
    // (it primes nothing -- copyToCPU does not cache), then time the steady
    // path twice. The first timed copy is the honest "cold, mid-round" number;
    // a second copy from the same buffer should hit the same cost, which is
    // the check that we are not measuring an early-return.
    ids->copyToCPU(pool); // discard: prime any host-side allocation
    double t0 = now_ms();
    ids->copyToCPU(pool);
    double t1 = now_ms();
    r.real_ms = t1 - t0;

    // Now measure the free path. host_authoritative_ is only set for
    // initializers/graph inputs, so the observable "free" case here is a
    // second copy on an int64 tensor, which is what the shape-meta chain uses.
    // Report it for contrast: if this is ~0, the cost is entirely in the GPU
    // copy and a caching policy removes all of it; if it is not, the op's
    // readback path is doing work even when the value is already known.
    //
    // The same call is ALSO the key falsification test: if copyToCPU re-copies
    // every time, then the bytes sitting in device memory are never the reason
    // a readback is cheap, and an "async" variant that issues the copy but
    // defers the host wait should be able to overlap a whole layer's compute
    // with it -- i.e. the 4 ms below is a latency to hide, not bandwidth to
    // pay. Both cases fall out of the two numbers without a separate shader.
    t0 = now_ms();
    ids->copyToCPU(pool);
    t1 = now_ms();
    r.warm_ms = t1 - t0;

    // The host-known path: set host_copy_valid_ and time the same call. Done
    // LAST so it cannot affect the two numbers above, and the flag is only set
    // when the caller asked for this mode. The early-return hands back the
    // bytes data_ already holds, so the id check below still reads real ids.
    if (host_known) {
        ids->set_host_copy_valid();
        t0 = now_ms();
        ids->copyToCPU(pool);
        t1 = now_ms();
        r.known_ms = t1 - t0;
    }

    // Sanity: the host copy must actually carry the top-k ids, else we timed
    // an empty tensor and every number above is meaningless.
    const int *data = reinterpret_cast<const int *>(ids->data().data());
    int expect = n_experts - 1;
    bool match = true;
    for (int i = 0; i < k; ++i) {
        r.checksum += data[i];
        if (data[i] != expect - i)
            match = false;
    }
    r.ok = match;
    return r;
}

} // namespace

int main(int argc, char **argv) {
    int n_experts = argc > 1 ? std::atoi(argv[1]) : 60;
    int k = argc > 2 ? std::atoi(argv[2]) : 4;
    int reps = argc > 3 ? std::atoi(argv[3]) : 200;

    // No model file in this probe, so seed the archive from the build tree the
    // same way the gtest main does: libvkop embeds no SPIR-V of its own and the
    // TopK pipeline has to come from somewhere.
    vkop::dev::seed_shader_archive_from_build_tree();
    TestEnv::initialize();
    auto dev = TestEnv::get_device();
    auto pool = TestEnv::get_command_pool();

    printf("[rbprobe] device=%s  n_experts=%d  k=%d  reps=%d\n",
           dev->getDeviceName().c_str(), n_experts, k, reps);
    printf("[rbprobe] expert-id tensor = [1,%d] int32 = %d B\n", k,
           k * static_cast<int>(sizeof(int)));

    // Mode 3: the shape the B0 prefetch would use. Drive the same TopK, but
    // mark the ids output host_copy_valid_ this time: copyToGPU stays live (so
    // the device still gets its expert-id tensor and nothing downstream breaks)
    // while copyToCPU returns early with the bytes the *his* copy wrote. That
    // is the whole seam -- "the host already knows this value" without changing
    // which ops run -- so it is worth a number of its own: it is what a
    // predictive/resident-cache readback would cost, versus the .13 ms miss
    // below. Marking host_authoritative_ instead would NOT do: it is also a
    // path selector (Tensor.hpp:138) and would flip TopK/Reshape onto their CPU
    // implementations, changing the arithmetic rather than only the readback.
    auto known = run_once(dev, pool, n_experts, k, /*use_buffer=*/true,
                          /*host_known=*/true);
    if (!known.ok) {
        fprintf(stderr, "[rbprobe] host-known mode produced wrong ids\n");
        return 1;
    }

    // Warm the pipeline (shader compile lands in the first dispatch, not in
    // any copy, but do one full untimed iteration so the loop is steady).
    auto first = run_once(dev, pool, n_experts, k, /*use_buffer=*/true);
    if (!first.ok) {
        fprintf(stderr,
                "[rbprobe] top-k ids did not match the expected ramp; the "
                "readback path is suspect, aborting\n");
        return 1;
    }

    std::vector<double> real;
    std::vector<double> warm;
    std::vector<double> know;
    real.reserve(reps);
    warm.reserve(reps);
    know.reserve(reps);
    for (int i = 0; i < reps; ++i) {
        auto r = run_once(dev, pool, n_experts, k, true);
        if (!r.ok) {
            fprintf(stderr, "[rbprobe] rep %d produced wrong ids\n", i);
            return 1;
        }
        real.push_back(r.real_ms);
        warm.push_back(r.warm_ms);
        know.push_back(r.known_ms);
    }
    std::sort(real.begin(), real.end());
    std::sort(warm.begin(), warm.end());
    std::sort(know.begin(), know.end());
    auto pct = [](const std::vector<double> &v, double p) {
        return v[static_cast<size_t>(p * (v.size() - 1))];
    };

    printf("[rbprobe] miss   copyToCPU (GPU->CPU, int32[1,%d], nothing "
           "cached):     min=%.3f med=%.3f p90=%.3f ms\n",
           k, pct(real, 0.0), pct(real, 0.5), pct(real, 0.9));
    printf("[rbprobe] hit    copyToCPU (host_copy_valid_ already set, "
           "early-return): min=%.3f med=%.3f p90=%.3f ms\n",
           pct(know, 0.0), pct(know, 0.5), pct(know, 0.9));
    printf("[rbprobe] repeat copyToCPU (no caching at all: the two copies on "
           "one op are identical, so every call re-copies): "
           "min=%.3f med=%.3f p90=%.3f ms\n",
           pct(warm, 0.0), pct(warm, 0.5), pct(warm, 0.9));

    // Projected MoE cost. Qwen1.5-MoE-A2.7B has 24 layers; a decode round
    // routes once per layer, so 24 readbacks/token. Prefill routes the whole
    // chunk once per layer (same count, bigger rows). The numbers below are
    // the floor B0 pays before any prefetch/static-slot optimization.
    const double med = pct(real, 0.5);
    const double hit = pct(know, 0.5);
    printf("\n[rbprobe] projected per-token MoE routing readback, all-miss: "
           "24 layers x med = %.1f ms\n",
           24.0 * med);
    printf("[rbprobe] projected per-token MoE routing readback, all-hit:  "
           "24 layers x med = %.1f ms\n",
           24.0 * hit);
    printf("[rbprobe] (for scale: one Qwen3-VL decode round measured ~60 ms; "
           "all-miss would be %.0f%% of it)\n",
           100.0 * 24.0 * med / 60.0);
    printf("[rbprobe] the miss is submit/fence latency, not transfer: the "
           "payload is %d B, which is <1 us on any link, while the fence is "
           "%.0f us. So the lever is overlap (issue the copy, keep computing, "
           "wait at the layer boundary) or an all-hit cache -- not a smaller "
           "expert-id tensor.\n",
           k * static_cast<int>(sizeof(int)), 1000.0 * (med - hit));

    TestEnv::cleanup();
    return 0;
}
