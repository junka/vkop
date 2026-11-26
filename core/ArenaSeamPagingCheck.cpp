// Does the shared paged arena stay correct when bands are paged in through the
// per-level seam?
//
// The two shipped pieces were each proven on their own:
//   * the level seam (LevelHookPagingCheck) pages each band into that weight's
//     OWN resident buffer, so every band has a distinct address;
//   * the shared arena (UnloadedInitializerCheck) aliases every paged weight
//     onto ONE slot, so every band has the SAME address.
// Neither check exercises both at once, and that combination is exactly what a
// driver that wants to save VRAM must use.
//
// The combination has a hazard the separate proofs cannot see. In graph-submit
// mode a segment packs many levels into one command buffer that is submitted
// only when the segment closes. If every band shares offset 0, then by the time
// the segment is submitted, the copy for the LAST level of the segment has
// overwritten the slot that the FIRST level's dispatches read. Unshared bands
// do not have this problem: each band's bytes live at their own address, so a
// later page-in cannot move an earlier one.
//
// So this check builds the situation and requires the two arms to agree:
//   shared arena  + seam paging  ==  fresh LoadModel of the paged-in artifact
// If they disagree, the arena is only safe with one band per segment (or with a
// flush before each page-in), and the runtime must say so rather than silently
// compute garbage.
//
// Build:
//   g++ -std=gnu++17 -O2 -mf16c -DVK_ENABLE_BETA_EXTENSIONS \
//       -I<src> -I<src>/model -I<src>/model/SYSTEM -I<src>/model/generated \
//       -xc++ core/ArenaSeamPagingCheck.cpp -o asp \
//       <build>/libvkop.a <build>/model/libvload.a -ldl -lpthread
//
//   ./asp <chain_a.vkopbin> <chain_b.vkopbin>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/Tensor.hpp"
#include "core/WeightPager.hpp"
#include "core/runtime.hpp"
#include "include/logger.hpp"
#include "model/load.hpp"
#include "vulkan/VulkanCommandPool.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"

using vkop::VulkanCommandPool;
using vkop::VulkanDevice;
using vkop::VulkanInstance;
using vkop::core::ITensor;
using vkop::core::Runtime;
using vkop::core::WeightPager;

namespace {

int g_failures = 0;
void Check(bool ok, const std::string &what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

void fill_input(const std::shared_ptr<Runtime> &rt,
                const std::shared_ptr<VulkanCommandPool> &pool) {
    auto t = rt->GetInput("x");
    if (!t) {
        std::printf("FAIL: the model has no input named 'x'\n");
        ++g_failures;
        return;
    }
    size_t n = 1;
    for (int d : t->getShape()) n *= static_cast<size_t>(d);
    std::uint32_t s = 0xC0FFEEu;
    auto rnd = [&s]() {
        s = s * 1664525u + 1013904223u;
        return (static_cast<float>((s >> 8) & 0xFFFF) / 65535.0f) - 0.5f;
    };
    std::vector<uint16_t> v(n);
    for (auto &x : v) x = ITensor::fp32_to_fp16(rnd());
    if (auto h = vkop::core::as_tensor<uint16_t>(t)) {
        h->fillToCPU(v);
        // Without this the GPU reads an empty buffer and every comparison in
        // this check degenerates to 0 == 0 -- which is exactly how the first
        // run of this check "passed" on a fixture whose output was all zeros.
        h->copyToGPU(pool);
    } else if (auto f = vkop::core::as_tensor<float>(t)) {
        std::vector<float> w(n);
        for (size_t i = 0; i < n; ++i) w[i] = ITensor::fp16_to_fp32(v[i]);
        f->fillToCPU(w);
        f->copyToGPU(pool);
    }
}

std::vector<uint16_t> run_and_read(const std::shared_ptr<Runtime> &rt,
                                  const std::shared_ptr<VulkanCommandPool> &pool) {
    rt->Run();
    auto h = vkop::core::as_tensor<uint16_t>(rt->GetOutput());
    h->copyToCPU(pool);
    return h->data();
}

int diff(const std::vector<uint16_t> &a, const std::vector<uint16_t> &b) {
    if (a.size() != b.size()) return -1;
    int bad = 0;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) ++bad;
    return bad;
}

std::shared_ptr<Runtime> load_plain(const std::shared_ptr<VulkanCommandPool> &pool,
                                    const std::string &path) {
    auto rt = std::make_shared<Runtime>(pool, path, /*precision=*/0);
    rt->set_backend_buffer(true);
    rt->LoadModel();
    fill_input(rt, pool);
    return rt;
}

// weight name -> the level that reads it
std::map<std::string, size_t> level_of_weight(const vkop::load::VkModel &m) {
    std::map<std::string, size_t> out;
    std::map<std::string, size_t> level_of_node;
    for (size_t li = 0; li < m.concurrent_execution_levels.size(); ++li)
        for (const auto &n : m.concurrent_execution_levels[li])
            level_of_node[n] = li;
    for (const auto &n : m.nodes) {
        auto it = level_of_node.find(n.name);
        if (it == level_of_node.end()) continue;
        for (const auto &in : n.inputs)
            if (m.initializers.count(in.name)) out[in.name] = it->second;
    }
    return out;
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: asp <chain_a.vkopbin> <chain_b.vkopbin>\n");
        return 2;
    }
    const std::string path_a = argv[1], path_b = argv[2];

    Logger::getInstance().setLevel(LOG_WARNING);
    const auto &phy = VulkanInstance::getVulkanInstance().getPhysicalDevices();
    auto dev = std::make_shared<VulkanDevice>(phy[0]);
    if (dev->getDeviceName().find("llvmpipe") != std::string::npos) {
        std::printf("no valid vulkan device\n");
        return 1;
    }
    auto pool = std::make_shared<VulkanCommandPool>(dev);
    std::printf("[asp] device: %s\n", dev->getDeviceName().c_str());

    // Ground truth: B loaded normally, every weight resident at its own address.
    auto rt_b = load_plain(pool, path_b);
    const auto gt_b = run_and_read(rt_b, pool);
    rt_b.reset();
    // A degenerate ground truth makes every comparison below trivially true:
    // an all-zero output "matches" any other all-zero output, so a check
    // written this way reports PASS while proving nothing. That happened on a
    // Relu-chained fixture whose output collapsed to zero. Refuse to proceed.
    {
        size_t nonzero = 0;
        for (auto v : gt_b)
            if (v != 0) ++nonzero;
        std::printf("[asp] ground truth: %zu of %zu element(s) non-zero\n",
                    nonzero, gt_b.size());
        Check(nonzero * 4 >= gt_b.size(),
              "the ground truth is not degenerate (>=25% non-zero), so a "
              "mismatch is actually detectable");
        if (nonzero * 4 < gt_b.size()) {
            std::printf("[asp] refusing to compare against a near-zero ground "
                        "truth; use a fixture whose output depends on the "
                        "weights\n");
            std::fflush(stdout);
            _exit(1);
        }
    }

    vkop::load::VkModel mb(path_b);
    const auto lvl_of = level_of_weight(mb);
    std::printf("[asp] %zu weight(s), %zu with a single reading level\n",
                mb.initializers.size(), lvl_of.size());

    // Everything the plan needs, declared once.
    std::vector<std::string> names;
    for (const auto &kv : mb.initializers) names.push_back(kv.first);

    // ---- the two arms differ ONLY in whether the bands share a slot ----
    auto run_arm = [&](bool share) -> std::vector<uint16_t> {
        auto rt = std::make_shared<Runtime>(pool, path_a, 0);
        rt->set_backend_buffer(true);
        vkop::core::PagedArenaPlan plan;
        plan.names = names;
        plan.share_slots = share;
        rt->set_paged_initializers(std::move(plan));
        rt->LoadModel();
        fill_input(rt, pool);

        WeightPager pager(rt, pool);
        for (const auto &kv : mb.initializers) {
            vkop::core::PagedSource s;
            s.name = kv.first;
            s.path = path_b;
            s.offset =
                mb.initializer_file_offset + mb.initializer_offsets.at(kv.first);
            s.size = kv.second.size;
            auto it = lvl_of.find(kv.first);
            s.group = "level:" +
                      std::to_string(it == lvl_of.end() ? 1u << 30 : it->second);
            pager.add(std::move(s));
        }
        size_t calls = 0, bytes = 0;
        pager.attach_to(rt, [&](const vkop::core::PageStats &st) {
            ++calls;
            bytes += st.bytes;
        });
        auto out = run_and_read(rt, pool);
        std::printf("[asp] %s slots: %zu page-in(s), %zu bytes, arena %.2f MiB"
                    ", %zu level(s)\n",
                    share ? "shared" : "per-weight", calls, bytes,
                    rt->paged_arena() ? rt->paged_arena()->getSize() / 1048576.0
                                      : 0.0,
                    rt->level_count());
        Check(calls == lvl_of.size(),
              share ? "shared arm pages once per band-owning level"
                    : "per-weight arm pages once per band-owning level");
        return out;
    };

    const auto per_weight = run_arm(false);
    const auto shared = run_arm(true);

    std::printf("[asp] per-weight slots vs B: %d mismatch(es)\n",
                diff(per_weight, gt_b));
    std::printf("[asp] shared slots    vs B: %d mismatch(es); vs per-weight: %d\n",
                diff(shared, gt_b), diff(shared, per_weight));
    if (diff(shared, per_weight) != 0) {
        auto f16 = [](uint16_t v) { return ITensor::fp16_to_fp32(v); };
        std::printf("[asp] first 8 shared / per-weight / B:\n");
        for (size_t i = 0; i < 8 && i < shared.size(); ++i)
            std::printf("  [%zu] %.5f / %.5f / %.5f\n", i, f16(shared[i]),
                        f16(per_weight[i]), f16(gt_b[i]));
    }

    // The per-weight arm is the already-proven behaviour; keep it as a
    // reference so a failure here is attributed to sharing, not to the seam.
    Check(diff(per_weight, gt_b) == 0,
          "unshared bands paged through the seam reproduce B exactly (the "
          "already-proven behaviour)");
    Check(diff(shared, per_weight) == 0,
          "sharing a slot does not change the result: every band a level reads "
          "is the band that level's dispatches see");

    if (g_failures == 0) {
        std::printf("[asp] PASS: a shared arena is safe across the level seam "
                    "on this graph\n");
    } else {
        std::printf("[asp] %d FAILURE(S) -- a shared slot is NOT safe with the "
                    "seam as it stands: an earlier level's unsubmitted segment "
                    "reads the slot after a later level overwrote it\n",
                    g_failures);
    }
    std::fflush(stdout);
    _exit(g_failures == 0 ? 0 : 1);
}
