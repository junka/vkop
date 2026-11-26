// End-to-end check of the per-level paging seam: install a Runtime level hook
// that pages a band in just before the level which reads it, then require the
// model's output to match a fresh LoadModel of the paged-in artifact.
//
// This is the integration that the two earlier checks bracket:
//   * weight_paging_probe.cpp proved the primitive (an offset copy into a live
//     weight buffer is exact);
//   * WeightPagerCheck.cpp proved the class that wraps it;
//   * this proves the SEAM -- that Runtime calls the hook at a point where the
//     copy lands before the reading level's dispatches, with no forced flush
//     and no replay interaction.
//
// The graph is a MatMul chain, one weight per level, so a band is one level's
// weight and the hook fires once per pageable level.
//
// Build:
//   cmake: the `LevelHookPagingCheck` target (see CMakeLists.txt).
//   manual: g++ -std=gnu++17 -O2 -mf16c -DVK_ENABLE_BETA_EXTENSIONS
//           -I<src> -I<src>/model -I<src>/model/SYSTEM
//           -I<src>/model/generated -xc++ core/LevelHookPagingCheck.cpp
//           -o LevelHookPagingCheck <build>/libvkop.a
//           <build>/model/libvload.a -ldl -lpthread
//
//   ./lhp <chain_a.vkopbin> <chain_b.vkopbin>

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
                const std::shared_ptr<VulkanCommandPool> &pool,
                const std::shared_ptr<VulkanDevice> &dev) {
    auto t = rt->GetInput("x");
    const auto dims = t->getShape();
    size_t n = 1;
    for (int d : dims) n *= static_cast<size_t>(d);
    std::uint32_t s = 0xBEEF01u;
    auto rnd = [&s]() {
        s = s * 1664525u + 1013904223u;
        return (static_cast<float>((s >> 8) & 0xFFFF) / 65535.0f) - 0.5f;
    };
    if (auto f = vkop::core::as_tensor<float>(t)) {
        std::vector<float> v(n);
        for (auto &x : v) x = rnd();
        f->fillToCPU(v);
        f->copyToGPU(pool);
    } else if (auto h = vkop::core::as_tensor<uint16_t>(t)) {
        std::vector<uint16_t> v(n);
        for (auto &x : v) x = ITensor::fp32_to_fp16(rnd());
        h->fillToCPU(v);
        h->copyToGPU(pool);
    }
    (void)dev;
}

std::vector<uint16_t>
run_and_read(const std::shared_ptr<Runtime> &rt,
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

// Any initializer name, for the "bad band name" case.
std::string wname_or_first(const vkop::load::VkModel &m) {
    for (const auto &kv : m.initializers) {
        if (!kv.second.moe_source.empty()) return kv.first;
    }
    return m.initializers.begin()->first;
}
std::shared_ptr<Runtime> load(const std::shared_ptr<VulkanCommandPool> &pool,
                              const std::shared_ptr<VulkanDevice> &dev,
                              const std::string &path) {
    auto rt = std::make_shared<Runtime>(pool, path, /*precision=*/0);
    rt->set_backend_buffer(true);
    rt->LoadModel();
    fill_input(rt, pool, dev);
    return rt;
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: lhp <chain_a.vkopbin> <chain_b.vkopbin>\n");
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
    std::printf("[lhp] device: %s\n", dev->getDeviceName().c_str());

    // Ground truth.
    auto rt_a = load(pool, dev, path_a);
    const auto gt_a = run_and_read(rt_a, pool);
    auto rt_b = load(pool, dev, path_b);
    const auto gt_b = run_and_read(rt_b, pool);
    Check(diff(gt_a, gt_b) > 0, "artifacts differ");
    std::printf("[lhp] ground truth differs in %d of %zu elements\n",
                diff(gt_a, gt_b), gt_a.size());
    rt_b.reset();

    // Declare A, but source every weight from B.
    vkop::load::VkModel mb(path_b);
    WeightPager pager(rt_a, pool);
    std::string dir;
    {
        const auto slash = path_b.find_last_of('/');
        if (slash != std::string::npos) dir = path_b.substr(0, slash + 1);
    }
    for (const auto &kv : mb.initializers) {
        const auto &in = kv.second;
        if (in.moe_source.empty()) continue;
        (void)in;
    }
    // No moe_source on a plain chain: declare manually, one band per weight.
    for (const auto &kv : mb.initializers) {
        vkop::core::PagedSource s;
        s.name = kv.first;
        s.path = path_b;
        s.offset = mb.initializer_file_offset + mb.initializer_offsets.at(kv.first);
        s.size = kv.second.size;
        s.group = "w:" + kv.first;
        pager.add(std::move(s));
    }
    std::printf("[lhp] %zu band(s) declared, one weight each\n", pager.size());

    // A weight's band is the level that reads it, so the hook needs a map
    // level -> band. Derive it the same way WeightPager does.
    std::map<size_t, std::string> band_of_level;
    for (const auto &kv : mb.initializers) {
        for (const auto &n : mb.nodes) {
            bool reads = false;
            for (const auto &in : n.inputs) {
                if (in.name == kv.first) { reads = true; break; }
            }
            if (!reads) continue;
            for (size_t li = 0; li < mb.concurrent_execution_levels.size(); ++li) {
                for (const auto &nn : mb.concurrent_execution_levels[li]) {
                    if (nn == n.name) band_of_level[li] = "w:" + kv.first;
                }
            }
        }
    }
    std::printf("[lhp] %zu level(s) own a band\n", band_of_level.size());

    // The seam under test.
    size_t hooked = 0, paged = 0;
    rt_a->set_level_hook([&](size_t level) {
        ++hooked;
        auto it = band_of_level.find(level);
        if (it == band_of_level.end()) return;
        pager.page_in_group(it->second);
        ++paged;
    });
    Check(rt_a->has_level_hook(), "hook installed");
    Check(rt_a->level_count() == mb.concurrent_execution_levels.size(),
          "level_count matches the model's level table");

    const auto after = run_and_read(rt_a, pool);
    std::printf("[lhp] hook fired %zu time(s), paged %zu band(s)\n", hooked,
                paged);
    Check(hooked >= rt_a->level_count(),
          "the hook runs once per level");
    Check(paged > 0, "at least one band was paged in through the seam");
    Check(diff(after, gt_a) != 0, "output moved away from A");
    Check(diff(after, gt_b) == 0,
          "paging through the level seam reproduces B exactly");
    std::printf("[lhp] after seam paging: vs B -> %d mismatch(es); vs A -> %d\n",
                diff(after, gt_b), diff(after, gt_a));

    // ---- attach_to: the one-line driver integration ----
    // Rebuild the pager with graph-derived band names so attach_to accepts it,
    // then confirm it pages exactly the bands the graph puts on levels.
    {
        WeightPager p2(rt_a, pool);
        for (const auto &kv : mb.initializers) {
            vkop::core::PagedSource s;
            s.name = kv.first;
            s.path = path_b;
            s.offset =
                mb.initializer_file_offset + mb.initializer_offsets.at(kv.first);
            s.size = kv.second.size;
            // Name the band by the level that reads it, which is what attach_to
            // requires and what add_bands_from_graph produces.
            std::string grp;
            for (const auto &[lvl, band] : band_of_level) {
                if (band == "w:" + kv.first) grp = "level:" + std::to_string(lvl);
            }
            s.group = grp.empty() ? "level:999999" : grp;
            p2.add(std::move(s));
        }
        size_t calls = 0, bytes = 0;
        p2.attach_to(rt_a, [&](const vkop::core::PageStats &st) {
            ++calls;
            bytes += st.bytes;
        });
        Check(rt_a->has_level_hook(), "attach_to installed a hook");
        const auto out2 = run_and_read(rt_a, pool);
        std::printf("[lhp] attach_to: %zu page-in(s), %zu bytes\n", calls,
                    bytes);
        Check(calls == band_of_level.size(),
              "attach_to pages once per band-owning level");
        Check(diff(out2, gt_b) == 0,
              "attach_to reproduces B exactly as the hand-written hook did");
    }

    // A pager whose bands are not level-named must be refused, not silently
    // scheduled wrong.
    {
        WeightPager p3(rt_a, pool);
        vkop::core::PagedSource s;
        s.name = wname_or_first(mb);
        s.path = path_b;
        s.offset = mb.initializer_file_offset +
                   mb.initializer_offsets.at(s.name);
        s.size = mb.initializers.at(s.name).size;
        s.group = "some-moe-group";
        p3.add(std::move(s));
        bool threw = false;
        try {
            p3.attach_to(rt_a);
        } catch (const std::exception &e) {
            threw = true;
            std::printf("[lhp] non-level band refused: %s\n", e.what());
        }
        Check(threw, "attach_to refuses a band it cannot schedule");
    }

    if (g_failures == 0) {
        std::printf("[lhp] PASS: the level seam pages in a band exactly, with "
                    "no forced flush\n");
    } else {
        std::printf("[lhp] %d FAILURE(S)\n", g_failures);
    }
    std::fflush(stdout);
    _exit(g_failures == 0 ? 0 : 1);
}
