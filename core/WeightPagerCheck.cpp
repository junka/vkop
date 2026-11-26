// Verify core/WeightPager.hpp against the thing it claims: after
// page_in_group(), the model computes what a fresh LoadModel of the paged-in
// artifact computes.
//
// This is the layer above tools/probes/weight_paging_probe.cpp. That one proved
// the primitive (an offset copy into a live weight buffer is exact); this one
// proves the class that a driver would actually call, including its two
// failure modes -- a source larger than its buffer, and a name the runtime does
// not hold -- fail closed instead of writing a partial band.
//
// Two artifacts are converted from the same graph with different weight bytes,
// so "did the page-in take effect" and "is the result right" are the same
// question with an independent answer (model_b loaded fresh).
//
// Build (against a libvkop from the same source revision):
//   cmake: the `WeightPagerCheck` target (see CMakeLists.txt).
//   manual: g++ -std=gnu++17 -O2 -mf16c -DVK_ENABLE_BETA_EXTENSIONS
//           -I<src> -I<src>/model -I<src>/model/SYSTEM
//           -I<src>/model/generated -xc++ core/WeightPagerCheck.cpp
//           -o WeightPagerCheck <build>/libvkop.a
//           <build>/model/libvload.a -ldl -lpthread
//
//   ./wpcheck <model_a.vkopbin> <model_b.vkopbin> [weight_name]

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    const auto dims = t->getShape();
    size_t n = 1;
    for (int d : dims) n *= static_cast<size_t>(d);
    std::uint32_t s = 0xC0FFEEu;
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
}

std::vector<uint16_t> run_and_read(const std::shared_ptr<Runtime> &rt,
                                   const std::shared_ptr<VulkanCommandPool> &pool,
                                   bool refill) {
    if (refill) fill_input(rt, pool);
    rt->Run();
    auto out = rt->GetOutput();
    auto h = vkop::core::as_tensor<uint16_t>(out);
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

std::shared_ptr<Runtime> load(const std::shared_ptr<VulkanCommandPool> &pool,
                              const std::string &path) {
    auto rt = std::make_shared<Runtime>(pool, path, /*precision=*/0);
    rt->set_backend_buffer(true);
    rt->LoadModel();
    fill_input(rt, pool);
    return rt;
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: wpcheck <model_a.vkopbin> <model_b.vkopbin> "
                     "[weight_name]\n");
        return 2;
    }
    const std::string path_a = argv[1], path_b = argv[2];
    const std::string wname = argc >= 4 ? argv[3] : "w";

    Logger::getInstance().setLevel(LOG_WARNING);
    const auto &phydevs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
    auto dev = std::make_shared<VulkanDevice>(phydevs[0]);
    if (dev->getDeviceName().find("llvmpipe") != std::string::npos) {
        std::printf("no valid vulkan device\n");
        return 1;
    }
    auto pool = std::make_shared<VulkanCommandPool>(dev);
    std::printf("[wpc] device: %s\n", dev->getDeviceName().c_str());

    // Ground truth: each artifact with its own weight resident.
    auto rt_a = load(pool, path_a);
    const auto gt_a = run_and_read(rt_a, pool, false);
    auto rt_b = load(pool, path_b);
    const auto gt_b = run_and_read(rt_b, pool, false);
    if (diff(gt_a, gt_b) <= 0) {
        std::printf("[wpc] FAIL: artifacts compute the same output\n");
        return 1;
    }
    std::printf("[wpc] ground truth: A and B differ (%d of %zu)\n",
                diff(gt_a, gt_b), gt_a.size());
    rt_b.reset(); // free B's weights; the pager works on A

    // Locate B's weight bytes.
    vkop::load::VkModel mb(path_b);
    auto it = mb.initializers.find(wname);
    if (it == mb.initializers.end()) {
        std::printf("[wpc] FAIL: %s has no initializer '%s'\n", path_b.c_str(),
                    wname.c_str());
        return 1;
    }
    const size_t nbytes = static_cast<size_t>(it->second.size);

    // ---- the class, as a driver would use it ----
    WeightPager pager(rt_a, pool);
    vkop::core::PagedSource src;
    src.name = wname;
    src.path = path_b;
    src.offset = mb.initializer_file_offset + mb.initializer_offsets.at(wname);
    src.size = nbytes;
    src.group = "layer0";
    pager.add(src);
    Check(pager.size() == 1, "one source declared");
    {
        auto gs = pager.groups();
        Check(gs.size() == 1 && gs[0] == "layer0", "one group, named layer0");
    }
    // Same name again must replace, not duplicate (a model swap re-declares).
    pager.add(src);
    Check(pager.size() == 1, "re-declaring a name does not duplicate it");

    auto st = pager.page_in_group("layer0");
    std::printf("[wpc] page_in_group: %zu weight(s), %zu bytes, "
                "host %.3f ms + device %.3f ms (%.1f GB/s)\n",
                st.weights, st.bytes, st.host_ms, st.device_ms,
                st.device_gbps());
    // Second call: same band, staging already allocated, file already in page
    // cache. This is the steady-state cost a paging schedule actually pays.
    // The first call's host time includes a one-off staging allocation and the
    // cold file read, which the budget must account for separately.
    {
        auto st2 = pager.page_in_group("layer0");
        std::printf("[wpc] page_in_group (steady state): host %.3f ms + "
                    "device %.3f ms (%.1f GB/s)\n",
                    st2.host_ms, st2.device_ms, st2.device_gbps());
    }
    Check(st.weights == 1 && st.bytes == nbytes, "stats cover the whole band");

    const auto after = run_and_read(rt_a, pool, false);
    Check(diff(after, gt_a) != 0,
          "output changed, so the page-in took effect");
    Check(diff(after, gt_b) == 0,
          "paged model reproduces a fresh LoadModel of B exactly");
    std::printf("[wpc] after page-in: vs B -> %d mismatch(es); vs A -> %d\n",
                diff(after, gt_b), diff(after, gt_a));

    // ---- fail-closed: a source bigger than the buffer ----
    {
        WeightPager bad(rt_a, pool);
        auto big = src;
        big.size = nbytes + 4096;
        bad.add(big);
        bool threw = false;
        try {
            bad.page_in(wname);
        } catch (const std::exception &e) {
            threw = true;
            std::printf("[wpc] oversized source refused: %s\n", e.what());
        }
        Check(threw, "an oversized source is refused, not partially written");
    }
    // ---- fail-closed: a name the runtime does not hold ----
    {
        WeightPager bad(rt_a, pool);
        auto ghost = src;
        ghost.name = "no_such_weight";
        bad.add(ghost);
        bool threw = false;
        try {
            bad.page_in("no_such_weight");
        } catch (const std::exception &e) {
            threw = true;
            std::printf("[wpc] unknown weight refused: %s\n", e.what());
        }
        Check(threw, "an unknown weight is refused");
    }

    // The failed attempts must not have touched the live weights.
    const auto end_state = run_and_read(rt_a, pool, false);
    Check(diff(end_state, gt_b) == 0,
          "failed page-in attempts left the resident band intact");

    if (g_failures == 0) {
        std::printf("[wpc] PASS: WeightPager pages a band in exactly, and both "
                    "failure modes fail closed\n");
    } else {
        std::printf("[wpc] %d FAILURE(S)\n", g_failures);
    }
    std::fflush(stdout);
    _exit(g_failures == 0 ? 0 : 1); // NVIDIA's EGL atexit handler segfaults
}
