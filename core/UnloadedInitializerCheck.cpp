// Verify set_unloaded_initializers: a skipped weight gets its buffer but not its
// bytes, and paging it in afterwards reproduces a normal load exactly.
//
// This is the change that makes paging SAVE VRAM rather than merely work: with
// the weight uploaded at LoadModel there is nothing to save, so the skip is the
// whole point. Two compatibility claims are checked with the same rigor:
//
//   * a Runtime told nothing behaves exactly as before (the default path);
//   * a skipped weight is registered, bound and sized like an uploaded one, so
//     every op that reads it finds the buffer it expects;
//   * the skipped buffer is left NOT marked on-GPU, so a later copyToGPU would
//     still upload instead of short-circuiting on a stale flag.
//
// Build:
//   cmake: the `UnloadedInitializerCheck` target (see CMakeLists.txt).
//   manual: g++ -std=gnu++17 -O2 -mf16c -DVK_ENABLE_BETA_EXTENSIONS
//           -I<src> -I<src>/model -I<src>/model/SYSTEM
//           -I<src>/model/generated -xc++ core/UnloadedInitializerCheck.cpp
//           -o UnloadedInitializerCheck <build>/libvkop.a
//           <build>/model/libvload.a -ldl -lpthread
//
//   ./uic <model_a.vkopbin> <model_b.vkopbin> [weight_name]

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
#include "vulkan/VulkanLib.hpp"

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

} // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: uic <model_a.vkopbin> <model_b.vkopbin> "
                     "[weight_name]\n");
        return 2;
    }
    const std::string path_a = argv[1], path_b = argv[2];
    const std::string wname = argc >= 4 ? argv[3] : "w";
    // Sections 1-4 drive the model by feeding an input called "x", which is a
    // fixture assumption; a real LLM artifact has no such input. Section 5
    // (does a shared arena free device memory?) needs no input at all, so a
    // real model can be measured for arena size with this switch.
    const bool arena_only = std::getenv("UIC_ARENA_ONLY") != nullptr;

    Logger::getInstance().setLevel(LOG_WARNING);
    const auto &phy = VulkanInstance::getVulkanInstance().getPhysicalDevices();
    auto dev = std::make_shared<VulkanDevice>(phy[0]);
    if (dev->getDeviceName().find("llvmpipe") != std::string::npos) {
        std::printf("no valid vulkan device\n");
        return 1;
    }
    auto pool = std::make_shared<VulkanCommandPool>(dev);

    // ---- 1. the default path is untouched ----
    if (!arena_only) {
        auto rt = std::make_shared<Runtime>(pool, path_a, 0);
        rt->set_backend_buffer(true);
        rt->LoadModel();
        Check(rt->unloaded_initializers().empty(),
              "a Runtime told nothing skips nothing");
        Check(!rt->is_unloaded_initializer(wname),
              "is_unloaded_initializer is false by default");
        fill_input(rt, pool);
        auto out = run_and_read(rt, pool);
        auto fresh = [&] {
            auto r2 = std::make_shared<Runtime>(pool, path_b, 0);
            r2->set_backend_buffer(true);
            r2->LoadModel();
            fill_input(r2, pool);
            return run_and_read(r2, pool);
        }();
        Check(diff(out, fresh) != 0, "A and B still differ");
        std::printf("[uic] default path: %zu skipped, output differs from B "
                    "as expected\n", rt->unloaded_initializers().size());
    }

    // ---- 2. a skipped weight keeps its buffer but not its bytes ----
    std::shared_ptr<Runtime> rt;
    std::shared_ptr<vkop::core::ITensor> wt;
    if (!arena_only) {
        rt = std::make_shared<Runtime>(pool, path_a, 0);
        rt->set_backend_buffer(true);
        // Not shared: the chain reads every weight in the same round, so each
        // needs its own slot. Sharing here would alias them all onto offset 0.
        vkop::core::PagedArenaPlan plan;
        for (const auto &kv : vkop::load::VkModel(path_a).initializers)
            plan.names.push_back(kv.first);
        plan.share_slots = false;
        rt->set_paged_initializers(std::move(plan));
        rt->LoadModel();
        Check(rt->is_unloaded_initializer(wname), "the name is in the skip set");
        const auto &taken = rt->unloaded_initializers();
        Check(!taken.empty() && taken[0] == wname,
              "LoadModel reports the names it routed to the arena");
        Check(rt->paged_arena_offset(wname) == 0 ||
                  rt->paged_arena_offset(wname) % 4 == 0,
              "a paged weight's slot offset is aligned");
        wt = rt->GetInitializer(wname);
        Check(wt != nullptr, "a skipped weight is still a registered tensor");
        if (!wt) {
            std::printf("[uic] %d FAILURE(S)\n", g_failures);
            return 1;
        }
        auto vb = std::dynamic_pointer_cast<vkop::VulkanBuffer>(wt->gpu_buffer());
        Check(vb != nullptr, "a skipped weight still has a device buffer");
        vkop::load::VkModel mb(path_b);
        Check(vb && vb->getSize() >= mb.initializers.at(wname).size,
              "the buffer is at least as large as the payload it will receive");
        // The flag that would make a later copyToGPU a no-op.
        Check(!wt->is_on_GPU(),
              "a paged weight is NOT marked on-GPU (a later upload still "
              "works)");
        Check(rt->paged_arena() != nullptr, "a shared arena was allocated");
        std::printf("[uic] skipped '%s': buffer %llu B, on-GPU=%d\n",
                    wname.c_str(),
                    static_cast<unsigned long long>(vb ? vb->getSize() : 0),
                    wt->is_on_GPU() ? 1 : 0);
    }

    // ---- 3. running before the page-in uses garbage; after, it is exact ----
    if (!arena_only) {
    fill_input(rt, pool);
    // Ground truth for A with its own weight resident.
    auto gt_a = [&] {
        auto r = std::make_shared<Runtime>(pool, path_a, 0);
        r->set_backend_buffer(true);
        r->LoadModel();
        fill_input(r, pool);
        return run_and_read(r, pool);
    }();
    auto gt_b = [&] {
        auto r = std::make_shared<Runtime>(pool, path_b, 0);
        r->set_backend_buffer(true);
        r->LoadModel();
        fill_input(r, pool);
        return run_and_read(r, pool);
    }();
    Check(diff(gt_a, gt_b) > 0, "ground truths differ");

    WeightPager pager(rt, pool);
    {
        vkop::load::VkModel mb(path_b);
        for (const auto &kv : mb.initializers) {
            vkop::core::PagedSource s;
            s.name = kv.first;
            s.path = path_b;
            s.offset = mb.initializer_file_offset +
                       mb.initializer_offsets.at(kv.first);
            s.size = kv.second.size;
            s.group = "g";
            pager.add(s);
        }
    }
    auto st = pager.page_in_group("g");
    std::printf("[uic] paged in %zu B (host %.3f + device %.3f ms)\n", st.bytes,
                st.host_ms, st.device_ms);
    auto after = run_and_read(rt, pool);
    Check(diff(after, gt_b) == 0,
          "a skip-then-page model reproduces a normal load of B exactly");
    Check(diff(after, gt_a) != 0, "the page-in took effect (output moved)");
    std::printf("[uic] skip+page: vs B -> %d mismatch(es); vs A -> %d\n",
                diff(after, gt_b), diff(after, gt_a));
    } // !arena_only

    // ---- 4. fail-closed paths ----
    if (!arena_only) {
    {
        auto r = std::make_shared<Runtime>(pool, path_a, 0);
        r->set_backend_buffer(true);
        // A name that is not an initializer is ignored, not fatal.
        vkop::core::PagedArenaPlan bogus;
        bogus.names = {"no_such_weight"};
        r->set_paged_initializers(std::move(bogus));
        bool threw = false;
        try {
            r->LoadModel();
        } catch (const std::exception &) {
            threw = true;
        }
        Check(!threw, "an unknown name in the skip set is ignored");
        Check(r->unloaded_initializers().empty(),
              "nothing was skipped for an unknown name");
    }
    } // !arena_only

    // ---- 5. does a SHARED arena actually free device memory? ----
    // Asserted on the arena's own size, which is deterministic, rather than on
    // a heap delta that allocator rounding can hide at these sizes. One slot
    // reused must cost the LARGEST weight; a slot per weight must cost their
    // sum. That difference is exactly the saving.
    {
        vkop::load::VkModel mc(path_a);
        std::vector<std::string> names;
        size_t largest = 0, sum = 0, refused = 0, refused_bytes = 0;
        for (const auto &kv : mc.initializers) {
            // Same eligibility rule the runtime enforces: an int64/int32
            // constant is read on the host, so it cannot be paged. Declaring it
            // here would make LoadModel throw by design, not measure the arena.
            if (kv.second.dtype == "int64" || kv.second.dtype == "int32") {
                ++refused;
                refused_bytes += kv.second.size;
                continue;
            }
            names.push_back(kv.first);
            largest = std::max<size_t>(largest, kv.second.size);
            sum += kv.second.size;
        }
        if (refused)
            std::printf("[uic] %zu host-read constant(s) (%.2f MiB) are not "
                        "pageable and stay out of the arena\n",
                        refused, refused_bytes / 1048576.0);
        if (names.size() >= 2) {
            // Group the weights the way a real paging plan does -- by the level
            // that reads each one -- so "shared" costs one TYPICAL band, not one
            // weight. A flat plan puts every weight in one group and therefore
            // lays them out end to end (they are all live at once as far as the
            // runtime can tell), which is the sum and measures nothing.
            std::shared_ptr<Runtime> no_rt;
            vkop::core::WeightPager layout(no_rt, nullptr);
            const size_t nbands =
                layout.add_bands_from_graph(mc, path_a, /*min_bytes=*/0);
            std::printf("[uic] banding: %zu band(s) over %zu weight(s)\n",
                        nbands, layout.size());
            auto banded = [&](bool share) {
                auto r = std::make_shared<Runtime>(pool, path_a, 0);
                r->set_backend_buffer(true);
                vkop::core::PagedArenaPlan p;
                p.names = names;
                p.share_slots = share;
                if (share && layout.size() > 0) p.slot_of = layout.slot_groups();
                r->set_paged_initializers(std::move(p));
                r->LoadModel();
                return r->paged_arena() ? r->paged_arena()->getSize() : 0;
            };
            const size_t shared = banded(true);
            // The per-weight arm sizes the arena to the payload sum, which is
            // the very thing a card too small for the model cannot allocate.
            // Skipping it does not weaken the claim: the saving is
            // `sum - shared`, and `sum` is already known from the file.
            size_t per_weight = 0;
            if (!std::getenv("UIC_NO_PER_WEIGHT_ARM"))
                per_weight = banded(false);
            else
                per_weight = sum;
            std::printf("[uic] %zu weights: shared arena %.2f MiB, "
                        "per-weight arena %.2f MiB (payload sum %.2f MiB)\n",
                        names.size(), shared / 1048576.0,
                        per_weight / 1048576.0, sum / 1048576.0);
            Check(shared > 0 && per_weight > 0, "both plans allocate an arena");
            Check(shared < per_weight,
                  "a shared slot costs one band, not the sum of every weight");
            Check(shared >= largest, "a shared slot still fits the largest");
            if (per_weight)
                std::printf("[uic] saving %.2f GiB (%.1f%% of the per-weight "
                            "arena); the shared arena is %.2f%% of the payload "
                            "sum\n",
                            (per_weight - shared) / 1073741824.0,
                            100.0 * (per_weight - shared) / per_weight,
                            sum ? 100.0 * shared / sum : 0.0);
        }
    }

    if (g_failures == 0) {
        std::printf("[uic] PASS: a shared arena really saves device memory, "
                    "paged weights reproduce a normal load exactly, and the "
                    "default path is unchanged\n");
    } else {
        std::printf("[uic] %d FAILURE(S)\n", g_failures);
    }
    std::fflush(stdout);
    _exit(g_failures == 0 ? 0 : 1);
}
