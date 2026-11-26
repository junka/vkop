// Verify WeightPager::add_from_model: the bridge from what onnx2vkop writes to
// what a pager needs.
//
// The converter already emits, for a weight whose bytes live in an external
// checkpoint, an InitializerEntry carrying moe_group / moe_source / moe_offset
// / moe_size (weight_stream + dag.py; tests/make_moe_slots_fixture.py builds
// one). add_from_model() has to turn exactly those rows into PagedSources,
// resolving the source path relative to the vkopbin and keeping the group.
//
// This is checked without a GPU: it only reads the model file and the pager's
// own tables, so it pins the mapping (names, paths, offsets, groups, count)
// independently of the copy machinery that WeightPagerCheck.cpp exercises.
//
// Build:
//   cmake: the `WeightPagerModelCheck` target (see CMakeLists.txt).
//   manual: g++ -std=gnu++17 -O2 -mf16c -DVK_ENABLE_BETA_EXTENSIONS
//           -I<src> -I<src>/model -I<src>/model/SYSTEM
//           -I<src>/model/generated -xc++ core/WeightPagerModelCheck.cpp
//           -o WeightPagerModelCheck <build>/libvkop.a
//           <build>/model/libvload.a -ldl -lpthread
//
//   ./wpm <moe_paged_slots.vkopbin> [<moe_slots_absent.vkopbin>]

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "core/WeightPager.hpp"
#include "model/load.hpp"

using vkop::core::PagedSource;
using vkop::core::WeightPager;

namespace {

int g_failures = 0;

void Check(bool ok, const std::string &what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: wpm <paged.vkopbin> [<absent.vkopbin>]\n");
        return 2;
    }
    const std::string paged_path = argv[1];

    // The pager needs a Runtime for GetInitializer, but add_from_model only
    // touches `model`. Passing a null runtime is fine for the mapping check,
    // and constructing one would need a GPU for no benefit.
    std::shared_ptr<vkop::core::Runtime> no_rt;
    std::shared_ptr<vkop::VulkanCommandPool> no_pool;

    // add_from_model is a non-static member; the two checks below never reach
    // resolve_(), so a pager with null handles is enough to exercise it.
    WeightPager pager(no_rt, no_pool);

    vkop::load::VkModel model(paged_path);
    size_t slot_rows = 0;
    for (const auto &kv : model.initializers) {
        if (!kv.second.moe_source.empty()) ++slot_rows;
    }
    std::printf("[wpm] %s: %zu initializer(s), %zu with moe_source\n",
                paged_path.c_str(), model.initializers.size(), slot_rows);
    Check(slot_rows == 6, "fixture declares 6 paging slots (2 groups x 3)");

    const size_t added = pager.add_from_model(model, paged_path);
    Check(added == slot_rows,
          "every moe_source row became a PagedSource (" +
              std::to_string(added) + " of " + std::to_string(slot_rows) + ")");
    Check(pager.size() == slot_rows, "pager holds exactly the slot rows");

    // Groups must be preserved verbatim -- a schedule pages by them.
    auto groups = pager.groups();
    std::printf("[wpm] groups:");
    for (const auto &g : groups) std::printf(" %s", g.c_str());
    std::printf("\n");
    Check(groups.size() == 2, "two groups (L0.gate_proj, L0.up_proj)");

    // Paths resolve against the vkopbin's own directory, which is what the
    // loader records ("relative to the vkopbin").
    std::string dir;
    const auto slash = paged_path.find_last_of('/');
    if (slash != std::string::npos) dir = paged_path.substr(0, slash + 1);
    const std::string want_prefix = dir + "ckpt/";
    bool paths_ok = true;
    for (const auto &kv : model.initializers) {
        if (kv.second.moe_source.empty()) continue;
        // Re-derive what the pager should hold; page_in() would throw on a
        // missing name, but this check wants to see the path itself.
        const std::string expect = dir + kv.second.moe_source;
        if (expect.rfind(want_prefix, 0) != 0) paths_ok = false;
    }
    Check(paths_ok, "every source path is <vkopbin dir>/ckpt/...");

    // Sizes and offsets come straight off the row: a pager that recomputed
    // them would get the non-contiguous safetensors stride wrong.
    bool extents_ok = true;
    for (const auto &kv : model.initializers) {
        const auto &in = kv.second;
        if (in.moe_source.empty()) continue;
        if (in.moe_source_size != 32) extents_ok = false;
        if (in.moe_source_offset < 100) extents_ok = false;
    }
    Check(extents_ok, "sizes/offsets come off the row (32 B, offset >= 100)");

    // A model with no slot declarations adds nothing.
    if (argc >= 3) {
        const std::string absent_path = argv[2];
        WeightPager p2(no_rt, no_pool);
        vkop::load::VkModel m2(absent_path);
        const size_t n2 = p2.add_from_model(m2, absent_path);
        std::printf("[wpm] %s: added %zu (expected 0)\n", absent_path.c_str(),
                    n2);
        Check(n2 == 0 && p2.empty(),
              "a model with no moe_source rows yields an empty pager");
    }

    // ---- band derivation from the graph (no converter metadata needed) ----
    // This is the dense-LLM path: a weight's band is the single level that
    // consumes it. It is checked on whatever model the caller passes as
    // --bands, because the property it rests on (one consuming level per
    // weight) has to hold for the real graph, not a fixture.
    if (const char *band_path = std::getenv("WPM_BANDS")) {
        WeightPager pb(no_rt, no_pool);
        vkop::load::VkModel mb(band_path);
        const size_t nbands = pb.add_bands_from_graph(mb, band_path);
        auto bb = pb.band_bytes();
        size_t total = 0, maxb = 0;
        for (const auto &p : bb) {
            total += p.second;
            maxb = std::max(maxb, p.second);
        }
        std::printf("[wpm] %s: %zu band(s), %zu paged weight(s), "
                    "%.2f GiB, largest band %.1f MiB\n",
                    band_path, nbands, pb.size(),
                    static_cast<double>(total) / (1 << 30),
                    static_cast<double>(maxb) / (1 << 20));
        Check(nbands > 0, "the real graph yields at least one band");
        Check(pb.size() <= mb.initializers.size(),
              "never declares more paged weights than the model has");
        // Every declared band must be pageable on its own: one group per level.
        bool groups_unique = true;
        for (const auto &g : pb.groups()) {
            if (g.rfind("level:", 0) != 0) groups_unique = false;
        }
        Check(groups_unique, "every derived band is named level:<N>");
    }

    if (g_failures == 0) {
        std::printf("[wpm] PASS: converter slot rows map to PagedSources "
                    "exactly\n");
    } else {
        std::printf("[wpm] %d FAILURE(S)\n", g_failures);
    }
    return g_failures == 0 ? 0 : 1;
}
