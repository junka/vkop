// The MoE paging slot table, as the C++ loader reads it off disk.
//
// Two contracts this pins down, both silent-failure shaped:
//
//  1. A slot row is a *declaration*, not a payload: `size`/`offset` are 0 and
//     the bytes live in the file named by moe_source. A loader that trusted
//     `offset`/`size` would hand the runtime a zero-byte buffer at blob
//     offset 0 -- the header of the blob -- for every expert.
//  2. A pre-paging vkopbin must load unchanged: the fields are appended to the
//     FlatBuffers table, so an older file's accessors return ""/-1, which is
//     exactly the "no paging" case.
//
// The fixture is written by the Python writer at conversion level (the same
// code path the real `llm_moe_paged.vkopbin` takes), so this also checks the
// schema field order the two languages agree on.
#include "include/logger.hpp"
#include "model/load.hpp"

#include <cstdio>
#include <string>

using vkop::load::VkModel;

namespace {

int failures = 0;

void Check(bool ok, const std::string &what) {
    if (!ok) {
        printf("FAIL: %s\n", what.c_str());
        ++failures;
    }
}

} // namespace

int main() {
    Logger::getInstance().setLevel(LOG_INFO);

    // --- the paged fixture: two groups, 3 experts each, gap-free ids ---
    VkModel paged(TEST_DATA_PATH "/moe_paged_slots.vkopbin");
    // 6 slot rows + d1, the one weight a live node reads. The writer's
    // liveness prune drops every initializer nothing reads -- the fixture's `d0`
    // among them -- except the slot names, which `moe_keep_alive` holds open.
    Check(paged.initializers.size() == 7,
          "expected 6 slot rows + 1 dense weight, got " +
              std::to_string(paged.initializers.size()));

    int slots = 0;
    for (const auto &[name, init] : paged.initializers) {
        if (init.moe_group.empty()) {
            Check(init.moe_expert_id == -1,
                  name + ": a dense weight must have expert id -1");
            Check(init.moe_source.empty(),
                  name + ": a dense weight must page from nowhere");
            continue;
        }
        ++slots;
        Check(init.size == 0, name + ": a slot must declare no payload bytes");
        Check(paged.initializer_offsets.at(name) == 0,
              name + ": a slot's blob offset is 0 (there are no bytes there)");
        Check(init.moe_expert_id >= 0 && init.moe_expert_id < 3,
              name + ": expert id out of range");
        Check(!init.moe_source.empty(),
              name + ": a slot with no moe_source has no bytes anywhere");
        Check(init.moe_source == "ckpt/shard-00001.safetensors",
              name + ": wrong source file, got " + init.moe_source);
        Check(init.moe_source_size == 32,
              name + ": expected a 32-byte slice, got " +
                  std::to_string(init.moe_source_size));
        Check(init.moe_source_dtype == "bfloat16",
              name + ": the checkpoint's bytes are bf16, got " +
                  init.moe_source_dtype);
        Check(init.dtype == "float16",
              name + ": the slot's buffer is fp16, got " + init.dtype);
        Check(init.dims.size() == 2 && init.dims[0] == 4 && init.dims[1] == 4,
              name + ": wrong dims, so the buffer is the wrong size");
    }
    Check(slots == 6, "expected 6 slot rows, got " + std::to_string(slots));
    Check(paged.initializers.count("d1") == 1,
          "the dense weight the graph reads must survive");

    // The slices are the checkpoint's, not arithmetic on the expert id: the
    // fixture strides them 40 bytes for a 32-byte tensor, so `first + id*size`
    // lands on the wrong expert (this is why the runtime builds a lookup).
    for (int e = 0; e < 3; ++e) {
        for (const char *group : {"gate_proj", "up_proj"}) {
            const std::string name =
                std::string("moe.L0.") + group + ".e" + std::to_string(e);
            const auto &init = paged.initializers.at(name);
            Check(init.moe_source_offset == static_cast<uint64_t>(100 + 40 * e),
                  name + ": slice offset does not match the exporter's layout");
        }
    }

    // --- a pre-paging file: the same graph, no MoE fields at all ---
    VkModel dense(TEST_DATA_PATH "/moe_slots_absent.vkopbin");
    // Same graph, nothing held open: d1 (read) and moe.L0.gate_proj.e0 (read by
    // mm1) are what is left, and the expert matrix is an ordinary weight.
    Check(dense.initializers.size() == 2,
          "the pre-paging fixture must carry d1 + the expert matrix, got " +
              std::to_string(dense.initializers.size()));
    for (const auto &[name, init] : dense.initializers) {
        Check(init.moe_group.empty(), name + ": no group in an old file");
        Check(init.moe_expert_id == -1, name + ": no expert id in an old file");
        Check(init.moe_source.empty(), name + ": no source in an old file");
        Check(init.moe_source_size == 0, name + ": no slice in an old file");
        Check(init.moe_source_dtype.empty(),
              name + ": no source dtype in an old file");
    }
    // ...and the weights are still readable through the ordinary path: the
    // expert matrix has real bytes at a real blob offset, not a slot's 0/0.
    const auto &expert = dense.initializers.at("moe.L0.gate_proj.e0");
    Check(expert.size == 32, "the expert matrix must still declare its 32 bytes");
    Check(dense.initializer_offsets.at("moe.L0.gate_proj.e0") + 32 <=
              dense.initializer_memory_size,
          "a pre-paging weight must address bytes inside its own blob");
    const auto &w = dense.initializers.at("d1");
    Check(w.size == 32, "d1 must still be loadable as an ordinary weight");

    if (failures) {
        printf("%d check(s) failed\n", failures);
        return 1;
    }
    printf("ok: %d slots paged from the checkpoint, %zu weights intact, "
           "pre-paging file loads unchanged\n",
           slots, dense.initializers.size());
    return 0;
}
