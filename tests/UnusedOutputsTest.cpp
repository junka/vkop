// Skipping the dead tail of a round: Runtime::live_level_prefix.
//
// Standalone (no gtest, no GPU, no model): the function takes plain vectors,
// so the policy can be pinned here rather than inferred from a benchmark.
//
// The three properties that matter, each of which was a real failure mode
// somewhere else in the scheduler:
//
//   1. A dead TAIL is dropped (that is the whole point: an LM head and the
//      TopK on it, on every prefill chunk whose logits nobody reads).
//   2. A hole is NOT dropped. The answer must be a prefix, because the
//      scheduler numbers replayable segments by level ordinal and a gap would
//      shift every later ordinal -- recorded as "garbage tokens" in Run().
//   3. A dead node with a side effect still runs (in-place KV writes, host
//      readback shape chains), because "nobody reads its output" is not the
//      same as "it does nothing".
//
// Run: ./UnusedOutputsTest

#include <cstdio>
#include <vector>

#include "core/runtime.hpp"

namespace {

int g_fail = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);        \
            ++g_fail;                                                          \
        }                                                                      \
    } while (0)

using Levels = std::vector<std::vector<size_t>>;
using Flags = std::vector<bool>;

// levels[i] = the node indices in level i.
size_t run(const Levels &levels, const Flags &live, const Flags &side) {
    return vkop::core::Runtime::live_level_prefix(levels, live, side);
}

// A straight chain 0->1->2->3, one node per level.
void test_dead_tail_is_dropped() {
    Levels lv{{0}, {1}, {2}, {3}};
    Flags live{true, true, false, false};
    Flags side(4, false);
    CHECK(run(lv, live, side) == 2);

    // Only the very last level dead -> keep 3.
    Flags live2{true, true, true, false};
    CHECK(run(lv, live2, side) == 3);

    // Nothing dead -> keep everything (the default driver behaviour).
    Flags live3(4, true);
    CHECK(run(lv, live3, side) == 4);
}

// A dead level with a live one after it: the scan must stop BEFORE the hole.
void test_hole_is_not_dropped() {
    Levels lv{{0}, {1}, {2}, {3}};
    Flags live{true, false, true, false};
    Flags side(4, false);
    // Level 1 is dead but level 2 is live, so only level 3 is droppable and
    // the prefix ends at 3.
    CHECK(run(lv, live, side) == 3);

    // A hole immediately before a live tail: nothing can be dropped, because
    // dropping the hole would renumber the live levels after it.
    Flags live2{false, true, true, true};
    CHECK(run(lv, live2, side) == 4);
}

// A dead node that does host work (or whose recording changes per round) has
// to run: it is not observable through its output bytes alone.
void test_side_effect_blocks_the_skip() {
    Levels lv{{0}, {1}, {2}};
    Flags live{true, true, false};
    Flags side{false, false, true}; // the dead tail node has a side effect
    CHECK(run(lv, live, side) == 3);

    // Same shape, no side effect -> the tail goes.
    Flags no_side(3, false);
    Flags live_ab{true, true, false};
    CHECK(run(lv, live_ab, no_side) == 2);
}

// Several nodes in one level: the level is kept if ANY of them must run.
void test_level_kept_if_any_node_is_live() {
    Levels lv{{0, 1}, {2, 3}, {4, 5}};
    Flags live{true, false, false, false, false, false};
    Flags side(6, false);
    // Level 1 has one live node (2) -> keep it; level 2 is dead -> drop.
    Flags live_b{false, false, true, false, false, false};
    CHECK(run(lv, live_b, side) == 2);

    // A side effect on ANY node of the last level keeps that level.
    Flags side_b{false, false, false, false, false, true};
    CHECK(run(lv, live_b, side_b) == 3);
}

// Out-of-range indices are kept (conservative: an index we cannot classify is
// not evidence of deadness).
void test_unknown_index_is_kept() {
    Levels lv{{0}, {99}};
    Flags live{true};
    Flags side{false};
    CHECK(run(lv, live, side) == 2);
}

// An empty level table must not crash and must report zero.
void test_empty() {
    Levels lv;
    Flags live;
    Flags side;
    CHECK(run(lv, live, side) == 0);
}

} // namespace

int main() {
    test_dead_tail_is_dropped();
    test_hole_is_not_dropped();
    test_side_effect_blocks_the_skip();
    test_level_kept_if_any_node_is_live();
    test_unknown_index_is_kept();
    test_empty();
    if (g_fail) {
        std::printf("%d check(s) FAILED\n", g_fail);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
