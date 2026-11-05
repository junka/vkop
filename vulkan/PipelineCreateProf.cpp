// Copyright 2026 @junka
#include "vulkan/PipelineCreateProf.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

namespace vkop {
namespace {

// Display bucket only, NOT a compile/hit detector. Measured on llm.vkopbin
// (GTX 1070): deserializing one large cached SPIR-V took 8.5ms while a cold
// compile of the same shader took 959ms, and 715 of 738 load-phase creates came
// in under 40us even with no VkPipelineCache at all -- duration alone cannot
// separate the two populations. VK_EXT_pipeline_creation_feedback is the
// extension that reports an actual cache hit; wire that in if this split ever
// needs to be a fact rather than a hint.
constexpr uint64_t kSlowThresholdNs = 200000;

constexpr int kSlowestShown = 8;

struct Event {
    uint64_t create_ns = 0;
    uint64_t ctor_ns = 0;
    uint32_t spv_bytes = 0;
    uint32_t subgroup = 0;
    std::string context;
};

struct Bucket {
    std::string phase;
    std::vector<Event> events;
};

struct State {
    bool enabled = []() {
        const char *env = std::getenv("VK_PIPELINE_CREATE_PROF");
        return env && env[0] == '1';
    }();
    std::mutex mutex;
    std::vector<Bucket> buckets;
    std::string phase = "other";
    std::string context;

    Bucket &bucket(const std::string &phase) {
        for (auto &b : buckets) {
            if (b.phase == phase) {
                return b;
            }
        }
        buckets.push_back(Bucket{phase, {}});
        return buckets.back();
    }
};

State &state() {
    static State s;
    return s;
}

double ms(uint64_t ns) { return static_cast<double>(ns) / 1e6; }

// Sorted-order lookup over the create durations of one bucket.
uint64_t percentile(std::vector<uint64_t> &sorted, double q) {
    if (sorted.empty()) {
        return 0;
    }
    size_t idx =
        static_cast<size_t>(q * static_cast<double>(sorted.size() - 1));
    return sorted[idx];
}

} // namespace

bool pipeline_prof_enabled() { return state().enabled; }

uint64_t pipeline_prof_now() {
    if (!state().enabled) {
        return 0;
    }
    return static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
}

void pipeline_prof_set_phase(const char *phase) {
    State &s = state();
    if (!s.enabled || !phase) {
        return;
    }
    std::lock_guard<std::mutex> lock(s.mutex);
    s.phase = phase;
    s.context.clear();
}

void pipeline_prof_set_context(const char *context) {
    State &s = state();
    if (!s.enabled) {
        return;
    }
    std::lock_guard<std::mutex> lock(s.mutex);
    s.context = context ? context : "";
}

void pipeline_prof_record(uint64_t create_ns, uint64_t ctor_ns,
                          uint32_t spv_bytes, uint32_t required_subgroup_size) {
    State &s = state();
    if (!s.enabled) {
        return;
    }
    std::lock_guard<std::mutex> lock(s.mutex);
    s.bucket(s.phase).events.push_back(Event{
        create_ns, ctor_ns, spv_bytes, required_subgroup_size, s.context});
}

void pipeline_prof_report(const char *header, double span_ms) {
    State &s = state();
    if (!s.enabled) {
        return;
    }
    std::lock_guard<std::mutex> lock(s.mutex);
    std::fprintf(stderr, "[pcreate] === %s ===\n",
                 header && *header ? header : "report");
    if (s.buckets.empty()) {
        std::fprintf(stderr, "[pcreate]   no compute pipelines created\n");
        return;
    }

    uint64_t all_create = 0;
    size_t all_count = 0;
    for (const auto &b : s.buckets) {
        std::vector<uint64_t> sorted;
        sorted.reserve(b.events.size());
        uint64_t create = 0;
        uint64_t ctor = 0;
        uint64_t bytes = 0;
        size_t compiled = 0;
        for (const auto &e : b.events) {
            sorted.push_back(e.create_ns);
            create += e.create_ns;
            ctor += e.ctor_ns;
            bytes += e.spv_bytes;
            if (e.create_ns >= kSlowThresholdNs) {
                compiled++;
            }
        }
        std::sort(sorted.begin(), sorted.end());
        all_create += create;
        all_count += b.events.size();

        std::fprintf(stderr,
                     "[pcreate] phase=%-5s n=%zu create=%.1fms "
                     "(mean %.2fms p50 %.2fms p95 %.2fms max %.2fms) "
                     "ctor=%.1fms\n",
                     b.phase.c_str(), b.events.size(), ms(create),
                     ms(create) / b.events.size(), ms(percentile(sorted, 0.50)),
                     ms(percentile(sorted, 0.95)), ms(sorted.back()), ms(ctor));
        std::fprintf(
            stderr,
            "[pcreate]   slow(>=%lluus)=%zu  fast(<)=%zu  "
            "spirv_total=%.1fKB\n",
            static_cast<unsigned long long>(kSlowThresholdNs / 1000),
            compiled, b.events.size() - compiled, bytes / 1024.0);

        std::vector<const Event *> slowest = {};
        slowest.reserve(b.events.size());
        for (const auto &e : b.events) {
            slowest.push_back(&e);
        }
        std::sort(slowest.begin(), slowest.end(),
                  [](const Event *a, const Event *b) {
                      return a->create_ns > b->create_ns;
                  });
        const size_t show = std::min<size_t>(kSlowestShown, slowest.size());
        if (show == 0) {
            continue;
        }
        std::fprintf(stderr, "[pcreate]   slowest %zu of phase=%s:\n", show,
                     b.phase.c_str());
        for (size_t i = 0; i < show; ++i) {
            const Event &e = *slowest[i];
            std::fprintf(
                stderr, "[pcreate]     %8.2fms  spv=%uB subgroup=%u  %s\n",
                ms(e.create_ns), e.spv_bytes, e.subgroup,
                e.context.empty() ? "(no node context)" : e.context.c_str());
        }
    }

    std::fprintf(stderr, "[pcreate] total: n=%zu create=%.1fms\n", all_count,
                 ms(all_create));
    if (span_ms > 0.0) {
        std::fprintf(stderr,
                     "[pcreate]   = %.1f%% of the %.1fms enclosing span\n",
                     ms(all_create) * 100.0 / span_ms, span_ms);
    }
}

} // namespace vkop
