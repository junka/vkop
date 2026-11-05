// Copyright 2026 @junka
#include "vulkan/DescriptorWriteProf.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace vkop {
namespace {

// Log2-nsec histogram: bucket i covers [2^i, 2^(i+1)) ns, so percentiles are
// recoverable at one-bit resolution while memory stays constant. A decode step
// makes thousands of these calls, so unlike the pipeline-create probe this one
// cannot keep every event, and nothing may allocate or sort per call -- the
// probe sits in the dispatch hot path and would otherwise contaminate the very
// host time it measures.
constexpr int kHistN = 40; // up to ~17min per call, nothing reaches here

inline int hist_idx(uint64_t ns) {
    int i = 0;
    while (i < kHistN - 1 && (1ULL << (i + 1)) <= ns) {
        i++;
    }
    return i;
}

struct Bucket {
    uint64_t count = 0;
    uint64_t total_ns = 0;
    uint64_t max_ns = 0;
    uint64_t writes = 0; // sum of the per-call write counts
    uint64_t hist[kHistN] = {};

    void add(uint64_t ns, uint32_t write_count) {
        count++;
        total_ns += ns;
        if (ns > max_ns) {
            max_ns = ns;
        }
        writes += write_count;
        hist[hist_idx(ns)]++;
    }

    // Upper edge of the bucket holding the q-quantile, so it rounds up.
    uint64_t percentile(double q) const {
        if (count == 0) {
            return 0;
        }
        const uint64_t want =
            static_cast<uint64_t>(q * static_cast<double>(count - 1)) + 1;
        uint64_t acc = 0;
        for (int i = 0; i < kHistN; i++) {
            acc += hist[i];
            if (acc >= want) {
                return 1ULL << (i + 1);
            }
        }
        return max_ns;
    }
};

using KindBuckets = std::vector<std::pair<std::string, Bucket>>;

struct State {
    bool enabled = []() {
        const char *env = std::getenv("VK_DS_WRITE_PROF");
        return env && env[0] == '1';
    }();
    std::mutex mutex;
    // Phase first, so the load/run split reads directly off the report.
    std::vector<std::pair<std::string, KindBuckets>> phases;
    std::string phase = "other";

    Bucket &bucket(const std::string &phase, const std::string &kind) {
        for (auto &p : phases) {
            if (p.first == phase) {
                for (auto &k : p.second) {
                    if (k.first == kind) {
                        return k.second;
                    }
                }
                p.second.emplace_back(kind, Bucket{});
                return p.second.back().second;
            }
        }
        phases.emplace_back(phase, KindBuckets{});
        return bucket(phase, kind);
    }
};

State &state() {
    static State s;
    return s;
}

double ms(uint64_t ns) { return static_cast<double>(ns) / 1e6; }
double us(uint64_t ns) { return static_cast<double>(ns) / 1e3; }

} // namespace

bool ds_prof_enabled() { return state().enabled; }

uint64_t ds_prof_now() {
    if (!state().enabled) {
        return 0;
    }
    return static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
}

void ds_prof_set_phase(const char *phase) {
    State &s = state();
    if (!s.enabled || !phase) {
        return;
    }
    std::lock_guard<std::mutex> lock(s.mutex);
    s.phase = phase;
}

void ds_prof_record(const char *kind, uint64_t ns, uint32_t count) {
    State &s = state();
    if (!s.enabled || !kind) {
        return;
    }
    std::lock_guard<std::mutex> lock(s.mutex);
    // Linear scan over at most three kinds; the phase/kind strings are only
    // built on the first call for a pair.
    s.bucket(s.phase, kind).add(ns, count);
}

void ds_prof_report(const char *header, double span_ms) {
    State &s = state();
    if (!s.enabled) {
        return;
    }
    std::lock_guard<std::mutex> lock(s.mutex);
    std::fprintf(stderr, "[dswrite] === %s ===\n",
                 header && *header ? header : "report");
    if (s.phases.empty()) {
        std::fprintf(stderr, "[dswrite]   no descriptor calls observed\n");
        s.phases.clear();
        return;
    }

    uint64_t all_update = 0;
    uint64_t all_update_n = 0;
    uint64_t all_alloc = 0;
    uint64_t all_free = 0;
    for (const auto &p : s.phases) {
        std::fprintf(stderr, "[dswrite] phase=%s\n", p.first.c_str());
        for (const auto &k : p.second) {
            const Bucket &b = k.second;
            const std::string &kind = k.first;
            if (kind == "update") {
                all_update += b.total_ns;
                all_update_n += b.count;
            } else if (kind == "alloc") {
                all_alloc += b.total_ns;
            } else {
                all_free += b.total_ns;
            }
            std::fprintf(
                stderr,
                "[dswrite]   %-6s n=%-7llu total=%8.1fms mean=%7.2fus "
                "p50=%7.2fus p95=%7.2fus max=%8.1fus writes/call=%.2f "
                "writes_total=%llu\n",
                kind.c_str(), static_cast<unsigned long long>(b.count),
                ms(b.total_ns), us(b.total_ns) / b.count,
                us(b.percentile(0.50)), us(b.percentile(0.95)), us(b.max_ns),
                static_cast<double>(b.writes) / b.count,
                static_cast<unsigned long long>(b.writes));
        }
    }

    std::fprintf(stderr,
                 "[dswrite] total: update=%.1fms (n=%llu) alloc=%.1fms "
                 "free=%.1fms  push_descriptor-replaceable=%.1fms\n",
                 ms(all_update), static_cast<unsigned long long>(all_update_n),
                 ms(all_alloc), ms(all_free), ms(all_update + all_alloc));
    if (span_ms > 0.0) {
        std::fprintf(
            stderr,
            "[dswrite]   update = %.2f%% / replaceable = %.2f%% of the "
            "%.1fms enclosing span\n",
            ms(all_update) * 100.0 / span_ms,
            ms(all_update + all_alloc) * 100.0 / span_ms, span_ms);
    }
    s.phases.clear();
}

void ds_prof_round(double span_ms) {
    State &s = state();
    if (!s.enabled) {
        return;
    }
    std::lock_guard<std::mutex> lock(s.mutex);
    // Sum each kind over every phase: one round is one phase in practice, and
    // the fallback keeps the line honest if a round straddles a phase change.
    struct KindTotal {
        const char *kind;
        uint64_t ns = 0;
        uint64_t n = 0;
    } kinds[] = {{"update", 0, 0},   {"bind_ds", 0, 0},
                 {"push_const", 0, 0}, {"bind_pipeline", 0, 0},
                 {"alloc", 0, 0},    {"free", 0, 0}};
    for (const auto &p : s.phases) {
        for (const auto &k : p.second) {
            for (auto &t : kinds) {
                if (k.first == t.kind) {
                    t.ns += k.second.total_ns;
                    t.n += k.second.count;
                }
            }
        }
    }
    const double replaceable =
        ms(kinds[0].ns + kinds[1].ns + kinds[4].ns); // update + bind_ds + alloc
    std::fprintf(stderr,
                 "[dswrite] round span=%7.1fms  update=%6.2fms(n=%llu) "
                 "bind_ds=%6.2fms(n=%llu) alloc=%5.2fms  = kept: "
                 "bind_pipe=%5.2fms push_const=%5.2fms  | push_desc ceiling "
                 "=%.2fms (%.1f%%)\n",
                 span_ms, ms(kinds[0].ns),
                 static_cast<unsigned long long>(kinds[0].n),
                 ms(kinds[1].ns), static_cast<unsigned long long>(kinds[1].n),
                 ms(kinds[4].ns), ms(kinds[3].ns), ms(kinds[2].ns),
                 replaceable, span_ms > 0.0 ? replaceable * 100.0 / span_ms : 0.0);
    s.phases.clear();
}

} // namespace vkop
