// Copyright 2026 @junka
#ifndef VKOP_TOOLS_PROBES_PROBE_COMMON_HPP_
#define VKOP_TOOLS_PROBES_PROBE_COMMON_HPP_

// Dispatcher plumbing for the vkop op probes.
//
// These were 23 standalone executables, each with its own main(), its own
// device open and its own copy of the same scaffolding. Compiling any one of
// them spends ~5.5s on the shared header prefix against ~2.8s on the probe's
// own code, so merging them into one binary pays for itself: four probes in
// a single translation unit measured 10.2s against 4 x 10.3s.
//
// What this header does NOT do is lift the device setup out of the probes.
// Each probe's run() still does exactly what its main() used to -- creates the
// device, applies its own env knobs in the same order, and keeps its own
// _exit() teardown workaround. That keeps behaviour byte-for-byte identical;
// these probes are the only regression coverage for the BDA and descriptor
// routes, so a silent semantic drift here would be worse than the build time.

#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

namespace vkop {
namespace probes {

struct Probe {
    const char *name;
    const char *help;
    // Env var that selects the descriptor-fallback pass, e.g.
    // "GAP_BDA_DESC_ONLY". The dispatcher's --desc sets it and the probe reads
    // it exactly as before, so existing ctest ENVIRONMENT properties and
    // hand-run commands keep working unchanged.
    const char *desc_env;
    // The probe's former main(), kept verbatim.
    int (*run)();
};

void register_probe(Probe probe);
const std::vector<Probe> &all_probes();

// Per-probe registration, at the end of each probe file just inside that
// probe's own namespace, so probe_run is the bare function name:
//   REGISTER_PROBE("gap", "GlobalAveragePool", "GAP_BDA_DESC_ONLY", run_gap);
#define REGISTER_PROBE(probe_name, probe_help, probe_desc_env, probe_run)    \
    struct RegisterProbe_##probe_run {                                       \
        RegisterProbe_##probe_run() {                                        \
            ::vkop::probes::register_probe(                                  \
                {probe_name, probe_help, probe_desc_env, probe_run});        \
        }                                                                    \
    };                                                                       \
    [[maybe_unused]] RegisterProbe_##probe_run register_probe_##probe_run

} // namespace probes
} // namespace vkop

#endif // VKOP_TOOLS_PROBES_PROBE_COMMON_HPP_
