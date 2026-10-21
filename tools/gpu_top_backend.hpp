// Copyright 2025 @junka
// Backend interface for vkop_gpu_top: the one piece of platform-specific work
// is *where the cumulative GPU-time counters live*. Everything above this
// (delta math, aggregation, CLI, table rendering) is shared.
//
// The contract is deliberately narrow: a backend only ever hands the core a
// list of GPU *clients*, each carrying cumulative nanosecond counters keyed by
// the engine names it declares in engines(). The core then does what
// intel_gpu_top does and never interprets an absolute counter -- it only ever
// divides a delta by the wall-clock delta of the same window.
//
// Two consequences worth knowing before adding a third backend:
//   * Counters must be cumulative and monotonic per client. A backend that can
//     only report an instantaneous percentage cannot implement this interface
//     honestly (see the macOS note on PerformanceStatistics below).
//   * The client list is matched *by identity across samples*, not by
//     position: clients come and go (fds closed, registry nodes removed), and
//     diffing a shifted list fabricates enormous busy times (a macOS
//     prototype that keyed by list position reported WindowServer at 320x the
//     wall clock). Each backend is therefore responsible for reporting a
//     stable identity per client.
#ifndef TOOLS_GPU_TOP_BACKEND_HPP_
#define TOOLS_GPU_TOP_BACKEND_HPP_

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace gputop {

// One engine counter column. `key` is the name used in ClientInfo::engine_ns,
// `label` the column header. Column order here is the display order.
struct EngineSpec {
    const char *key;
    const char *label;
};

// Which of the optional device-level columns a given platform can fill.
// Apple silicon has no unprivileged equivalent of the i915 clock or RC6
// sysfs files (powermetrics owns them and insists on root), so the columns
// are dropped rather than printed as a permanent "--".
struct Columns {
    const char *device_label = "pdev"; // header key for ClientInfo::device
    const char *mem_label = "ResMB";   // per-client memory column header
    bool freq = false;                 // instantaneous clock, MHz
    bool idle = false;                 // idle residency, % of the window
    bool device_mem = false;           // memory is summable over all clients
};

// One GPU client as the OS sees it. On Linux that is one open fd on
// /dev/dri/* (one drm_file); on macOS one AGXDeviceUserClient registry node.
// A process typically holds several of them, so the core always sums per pid.
struct ClientInfo {
    int pid = 0;
    std::string proc;     // process name, for the Client column
    std::string device;   // PCI slot / accelerator class name
    std::string identity; // stable across samples; see the file header
    // Cumulative busy time per engine, nanoseconds, keyed by EngineSpec::key.
    std::map<std::string, uint64_t> engine_ns;
    uint64_t mem_bytes = 0; // meaning is backend-defined, see mem_valid
    bool mem_valid = false;
    // False when the number belongs to the *process* rather than to this one
    // client, so that summing it across a process's clients would multiply it.
    // macOS has to report the process footprint this way: Apple exposes no
    // per-client (let alone per-process) count of GPU-resident bytes, which is
    // what drm-resident-<region> is on Linux.
    bool mem_is_per_process = false;
    int nhandles = 0; // fds / command queues folded into this one client
};

class Backend {
  public:
    virtual ~Backend() = default;
    virtual const char *name() const = 0;
    virtual const std::vector<EngineSpec> &engines() const = 0;
    virtual Columns columns() const = 0;

    // Clients for `pid`, or every readable client when pid <= 0. A client that
    // cannot be read (other users, exited, no permission) is omitted rather
    // than reported with zeroed counters -- a missing row is honest, a zero is
    // indistinguishable from an idle GPU.
    virtual std::vector<ClientInfo> collect(int pid) = 0;

    // Device-level extras, only called when columns() enables them. idle_pct()
    // is called once per window with that window's length and keeps its own
    // previous reading, because idle residency is itself a cumulative counter.
    virtual uint64_t freq_mhz() { return 0; }
    virtual double idle_pct(double /*wall_s*/) { return -1.0; }

    // Choose one of several implementations when a single platform offers more
    // than one -- Linux has both Intel (i915/xe) and NVIDIA cards, and a
    // workstation usually has both drivers loaded. Called once, before the
    // first collect(); "auto" returns to the platform's own preference.
    // Returns false when the name is not offered here, which is also the
    // default for platforms that have nothing to choose (macOS).
    virtual bool select(const std::string &name) {
        (void)name;
        return false;
    }
};

// Provided by the platform source selected in CMake
// (gpu_top_linux.cpp / gpu_top_darwin.cpp).
Backend &backend();

} // namespace gputop

#endif // TOOLS_GPU_TOP_BACKEND_HPP_
