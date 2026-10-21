// Copyright 2025 @junka
// Linux backends for vkop_gpu_top: Intel i915/xe and NVIDIA, through the two
// interfaces the kernel and the driver expose to ordinary users.
//
//   Intel
//   /proc/<pid>/fdinfo/<fd>   per-DRM-client accounting: cumulative nanosecond
//                             busy time per hardware engine (drm-engine-render
//                             / -compute / -copy / ...), plus current resident
//                             memory (drm-resident-<region>). No privileges
//                             needed for one's own -- or same-uid -- processes.
//   /sys/class/drm/cardN/gt/* i915 GT sysfs: rps_act_freq_mhz and
//                             rc6_residency_ms. World-readable.
//
//   NVIDIA (NVML, dlopen'd -- the tool links nothing)
//   nvmlDeviceGetProcessUtilization   per-process SM/memory/encoder/decoder
//                                     utilization, one sample per running
//                                     process per ~1s driver sample period.
//   nvmlDeviceGetCompute/GraphicsRunningProcesses_v2   per-process device
//                                     memory, and the process list itself.
//   nvmlDeviceGetClockInfo(SM)        SM clock, MHz.
// CUDA opens /dev/nvidia* character devices rather than a DRM render node, so
// nvidia-drm's fdinfo carries no engine counters at all -- there is no
// cumulative NVIDIA counter to read, which is why this backend is built
// differently from the Intel one (see the trap on synthesized counters below).
//
// intel_gpu_top itself needs root on most hosts: it opens the i915 PMU via
// perf_event_open, which /proc/sys/kernel/perf_event_paranoid>=2 blocks for
// unprivileged users, and its OA path is absent on integrated i915 anyway.
//
// Two implementation traps, both verified on ARL-P / i915 (card1):
//   * A process usually holds *several* i915 render-node fds (the Vulkan
//     loader opens one, the ICD opens the working one). The kernel accounts
//     each drm_file separately, so the busy time is split across them -- the
//     core sums them per pid. Taking the first fd yields a permanent 0.
//   * drm-engine-capacity-* lines carry a queue count ("...: 2"), not a
//     nanosecond total; they are skipped by requiring the " ns" suffix.
//   * The per-client engine time is context-residency based, so on a *shared*
//     engine the sum across clients can exceed the wall clock (i915 overlaps
//     preempted contexts). Whole-GPU mode (no -p) therefore prints a
//     per-client sum that is an upper bound; use -p PID for the exact figure
//     of a single profiled process.
//
// Traps on the NVIDIA side, all hit and measured on b3 (RTX 4000 Ada, driver
// 595.99.02):
//   * NVML reports utilization as a *percentage per sample period*, not as a
//     cumulative counter, and this interface is cumulative by construction. The
//     backend therefore keeps its own nanosecond accumulators and advances them
//     sample-and-hold: each process's last measured percentage is held for the
//     wall time until the next sample arrives. That keeps the counters
//     monotonic, keeps -s 500 from alternating between 0% and 200% on a source
//     that only updates once a second, and still averages to the true figure
//     over any window longer than the driver's sample period.
//   * nvmlDeviceGetProcessUtilization returns only processes with *non-zero*
//     utilization during the last period, and returns NVML_ERROR_NOT_FOUND (6)
//     when nothing new has arrived. On any non-zero return the count is left
//     untouched and the buffer must not be read -- reading it yields pid 0.
//   * MIG instances do not support the process-utilization query at all.
//   * The process list is the authority on which pids exist: a pid absent from
//     both lists has exited, so its accumulator is dropped rather than left to
//     reappear as a huge delta when the pid number is recycled.
#include "gpu_top_backend.hpp"

#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

namespace gputop {
namespace {

// ---------------------------------------------------------------------------
// Small file helpers (sysfs/proc only ever holds tiny files).
// ---------------------------------------------------------------------------

std::string read_file(const std::string &path) {
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        return {};
    std::string out;
    char buf[4096];
    ssize_t n;
    while ((n = ::read(fd, buf, sizeof(buf))) > 0)
        out.append(buf, static_cast<size_t>(n));
    ::close(fd);
    return out;
}

std::string trim(const std::string &s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a])))
        ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1])))
        --b;
    return s.substr(a, b - a);
}

// Read a single-integer sysfs file. Returns false on a missing/unreadable
// file or a non-numeric body (rps_act_freq_mhz reads "0" when idle, which is
// a valid value, not an error).
bool read_u64_file(const std::string &path, uint64_t &out) {
    std::string s = read_file(path);
    if (s.empty())
        return false;
    char *end = nullptr;
    unsigned long long v = std::strtoull(s.c_str(), &end, 10);
    if (end == s.c_str())
        return false;
    out = static_cast<uint64_t>(v);
    return true;
}

std::string read_comm(int pid) {
    return trim(read_file("/proc/" + std::to_string(pid) + "/comm"));
}

// ---------------------------------------------------------------------------
// fdinfo parsing.
// ---------------------------------------------------------------------------

// One DRM client (one open fd on /dev/dri/*). Every field is optional: a
// non-DRM fd yields is_drm=false and is discarded by the caller.
struct FdInfo {
    bool is_drm = false;
    std::string driver; // i915 / xe
    std::string pdev;   // e.g. 0000:00:02.0
    std::map<std::string, uint64_t> engine_ns;
    uint64_t resident_bytes = 0;
};

FdInfo parse_fdinfo(const std::string &text) {
    FdInfo fi;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t eol = text.find('\n', pos);
        if (eol == std::string::npos)
            eol = text.size();
        std::string line = text.substr(pos, eol - pos);
        pos = eol + 1;
        if (line.rfind("drm-driver:", 0) == 0) {
            fi.is_drm = true;
            fi.driver = trim(line.substr(std::strlen("drm-driver:")));
        } else if (line.rfind("drm-pdev:", 0) == 0) {
            fi.pdev = trim(line.substr(std::strlen("drm-pdev:")));
        } else if (line.rfind("drm-engine-", 0) == 0) {
            size_t colon = line.find(':');
            size_t ns = line.rfind(" ns");
            // A capacity line has no " ns" -- skip it (see file header).
            if (colon == std::string::npos || ns == std::string::npos)
                continue;
            std::string name = line.substr(std::strlen("drm-engine-"),
                                           colon - std::strlen("drm-engine-"));
            fi.engine_ns[name] =
                std::strtoull(line.c_str() + colon + 1, nullptr, 10);
        } else if (line.rfind("drm-resident-", 0) == 0) {
            size_t colon = line.find(':');
            if (colon == std::string::npos)
                continue;
            fi.resident_bytes +=
                std::strtoull(line.c_str() + colon + 1, nullptr, 10);
        }
    }
    return fi;
}

// ---------------------------------------------------------------------------
// i915 GT sysfs.
// ---------------------------------------------------------------------------

// Map a drm-pdev ("0000:00:02.0") to /sys/class/drm/cardN/gt by comparing the
// basename of each card's device symlink (its target ends in the PCI slot
// name). Returns "" when the card is not found or not i915-shaped.
std::string find_gt_base(const std::string &pdev) {
    if (pdev.empty())
        return {};
    DIR *d = opendir("/sys/class/drm");
    if (!d)
        return {};
    std::string result;
    while (dirent *e = readdir(d)) {
        const char *n = e->d_name;
        if (std::strncmp(n, "card", 4) != 0 ||
            !std::isdigit(static_cast<unsigned char>(n[4])))
            continue;
        std::string dev = std::string("/sys/class/drm/") + n + "/device";
        char tgt[512];
        ssize_t len = readlink(dev.c_str(), tgt, sizeof(tgt) - 1);
        if (len <= 0)
            continue;
        tgt[len] = '\0';
        std::string base(tgt);
        size_t slash = base.rfind('/');
        if (slash != std::string::npos)
            base = base.substr(slash + 1);
        if (base == pdev) {
            result = std::string("/sys/class/drm/") + n + "/gt";
            break;
        }
    }
    closedir(d);
    return result;
}

// Max current request frequency over all GT tiles, MHz (0 = unknown). ARL-P
// exposes gt0 and gt1; gt0 carries the render/compute tile, so the max is the
// interesting number for a compute-bound workload.
uint64_t read_freq_mhz(const std::string &gt_base) {
    if (gt_base.empty())
        return 0;
    DIR *d = opendir(gt_base.c_str());
    if (!d)
        return 0;
    uint64_t best = 0;
    while (dirent *e = readdir(d)) {
        const char *n = e->d_name;
        if (std::strncmp(n, "gt", 2) != 0 ||
            !std::isdigit(static_cast<unsigned char>(n[2])))
            continue;
        uint64_t v = 0;
        if (read_u64_file(gt_base + "/" + n + "/rps_act_freq_mhz", v) &&
            v > best)
            best = v;
    }
    closedir(d);
    return best;
}

// Per-tile cumulative RC6 (engine-idle) residency in ms, keyed by tile name.
// ARL-P exposes two tiles (gt0 = render/compute, gt1 = media); they idle
// independently, so they must be kept apart rather than summed -- summing
// reports the idle media tile's residency and hides the busy render tile's.
std::map<std::string, uint64_t> read_rc6_all(const std::string &gt_base) {
    std::map<std::string, uint64_t> out;
    if (gt_base.empty())
        return out;
    DIR *d = opendir(gt_base.c_str());
    if (!d)
        return out;
    while (dirent *e = readdir(d)) {
        const char *n = e->d_name;
        if (std::strncmp(n, "gt", 2) != 0 ||
            !std::isdigit(static_cast<unsigned char>(n[2])))
            continue;
        uint64_t v = 0;
        if (read_u64_file(gt_base + "/" + n + "/rc6_residency_ms", v))
            out[n] = v;
    }
    closedir(d);
    return out;
}

// ---------------------------------------------------------------------------

class Intel : public Backend {
  public:
    const char *name() const override { return "i915 fdinfo"; }

    // Display order/labels, mirroring intel_gpu_top's engine columns.
    const std::vector<EngineSpec> &engines() const override {
        static const std::vector<EngineSpec> kEngines = {
            {"render", "Render"},
            {"compute", "Compute"},
            {"copy", "Copy"},
            {"video", "Video"},
            {"video-enhance", "VEnh"}};
        return kEngines;
    }

    Columns columns() const override {
        Columns c;
        c.device_label = "pdev";
        c.mem_label = "ResMB";
        c.freq = true;
        c.idle = true;
        c.device_mem = true;
        return c;
    }

    // Walk /proc/<pid>/fd and report every DRM client found. When pid <= 0,
    // walk every readable process (whole-GPU view). Unreadable /proc entries
    // (other users, kernel threads, races with process exit) are skipped
    // silently -- no privilege escalation is attempted.
    std::vector<ClientInfo> collect(int pid) override {
        std::vector<ClientInfo> out;
        std::vector<int> pids;
        if (pid > 0) {
            pids.push_back(pid);
        } else {
            DIR *d = opendir("/proc");
            if (!d)
                return out;
            while (dirent *e = readdir(d)) {
                const char *n = e->d_name;
                bool digits = *n != '\0';
                for (const char *p = n; *p != '\0' && digits; ++p)
                    digits = std::isdigit(static_cast<unsigned char>(*p)) != 0;
                if (digits)
                    pids.push_back(std::atoi(n));
            }
            closedir(d);
        }
        for (int p : pids) {
            std::string fddir = "/proc/" + std::to_string(p) + "/fd";
            DIR *d = opendir(fddir.c_str());
            if (!d)
                continue;
            std::string comm; // read once per pid, only if it has a client
            while (dirent *e = readdir(d)) {
                const char *n = e->d_name;
                if (n[0] < '0' || n[0] > '9')
                    continue;
                std::string fdpath = fddir + "/" + n;
                char tgt[512];
                ssize_t len = readlink(fdpath.c_str(), tgt, sizeof(tgt) - 1);
                if (len <= 0)
                    continue;
                tgt[len] = '\0';
                if (std::strncmp(tgt, "/dev/dri/", 9) != 0)
                    continue;
                FdInfo fi = parse_fdinfo(
                    read_file("/proc/" + std::to_string(p) + "/fdinfo/" + n));
                // Intel only: the fdinfo engine names and the GT sysfs below
                // are i915/xe-specific. On a hybrid host (one that also has an
                // NVIDIA card) including another vendor's clients would both
                // mis-resolve the pdev -> card mapping and fold foreign engine
                // counters into the Util% columns.
                if (!fi.is_drm || (fi.driver != "i915" && fi.driver != "xe"))
                    continue;
                if (comm.empty())
                    comm = read_comm(p);
                ClientInfo ci;
                ci.pid = p;
                ci.proc = comm;
                ci.device = fi.pdev;
                // The drm_file lives and dies with the fd, so the fd path is a
                // usable identity: it is stable while open and disappears with
                // the client, which the core treats as a one-frame gap.
                ci.identity = fdpath;
                ci.engine_ns = std::move(fi.engine_ns);
                ci.mem_bytes = fi.resident_bytes;
                ci.mem_valid = true;
                ci.nhandles = 1;
                // The card is whatever the first client we ever see is using;
                // a monitor started before any GPU process exists simply
                // reports no clock/idle numbers until one shows up.
                if (!gt_base_.resolved && !ci.device.empty()) {
                    gt_base_.base = find_gt_base(ci.device);
                    gt_base_.resolved = !gt_base_.base.empty();
                }
                out.push_back(std::move(ci));
            }
            closedir(d);
        }
        return out;
    }

    uint64_t freq_mhz() override { return read_freq_mhz(gt_base_.base); }

    // Idle fraction of the *least idle* tile over the window -- i.e. of the
    // tile actually doing the profiled work. Reporting the min rather than a
    // sum keeps the number honest on a multi-tile part: a GPU-bound workload
    // shows ~0% (the render tile never enters RC6 while busy) whereas a fully
    // idle GPU shows ~100%. Summing the tiles instead reports the idle media
    // tile.
    double idle_pct(double wall_s) override {
        std::map<std::string, uint64_t> cur = read_rc6_all(gt_base_.base);
        double best = -1.0;
        for (const auto &kv : cur) {
            auto it = rc6_prev_.find(kv.first);
            if (it == rc6_prev_.end() || kv.second < it->second || wall_s <= 0)
                continue;
            // rc6_residency_ms is milliseconds, so the denominator is 1e3.
            double pct =
                100.0 * double(kv.second - it->second) / (wall_s * 1e3);
            if (best < 0.0 || pct < best)
                best = pct;
        }
        rc6_prev_.swap(cur);
        return best < 0.0 ? 0.0 : best;
    }

  private:
    struct {
        std::string base;
        bool resolved = false;
    } gt_base_;
    std::map<std::string, uint64_t> rc6_prev_;
};

} // namespace

// ---------------------------------------------------------------------------
// NVIDIA, through NVML.
// ---------------------------------------------------------------------------

namespace {

// NVML's ABI, declared here rather than including <nvml.h> or linking
// libnvidia-ml: the header is part of the CUDA toolkit (absent on a host with
// only the driver) and the library is dlopen'd, so a machine without an NVIDIA
// card never notices. Layouts are copied from nvml.h and must match it exactly.
// The enum values are the ones that change behaviour rather than the ones we
// print.
enum {
    kNvmlSuccess = 0,
    kNvmlNoPermission = 4,
    kNvmlNotFound = 6,
    kNvmlInsufficientSize = 7,
};
enum { kNvmlClockSm = 1 };

struct NvmlProcessUtil {
    unsigned pid;
    unsigned long long time_stamp; // microseconds, driver clock
    unsigned sm, mem, enc, dec;    // percent over the driver's sample period
};
struct NvmlProcessInfoV1 {
    unsigned pid;
    unsigned long long used_gpu_memory;
};
struct NvmlProcessInfoV2 {
    unsigned pid;
    unsigned long long used_gpu_memory;
    unsigned gpu_instance_id, compute_instance_id;
};
struct NvmlPciInfo {
    char bus_id_legacy[16];
    unsigned domain, bus, device, pci_device_id, pci_sub_system_id;
    char bus_id[32];
};

// The symbols we need, resolved by name so a missing optional entry point
// degrades to a fallback instead of failing the whole backend.
struct NvmlApi {
    void *lib = nullptr;
    bool tried = false;
    int (*init)() = nullptr;
    int (*shutdown)() = nullptr;
    int (*device_count)(unsigned *) = nullptr;
    int (*handle_by_index)(unsigned, void **) = nullptr;
    int (*pci_info)(void *, NvmlPciInfo *) = nullptr;
    int (*clock_info)(void *, int, unsigned *) = nullptr;
    int (*process_util)(void *, NvmlProcessUtil *, unsigned *,
                        unsigned long long) = nullptr;
    int (*compute_v2)(void *, unsigned *, NvmlProcessInfoV2 *) = nullptr;
    int (*compute_v1)(void *, unsigned *, NvmlProcessInfoV1 *) = nullptr;
    int (*graphics_v2)(void *, unsigned *, NvmlProcessInfoV2 *) = nullptr;
    int (*graphics_v1)(void *, unsigned *, NvmlProcessInfoV1 *) = nullptr;

    template <typename Fn> static bool grab(void *lib, const char *n, Fn &fn) {
        fn = reinterpret_cast<Fn>(dlsym(lib, n));
        return fn != nullptr;
    }

    // Load the library and the mandatory entry points. Called once; `tried`
    // keeps a host with no driver from re-attempting every sample.
    bool open() {
        if (tried)
            return lib != nullptr;
        tried = true;
        lib = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!lib)
            lib = dlopen("libnvidia-ml.so", RTLD_NOW | RTLD_LOCAL);
        if (!lib)
            return false;
        bool ok = grab(lib, "nvmlInit_v2", init) || grab(lib, "nvmlInit", init);
        ok = ok && grab(lib, "nvmlDeviceGetCount_v2", device_count) &&
             grab(lib, "nvmlDeviceGetHandleByIndex_v2", handle_by_index) &&
             grab(lib, "nvmlDeviceGetProcessUtilization", process_util);
        if (!ok || init() != kNvmlSuccess) {
            lib = nullptr;
            return false;
        }
        grab(lib, "nvmlDeviceGetPciInfo_v3", pci_info);
        if (!pci_info)
            grab(lib, "nvmlDeviceGetPciInfo", pci_info);
        grab(lib, "nvmlDeviceGetClockInfo", clock_info);
        // _v2 carries MIG instance ids and is absent before driver ~470; the
        // _v1 entry points report the same pid and memory with a shorter
        // struct.
        if (!grab(lib, "nvmlDeviceGetComputeRunningProcesses_v2", compute_v2))
            grab(lib, "nvmlDeviceGetComputeRunningProcesses", compute_v1);
        if (!grab(lib, "nvmlDeviceGetGraphicsRunningProcesses_v2", graphics_v2))
            grab(lib, "nvmlDeviceGetGraphicsRunningProcesses", graphics_v1);
        return true;
    }

    // Query a "count + array" entry point, growing once if the first guess was
    // too small (the buffer is caller-sized on input, filled count on output).
    template <typename T, typename Call>
    static unsigned query(Call call, std::vector<T> &buf) {
        unsigned n = static_cast<unsigned>(buf.size());
        int rc = call(&n, buf.data());
        if (rc == kNvmlInsufficientSize) {
            buf.resize(n > buf.size() ? n : buf.size() * 2);
            n = static_cast<unsigned>(buf.size());
            rc = call(&n, buf.data());
        }
        // Anything else (no permission, not found, MIG unsupported) means "no
        // data this round"; the buffer is untouched and must not be read.
        return rc == kNvmlSuccess ? n : 0u;
    }
};

uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

// One GPU as NVML sees it.
struct NvDevice {
    void *handle = nullptr;
    std::string bus;
    uint64_t last_stamp = 0; // lastSeenTimeStamp, microseconds
};

// Everything this backend remembers about one (GPU, pid) pair between samples.
// The accumulators are the synthesized cumulative counters the core diffs.
struct NvProc {
    int pid = 0;
    std::string bus;
    uint64_t accum_ns[4] = {0, 0, 0, 0};
    unsigned pct[4] = {0, 0, 0, 0};
    bool have_pct = false;
    uint64_t tick_ns = 0;
    uint64_t mem_bytes = 0;
    bool mem_known = false;
};

class Nvidia : public Backend {
  public:
    const char *name() const override { return "NVML process util"; }

    // NVML's own split. "SM" is 3D/compute, "Mem" the frame-buffer controller,
    // "Enc"/"Dec" the video engines -- the same four nvtop shows.
    const std::vector<EngineSpec> &engines() const override {
        static const std::vector<EngineSpec> kEngines = {
            {"sm", "SM"}, {"mem", "Mem"}, {"enc", "Enc"}, {"dec", "Dec"}};
        return kEngines;
    }

    Columns columns() const override {
        Columns c;
        c.device_label = "bus";
        c.mem_label = "VRAMMB"; // per-process usedGpuMemory, not drm-resident
        c.freq = true;          // SM clock
        c.idle = false;         // NVML has no unprivileged idle-residency count
        c.device_mem = true;    // one process's memory is nobody else's
        return c;
    }

    // True when a driver and at least one GPU are present. Never escalates.
    bool available() {
        if (!api_.open())
            return false;
        if (devices_.empty() && !enumerated_) {
            enumerated_ = true;
            unsigned n = 0;
            if (api_.device_count(&n) != kNvmlSuccess || n == 0)
                return false;
            for (unsigned i = 0; i < n; ++i) {
                NvDevice d;
                if (api_.handle_by_index(i, &d.handle) != kNvmlSuccess)
                    continue;
                NvmlPciInfo pci{};
                d.bus = api_.pci_info &&
                                api_.pci_info(d.handle, &pci) == kNvmlSuccess
                            ? pci.bus_id
                            : std::to_string(i);
                devices_.push_back(std::move(d));
            }
        }
        return !devices_.empty();
    }

    std::vector<ClientInfo> collect(int pid) override {
        std::vector<ClientInfo> out;
        if (!available())
            return out;
        const uint64_t now = now_ns();

        // Which (GPU, pid) pairs the driver still knows about, and a refresh of
        // their percentages and memory.
        std::map<std::string, bool> present;
        for (NvDevice &d : devices_) {
            drain_utilization(d, present);
            drain_processes(d, present);
        }

        // Advance every surviving accumulator to `now` and drop the rest. A pid
        // that has left both process lists has exited; keeping its counter
        // would turn a recycled pid number into a burst of fabricated busy
        // time.
        for (auto it = procs_.begin(); it != procs_.end();) {
            auto p = present.find(it->first);
            if (p == present.end()) {
                it = procs_.erase(it);
                continue;
            }
            NvProc &ps = it->second;
            if (ps.tick_ns == 0)
                ps.tick_ns = now;
            const uint64_t dt = now > ps.tick_ns ? now - ps.tick_ns : 0;
            if (ps.have_pct && dt > 0)
                for (int i = 0; i < 4; ++i)
                    ps.accum_ns[i] += uint64_t(ps.pct[i]) * (dt / 100);
            ps.tick_ns = now;
            ++it;
        }

        for (const auto &kv : procs_) {
            const NvProc &ps = kv.second;
            if (pid > 0 && ps.pid != pid)
                continue;
            ClientInfo ci;
            ci.pid = ps.pid;
            ci.proc = read_comm(ps.pid);
            ci.device = ps.bus;
            ci.identity = kv.first;
            ci.engine_ns = {{"sm", ps.accum_ns[0]},
                            {"mem", ps.accum_ns[1]},
                            {"enc", ps.accum_ns[2]},
                            {"dec", ps.accum_ns[3]}};
            ci.mem_bytes = ps.mem_bytes;
            ci.mem_valid = ps.mem_known;
            ci.nhandles = 1;
            out.push_back(std::move(ci));
        }
        return out;
    }

    // Max SM clock over the GPUs, MHz. 0 when NVML will not say.
    uint64_t freq_mhz() override {
        if (!available() || !api_.clock_info)
            return 0;
        uint64_t best = 0;
        for (NvDevice &d : devices_) {
            unsigned mhz = 0;
            if (api_.clock_info(d.handle, kNvmlClockSm, &mhz) == kNvmlSuccess &&
                mhz > best)
                best = mhz;
        }
        return best;
    }

  private:
    static std::string key_of(const std::string &bus, int pid) {
        return bus + ":" + std::to_string(pid);
    }

    // Pull whatever the driver has recorded since the last call. One sample per
    // running process per driver sample period (~1s), holding its percentages.
    void drain_utilization(NvDevice &d, std::map<std::string, bool> &present) {
        // Sized so the usual case is one call: NVML reports the *driver's whole
        // sample buffer* size as required space when lastSeenTimeStamp is 0
        // (72 entries on the 595 driver, far more than the 3 processes that had
        // run). NvmlApi::query grows and retries if this is ever too small.
        std::vector<NvmlProcessUtil> buf(128);
        auto call = [&](unsigned *n, NvmlProcessUtil *p) {
            return api_.process_util(d.handle, p, n, d.last_stamp);
        };
        unsigned n = NvmlApi::query(call, buf);
        for (unsigned i = 0; i < n; ++i) {
            const NvmlProcessUtil &s = buf[i];
            if (s.pid == 0)
                continue;
            if (s.time_stamp > d.last_stamp)
                d.last_stamp = s.time_stamp;
            const std::string k = key_of(d.bus, int(s.pid));
            NvProc &ps = procs_[k];
            ps.pid = int(s.pid);
            ps.bus = d.bus;
            ps.pct[0] = s.sm;
            ps.pct[1] = s.mem;
            ps.pct[2] = s.enc;
            ps.pct[3] = s.dec;
            ps.have_pct = true;
            present[k] = true;
        }
    }

    // The process list, which is what tells this backend a pid still exists --
    // and the only per-process memory NVML offers. Union of the compute and
    // graphics lists: a process doing only presentation work appears in the
    // latter alone, and one that is merely holding a context appears in neither
    // the utilization samples nor (sometimes) the other list.
    void drain_processes(NvDevice &d, std::map<std::string, bool> &present) {
        if (api_.compute_v2) {
            std::vector<NvmlProcessInfoV2> buf(64);
            auto call = [&](unsigned *n, NvmlProcessInfoV2 *p) {
                return api_.compute_v2(d.handle, n, p);
            };
            unsigned n = NvmlApi::query(call, buf);
            for (unsigned i = 0; i < n; ++i)
                note(d, buf[i].pid, buf[i].used_gpu_memory, present);
        } else if (api_.compute_v1) {
            std::vector<NvmlProcessInfoV1> buf(64);
            auto call = [&](unsigned *n, NvmlProcessInfoV1 *p) {
                return api_.compute_v1(d.handle, n, p);
            };
            unsigned n = NvmlApi::query(call, buf);
            for (unsigned i = 0; i < n; ++i)
                note(d, buf[i].pid, buf[i].used_gpu_memory, present);
        }
        if (api_.graphics_v2) {
            std::vector<NvmlProcessInfoV2> buf(64);
            auto call = [&](unsigned *n, NvmlProcessInfoV2 *p) {
                return api_.graphics_v2(d.handle, n, p);
            };
            unsigned n = NvmlApi::query(call, buf);
            for (unsigned i = 0; i < n; ++i)
                note(d, buf[i].pid, buf[i].used_gpu_memory, present);
        } else if (api_.graphics_v1) {
            std::vector<NvmlProcessInfoV1> buf(64);
            auto call = [&](unsigned *n, NvmlProcessInfoV1 *p) {
                return api_.graphics_v1(d.handle, n, p);
            };
            unsigned n = NvmlApi::query(call, buf);
            for (unsigned i = 0; i < n; ++i)
                note(d, buf[i].pid, buf[i].used_gpu_memory, present);
        }
    }

    void note(NvDevice &d, unsigned pid, unsigned long long bytes,
              std::map<std::string, bool> &present) {
        if (pid == 0)
            return;
        const std::string k = key_of(d.bus, int(pid));
        NvProc &ps = procs_[k];
        ps.pid = int(pid);
        ps.bus = d.bus;
        // NVML_VALUE_NOT_AVAILABLE is 0xFFFFFFFFFFFFFFFF (WDDM, or a process
        // whose memory the driver will not report); leave the column as "--".
        if (bytes != ~0ull) {
            ps.mem_bytes = bytes;
            ps.mem_known = true;
        }
        present[k] = true;
    }

    NvmlApi api_;
    std::vector<NvDevice> devices_;
    bool enumerated_ = false;
    std::map<std::string, NvProc> procs_;
};

// Is an Intel DRM card bound? Used to decide between the two backends on a
// host that has both.
bool intel_drm_present() {
    DIR *d = opendir("/sys/class/drm");
    if (!d)
        return false;
    bool found = false;
    while (dirent *e = readdir(d)) {
        const char *n = e->d_name;
        if (std::strncmp(n, "card", 4) != 0 ||
            !std::isdigit(static_cast<unsigned char>(n[4])))
            continue;
        std::string drv = std::string("/sys/class/drm/") + n + "/device/driver";
        char tgt[512];
        ssize_t len = readlink(drv.c_str(), tgt, sizeof(tgt) - 1);
        if (len <= 0)
            continue;
        tgt[len] = '\0';
        std::string base(tgt);
        size_t slash = base.rfind('/');
        if (slash != std::string::npos)
            base = base.substr(slash + 1);
        if (base == "i915" || base == "xe") {
            found = true;
            break;
        }
    }
    closedir(d);
    return found;
}

// Picks between the two Linux backends. Decides by *driver presence* rather
// than by which one happens to have clients right now: a monitor started before
// the workload would otherwise lock in the wrong choice for the whole run.
// Auto prefers Intel, because a hybrid workstation's i915/xe is where vkop's
// own Vulkan work runs (and it is the only backend that can report engines);
// VKOP_GPU_TOP_BACKEND=nvidia or -g nvidia overrides.
class Dispatcher : public Backend {
  public:
    const char *name() const override { return pick().name(); }
    const std::vector<EngineSpec> &engines() const override {
        return pick().engines();
    }
    Columns columns() const override { return pick().columns(); }
    std::vector<ClientInfo> collect(int pid) override {
        return pick().collect(pid);
    }
    uint64_t freq_mhz() override { return pick().freq_mhz(); }
    double idle_pct(double wall_s) override { return pick().idle_pct(wall_s); }

    bool select(const std::string &name) override {
        if (name == "intel" || name == "i915" || name == "xe") {
            // Always satisfiable -- the code path exists everywhere -- but on a
            // pure-NVIDIA host it is a mistake, and an all-dashes table with no
            // explanation reads like the tool is broken.
            if (!intel_drm_present())
                std::fprintf(stderr,
                             "vkop_gpu_top: -g intel, but no i915/xe card is "
                             "bound on this host; the table will stay empty\n");
            forced_ = "intel";
            return true;
        }
        if (name == "nvidia" || name == "nvml") {
            if (!nv().available())
                return false;
            forced_ = "nvidia";
            return true;
        }
        if (name == "auto" || name.empty()) {
            forced_.clear();
            return true;
        }
        return false;
    }

  private:
    // The two implementations are process-wide singletons: they hold error
    // state (a missing NVML library, a resolved sysfs path) that must survive
    // across windows, and the run only ever uses one of them.
    static Intel &in() {
        static Intel b;
        return b;
    }
    static Nvidia &nv() {
        static Nvidia b;
        return b;
    }
    Backend &pick() const {
        if (choice_.empty()) {
            std::string want = forced_;
            if (want.empty()) {
                if (const char *env = std::getenv("VKOP_GPU_TOP_BACKEND"))
                    want = env;
            }
            if (want == "nvidia" || want == "nvml")
                choice_ = nv().available() ? "nvidia" : "intel";
            else if (want == "intel" || want == "i915" || want == "xe")
                choice_ = "intel";
            else
                choice_ = (nv().available() && !intel_drm_present()) ? "nvidia"
                                                                     : "intel";
        }
        return choice_ == "nvidia" ? static_cast<Backend &>(nv())
                                   : static_cast<Backend &>(in());
    }

    mutable std::string choice_;
    std::string forced_;
};

} // namespace

Backend &backend() {
    static Dispatcher d;
    return d;
}

} // namespace gputop
