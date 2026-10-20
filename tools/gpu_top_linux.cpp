// Copyright 2025 @junka
// Linux backend for vkop_gpu_top: Intel i915/xe through the two interfaces the
// kernel exposes to ordinary users.
//
//   /proc/<pid>/fdinfo/<fd>   per-DRM-client accounting: cumulative nanosecond
//                             busy time per hardware engine (drm-engine-render
//                             / -compute / -copy / ...), plus current resident
//                             memory (drm-resident-<region>). No privileges
//                             needed for one's own -- or same-uid -- processes.
//   /sys/class/drm/cardN/gt/* i915 GT sysfs: rps_act_freq_mhz and
//                             rc6_residency_ms. World-readable.
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
#include "gpu_top_backend.hpp"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <dirent.h>
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

Backend &backend() {
    static Intel b;
    return b;
}

} // namespace gputop
