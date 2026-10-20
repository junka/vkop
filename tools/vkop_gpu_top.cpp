// Copyright 2025 @junka
// vkop_gpu_top -- a rootless, Intel-flavoured "gpu top" for profiling vkop
// runs.
//
// intel_gpu_top needs root on most hosts (it opens the i915 PMU via
// perf_event_open, which /proc/sys/kernel/perf_event_paranoid>=2 blocks for
// unprivileged users, and its OA path is absent on integrated i915 anyway).
// This tool instead reads the two interfaces the kernel exposes to ordinary
// users:
//
//   /proc/<pid>/fdinfo/<fd>   per-DRM-client accounting: cumulative
//                             nanosecond busy time per hardware engine
//                             (drm-engine-render / -compute / -copy / ...),
//                             plus current resident memory
//                             (drm-resident-<region>). No privileges needed
//                             for one's own -- or same-uid -- processes.
//   /sys/class/drm/cardN/gt/* i915 GT sysfs: rps_act_freq_mhz and
//                             rc6_residency_ms. World-readable.
//
// What it gives you that the existing VKOP_SUBMIT_PROF cannot: the engine
// busy time is measured by the kernel's own scheduler accounting, so it is
// the true hardware utilization. VKOP_SUBMIT_PROF's per-op query-pool
// timestamps measure the *span* of a recorded op, which includes queue
// drain / stall bubbles. busy% over the same window therefore separates
// "compute-bound" (busy ~= 100%) from "bubble-bound" (busy well under
// 100%), the distinction behind the still-open latency/occupancy questions
// (see coopmat-not-dram-bound.md).
//
// The percentage is computed the same way intel_gpu_top does: read every
// counter, sleep one period, read again, and divide the delta by the wall
// delta -- so a single read is never interpreted, only a delta.
//
// Two implementation traps, both verified on ARL-P / i915 (card1):
//   * A process usually holds *several* i915 render-node fds (the Vulkan
//     loader opens one, the ICD opens the working one). The kernel accounts
//     each drm_file separately, so the busy time is split across them --
//     sum over all of them. Taking the first fd yields a permanent 0.
//   * drm-engine-capacity-* lines carry a queue count ("...: 2"), not a
//     nanosecond total; they are skipped by requiring the " ns" suffix.
//   * The per-client engine time is context-residency based, so on a *shared*
//     engine the sum across clients can exceed the wall clock (i915 overlaps
//     preempted contexts). Whole-GPU mode (no -p) therefore prints a
//     per-client sum that is an upper bound; use -p PID for the exact figure
//     of a single profiled process.
//
// Usage:
//   vkop_gpu_top [-p PID] [-s MS] [-n COUNT] [-r ROWS] [-P] [--no-proc] [-l]
//
//   -p PID    monitor this process (default: aggregate over every readable
//             DRM client, i.e. whole-GPU busy)
//   -s MS     sample period in milliseconds (default 500, min 10)
//   -n COUNT  stop after COUNT samples (default: run until Ctrl-C)
//   -r ROWS   rows kept on screen in place mode (default: as many as fit in
//             the terminal; ignored with -l)
//   -P        force per-process rows on. They are already on by default when
//             monitoring the whole GPU; with -p the breakdown would just echo
//             the summary row, so it is off there unless -P is given.
//   --no-proc force them off (compact one-line summary per sample).
//
//             The per-process list is the fdinfo equivalent of nvtop's process
//             list -- intel_gpu_top has no such view (its multiple rows are
//             *engines*), but fdinfo carries the pid, so it is free here.
//   -l        line mode: append one row per sample -- a scrolling log that
//             survives in the terminal and can be piped to a file -- instead
//             of redrawing a fixed window in place (the default on a tty)
//
// Build: plain C++17, no dependencies -- it does not link libvkop.
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include <dirent.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

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
    std::string s = read_file("/proc/" + std::to_string(pid) + "/comm");
    return trim(s);
}

// ---------------------------------------------------------------------------
// fdinfo parsing.
// ---------------------------------------------------------------------------

// One DRM client (one open fd on /dev/dri/*). Every field is optional: a
// non-DRM fd yields is_drm=false and is discarded by the caller.
struct ClientInfo {
    bool is_drm = false;
    int pid = 0;
    std::string driver;                        // i915 / xe
    std::string pdev;                          // e.g. 0000:00:02.0
    std::map<std::string, uint64_t> engine_ns; // engine -> cumulative busy ns
    uint64_t resident_bytes = 0;               // sum over memory regions
};

ClientInfo parse_fdinfo(const std::string &text) {
    ClientInfo ci;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t eol = text.find('\n', pos);
        if (eol == std::string::npos)
            eol = text.size();
        std::string line = text.substr(pos, eol - pos);
        pos = eol + 1;
        if (line.rfind("drm-driver:", 0) == 0) {
            ci.is_drm = true;
            ci.driver = trim(line.substr(std::strlen("drm-driver:")));
        } else if (line.rfind("drm-pdev:", 0) == 0) {
            ci.pdev = trim(line.substr(std::strlen("drm-pdev:")));
        } else if (line.rfind("drm-engine-", 0) == 0) {
            size_t colon = line.find(':');
            size_t ns = line.rfind(" ns");
            // A capacity line has no " ns" -- skip it (see file header).
            if (colon == std::string::npos || ns == std::string::npos)
                continue;
            std::string name = line.substr(std::strlen("drm-engine-"),
                                           colon - std::strlen("drm-engine-"));
            ci.engine_ns[name] =
                std::strtoull(line.c_str() + colon + 1, nullptr, 10);
        } else if (line.rfind("drm-resident-", 0) == 0) {
            size_t colon = line.find(':');
            if (colon == std::string::npos)
                continue;
            ci.resident_bytes +=
                std::strtoull(line.c_str() + colon + 1, nullptr, 10);
        }
    }
    return ci;
}

// Walk /proc/<pid>/fd and append every DRM client found. When pid == 0,
// walk every readable process (whole-GPU view). Unreadable /proc entries
// (other users, kernel threads, races with process exit) are skipped
// silently -- no privilege escalation is attempted.
void collect_clients(int pid, std::vector<ClientInfo> &out) {
    std::vector<int> pids;
    if (pid > 0) {
        pids.push_back(pid);
    } else {
        DIR *d = opendir("/proc");
        if (!d)
            return;
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
            ClientInfo ci = parse_fdinfo(
                read_file("/proc/" + std::to_string(p) + "/fdinfo/" + n));
            // Intel only: the fdinfo engine names and the GT sysfs below are
            // i915/xe-specific. On a hybrid host (this one also has an NVIDIA
            // card) including another vendor's clients would both mis-resolve
            // the pdev -> card mapping and fold foreign engine counters into
            // the Util% columns.
            if (!ci.is_drm || (ci.driver != "i915" && ci.driver != "xe"))
                continue;
            ci.pid = p;
            out.push_back(std::move(ci));
        }
        closedir(d);
    }
}

// ---------------------------------------------------------------------------
// Snapshot / aggregation.
// ---------------------------------------------------------------------------

// Per-process aggregation of the fdinfo clients: a process may hold several
// DRM fds (loader + ICD), all of which fold into one row.
struct PidAgg {
    std::map<std::string, uint64_t> engine_ns;
    uint64_t resident_bytes = 0;
    int nfds = 0;
};

struct Sample {
    std::map<std::string, uint64_t> engine_ns; // summed over all clients
    std::map<int, PidAgg> by_pid; // per-process rows + the "busiest" column
    uint64_t resident_bytes = 0;
    int nclients = 0;
    std::string pdev; // dominant GPU among the clients
};

Sample take_snapshot(int pid) {
    std::vector<ClientInfo> clients;
    collect_clients(pid, clients);

    Sample s;
    std::map<std::string, int> pdev_hits;
    for (const auto &c : clients) {
        PidAgg &pa = s.by_pid[c.pid];
        for (const auto &kv : c.engine_ns) {
            s.engine_ns[kv.first] += kv.second;
            pa.engine_ns[kv.first] += kv.second;
        }
        s.resident_bytes += c.resident_bytes;
        pa.resident_bytes += c.resident_bytes;
        ++pa.nfds;
        ++s.nclients;
        if (!c.pdev.empty())
            ++pdev_hits[c.pdev];
    }
    int best = 0;
    for (const auto &kv : pdev_hits)
        if (kv.second > best) {
            best = kv.second;
            s.pdev = kv.first;
        }
    return s;
}

// Look up an engine counter, treating "absent" as zero (const-safe: the
// callers hold const snapshots).
uint64_t engine_of(const std::map<std::string, uint64_t> &m,
                   const std::string &k) {
    auto it = m.find(k);
    return it == m.end() ? 0 : it->second;
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
// Output.
// ---------------------------------------------------------------------------

// Display order/labels, mirroring intel_gpu_top's engine columns.
const char *kEngineOrder[] = {"render", "compute", "copy", "video",
                              "video-enhance"};
const char *kEngineLabel[] = {"Render", "Compute", "Copy", "Video", "VEnh"};
const size_t kEngineCount = sizeof(kEngineOrder) / sizeof(kEngineOrder[0]);

// Engines present in either snapshot, in display order.
std::vector<size_t> active_engines(const Sample &a, const Sample &b) {
    std::vector<size_t> out;
    for (size_t i = 0; i < kEngineCount; ++i) {
        std::string k = kEngineOrder[i];
        if (a.engine_ns.count(k) != 0 || b.engine_ns.count(k) != 0)
            out.push_back(i);
    }
    return out;
}

// Column widths, shared by the header and every row so the table lines up.
// Note the value formats carry a unit suffix *outside* the width ("%8.2fs",
// "%8.1f%%"), so those columns are one wider than the 8 they look like -- the
// header must use the same total width or the drift accumulates across the
// engine columns.
constexpr int kTimeW = 9;  // "%8.2fs" / "%9s"
constexpr int kEngW = 9;   // "%8.1f%%" / "%9s"
constexpr int kFreqW = 10; // "%10s"
constexpr int kRc6W = 9;   // "%8.1f%%" / "%9s"
constexpr int kResW = 9;   // "%9.1f" / "%9s"

void print_header(int pid, const Sample &s, int period_ms,
                  const std::vector<size_t> &eng) {
    std::printf("vkop_gpu_top  pid=%s  pdev=%s  clients=%d  period=%dms%s\n",
                pid > 0 ? std::to_string(pid).c_str() : "all",
                s.pdev.empty() ? "?" : s.pdev.c_str(), s.nclients, period_ms,
                pid > 0 ? "" : "  (Util% = per-client sum; may exceed 100%)");
    std::printf("%*s", kTimeW, "Time");
    for (size_t i : eng)
        std::printf("%*s", kEngW, kEngineLabel[i]);
    std::printf("%*s%*s%*s  %s\n", kFreqW, "FreqMHz", kRc6W, "RC6%", kResW,
                "ResMB", "Client");
}

// A delta-based percentage: (cur - prev) / wall. Never reads a counter
// absolutely, so a wrap or a re-used client-id would show as a one-frame
// glitch rather than a bogus steady value.
double delta_pct(uint64_t cur, uint64_t prev, double wall_s,
                 double unit_per_s) {
    if (wall_s <= 0 || cur < prev)
        return 0.0;
    return 100.0 * static_cast<double>(cur - prev) / (wall_s * unit_per_s);
}

// Idle fraction of the *least idle* tile over the window -- i.e. of the tile
// actually doing the profiled work. Reporting the min rather than a sum keeps
// the number honest on a multi-tile part: a GPU-bound workload shows ~0%
// (the render tile never enters RC6 while busy) whereas a fully idle GPU
// shows ~100%. Summing the tiles instead reports the idle media tile.
double min_tile_rc6_pct(const std::map<std::string, uint64_t> &cur,
                        const std::map<std::string, uint64_t> &prev,
                        double wall_s) {
    double best = -1.0;
    for (const auto &kv : cur) {
        auto it = prev.find(kv.first);
        if (it == prev.end())
            continue;
        double pct = delta_pct(kv.second, it->second, wall_s, 1e3);
        if (best < 0.0 || pct < best)
            best = pct;
    }
    return best < 0.0 ? 0.0 : best;
}

// Terminal height in rows, or 24 when stdout is not a tty (in which case the
// callers do not redraw anyway).
int terminal_height() {
    struct winsize ws {};
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_row > 0)
        return ws.ws_row;
    return 24;
}

// Format one data row. Returned as a string (not streamed) so the in-place
// mode can keep the last N rows and rewrite the whole block atomically each
// frame instead of drawing a single row that the next frame overwrites.
std::string format_row(double elapsed_s, const Sample &prev, const Sample &cur,
                       double wall_s, const std::vector<size_t> &eng,
                       uint64_t freq, double rc6_pct) {
    char buf[512];
    std::string out;

    auto emit = [&](const char *fmt, auto... args) {
        int n = std::snprintf(buf, sizeof(buf), fmt, args...);
        if (n > 0)
            out.append(buf, static_cast<size_t>(n));
    };

    emit("%8.2fs", elapsed_s);
    for (size_t i : eng) {
        const std::string &k = kEngineOrder[i];
        emit("%8.1f%%", delta_pct(engine_of(cur.engine_ns, k),
                                  engine_of(prev.engine_ns, k), wall_s, 1e9));
    }
    emit("%*s%8.1f%%%9.1f  ", kFreqW,
         freq ? std::to_string(freq).c_str() : "-", rc6_pct,
         static_cast<double>(cur.resident_bytes) / (1024.0 * 1024.0));

    // Busiest client by render-busy delta this window.
    int best_pid = 0;
    double best_pct = 0;
    for (const auto &kv : cur.by_pid) {
        auto it = prev.by_pid.find(kv.first);
        uint64_t p = it == prev.by_pid.end()
                         ? 0
                         : engine_of(it->second.engine_ns, "render");
        double v =
            delta_pct(engine_of(kv.second.engine_ns, "render"), p, wall_s, 1e9);
        if (v > best_pct) {
            best_pct = v;
            best_pid = kv.first;
        }
    }
    if (best_pid > 0)
        emit("%s(%d) %.0f%%", read_comm(best_pid).c_str(), best_pid, best_pct);
    out.push_back('\n');
    return out;
}

// Per-process rows: one line per client that used the GPU this window,
// hottest first. They use the *same columns* as the summary row -- the engine
// percentages, the resident size and the trailing client name all line up --
// with the fields that only make sense for the whole GPU (the sample clock,
// the frequency, the RC6 residency) blanked out as "--", so a process row
// reads as a continuation of the table rather than a separate block.
// Near-idle clients are dropped so the profiled run stands out from the
// compositor's background noise.
std::vector<std::string> format_proc_lines(const Sample &prev,
                                           const Sample &cur, double wall_s,
                                           const std::vector<size_t> &eng,
                                           int cap) {
    struct Row {
        int pid;
        std::string name;
        std::vector<double> pct;
        double res_mb;
        double hot;
    };
    std::vector<Row> rows;
    const PidAgg empty;
    for (const auto &kv : cur.by_pid) {
        auto it = prev.by_pid.find(kv.first);
        const PidAgg &pv = it == prev.by_pid.end() ? empty : it->second;
        Row r;
        r.pid = kv.first;
        r.name = read_comm(kv.first);
        r.res_mb =
            static_cast<double>(kv.second.resident_bytes) / (1024.0 * 1024.0);
        r.hot = 0;
        for (size_t i : eng) {
            const std::string &k = kEngineOrder[i];
            double v = delta_pct(engine_of(kv.second.engine_ns, k),
                                 engine_of(pv.engine_ns, k), wall_s, 1e9);
            r.pct.push_back(v);
            if (v > r.hot)
                r.hot = v;
        }
        if (r.hot > 0.05)
            rows.push_back(std::move(r));
    }
    std::sort(rows.begin(), rows.end(), [](const Row &a, const Row &b) {
        if (a.hot != b.hot)
            return a.hot > b.hot;
        return a.pid < b.pid;
    });

    std::vector<std::string> out;
    const int n = std::min<int>(cap, static_cast<int>(rows.size()));
    for (int i = 0; i < n; ++i) {
        char buf[512];
        std::string line;
        auto emit = [&](const char *fmt, auto... args) {
            int c = std::snprintf(buf, sizeof(buf), fmt, args...);
            if (c > 0)
                line.append(buf, static_cast<size_t>(c));
        };
        emit("%*s", kTimeW, "--");
        for (double v : rows[i].pct)
            emit("%8.1f%%", v);
        emit("%*s%*s%9.1f  %s(%d)\n", kFreqW, "--", kRc6W, "--", rows[i].res_mb,
             rows[i].name.c_str(), rows[i].pid);
        out.push_back(std::move(line));
    }
    return out;
}

} // namespace

static std::atomic<bool> g_stop{false};
static void on_sigint(int) { g_stop = true; }

static void usage(const char *argv0) {
    std::fprintf(
        stderr,
        "usage: %s [-p PID] [-s MS] [-n COUNT] [-r ROWS] [-P] [--no-proc] [-l]\n"
        "  -p PID    monitor one process (default: whole-GPU)\n"
        "  -s MS     sample period, ms (default 500)\n"
        "  -n COUNT  stop after COUNT samples (default: until ^C)\n"
        "  -r ROWS   rows kept on screen in place mode (default: fits\n"
        "            the terminal; ignored with -l)\n"
        "  -P        force per-process rows on (default: on for whole-GPU,\n"
        "            off with -p since a single process is already named)\n"
        "  --no-proc force per-process rows off (compact one-line summary)\n"
        "  -l        line mode: append every row (a scrolling log to\n"
        "            save/pipe) instead of redrawing a fixed window\n",
        argv0);
}

int main(int argc, char **argv) {
    int pid = 0;
    int period_ms = 500;
    long long count = 0;
    int max_rows = 0;  // 0 = auto
    int proc_mode = -1; // -1 auto (on for whole-GPU, off with -p), 0 off, 1 on
    bool line_mode = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need_next = [&](const char *what) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "vkop_gpu_top: %s needs a value\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return 0;
        } else if (a == "-P") {
            proc_mode = 1;
        } else if (a == "--no-proc") {
            proc_mode = 0;
        } else if (a == "-l") {
            line_mode = true;
        } else if (a == "-p") {
            pid = std::atoi(need_next("-p"));
        } else if (a == "-s") {
            period_ms = std::max(10, std::atoi(need_next("-s")));
        } else if (a == "-n") {
            count = std::atoll(need_next("-n"));
        } else if (a == "-r") {
            max_rows = std::max(1, std::atoi(need_next("-r")));
        } else if (!a.empty() &&
                   std::isdigit(static_cast<unsigned char>(a[0]))) {
            pid = std::atoi(a.c_str()); // bare PID shorthand
        } else {
            std::fprintf(stderr, "vkop_gpu_top: unknown argument '%s'\n",
                         a.c_str());
            usage(argv[0]);
            return 2;
        }
    }

    std::signal(SIGINT, on_sigint);
    std::signal(SIGTERM, on_sigint);
    // Whole-GPU mode is a *shared* device: list the clients by default, the way
    // a `top` would. With -p there is only one process, already named in the
    // summary row, so the breakdown would just echo it.
    const bool per_proc = proc_mode >= 0 ? (proc_mode == 1) : (pid == 0);
    const bool redraw = isatty(STDOUT_FILENO) && !line_mode;

    Sample prev = take_snapshot(pid);
    auto t_prev = std::chrono::steady_clock::now();
    auto t_start = t_prev;
    std::string gt_base = find_gt_base(prev.pdev);
    auto rc6_prev_map = read_rc6_all(gt_base);

    if (prev.nclients == 0) {
        std::fprintf(stderr,
                     "vkop_gpu_top: no DRM client%s found yet; "
                     "waiting for GPU activity\n",
                     pid > 0 ? " for the given pid" : "s");
    }

    long long iter = 0;
    bool header_done = false;
    // In-place mode keeps a rolling window of recent samples so the screen
    // shows history rather than a single row that the next frame erases. The
    // window is sized to the terminal (title + column header + one spare row)
    // unless -r overrides it.
    const int rows_cap =
        max_rows > 0 ? max_rows
                     : (redraw ? std::max(1, terminal_height() - 3) : 0);
    std::deque<std::string> history;
    int lines_drawn = 0;
    while (!g_stop && (count == 0 || iter < count)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(period_ms));

        Sample cur = take_snapshot(pid);
        auto t_cur = std::chrono::steady_clock::now();
        double wall_s = std::chrono::duration<double>(t_cur - t_prev).count();
        double elapsed_s =
            std::chrono::duration<double>(t_cur - t_start).count();

        if (gt_base.empty() && !cur.pdev.empty())
            gt_base = find_gt_base(cur.pdev);
        auto rc6_cur_map = read_rc6_all(gt_base);

        if (pid > 0 && cur.nclients == 0) {
            std::fprintf(stderr,
                         "vkop_gpu_top: pid %d holds no DRM client "
                         "(exited?)\n",
                         pid);
            break;
        }

        std::vector<size_t> eng = active_engines(prev, cur);
        uint64_t freq = read_freq_mhz(gt_base);
        double rc6_pct = min_tile_rc6_pct(rc6_cur_map, rc6_prev_map, wall_s);

        // One sample is one summary line, optionally followed by its
        // per-process breakdown (same engine columns, process named in the
        // leading field).
        std::vector<std::string> lines;
        lines.push_back(
            format_row(elapsed_s, prev, cur, wall_s, eng, freq, rc6_pct));
        if (per_proc) {
            for (auto &l : format_proc_lines(prev, cur, wall_s, eng, 10))
                lines.push_back(std::move(l));
        }

        if (redraw) {
            for (auto &l : lines)
                history.push_back(std::move(l));
            while (static_cast<int>(history.size()) > rows_cap)
                history.pop_front();
            // Move back over the block drawn last frame and clear from there
            // down, then repaint title + column header + the window. Cursor-up
            // (never ESC[H, which homes to the *screen* origin) keeps whatever
            // the shell printed above the tool intact.
            if (lines_drawn > 0)
                std::printf("\033[%dA", lines_drawn);
            std::printf("\033[J");
            print_header(pid, cur, period_ms, eng);
            for (const auto &r : history)
                std::fputs(r.c_str(), stdout);
            lines_drawn = 2 + static_cast<int>(history.size());
            std::fflush(stdout);
        } else {
            if (!header_done) {
                print_header(pid, cur, period_ms, eng);
                header_done = true;
            }
            for (const auto &l : lines)
                std::fputs(l.c_str(), stdout);
        }

        prev = std::move(cur);
        t_prev = t_cur;
        rc6_prev_map = std::move(rc6_cur_map);
        ++iter;
    }
    return 0;
}
