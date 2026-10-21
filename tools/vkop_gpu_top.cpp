// Copyright 2025 @junka
// vkop_gpu_top -- a rootless "gpu top" for profiling vkop runs.
//
// It answers the one question VKOP_SUBMIT_PROF cannot: whether a run is
// *compute-bound* or *bubble-bound*. vkop's own per-op query pools measure the
// span of a recorded op, which includes queue drain and stall bubbles; the
// kernel's own per-client accounting measures how long the hardware was
// actually resident. busy% over the same wall window separates the two, and is
// the number behind the still-open latency/occupancy questions (see
// coopmat-not-dram-bound.md).
//
// So this tool reads only what the OS already hands to ordinary users -- no
// perf_event, no powermetrics, no root:
//   Linux (i915/xe)   /proc/<pid>/fdinfo cumulative ns per engine + GT sysfs
//                     (see gpu_top_linux.cpp)
//   Linux (NVIDIA)    NVML per-process SM/memory/encoder/decoder utilization,
//                     dlopen'd at runtime (see gpu_top_linux.cpp; NVML reports
//                     percentages, so that backend synthesizes the cumulative
//                     counters this core needs)
//   macOS (AGX)       the AGXDeviceUserClient nodes' AppUsage cumulative ns
//                     per GPU client (see gpu_top_darwin.cpp)
// The platform-specific half is small and narrow: a backend produces a list of
// GPU clients, each carrying cumulative nanosecond counters. Everything here
// (matching clients across samples, differencing, aggregating per process,
// rendering) is shared. A platform that has several stacks (Linux usually has
// an Intel card *and* an NVIDIA one) picks between them with -g.
//
// Two rules the whole design rests on, both of which a naive first attempt
// breaks:
//   * A single read of a counter means nothing. Every number printed here is
//     (cur - prev) over the same window's wall delta, exactly as
//     intel_gpu_top does it, so a counter reset or a recycled client shows up
//     as a one-frame gap rather than a bogus steady value.
//   * Clients are matched by identity, not by position, and diffed *before*
//     being summed per process. A process holds several clients (on Linux the
//     Vulkan loader and the ICD each open a render node; on macOS one vkop run
//     was measured with 4 command queues on 1 client node) and those clients
//     appear and disappear while it runs. Sum first and diff second and a
//     vanished client turns into fabricated busy time.
//
// On a *shared* engine the per-client sum is an upper bound -- both kernels
// overlap preempted contexts, so the clients' times can add up to more than the
// wall clock. Whole-GPU mode (no -p) prints that sum and says so; -p PID gives
// the exact figure for one profiled process.
//
// Usage:
//   vkop_gpu_top [-p PID] [-s MS] [-n COUNT] [-r ROWS] [-P] [--no-proc] [-g N]
//                [-l]
//
//   -p PID    monitor this process (default: aggregate over every readable
//             GPU client, i.e. whole-GPU busy)
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
//             *engines*), but the client records carry the pid, so it is free
//             here.
//   -l        line mode: append one row per sample -- a scrolling log that
//             survives in the terminal and can be piped to a file -- instead
//             of redrawing a fixed window in place (the default on a tty)
//
// Build: plain C++17, no dependencies -- it does not link libvkop, and on
// macOS only the system IOKit.
#include "gpu_top_backend.hpp"

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

#include <sys/ioctl.h>
#include <unistd.h>

namespace {

// ---------------------------------------------------------------------------
// Snapshot / aggregation.
// ---------------------------------------------------------------------------

// Per-process aggregation of the client deltas belonging to it in one window.
struct PidAgg {
    std::map<std::string, uint64_t> delta_ns;
    uint64_t bytes = 0;
    bool bytes_valid = false;
    int nhandles = 0;
    std::string proc;
};

// One window: everything that happened between two consecutive samples.
struct Window {
    std::map<std::string, uint64_t> delta_ns; // summed over all clients
    std::map<int, PidAgg> by_pid;
    uint64_t bytes = 0; // device-level total, when the backend can sum it
    bool bytes_valid = false;
    int nclients = 0;   // clients that contributed a delta
    int npending = 0;   // clients seen for the first time, no delta yet
    std::string device; // dominant GPU among the contributing clients
};

using Clients = std::map<std::string, gputop::ClientInfo>; // keyed by identity

// The one place counters are differenced. `prev` holds the absolute readings
// of the last sample; a client absent from it contributes nothing this window
// and is picked up from the next one onward.
Window diff_windows(const Clients &prev, const Clients &cur) {
    Window w;
    std::map<std::string, int> device_hits;
    for (const auto &kv : cur) {
        const gputop::ClientInfo &c = kv.second;
        auto p = prev.find(kv.first);
        if (p == prev.end()) {
            ++w.npending;
            continue;
        }
        const gputop::ClientInfo &pc = p->second;
        PidAgg &pa = w.by_pid[c.pid];
        pa.proc = c.proc;
        pa.nhandles += c.nhandles;
        ++w.nclients;
        for (const auto &en : c.engine_ns) {
            auto it = pc.engine_ns.find(en.first);
            // A counter that went backwards was reset or recycled: no delta.
            if (it == pc.engine_ns.end() || en.second < it->second)
                continue;
            uint64_t d = en.second - it->second;
            w.delta_ns[en.first] += d;
            pa.delta_ns[en.first] += d;
        }
        if (c.mem_valid && c.mem_is_per_process) {
            // Assign, never add: this number describes the process, and the
            // process may own several of the clients being folded here. Every
            // client of one process reports the same figure.
            pa.bytes = c.mem_bytes;
            pa.bytes_valid = true;
        } else if (c.mem_valid) {
            w.bytes += c.mem_bytes;
            pa.bytes += c.mem_bytes;
            pa.bytes_valid = true;
            w.bytes_valid = true;
        }
        if (!c.device.empty())
            ++device_hits[c.device];
    }
    int best = 0;
    for (const auto &kv : device_hits)
        if (kv.second > best) {
            best = kv.second;
            w.device = kv.first;
        }
    return w;
}

Clients to_map(const std::vector<gputop::ClientInfo> &v) {
    Clients out;
    for (const auto &c : v)
        if (!c.identity.empty())
            out[c.identity] = c;
    return out;
}

uint64_t counter_of(const std::map<std::string, uint64_t> &m,
                    const std::string &k) {
    auto it = m.find(k);
    return it == m.end() ? 0 : it->second;
}

// ---------------------------------------------------------------------------
// Output.
// ---------------------------------------------------------------------------

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

// Engines this backend has ever reported, in its display order. Sticky rather
// than recomputed per window so the column set cannot jitter when one engine
// happens to idle.
struct EngineSet {
    std::vector<const gputop::EngineSpec *> cols;
    void seen(const Window &w) {
        // Walk the backend's declared order, not the window's: delta_ns is a
        // std::map, so iterating it would lay the columns out alphabetically
        // (compute, copy, render, ...) and make the *first* column -- the one
        // format_row uses to name the busiest client -- the least interesting
        // engine on the part.
        for (const auto &e : gputop::backend().engines())
            if (!has(e.key) && w.delta_ns.count(e.key) != 0)
                cols.push_back(&e);
    }
    bool has(const std::string &k) const {
        for (const auto *e : cols)
            if (k == e->key)
                return true;
        return false;
    }
};

// A delta-based percentage: a nanosecond counter divided by the window.
double delta_pct(uint64_t d, double wall_s, double unit_per_s) {
    if (wall_s <= 0)
        return 0.0;
    return 100.0 * static_cast<double>(d) / (wall_s * unit_per_s);
}

struct Cols {
    std::vector<const gputop::EngineSpec *> eng;
    bool freq;
    bool idle;
    // What the summary row prints in the memory column. Either the device
    // total (Linux: per-client figures that may be summed) or, in the
    // single-process view, that one process's own figure -- there is nothing
    // to sum, so a per-process number is also the device number.
    bool mem;
    double mem_mb;
};

void print_header(int pid, const Window &w, int period_ms, const Cols &cols) {
    const gputop::Columns c = gputop::backend().columns();
    std::printf("vkop_gpu_top  %s  pid=%s  %s=%s  clients=%d  period=%dms%s\n",
                gputop::backend().name(),
                pid > 0 ? std::to_string(pid).c_str() : "all", c.device_label,
                w.device.empty() ? "?" : w.device.c_str(), w.nclients,
                period_ms,
                pid > 0 ? "" : "  (Util% = per-client sum; may exceed 100%)");
    std::printf("%*s", kTimeW, "Time");
    for (const auto *e : cols.eng)
        std::printf("%*s", kEngW, e->label);
    if (cols.freq)
        std::printf("%*s", kFreqW, "FreqMHz");
    if (cols.idle)
        std::printf("%*s", kRc6W, "RC6%");
    std::printf("%*s  %s\n", kResW, c.mem_label, "Client");
}

// Format one data row. Returned as a string (not streamed) so the in-place
// mode can keep the last N rows and rewrite the whole block atomically each
// frame instead of drawing a single row that the next frame overwrites.
std::string format_row(double elapsed_s, const Window &cur, double wall_s,
                       const Cols &cols, uint64_t freq, double idle_pct,
                       int pending) {
    char buf[512];
    std::string out;

    auto emit = [&](const char *fmt, auto... args) {
        int n = std::snprintf(buf, sizeof(buf), fmt, args...);
        if (n > 0)
            out.append(buf, static_cast<size_t>(n));
    };

    emit("%8.2fs", elapsed_s);
    for (const auto *e : cols.eng)
        emit("%8.1f%%",
             delta_pct(counter_of(cur.delta_ns, e->key), wall_s, 1e9));
    if (cols.freq)
        emit("%*s", kFreqW, freq ? std::to_string(freq).c_str() : "-");
    if (cols.idle)
        emit("%8.1f%%", idle_pct);
    if (cols.mem)
        emit("%9.1f  ", cols.mem_mb);
    else
        emit("%*s  ", kResW, "--");

    // Busiest process by the first engine column's delta this window. On Linux
    // that is "render"; on macOS the single column is all there is.
    const char *hot = cols.eng.empty() ? "gpu" : cols.eng.front()->key;
    int best_pid = 0;
    double best_pct = 0;
    for (const auto &kv : cur.by_pid) {
        double v = delta_pct(counter_of(kv.second.delta_ns, hot), wall_s, 1e9);
        if (v > best_pct) {
            best_pct = v;
            best_pid = kv.first;
        }
    }
    if (best_pid > 0)
        emit("%s(%d) %.0f%%", cur.by_pid.at(best_pid).proc.c_str(), best_pid,
             best_pct);
    if (pending > 0)
        emit("  +%d new client", pending);
    out.push_back('\n');
    return out;
}

// Per-process rows: one line per client that used the GPU this window,
// hottest first. They use the *same columns* as the summary row -- the engine
// percentages, the memory figure and the trailing client name all line up --
// with the fields that only make sense for the whole GPU (the sample clock,
// the frequency, the idle residency) blanked out as "--", so a process row
// reads as a continuation of the table rather than a separate block.
// Near-idle clients are dropped so the profiled run stands out from the
// compositor's background noise.
std::vector<std::string> format_proc_lines(const Window &cur, double wall_s,
                                           const Cols &cols, int cap) {
    struct Row {
        int pid;
        std::string name;
        std::vector<double> pct;
        double mem;
        bool mem_valid;
        int nhandles;
        double hot;
    };
    std::vector<Row> rows;
    for (const auto &kv : cur.by_pid) {
        Row r;
        r.pid = kv.first;
        r.name = kv.second.proc;
        r.mem = double(kv.second.bytes) / (1024.0 * 1024.0);
        r.mem_valid = kv.second.bytes_valid;
        r.nhandles = kv.second.nhandles;
        r.hot = 0;
        for (const auto *e : cols.eng) {
            double v =
                delta_pct(counter_of(kv.second.delta_ns, e->key), wall_s, 1e9);
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
            int k = std::snprintf(buf, sizeof(buf), fmt, args...);
            if (k > 0)
                line.append(buf, static_cast<size_t>(k));
        };
        emit("%*s", kTimeW, "--");
        for (double v : rows[i].pct)
            emit("%8.1f%%", v);
        if (cols.freq)
            emit("%*s", kFreqW, "--");
        if (cols.idle)
            emit("%*s", kRc6W, "--");
        if (rows[i].mem_valid)
            emit("%9.1f  %s(%d)", rows[i].mem, rows[i].name.c_str(),
                 rows[i].pid);
        else
            emit("%*s  %s(%d)", kResW, "--", rows[i].name.c_str(), rows[i].pid);
        if (rows[i].nhandles > 1)
            emit("[%d]", rows[i].nhandles);
        line.push_back('\n');
        out.push_back(std::move(line));
    }
    return out;
}

// Terminal height in rows, or 24 when stdout is not a tty (in which case the
// callers do not redraw anyway).
int terminal_height() {
    struct winsize ws {};
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_row > 0)
        return ws.ws_row;
    return 24;
}

} // namespace

static std::atomic<bool> g_stop{false};
static void on_sigint(int) { g_stop = true; }

static void usage(const char *argv0) {
    std::fprintf(
        stderr,
        "usage: %s [-p PID] [-s MS] [-n COUNT] [-r ROWS] [-P] [--no-proc] "
        "[-l]\n"
        "  -p PID    monitor one process (default: whole-GPU)\n"
        "  -s MS     sample period, ms (default 500)\n"
        "  -n COUNT  stop after COUNT samples (default: until ^C)\n"
        "  -r ROWS   rows kept on screen in place mode (default: fits\n"
        "            the terminal; ignored with -l)\n"
        "  -P        force per-process rows on (default: on for whole-GPU,\n"
        "            off with -p since a single process is already named)\n"
        "  --no-proc force per-process rows off (compact one-line summary)\n"
        "  -g NAME   pick the GPU stack when a host has more than one\n"
        "            (Linux: intel | nvidia | auto). Auto prefers intel\n"
        "            when an i915/xe card is bound. VKOP_GPU_TOP_BACKEND\n"
        "            sets the same thing.\n"
        "  -l        line mode: append every row (a scrolling log to\n"
        "            save/pipe) instead of redrawing a fixed window\n",
        argv0);
}

int main(int argc, char **argv) {
    int pid = 0;
    int period_ms = 500;
    long long count = 0;
    int max_rows = 0;   // 0 = auto
    int proc_mode = -1; // -1 auto (on for whole-GPU, off with -p), 0 off, 1 on
    bool line_mode = false;
    std::string backend_want; // -g: only meaningful where a platform has two

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
        } else if (a == "-g") {
            backend_want = need_next("-g");
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
    gputop::Backend &be = gputop::backend();
    if (!backend_want.empty() && !be.select(backend_want)) {
        std::fprintf(stderr,
                     "vkop_gpu_top: no '%s' backend on this host "
                     "(try -g auto)\n",
                     backend_want.c_str());
        return 2;
    }
    const gputop::Columns cols = be.columns();
    // Whole-GPU mode is a *shared* device: list the clients by default, the way
    // a `top` would. With -p there is only one process, already named in the
    // summary row, so the breakdown would just echo it.
    const bool per_proc = proc_mode >= 0 ? (proc_mode == 1) : (pid == 0);
    const bool redraw = isatty(STDOUT_FILENO) && !line_mode;

    Clients prev = to_map(be.collect(pid));
    auto t_prev = std::chrono::steady_clock::now();
    auto t_start = t_prev;

    if (prev.empty()) {
        std::fprintf(stderr,
                     "vkop_gpu_top: no GPU client%s found yet; "
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
    EngineSet engines;
    while (!g_stop && (count == 0 || iter < count)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(period_ms));

        Clients cur = to_map(be.collect(pid));
        auto t_cur = std::chrono::steady_clock::now();
        double wall_s = std::chrono::duration<double>(t_cur - t_prev).count();
        double elapsed_s =
            std::chrono::duration<double>(t_cur - t_start).count();

        if (pid > 0 && cur.empty()) {
            std::fprintf(stderr,
                         "vkop_gpu_top: pid %d holds no GPU client "
                         "(exited?)\n",
                         pid);
            break;
        }

        Window w = diff_windows(prev, cur);
        engines.seen(w);
        Cols out_cols;
        out_cols.eng = engines.cols;
        out_cols.freq = cols.freq;
        out_cols.idle = cols.idle;
        out_cols.mem = false;
        out_cols.mem_mb = 0.0;
        if (cols.device_mem && w.bytes_valid) {
            out_cols.mem = true;
            out_cols.mem_mb = double(w.bytes) / (1024.0 * 1024.0);
        } else if (w.by_pid.size() == 1) {
            const PidAgg &only = w.by_pid.begin()->second;
            if (only.bytes_valid) {
                out_cols.mem = true;
                out_cols.mem_mb = double(only.bytes) / (1024.0 * 1024.0);
            }
        }

        // One sample is one summary line, optionally followed by its
        // per-process breakdown (same engine columns, process named in the
        // leading field).
        std::vector<std::string> lines;
        lines.push_back(format_row(
            elapsed_s, w, wall_s, out_cols, cols.freq ? be.freq_mhz() : 0,
            cols.idle ? be.idle_pct(wall_s) : -1.0, w.npending));
        if (per_proc) {
            for (auto &l : format_proc_lines(w, wall_s, out_cols, 10))
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
            print_header(pid, w, period_ms, out_cols);
            for (const auto &r : history)
                std::fputs(r.c_str(), stdout);
            lines_drawn = 2 + static_cast<int>(history.size());
            std::fflush(stdout);
        } else {
            if (!header_done) {
                print_header(pid, w, period_ms, out_cols);
                header_done = true;
            }
            for (const auto &l : lines)
                std::fputs(l.c_str(), stdout);
        }

        prev = std::move(cur);
        t_prev = t_cur;
        ++iter;
    }
    return 0;
}
