// Copyright 2025 @junka
// macOS backend for vkop_gpu_top: Apple silicon (AGX) through the interfaces
// the IORegistry exposes to ordinary users.
//
//   IOService:/.../AGXAccelerator<X>/AGXDeviceUserClient
//     "AppUsage"            = [{API, accumulatedGPUTime, lastSubmittedTime},
//                              ...] cumulative GPU nanoseconds per client --
//                              the direct equivalent of fdinfo's
//                              drm-engine-* totals.
//     "IOUserClientCreator" = "pid <N>, <process name>"
// Both are readable unprivileged and unmediated: verified on an M5 Max
// (AGXAcceleratorG17X, macOS 26.3), where this saw all 56 GPU clients on the
// box including root-owned ones such as WindowServer.
//
// What this platform does *not* offer, and why the columns are dropped rather
// than faked:
//   * Clock and idle residency (the rps_act_freq_mhz / rc6_residency_ms
//     equivalents). Both live behind powermetrics, which refuses to run
//     without root, or behind IOReport -- whose symbols no linkable framework
//     exports and which do not even resolve through dlsym(RTLD_DEFAULT).
//   * Per-engine busy time. AppUsage reports one total per client, so there is
//     no render/copy/video split to put in separate columns.
//   * GPU-resident bytes per client (the drm-resident-<region> equivalent).
//     Per-process memory here is the *footprint*, which counts the system RAM
//     a process holds for the GPU along with everything else it holds, so it
//     is labelled FootMB and is not summable across processes. It is
//     unavailable for another user's process (proc_pid_rusage -> EPERM), which
//     reads as "--".
//
// Traps, all hit and verified while bringing this backend up:
//   * GPU clients are *not* registered services, so
//     IOServiceGetMatchingServices("AGXDeviceUserClient") matches nothing. The
//     nodes have to be reached by walking the accelerator's subtree with
//     IORegistryEntryCreateIterator.
//   * IORegistryEntryCreateCFProperty(entry, "AppUsage") returns NULL for
//     every client but your own, which silently degrades the whole tool to
//     "only my process has rows". IORegistryEntryCreateCFProperties does
//     return other processes' clients.
//   * There is no "IORegistryEntryID" *property*; the stable id comes from
//     IORegistryEntryGetRegistryEntryID(). Keying clients any other way (list
//     position, or an already-per-pid total) lets a client node appearing or
//     disappearing masquerade as busy time -- a prototype that did so reported
//     WindowServer at 320x the wall clock.
//   * A process holds several client nodes and each node several AppUsage
//     entries: a vkop test run measured 4 entries on 1 node. Diff per client,
//     then sum per pid, never the other way round.
//   * The device-level "Device Utilization %" in PerformanceStatistics is NOT
//     the same quantity. Measured: a saturated compute-only client reads 0%
//     there while an idle machine reads ~35%, because that sampler is tied to
//     the display path. It is deliberately not used. The per-client
//     nanoseconds agree with it to ~1 point while windowed work is running,
//     which is the only cross-check available without root.
//
// Cost: ~0.5-1.6 ms to walk all 56 clients, so the 500 ms default period and
// even -s 10 are comfortably affordable.
#include "gpu_top_backend.hpp"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <libproc.h>
#include <sys/resource.h>

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>

namespace gputop {
namespace {

std::string cfstr(CFTypeRef v) {
    if (!v || CFGetTypeID(v) != CFStringGetTypeID())
        return {};
    char buf[512];
    if (!CFStringGetCString((CFStringRef)v, buf, sizeof(buf),
                            kCFStringEncodingUTF8))
        return {};
    return buf;
}

uint64_t cfu64(CFTypeRef v) {
    if (!v || CFGetTypeID(v) != CFNumberGetTypeID())
        return 0;
    int64_t n = 0;
    if (!CFNumberGetValue((CFNumberRef)v, kCFNumberSInt64Type, &n))
        return 0;
    return n < 0 ? 0 : uint64_t(n);
}

class Agx : public Backend {
  public:
    const char *name() const override { return "AGX AppUsage"; }

    // One column: this platform folds every engine into one per-client total.
    const std::vector<EngineSpec> &engines() const override {
        static const std::vector<EngineSpec> kOne = {{"gpu", "GPU"}};
        return kOne;
    }

    Columns columns() const override {
        Columns c;
        c.device_label = "node";
        c.mem_label = "FootMB";
        return c; // freq/idle/device_mem all unavailable
    }

    std::vector<ClientInfo> collect(int pid) override {
        std::vector<ClientInfo> out;
        if (!open())
            return out;

        io_iterator_t it;
        if (IORegistryEntryCreateIterator(acc_, kIOServicePlane,
                                          kIORegistryIterateRecursively,
                                          &it) != KERN_SUCCESS)
            return out;

        std::map<int, std::pair<bool, uint64_t>> footprint; // one call per pid
        io_registry_entry_t e;
        while ((e = IOIteratorNext(it)) != 0) {
            CFMutableDictionaryRef props = nullptr;
            if (IORegistryEntryCreateCFProperties(
                    e, &props, kCFAllocatorDefault, 0) == KERN_SUCCESS &&
                props) {
                ClientInfo ci;
                if (parse(e, props, ci) && (pid <= 0 || ci.pid == pid)) {
                    auto f = footprint.find(ci.pid);
                    if (f == footprint.end())
                        f = footprint.emplace(ci.pid, process_footprint(ci.pid))
                                .first;
                    ci.mem_valid = f->second.first;
                    ci.mem_bytes = f->second.second;
                    out.push_back(std::move(ci));
                }
                CFRelease(props);
            }
            IOObjectRelease(e);
        }
        IOObjectRelease(it);
        return out;
    }

  private:
    // Fill `ci` from one client node. Returns false for registry entries that
    // are not GPU clients (the accelerator subtree also holds the work queues
    // and the firmware buddy devices).
    bool parse(io_registry_entry_t e, CFDictionaryRef props, ClientInfo &ci) {
        CFTypeRef usage = CFDictionaryGetValue(props, CFSTR("AppUsage"));
        CFTypeRef creator =
            CFDictionaryGetValue(props, CFSTR("IOUserClientCreator"));
        if (!usage || !creator || CFGetTypeID(usage) != CFArrayGetTypeID() ||
            CFArrayGetCount((CFArrayRef)usage) == 0)
            return false;

        std::string who = cfstr(creator);
        // "pid 169, WindowServer". The name is the driver's own and already
        // truncated to the kernel's 16-byte comm limit, so there is no /proc
        // equivalent to fetch here.
        if (who.rfind("pid ", 0) != 0)
            return false;
        ci.pid = std::atoi(who.c_str() + 4);
        if (ci.pid <= 0)
            return false;
        size_t comma = who.find(", ");
        ci.proc = comma == std::string::npos ? who : who.substr(comma + 2);

        // Its AppUsage entries are the client's queue groups; their
        // nanoseconds fold into the single "gpu" counter this platform has.
        uint64_t total = 0;
        for (CFIndex i = 0; i < CFArrayGetCount((CFArrayRef)usage); ++i) {
            CFTypeRef el = CFArrayGetValueAtIndex((CFArrayRef)usage, i);
            if (CFGetTypeID(el) != CFDictionaryGetTypeID())
                continue;
            total += cfu64(CFDictionaryGetValue((CFDictionaryRef)el,
                                                CFSTR("accumulatedGPUTime")));
            ++ci.nhandles;
        }
        if (!ci.nhandles)
            return false;
        ci.engine_ns["gpu"] = total;

        uint64_t eid = 0;
        if (IORegistryEntryGetRegistryEntryID(e, &eid) != KERN_SUCCESS)
            return false;
        ci.identity = std::to_string(eid);
        ci.device = acc_name_;
        ci.mem_is_per_process = true;
        return true;
    }

    // The process's own memory, which is the closest thing macOS offers for a
    // per-process number -- see the file header for why it is not the
    // equivalent of drm-resident. first=false when the task belongs to another
    // user (proc_pid_rusage -> EPERM), which the core prints as "--".
    static std::pair<bool, uint64_t> process_footprint(int pid) {
        rusage_info_v4 ri{};
        if (::proc_pid_rusage(pid, RUSAGE_INFO_V4, (rusage_info_t *)&ri) != 0)
            return {false, 0};
        return {true, ri.ri_phys_footprint};
    }

    // The accelerator node, by class-name prefix, because the tail of the name
    // encodes the chip (AGXAcceleratorG17X on this M5, ...G15/...G16 on older
    // generations). Found once and kept for the lifetime of the run.
    bool open() {
        if (acc_)
            return true;
        if (tried_)
            return false;
        tried_ = true;
        io_registry_entry_t root = IORegistryGetRootEntry(kIOMainPortDefault);
        if (!root)
            return false;
        io_iterator_t it;
        if (IORegistryEntryCreateIterator(root, kIOServicePlane,
                                          kIORegistryIterateRecursively,
                                          &it) != KERN_SUCCESS) {
            IOObjectRelease(root);
            return false;
        }
        char nm[128];
        io_registry_entry_t e;
        while ((e = IOIteratorNext(it)) != 0) {
            bool hit = IORegistryEntryGetName(e, nm) == KERN_SUCCESS &&
                       std::strncmp(nm, "AGXAccelerator", 14) == 0;
            if (hit) {
                acc_ = e; // keep the reference
                acc_name_ = nm;
                break;
            }
            IOObjectRelease(e);
        }
        IOObjectRelease(it);
        IOObjectRelease(root);
        if (!acc_)
            std::fprintf(stderr,
                         "vkop_gpu_top: no AGXAccelerator* in the IORegistry "
                         "(not an Apple silicon GPU?)\n");
        return acc_ != 0;
    }

    io_registry_entry_t acc_ = 0;
    bool tried_ = false;
    std::string acc_name_;
};

} // namespace

Backend &backend() {
    static Agx b;
    return b;
}

} // namespace gputop
