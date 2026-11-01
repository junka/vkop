// Copyright 2025 @junka
#ifndef SRC_RENDERDOC_HPP_
#define SRC_RENDERDOC_HPP_

#include "include/logger.hpp"
#include "include/renderdoc_app.h"

#include <atomic>
#include <cstdlib>
#include <cstdint>
#include <vulkan/vulkan.h>

#ifdef __linux__
#include <dlfcn.h>
#endif

namespace vkop {

// In-application capture control for RenderDoc. The entry point is fetched
// from the capture layer via vkGetDeviceProcAddr/vkGetInstanceProcAddr with
// the name "RENDERDOC-API-GetAPI", which works identically on Windows and
// Linux whenever the process runs under RenderDoc injection. Without
// injection every call is a silent no-op.
//
// Usage (from Runtime::Run, after the device exists):
//   RenderdocRunCapture capture(instance, device);
// which opens a capture for exactly one Run() round selected by env:
//   VKOP_RDOC=1        enable capture machinery (also implies debug labels)
//   VKOP_RDOC_RUN=<n>  capture the n-th Run() call, 1-based (default 1)
class Renderdoc {
  public:
    static bool capture_enabled() {
        static const bool on = std::getenv("VKOP_RDOC") != nullptr;
        return on;
    }

    static uint64_t capture_run_index() {
        static const uint64_t n = [] {
            const char *v = std::getenv("VKOP_RDOC_RUN");
            if (v && *v) {
                char *end = nullptr;
                uint64_t parsed = std::strtoull(v, &end, 10);
                if (end && parsed > 0)
                    return parsed;
            }
            return static_cast<uint64_t>(1);
        }();
        return n;
    }

    // Resolved once per process; nullptr when RenderDoc is not injected.
    static RENDERDOC_API_1_6_0 *api(VkInstance instance, VkDevice device) {
        static RENDERDOC_API_1_6_0 *cached = load(instance, device);
        return cached;
    }

    static bool begin(VkInstance instance, VkDevice device) {
        auto *rdoc = api(instance, device);
        if (!rdoc)
            return false;
        rdoc->StartFrameCapture(
            RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(instance), nullptr);
        LOG_INFO("RenderDoc capture started");
        return true;
    }

    static void end(VkInstance instance, VkDevice device) {
        auto *rdoc = api(instance, device);
        if (!rdoc)
            return;
        rdoc->EndFrameCapture(
            RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(instance), nullptr);
        LOG_INFO("RenderDoc capture ended");
    }

  private:
    static RENDERDOC_API_1_6_0 *load(VkInstance instance, VkDevice device) {
        pRENDERDOC_GetAPI get_api = nullptr;

        // The capture layer exports the in-app API through the instance and
        // device dispatch tables. Prefer the device path (device-level
        // lookup); fall back to the instance and finally to an explicit
        // dlopen on Linux for layer-less injection modes.
        if (device != VK_NULL_HANDLE) {
            get_api = reinterpret_cast<pRENDERDOC_GetAPI>(
                vkGetDeviceProcAddr(device, "RENDERDOC-API-GetAPI"));
        }
        if (!get_api && instance != VK_NULL_HANDLE) {
            get_api = reinterpret_cast<pRENDERDOC_GetAPI>(
                vkGetInstanceProcAddr(instance, "RENDERDOC-API-GetAPI"));
        }
#ifdef __linux__
        if (!get_api) {
            void *mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD);
            if (!mod) {
                mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_LOCAL);
            }
            if (!mod) {
                LOG_WARN("VKOP_RDOC set but librenderdoc.so not loadable");
                return nullptr;
            }
            get_api = reinterpret_cast<pRENDERDOC_GetAPI>(
                dlsym(mod, "RENDERDOC_GetAPI"));
        }
#endif
        if (!get_api) {
            LOG_WARN("VKOP_RDOC set but RenderDoc layer not detected; "
                     "launch this binary under RenderDoc to capture");
            return nullptr;
        }
        RENDERDOC_API_1_6_0 *rdoc = nullptr;
        const int ret = get_api(eRENDERDOC_API_Version_1_6_0,
                                reinterpret_cast<void **>(&rdoc));
        if (ret != 1 || !rdoc) {
            LOG_WARN("RenderDoc GetAPI rejected version 1.6.0");
            return nullptr;
        }
        LOG_INFO("RenderDoc in-app API acquired (v1.6.0)");
        return rdoc;
    }
};

// RAII guard: opens a capture on construction when running under RenderDoc
// and the process-wide Run() counter matches VKOP_RDOC_RUN; always closes it
// on destruction so the capture boundary is exactly one Run() call.
class RenderdocRunCapture {
  public:
    RenderdocRunCapture(VkInstance instance, VkDevice device)
        : instance_(instance), device_(device) {
        if (!Renderdoc::capture_enabled())
            return;
        auto &counter = run_counter();
        if (counter.fetch_add(1) + 1 != Renderdoc::capture_run_index())
            return;
        active_ = Renderdoc::begin(instance_, device_);
    }
    ~RenderdocRunCapture() {
        if (active_)
            Renderdoc::end(instance_, device_);
    }

    RenderdocRunCapture(const RenderdocRunCapture &) = delete;
    RenderdocRunCapture &operator=(const RenderdocRunCapture &) = delete;

  private:
    static std::atomic<uint64_t> &run_counter() {
        static std::atomic<uint64_t> counter{0};
        return counter;
    }

    VkInstance instance_;
    VkDevice device_;
    bool active_ = false;
};

} // namespace vkop

#endif // SRC_RENDERDOC_HPP_
