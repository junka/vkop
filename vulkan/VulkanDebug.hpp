// Copyright 2025 @junka
#ifndef SRC_VULKANDEBUG_HPP_
#define SRC_VULKANDEBUG_HPP_

#include <vulkan/vulkan.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>

namespace vkop {

// VK_EXT_debug_utils annotations for capture and navigation (RenderDoc
// groups command-buffer regions, shows insert labels on the following
// dispatch, and displays object names in its resource browser). Everything
// is a no-op unless VKOP_DEBUG_LABELS (implied by VKOP_RDOC) is set AND the
// extension function pointers resolve, so the release path pays one static
// bool check per call.
//
// The device is registered once via set_device() (VulkanDevice's ctor), so
// the label helpers never need the VkDevice threaded through every dispatch.
class VulkanDebug {
  public:
    static void set_device(VkDevice device) { s_device_.store(device); }

    static bool enabled() {
        static const bool on = std::getenv("VKOP_RDOC") != nullptr ||
                               std::getenv("VKOP_DEBUG_LABELS") != nullptr;
        return on;
    }

    static void set_object_name(VkDevice device, uint64_t handle,
                                VkObjectType type, const char *name) {
        if (!enabled() || !name || device == VK_NULL_HANDLE)
            return;
        auto fn = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
            vkGetDeviceProcAddr(device, "vkSetDebugUtilsObjectNameEXT"));
        if (!fn)
            return;
        VkDebugUtilsObjectNameInfoEXT info = {};
        info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
        info.objectType = type;
        info.objectHandle = handle;
        info.pObjectName = name;
        fn(device, &info);
    }

    // Names the event that follows: inserted right before a dispatch, it is
    // what makes an Event Browser entry read like a graph node instead of
    // "vkCmdDispatch #512".
    static void insert_label(VkCommandBuffer cmd, const char *name) {
        if (!enabled() || !name || !*name || cmd == VK_NULL_HANDLE)
            return;
        auto fns = cmd_fns();
        if (!fns.insert)
            return;
        VkDebugUtilsLabelEXT label = label_info(name);
        fns.insert(cmd, &label);
    }

    // Region form: begin/end pairs nest and group every command in between,
    // so one label covers multi-dispatch ops (split-K fold passes, etc.).
    static void begin_region(VkCommandBuffer cmd, const char *name) {
        if (!enabled() || !name || !*name || cmd == VK_NULL_HANDLE)
            return;
        auto fns = cmd_fns();
        if (!fns.begin)
            return;
        VkDebugUtilsLabelEXT label = label_info(name);
        fns.begin(cmd, &label);
    }

    static void end_region(VkCommandBuffer cmd) {
        if (!enabled() || cmd == VK_NULL_HANDLE)
            return;
        auto fns = cmd_fns();
        if (fns.end)
            fns.end(cmd);
    }

  private:
    struct CmdFns {
        PFN_vkCmdBeginDebugUtilsLabelEXT begin = nullptr;
        PFN_vkCmdEndDebugUtilsLabelEXT end = nullptr;
        PFN_vkCmdInsertDebugUtilsLabelEXT insert = nullptr;
    };

    static std::atomic<VkDevice> s_device_;

    // Resolved lazily on the first labeled command; recording is
    // single-threaded in this engine, so a plain static is fine.
    static CmdFns cmd_fns() {
        static CmdFns cached = [] {
            CmdFns fns;
            VkDevice dev = s_device_.load();
            if (dev == VK_NULL_HANDLE)
                return fns;
            fns.begin = reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(
                vkGetDeviceProcAddr(dev, "vkCmdBeginDebugUtilsLabelEXT"));
            fns.end = reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(
                vkGetDeviceProcAddr(dev, "vkCmdEndDebugUtilsLabelEXT"));
            fns.insert = reinterpret_cast<PFN_vkCmdInsertDebugUtilsLabelEXT>(
                vkGetDeviceProcAddr(dev, "vkCmdInsertDebugUtilsLabelEXT"));
            return fns;
        }();
        return cached;
    }

    static VkDebugUtilsLabelEXT label_info(const char *name) {
        VkDebugUtilsLabelEXT label = {};
        label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
        label.pLabelName = name;
        // Stable per-name color: same node, same hue across captures.
        uint32_t h = 2166136261u;
        for (const char *p = name; *p; ++p) {
            h ^= static_cast<uint8_t>(*p);
            h *= 16777619u;
        }
        label.color[0] = ((h >> 0) & 0xFF) / 255.0f;
        label.color[1] = ((h >> 8) & 0xFF) / 255.0f;
        label.color[2] = ((h >> 16) & 0xFF) / 255.0f;
        label.color[3] = 1.0f;
        return label;
    }
};

} // namespace vkop

#endif // SRC_VULKANDEBUG_HPP_
