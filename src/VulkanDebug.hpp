// Copyright 2025 @junka
#ifndef SRC_VULKANDEBUG_HPP_
#define SRC_VULKANDEBUG_HPP_

#include "VulkanInstance.hpp"
#include "VulkanLib.hpp"

namespace vkop {

class VulkanDebug {
  public:
    VulkanDebug(VkDevice device, uint64_t obj, VkDebugReportObjectTypeEXT type);
    ~VulkanDebug();

    void enable() { m_enable_ = true; }
    void disable() { m_enable_ = false; }

    void setObjectName(const char *name);
    void setObjecTag(const char *name);

    static void begin(VkCommandBuffer commandBuffer, const char *name,
                      float *c) {
        VkDebugMarkerMarkerInfoEXT info = {};
        info.sType = VK_STRUCTURE_TYPE_DEBUG_MARKER_MARKER_INFO_EXT;
        info.pMarkerName = name;
        memcpy(info.color, c, 4 * sizeof(float));
        auto vkCmdDebugMarkerBeginEXT =
            reinterpret_cast<PFN_vkCmdDebugMarkerBeginEXT>(
                vkGetInstanceProcAddr(
                    VulkanInstance::getVulkanInstance().getInstance(),
                    "vkCmdDebugMarkerBeginEXT"));
        if (vkCmdDebugMarkerBeginEXT)
            vkCmdDebugMarkerBeginEXT(commandBuffer, &info);
    }

    static void insert(VkCommandBuffer commandBuffer, const char *name,
                       float *c) {
        VkDebugMarkerMarkerInfoEXT info = {};
        info.sType = VK_STRUCTURE_TYPE_DEBUG_MARKER_MARKER_INFO_EXT;
        info.pMarkerName = name;
        memcpy(info.color, c, 4 * sizeof(float));

        auto vkCmdDebugMarkerInsertEXT =
            reinterpret_cast<PFN_vkCmdDebugMarkerInsertEXT>(
                vkGetInstanceProcAddr(
                    VulkanInstance::getVulkanInstance().getInstance(),
                    "vkCmdDebugMarkerInsertEXT"));
        if (vkCmdDebugMarkerInsertEXT)
            vkCmdDebugMarkerInsertEXT(commandBuffer, &info);
    }

    static void end(VkCommandBuffer commandBuffer) {
        auto vkCmdDebugMarkerEndEXT =
            reinterpret_cast<PFN_vkCmdDebugMarkerEndEXT>(vkGetInstanceProcAddr(
                VulkanInstance::getVulkanInstance().getInstance(),
                "vkCmdDebugMarkerEndEXT"));
        if (vkCmdDebugMarkerEndEXT)
            vkCmdDebugMarkerEndEXT(commandBuffer);
    }

  private:
    VkDevice m_device_;
    uint64_t m_obj_;
    VkDebugReportObjectTypeEXT m_objtype_;
    bool m_enable_;
};

} // namespace vkop

#endif // SRC_VULKANDEBUG_HPP_
