// Copyright 2025 @junka
#include "VulkanDebug.hpp"
#include "VulkanLib.hpp"

namespace vkop {

VulkanDebug::VulkanDebug(VkDevice device, uint64_t obj,
                         VkDebugReportObjectTypeEXT objtype)
    : m_device_(device), m_obj_(obj), m_objtype_(objtype) {}

void VulkanDebug::setObjectName(const char *name) {
    VkDebugMarkerObjectNameInfoEXT nameinfo = {};
    nameinfo.sType = VK_STRUCTURE_TYPE_DEBUG_MARKER_OBJECT_NAME_INFO_EXT;
    nameinfo.objectType = m_objtype_;
    nameinfo.object = m_obj_;
    nameinfo.pObjectName = name;

    auto vkDebugMarkerSetObjectNameEXT =
        reinterpret_cast<PFN_vkDebugMarkerSetObjectNameEXT>(
            vkGetInstanceProcAddr(
                VulkanInstance::getVulkanInstance().getInstance(),
                "vkDebugMarkerSetObjectNameEXT"));
    if (vkDebugMarkerSetObjectNameEXT)
        vkDebugMarkerSetObjectNameEXT(m_device_, &nameinfo);
}

void VulkanDebug::setObjecTag(const char *name) {
    VkDebugMarkerObjectTagInfoEXT taginfo = {};
    taginfo.sType = VK_STRUCTURE_TYPE_DEBUG_MARKER_OBJECT_TAG_INFO_EXT;
    taginfo.object = m_obj_;
    taginfo.objectType = m_objtype_;
    taginfo.tagName = 1;
    taginfo.tagSize = sizeof(int);
    taginfo.pTag = name;
    auto vkDebugMarkerSetObjectTagEXT =
        reinterpret_cast<PFN_vkDebugMarkerSetObjectTagEXT>(
            vkGetInstanceProcAddr(
                VulkanInstance::getVulkanInstance().getInstance(),
                "vkDebugMarkerSetObjectTagEXT"));
    if (vkDebugMarkerSetObjectTagEXT)
        vkDebugMarkerSetObjectTagEXT(m_device_, &taginfo);
}

} // namespace vkop
