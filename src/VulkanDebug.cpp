
#include "VulkanInstance.hpp"
#include "VulkanLib.hpp"
#include "VulkanDebug.hpp"
#include <vulkan/vulkan_core.h>


namespace vkop {


VulkanDebug::VulkanDebug(VkDevice device, uint64_t obj, VkDebugReportObjectTypeEXT objtype):
     m_device(device), m_obj(obj), m_objtype(objtype)
{

}


void VulkanDebug::setObjectName(const char *name)
{
    VkDebugMarkerObjectNameInfoEXT nameinfo = {};
    nameinfo.sType = VK_STRUCTURE_TYPE_DEBUG_MARKER_OBJECT_NAME_INFO_EXT;
    nameinfo.objectType = m_objtype;
    nameinfo.object = m_obj;
    nameinfo.pObjectName = name;
    auto vkDebugMarkerSetObjectNameEXT =
        reinterpret_cast<PFN_vkDebugMarkerSetObjectNameEXT>(
            vkGetInstanceProcAddr(VulkanInstance::getVulkanInstance().getInstance(), "vkDebugMarkerSetObjectNameEXT"));
    if (vkDebugMarkerSetObjectNameEXT)
        vkDebugMarkerSetObjectNameEXT(m_device, &nameinfo);
}

void VulkanDebug::setObjecTag(const char *name)
{
    VkDebugMarkerObjectTagInfoEXT taginfo = {};
    taginfo.sType = VK_STRUCTURE_TYPE_DEBUG_MARKER_OBJECT_TAG_INFO_EXT;
    taginfo.object = m_obj;
    taginfo.objectType = m_objtype;
    taginfo.tagName = 1;
    taginfo.tagSize = sizeof(int);
    taginfo.pTag = name;
    auto vkDebugMarkerSetObjectTagEXT =
        reinterpret_cast<PFN_vkDebugMarkerSetObjectTagEXT>(
            vkGetInstanceProcAddr(VulkanInstance::getVulkanInstance().getInstance(), "vkDebugMarkerSetObjectTagEXT"));
    if (vkDebugMarkerSetObjectTagEXT)
        vkDebugMarkerSetObjectTagEXT(m_device, &taginfo);
}


void VulkanDebug::begin(VkCommandBuffer commandBuffer, const char *name, float *c)
{
    VkDebugMarkerMarkerInfoEXT info = {};
    info.sType = VK_STRUCTURE_TYPE_DEBUG_MARKER_MARKER_INFO_EXT;
    info.pMarkerName = name;
    memcpy(info.color, c, 4 * sizeof(float));
    auto vkCmdDebugMarkerBeginEXT =
        reinterpret_cast<PFN_vkCmdDebugMarkerBeginEXT>(
            vkGetInstanceProcAddr(VulkanInstance::getVulkanInstance().getInstance(), "vkCmdDebugMarkerBeginEXT"));
    if (vkCmdDebugMarkerBeginEXT)
        vkCmdDebugMarkerBeginEXT(commandBuffer, &info);
}

void VulkanDebug::insert(VkCommandBuffer commandBuffer, const char *name, float *c)
{
    VkDebugMarkerMarkerInfoEXT info = {};
    info.sType = VK_STRUCTURE_TYPE_DEBUG_MARKER_MARKER_INFO_EXT;
    info.pMarkerName = name;
    memcpy(info.color, c, 4 * sizeof(float));

    auto vkCmdDebugMarkerInsertEXT =
        reinterpret_cast<PFN_vkCmdDebugMarkerInsertEXT>(
            vkGetInstanceProcAddr(VulkanInstance::getVulkanInstance().getInstance(), "vkCmdDebugMarkerInsertEXT"));
    if (vkCmdDebugMarkerInsertEXT)
        vkCmdDebugMarkerInsertEXT(commandBuffer, &info);
}

void VulkanDebug::end(VkCommandBuffer commandBuffer)
{
    auto vkCmdDebugMarkerEndEXT =
        reinterpret_cast<PFN_vkCmdDebugMarkerEndEXT>(
            vkGetInstanceProcAddr(VulkanInstance::getVulkanInstance().getInstance(), "vkCmdDebugMarkerEndEXT"));
    if (vkCmdDebugMarkerEndEXT)
        vkCmdDebugMarkerEndEXT(commandBuffer);
}

}