#ifndef __VULKAN_DEBUG_HPP__
#define __VULKAN_DEBUG_HPP__

#include "VulkanLib.hpp"
#include "vulkan/vulkan.hpp"

namespace vkop {

class VulkanDebug {
public:
    VulkanDebug(VkDevice device, uint64_t obj, VkDebugReportObjectTypeEXT type);
    ~VulkanDebug();

    void enable() { m_enable = true; }
    void disable() { m_enable = false; }

    void setObjectName(const char *name);
    void setObjecTag(const char *name);
    void begin(VkCommandBuffer commandBuffer, const char *name, float *c);
    void insert(VkCommandBuffer, const char *name, float *c);
    void end(VkCommandBuffer commandBuffer);
private:
    VkDevice m_device;
    uint64_t m_obj;
    VkDebugReportObjectTypeEXT m_objtype;
    bool m_enable;

};

}


#endif