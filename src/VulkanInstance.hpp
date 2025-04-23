#ifndef VULKAN_INSTANCE_HPP
#define VULKAN_INSTANCE_HPP
#include "vulkan/vulkan.hpp"
#include <vector>
#include <string>

namespace vkop {

class VulkanInstance {
public:
    VulkanInstance();
    ~VulkanInstance();

    // Initialize the Vulkan instance
    void createInstance(const std::string& applicationName, uint32_t applicationVersion);

    // Destroy the Vulkan instance
    void destroyInstance();

    // Get the Vulkan instance handle
    VkInstance getInstance() const;

    std::vector<VkPhysicalDevice> getPhysicalDevices(void) { return physicalDevices;}

private:
    uint32_t getVulkanVersion(void);
    // Check if a specific extension is supported
    bool isExtensionSupported(const char* extensionName) const;
    // Check if validation layers are supported
    bool checkValidationLayerSupport() const;
    // Check for required extensions
    void getRequiredExtensions() const;

    void enumPhysicalDevices(void);

    void enumInstanceExtensions(void);

    
    void getToolinfo(VkPhysicalDevice physicalDevice);


#if VK_EXT_debug_utils
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugUtilsCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageTypes,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData)
    {
        (void)messageTypes;
        (void)pUserData;
        if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
            fprintf(stderr, "%s [%d]: %s\n", pCallbackData->pMessageIdName,
                pCallbackData->messageIdNumber,
                pCallbackData->pMessage);
        } else {
            fprintf(stdout, "%s [%d]: %s\n", pCallbackData->pMessageIdName,
                pCallbackData->messageIdNumber,
                pCallbackData->pMessage);
        }
        return VK_FALSE;
    }
#endif
#if VK_EXT_debug_report
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallback(
        VkDebugReportFlagsEXT flags,
        VkDebugReportObjectTypeEXT objectType,
        uint64_t object,
        size_t location,
        int32_t messageCode,
        const char *pLayerPrefix,
        const char *pMessage,
        void *pUserData)
    {
        (void)flags;
        (void)objectType;
        (void)object;
        (void)location;
        (void)messageCode;
        (void)pLayerPrefix;
        (void)pUserData;
        fprintf(stderr, "%s\n", pMessage);
        return VK_FALSE;
    }
#endif
#if VK_EXT_debug_report
    VkDebugReportCallbackEXT CreateDebugReportCallback(void);
#endif
#if VK_EXT_debug_utils
    VkDebugUtilsMessengerEXT CreateDebugUtilsMessenger(void);
#endif
    VkInstance instance;
    mutable std::vector<const char*> validationLayers;
    mutable std::vector<const char*> extensions;
    union {
#if VK_EXT_debug_utils
        VkDebugUtilsMessengerEXT utils;
#endif
#if VK_EXT_debug_report
        VkDebugReportCallbackEXT report;
#endif
    } callback;

    std::vector<VkExtensionProperties> availableExtensions;

    std::vector<VkPhysicalDevice> physicalDevices;
};

} // namespace vkop

#endif // VULKAN_INSTANCE_HPP