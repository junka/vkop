// Copyright 2025 @junka
#ifndef SRC_VULKANINSTANCE_HPP_
#define SRC_VULKANINSTANCE_HPP_

#include "vulkan/vulkan.hpp"

#include <string>
#include <vector>

namespace vkop {

class VulkanInstance {
  public:
    VulkanInstance();
    ~VulkanInstance();

    VulkanInstance(const VulkanInstance &) = delete;
    VulkanInstance &operator=(const VulkanInstance &) = delete;

    // Get the Vulkan instance handle
    VkInstance getInstance() const;

    static VulkanInstance &getVulkanInstance() {
        static VulkanInstance ins; // = new VulkanInstance();
        return ins;
    }

    void destroyInstance();
    std::vector<VkPhysicalDevice> getPhysicalDevices() {
        return m_physical_devices_;
    }

  private:
    // Initialize the Vulkan instance
    void createInstance(const std::string &app_name, uint32_t app_version);

    // Check if a specific extension is supported
    bool isExtensionSupported(const char *extensionName) const;
    // Check for required extensions
    void getRequiredExtensions() const;

    void enumPhysicalDevices();

    void enumInstanceExtensions();

    void getToolinfo(VkPhysicalDevice physicalDevice);
#ifdef USE_DEBUG_LAYERS
#if VK_EXT_debug_utils
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugUtilsCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageTypes,
        const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
        void *pUserData) {
        (void)messageTypes;
        (void)pUserData;
        if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
            fprintf(stderr, "%s [%d]: %s\n", pCallbackData->pMessageIdName,
                    pCallbackData->messageIdNumber, pCallbackData->pMessage);
        } else {
            fprintf(stdout, "%s [%d]: %s\n", pCallbackData->pMessageIdName,
                    pCallbackData->messageIdNumber, pCallbackData->pMessage);
        }
        return VK_FALSE;
    }
#endif
#if VK_EXT_debug_report
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallback(
        VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objectType,
        uint64_t object, size_t location, int32_t messageCode,
        const char *pLayerPrefix, const char *pMessage, void *pUserData) {
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
    VkDebugReportCallbackEXT CreateDebugReportCallback();
#endif
#if VK_EXT_debug_utils
    VkDebugUtilsMessengerEXT CreateDebugUtilsMessenger();
#endif
    union {
#if VK_EXT_debug_utils
        VkDebugUtilsMessengerEXT utils;
#endif
#if VK_EXT_debug_report
        VkDebugReportCallbackEXT report;
#endif
    } callback_;

#endif

    VkInstance m_instance_ = VK_NULL_HANDLE;
#ifdef USE_VALIDATION_LAYERS
    mutable std::vector<const char *> validation_layers_;
    // Check if validation layers are supported
    bool checkValidationLayerSupport() const;
#endif
    mutable std::vector<const char *> extensions_;

    std::vector<VkExtensionProperties> available_extensions_;

    std::vector<VkPhysicalDevice> m_physical_devices_;
};

} // namespace vkop

#endif // SRC_VULKANINSTANCE_HPP_
