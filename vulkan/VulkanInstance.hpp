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
    const std::vector<VkPhysicalDevice> &getPhysicalDevices() const {
        return m_physical_devices_;
    }
    // The apiVersion the VkInstance was created with. This is the ceiling on
    // the core Vulkan version an application may use, independent of what the
    // physical device reports: a driver may advertise a newer apiVersion than
    // the instance asked for (NVIDIA 595 reports 1.4.329 through an instance
    // created at 1.3.275), and using a 1.4-core feature struct in that state
    // is invalid. Feature selection must compare min(instance, device).
    uint32_t apiVersion() const { return m_api_version_; }

  private:
    // Initialize the Vulkan instance
    void createInstance(const std::string &app_name, uint32_t app_version);

    // Check for required extensions
    void getRequiredExtensions(std::vector<const char *> &extensions);

    void enumPhysicalDevices();

    void getToolinfo(VkPhysicalDevice physicalDevice);
#ifdef USE_DEBUG_LAYERS
#if VK_EXT_debug_utils
    VkDebugUtilsMessengerEXT CreateDebugUtilsMessenger();
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugUtilsCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageTypes,
        const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
        void *pUserData) {
        (void)messageTypes;
        (void)pUserData;
        if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
            fprintf(stderr, "Validation Error: %s [%d]: %s\n",
                    pCallbackData->pMessageIdName,
                    pCallbackData->messageIdNumber, pCallbackData->pMessage);
        } else if (messageSeverity &
                   VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
            fprintf(stdout, "Validation Warning: %s [%d]: %s\n",
                    pCallbackData->pMessageIdName,
                    pCallbackData->messageIdNumber, pCallbackData->pMessage);
        } else {
            fprintf(stdout, "Validation Info: %s [%d]: %s\n",
                    pCallbackData->pMessageIdName,
                    pCallbackData->messageIdNumber, pCallbackData->pMessage);
        }
        return VK_FALSE;
    }
#endif
#if VK_EXT_debug_report
    VkDebugReportCallbackEXT CreateDebugReportCallback();
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
    uint32_t m_api_version_ = VK_API_VERSION_1_0;
    std::vector<VkPhysicalDevice> m_physical_devices_;

#ifdef USE_VALIDATION_LAYERS
    // Check if validation layers are supported
    bool
    checkValidationLayerSupport(std::vector<const char *> &validation_layers);
#endif
};

} // namespace vkop

#endif // SRC_VULKANINSTANCE_HPP_
