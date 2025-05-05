#include "VulkanLib.hpp"
#include "VulkanInstance.hpp"
#include <stdexcept>
#include <string>
#include <vector>
#include "logger.hpp"
namespace vkop {

VulkanInstance::VulkanInstance() : m_instance(VK_NULL_HANDLE) {
    createInstance("vkop", VK_MAKE_VERSION(1, 0, 0));

#if VK_EXT_debug_utils
    CreateDebugUtilsMessenger();
#endif
#if VK_EXT_debug_report
    if (callback.utils == nullptr) {
        CreateDebugReportCallback();
    }
#endif
    enumPhysicalDevices();
}

VulkanInstance::~VulkanInstance() {
    destroyInstance();
}


uint32_t VulkanInstance::getVulkanVersion(void)
    {
        uint32_t version = VK_API_VERSION_1_0;
        vkEnumerateInstanceVersion(&version);
        LOG_INFO("vulkan version %d.%d.%d", VK_VERSION_MAJOR(version), VK_VERSION_MINOR(version),VK_VERSION_PATCH(version));
        return version;
    }
void VulkanInstance::createInstance(const std::string& applicationName, uint32_t appVersion) {
    if (m_instance != VK_NULL_HANDLE) {
        throw std::runtime_error("Vulkan instance is already initialized.");
    }

    auto version = getVulkanVersion();
    // Application info
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pEngineName = "vkop";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = version;
    appInfo.pApplicationName = applicationName.c_str();
    appInfo.applicationVersion = appVersion;

    // Instance creation info
    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    // Extensions
    enumInstanceExtensions();
    getRequiredExtensions();
#if VK_KHR_portability_enumeration
    createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();
    // Validation layers
    if (checkValidationLayerSupport()) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
        createInfo.enabledLayerCount = 0;
    }

    // Create Vulkan instance
    if (vkCreateInstance(&createInfo, nullptr, &m_instance) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan instance.");
    }
}

void VulkanInstance::destroyInstance() {
#if VK_EXT_debug_utils
    if (callback.utils && !extensions.empty() && 
        std::string(extensions[0]).compare(VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0) {
        auto vkDestroyDebugUtilsMessengerEXT =
        reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(m_instance, "vkDestroyDebugUtilsMessengerEXT"));
        if (vkDestroyDebugUtilsMessengerEXT)
            vkDestroyDebugUtilsMessengerEXT(m_instance, callback.utils, nullptr);
    }
#endif
#if VK_EXT_debug_report
    if (callback.report && !extensions.empty() &&
        std::string(extensions[0]).compare(VK_EXT_DEBUG_REPORT_EXTENSION_NAME) == 0) {
        auto vkDestroyDebugReportCallbackEXT =
        reinterpret_cast<PFN_vkDestroyDebugReportCallbackEXT>(
            vkGetInstanceProcAddr(m_instance, "vkDestroyDebugReportCallbackEXT"));
    
        if (vkDestroyDebugReportCallbackEXT)
            vkDestroyDebugReportCallbackEXT(m_instance, callback.report, nullptr);
    }
#endif

    if (m_instance != VK_NULL_HANDLE) {
        vkDestroyInstance(m_instance, nullptr);
        m_instance = VK_NULL_HANDLE;
    }
}

VkInstance VulkanInstance::getInstance() const {
    return m_instance;
}


bool VulkanInstance::isExtensionSupported(const char* extensionName) const {
    
    for (const auto& extension : availableExtensions) {
        if (strcmp(extension.extensionName, extensionName) == 0) {
            return true;
        }
    }
    return false;
}


void VulkanInstance::enumInstanceExtensions(void)
{
    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

    availableExtensions.resize(extensionCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, availableExtensions.data());
}
void VulkanInstance::getRequiredExtensions() const {
    // Add instance extensions
#if VK_EXT_debug_utils
    if (isExtensionSupported(VK_EXT_DEBUG_UTILS_EXTENSION_NAME))
       extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif
#if VK_EXT_debug_report
    if (extensions.empty() && isExtensionSupported(VK_EXT_DEBUG_REPORT_EXTENSION_NAME)) {
        extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
    }
#endif
#if VK_KHR_get_physical_device_properties2
    if (isExtensionSupported(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME)) {
        extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    }
#endif
#if VK_KHR_portability_enumeration
    if (isExtensionSupported(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    }
#endif
#ifdef VK_EXT_debug_marker
    if (isExtensionSupported(VK_EXT_DEBUG_MARKER_EXTENSION_NAME)) {
        extensions.push_back(VK_EXT_DEBUG_MARKER_EXTENSION_NAME);
    }
#endif
}

bool VulkanInstance::checkValidationLayerSupport() const{
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    // try VK_LAYER_KHRONOS_validation first, it depreated VK_LAYER_LUNARG_standard_validation
    for (auto layer : availableLayers) {
        const char *valid[] = {
            "VK_LAYER_KHRONOS_validation",
            "VK_LAYER_LUNARG_standard_validation"
        };
        for (auto layerName : valid) {
            if (strcmp(layer.layerName, layerName) == 0) {
                LOG_INFO("validation layer found %s", layer.layerName);
                validationLayers.push_back(layerName);
                return true;
            }
        }
    }

    return false;
}
#ifdef VK_EXT_debug_utils
VkDebugUtilsMessengerEXT VulkanInstance::CreateDebugUtilsMessenger(void)
{
    VkResult error;

    VkDebugUtilsMessengerCreateInfoEXT callbackCreateInfo = {};
    callbackCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    callbackCreateInfo.pNext = NULL;
    callbackCreateInfo.flags = 0;
    callbackCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    callbackCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                    VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                    VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    callbackCreateInfo.pfnUserCallback = &debugUtilsCallback;
    callbackCreateInfo.pUserData = nullptr;

    VkDebugUtilsMessengerEXT callback = VK_NULL_HANDLE;
    auto vkCreateDebugUtilsMessengerEXT =
        reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(m_instance, "vkCreateDebugUtilsMessengerEXT"));
    if (vkCreateDebugUtilsMessengerEXT) {
        error = vkCreateDebugUtilsMessengerEXT(m_instance, &callbackCreateInfo, nullptr,
                                        &callback);
        if (error != VK_SUCCESS) {
            LOG_ERROR("Failed to create debug callback");
            return nullptr;
        }
    }
    this->callback.utils = callback;
    return callback;
}
#endif
#if VK_EXT_debug_report
VkDebugReportCallbackEXT VulkanInstance::CreateDebugReportCallback(void)
{
    VkResult error;
    VkDebugReportCallbackCreateInfoEXT callbackCreateInfo = {};
    callbackCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CREATE_INFO_EXT;
    callbackCreateInfo.pNext = nullptr;
    callbackCreateInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT |
                               VK_DEBUG_REPORT_WARNING_BIT_EXT |
                               VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
    callbackCreateInfo.pfnCallback = &debugReportCallback;
    callbackCreateInfo.pUserData = nullptr;
    VkDebugReportCallbackEXT callback = VK_NULL_HANDLE;
    auto vkCreateDebugReportCallbackEXT =
        reinterpret_cast<PFN_vkCreateDebugReportCallbackEXT>(
            vkGetInstanceProcAddr(m_instance, "vkCreateDebugReportCallbackEXT"));
    if (vkCreateDebugReportCallbackEXT) {
        error = vkCreateDebugReportCallbackEXT(m_instance, &callbackCreateInfo, nullptr,
                                        &callback);
        if (error != VK_SUCCESS) {
            return nullptr;
        }
    }
    this->callback.report = callback;

    return callback;
}
#endif

void VulkanInstance::enumPhysicalDevices(void)
{
    uint32_t count;
    VkResult error = vkEnumeratePhysicalDevices(m_instance, &count, nullptr);
    if (error != VK_SUCCESS) {
        throw std::runtime_error("Failed to enumerate physical devices.");
    }
    m_physicalDevices.resize(count);
    error = vkEnumeratePhysicalDevices(m_instance, &count, m_physicalDevices.data());
    if (error != VK_SUCCESS) {
        throw std::runtime_error("Failed to enumerate physical devices.");
    }
    LOG_INFO("Found %d physical devices.", count);
}



#ifdef VK_EXT_tooling_info

void VulkanInstance::getToolinfo(VkPhysicalDevice physicalDevice) {
    uint32_t toolCount = 0;
    auto vkGetPhysicalDeviceToolPropertiesEXT =
        reinterpret_cast<PFN_vkGetPhysicalDeviceToolPropertiesEXT>(
            vkGetInstanceProcAddr(m_instance, "vkGetPhysicalDeviceToolPropertiesEXT"));
    if (vkGetPhysicalDeviceToolPropertiesEXT == nullptr) {
        LOG_WARN("vkGetPhysicalDeviceToolPropertiesEXT not found");
        return;
    }
    vkGetPhysicalDeviceToolPropertiesEXT(physicalDevice, &toolCount, nullptr);
    std::vector<VkPhysicalDeviceToolPropertiesEXT> toolinfos(toolCount);
    for (uint32_t i = 0; i < toolCount; i++) {
        toolinfos[i].sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TOOL_PROPERTIES_EXT;
        toolinfos[i].pNext = nullptr;
    }

    vkGetPhysicalDeviceToolPropertiesEXT(physicalDevice, &toolCount, toolinfos.data());
    for (uint32_t i = 0; i < toolCount; i++) {
        LOG_INFO("tool info: %s", toolinfos[i].name);
        LOG_INFO("tool info: %s", toolinfos[i].description);
        LOG_INFO("tool info: %s", toolinfos[i].version);
        LOG_INFO("tool info: %s", toolinfos[i].layer);
    }
}

#endif

} // namespace vkop