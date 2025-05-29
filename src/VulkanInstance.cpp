// Copyright 2025 @junka
#include "VulkanInstance.hpp"
#include "VulkanLib.hpp"
#include "logger.hpp"
#include <stdexcept>
#include <string>
#include <vector>
namespace vkop {

VulkanInstance::VulkanInstance() {
    createInstance("vkop", VK_MAKE_VERSION(1, 0, 0));

#if VK_EXT_debug_utils
    CreateDebugUtilsMessenger();
#endif
#if VK_EXT_debug_report
    if (callback_.utils == nullptr) {
        CreateDebugReportCallback();
    }
#endif
    enumPhysicalDevices();
}

VulkanInstance::~VulkanInstance() { destroyInstance(); }

void VulkanInstance::createInstance(const std::string &app_name,
                                    uint32_t app_version) {
    if (m_instance_ != VK_NULL_HANDLE) {
        throw std::runtime_error("Vulkan instance is already initialized.");
    }

    uint32_t version = VK_API_VERSION_1_0;
    vkEnumerateInstanceVersion(&version);
    // Application info
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pEngineName = "vkop";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = version;
    app_info.pApplicationName = app_name.c_str();
    app_info.applicationVersion = app_version;

    // Instance creation info
    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    // Extensions
    enumInstanceExtensions();
    getRequiredExtensions();
#if VK_KHR_portability_enumeration
    create_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
    create_info.enabledExtensionCount =
        static_cast<uint32_t>(extensions_.size());
    create_info.ppEnabledExtensionNames = extensions_.data();
    // Validation layers
    if (checkValidationLayerSupport()) {
        create_info.enabledLayerCount =
            static_cast<uint32_t>(validation_layers_.size());
        create_info.ppEnabledLayerNames = validation_layers_.data();
    } else {
        create_info.enabledLayerCount = 0;
    }

    // Create Vulkan instance
    if (vkCreateInstance(&create_info, nullptr, &m_instance_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan instance.");
    }
}

void VulkanInstance::destroyInstance() {
#if VK_EXT_debug_utils
    if ((callback_.utils != nullptr) && !extensions_.empty() &&
        (std::string(extensions_[0]) == VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
        auto vkDestroyDebugUtilsMessengerEXT =
            reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(m_instance_,
                                      "vkDestroyDebugUtilsMessengerEXT"));
        if (vkDestroyDebugUtilsMessengerEXT)
            vkDestroyDebugUtilsMessengerEXT(m_instance_, callback_.utils,
                                            nullptr);
    }
#endif
#if VK_EXT_debug_report
    if ((callback_.report != nullptr) && !extensions_.empty() &&
        (std::string(extensions_[0]) == VK_EXT_DEBUG_REPORT_EXTENSION_NAME)) {
        auto vkDestroyDebugReportCallbackEXT =
            reinterpret_cast<PFN_vkDestroyDebugReportCallbackEXT>(
                vkGetInstanceProcAddr(m_instance_,
                                      "vkDestroyDebugReportCallbackEXT"));

        if (vkDestroyDebugReportCallbackEXT)
            vkDestroyDebugReportCallbackEXT(m_instance_, callback_.report,
                                            nullptr);
    }
#endif

    if (m_instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(m_instance_, nullptr);
        m_instance_ = VK_NULL_HANDLE;
    }
}

VkInstance VulkanInstance::getInstance() const { return m_instance_; }

bool VulkanInstance::isExtensionSupported(const char *extensionName) const {
    for (const auto &extension : available_extensions_) {
        if (strcmp(extension.extensionName, extensionName) == 0) {
            return true;
        }
    }
    return false;
}

void VulkanInstance::enumInstanceExtensions() {
    uint32_t extension_count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr);

    available_extensions_.resize(extension_count);
    vkEnumerateInstanceExtensionProperties(nullptr, &extension_count,
                                           available_extensions_.data());
}
void VulkanInstance::getRequiredExtensions() const {
    // Add instance extensions
#if VK_EXT_debug_utils
    if (isExtensionSupported(VK_EXT_DEBUG_UTILS_EXTENSION_NAME))
        extensions_.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif
#if VK_EXT_debug_report
    if (extensions_.empty() &&
        isExtensionSupported(VK_EXT_DEBUG_REPORT_EXTENSION_NAME)) {
        extensions_.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
    }
#endif
#if VK_KHR_get_physical_device_properties2
    if (isExtensionSupported(
            VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME)) {
        extensions_.push_back(
            VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    }
#endif
#if VK_KHR_portability_enumeration
    if (isExtensionSupported(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        extensions_.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    }
#endif
#ifdef VK_EXT_debug_marker
    if (isExtensionSupported(VK_EXT_DEBUG_MARKER_EXTENSION_NAME)) {
        extensions_.push_back(VK_EXT_DEBUG_MARKER_EXTENSION_NAME);
    }
#endif
}

bool VulkanInstance::checkValidationLayerSupport() const {
    uint32_t layer_count;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);

    std::vector<VkLayerProperties> available_layers(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

    // try VK_LAYER_KHRONOS_validation first, it depreated
    // VK_LAYER_LUNARG_standard_validation
    for (auto layer : available_layers) {
        const char *valid[] = {"VK_LAYER_KHRONOS_validation",
                               "VK_LAYER_LUNARG_standard_validation"};
        for (const auto *layer_name : valid) {
            if (strcmp(layer.layerName, layer_name) == 0) {
                LOG_INFO("validation layer found %s", layer.layerName);
                validation_layers_.push_back(layer_name);
                return true;
            }
        }
    }

    return false;
}
#ifdef VK_EXT_debug_utils
VkDebugUtilsMessengerEXT VulkanInstance::CreateDebugUtilsMessenger() {
    VkResult error;

    VkDebugUtilsMessengerCreateInfoEXT callback_create_info = {};
    callback_create_info.sType =
        VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    callback_create_info.pNext = nullptr;
    callback_create_info.flags = 0;
    callback_create_info.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    callback_create_info.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    callback_create_info.pfnUserCallback = &debugUtilsCallback;
    callback_create_info.pUserData = nullptr;

    VkDebugUtilsMessengerEXT callback = VK_NULL_HANDLE;
    auto vkCreateDebugUtilsMessengerEXT =
        reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(m_instance_,
                                  "vkCreateDebugUtilsMessengerEXT"));
    if (vkCreateDebugUtilsMessengerEXT) {
        error = vkCreateDebugUtilsMessengerEXT(
            m_instance_, &callback_create_info, nullptr, &callback);
        if (error != VK_SUCCESS) {
            LOG_ERROR("Failed to create debug callback");
            return nullptr;
        }
    }
    this->callback_.utils = callback;
    return callback;
}
#endif
#if VK_EXT_debug_report
VkDebugReportCallbackEXT VulkanInstance::CreateDebugReportCallback() {
    VkResult error;
    VkDebugReportCallbackCreateInfoEXT callback_create_info = {};
    callback_create_info.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CREATE_INFO_EXT;
    callback_create_info.pNext = nullptr;
    callback_create_info.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT |
                                 VK_DEBUG_REPORT_WARNING_BIT_EXT |
                                 VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
    callback_create_info.pfnCallback = &debugReportCallback;
    callback_create_info.pUserData = nullptr;
    VkDebugReportCallbackEXT callback = VK_NULL_HANDLE;
    auto vkCreateDebugReportCallbackEXT =
        reinterpret_cast<PFN_vkCreateDebugReportCallbackEXT>(
            vkGetInstanceProcAddr(m_instance_,
                                  "vkCreateDebugReportCallbackEXT"));
    if (vkCreateDebugReportCallbackEXT) {
        error = vkCreateDebugReportCallbackEXT(
            m_instance_, &callback_create_info, nullptr, &callback);
        if (error != VK_SUCCESS) {
            return nullptr;
        }
    }
    this->callback_.report = callback;

    return callback;
}
#endif

void VulkanInstance::enumPhysicalDevices() {
    uint32_t count;
    VkResult error = vkEnumeratePhysicalDevices(m_instance_, &count, nullptr);
    if (error != VK_SUCCESS) {
        throw std::runtime_error("Failed to enumerate physical devices.");
    }
    m_physical_devices_.resize(count);
    error = vkEnumeratePhysicalDevices(m_instance_, &count,
                                       m_physical_devices_.data());
    if (error != VK_SUCCESS) {
        throw std::runtime_error("Failed to enumerate physical devices.");
    }
    LOG_INFO("Found %d physical devices.", count);
}

#ifdef VK_EXT_tooling_info

void VulkanInstance::getToolinfo(VkPhysicalDevice physicalDevice) {
    uint32_t tool_count = 0;
    auto vkGetPhysicalDeviceToolPropertiesEXT =
        reinterpret_cast<PFN_vkGetPhysicalDeviceToolPropertiesEXT>(
            vkGetInstanceProcAddr(m_instance_,
                                  "vkGetPhysicalDeviceToolPropertiesEXT"));
    if (vkGetPhysicalDeviceToolPropertiesEXT == nullptr) {
        LOG_WARN("vkGetPhysicalDeviceToolPropertiesEXT not found");
        return;
    }
    vkGetPhysicalDeviceToolPropertiesEXT(physicalDevice, &tool_count, nullptr);
    std::vector<VkPhysicalDeviceToolPropertiesEXT> toolinfos(tool_count);
    for (uint32_t i = 0; i < tool_count; i++) {
        toolinfos[i].sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TOOL_PROPERTIES_EXT;
        toolinfos[i].pNext = nullptr;
    }

    vkGetPhysicalDeviceToolPropertiesEXT(physicalDevice, &tool_count,
                                         toolinfos.data());
    for (uint32_t i = 0; i < tool_count; i++) {
        LOG_INFO("tool info: %s", toolinfos[i].name);
        LOG_INFO("tool info: %s", toolinfos[i].description);
        LOG_INFO("tool info: %s", toolinfos[i].version);
        LOG_INFO("tool info: %s", toolinfos[i].layer);
    }
}

#endif

} // namespace vkop
