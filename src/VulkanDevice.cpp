#include "VulkanLib.hpp"
#include "VulkanDevice.hpp"
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace vkop {

VulkanDevice::VulkanDevice(VkPhysicalDevice physicalDevice)
    : physicalDevice(physicalDevice) {
    if (physicalDevice == VK_NULL_HANDLE) {
        throw std::runtime_error("Invalid Vulkan physical device handle.");
    }
    if (logicalDevice != VK_NULL_HANDLE) {
        throw std::runtime_error("Logical device already created.");
    }
    if (computeQueue != VK_NULL_HANDLE) {
        throw std::runtime_error("Compute queue already created.");
    }
    create();
}

VulkanDevice::~VulkanDevice() {
    destroy();
}

void VulkanDevice::getProperties() {
    
    uint32_t pPropertyCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &pPropertyCount, nullptr);
    ext_properties.resize(pPropertyCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &pPropertyCount, ext_properties.data());
    for (auto ext : this->ext_properties) {
        std::cout << "device extension " << ext.extensionName << std::endl;
    }

    VkPhysicalDeviceProperties2 properties2 = {};
    properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;

#ifdef VK_EXT_host_image_copy
    VkPhysicalDeviceHostImageCopyProperties hostimagecopyproperty = {};
    hostimagecopyproperty.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_PROPERTIES;
    hostimagecopyproperty.pNext = nullptr;

    properties2.pNext = &hostimagecopyproperty;

    vkGetPhysicalDeviceProperties2(physicalDevice, &properties2);
    this->copySrcLayout.resize(hostimagecopyproperty.copySrcLayoutCount);
    this->copyDstLayout.resize(hostimagecopyproperty.copyDstLayoutCount);
    hostimagecopyproperty.pCopySrcLayouts = this->copySrcLayout.data();
    hostimagecopyproperty.pCopyDstLayouts = this->copyDstLayout.data();
#endif

    VkPhysicalDeviceSubgroupProperties subgroup_properties = {};
    subgroup_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
#ifdef VK_EXT_host_image_copy
    subgroup_properties.pNext = &hostimagecopyproperty;
#else
    subgroup_properties.pNext = nullptr;
#endif

    properties2.pNext = &subgroup_properties;

    vkGetPhysicalDeviceProperties2(physicalDevice, &properties2);
    this->deviceProperties = properties2.properties;
    this->timestampPeriod = deviceProperties.limits.timestampPeriod;
    std::cout << "GPU " << deviceProperties.deviceName << std::endl;
}

void VulkanDevice::create() {
    getProperties();
    if (!createLogicalDevice()) {
        std::cerr << "Failed to create logical device!" << std::endl;
        return;
    }
}

void VulkanDevice::destroy() {
    if (logicalDevice != VK_NULL_HANDLE) {
        vkDestroyDevice(logicalDevice, nullptr);
        logicalDevice = VK_NULL_HANDLE;
    }
}

bool VulkanDevice::createLogicalDevice() {
    computeQueueFamilyIndex = findComputeQueueFamily(physicalDevice);
    if (computeQueueFamilyIndex == -1) {
        std::cerr << "Failed to find a suitable compute queue family!" << std::endl;
        return false;
    }


    VkPhysicalDeviceFeatures deviceFeatures = {};
    vkGetPhysicalDeviceFeatures(physicalDevice, &deviceFeatures);

    VkPhysicalDeviceShaderFloat16Int8Features devicefloat16Int8Features = {};
    devicefloat16Int8Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR;
#if VK_KHR_shader_integer_dot_product
    VkPhysicalDeviceShaderIntegerDotProductFeatures integerDotProductFeatures = {};
    integerDotProductFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR;
    devicefloat16Int8Features.pNext = &integerDotProductFeatures;
#endif

    VkPhysicalDeviceFeatures2 features2 = {};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &devicefloat16Int8Features;

    vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);

    VkPhysicalDeviceFeatures features = {};
    features.robustBufferAccess = VK_TRUE;
    if (deviceFeatures.shaderInt64)
        features.shaderInt64 = VK_TRUE;
    if (deviceFeatures.shaderFloat64)
        features.shaderFloat64 = VK_TRUE;
    if (deviceFeatures.shaderInt16)
        features.shaderInt16 = VK_TRUE;
    features.shaderStorageImageWriteWithoutFormat = VK_TRUE;

    VkPhysicalDeviceFloat16Int8FeaturesKHR float16Int8Features = {};
    float16Int8Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR;

    VkPhysicalDevice8BitStorageFeatures storage8bitFeatures = {};
    storage8bitFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
    storage8bitFeatures.uniformAndStorageBuffer8BitAccess = VK_TRUE;
    storage8bitFeatures.storageBuffer8BitAccess = VK_TRUE;

#ifdef VK_KHR_16bit_storage
    VkPhysicalDevice16BitStorageFeatures storage16bitFeatures = {};
    storage16bitFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    storage16bitFeatures.uniformAndStorageBuffer16BitAccess = VK_TRUE;
    storage16bitFeatures.storageBuffer16BitAccess = VK_TRUE;
    storage16bitFeatures.storageInputOutput16 = VK_TRUE;
#elif defined VK_VERSION_1_1
    VkPhysicalDeviceVulkan11Features storage16bitFeatures = {};
    storage16bitFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    storage16bitFeatures.storageBuffer16BitAccess = VK_TRUE;
    storage16bitFeatures.storageInputOutput16 = VK_TRUE;
    storage16bitFeatures.uniformAndStorageBuffer16BitAccess = VK_TRUE;
#endif

#ifdef VK_KHR_shader_integer_dot_product 
    VkPhysicalDeviceShaderIntegerDotProductFeatures shaderIntegerDotProductFeatures = {};
    shaderIntegerDotProductFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR;
    shaderIntegerDotProductFeatures.shaderIntegerDotProduct = VK_TRUE;
#elif defined VK_VERSION_1_3
    VkPhysicalDeviceVulkan13Features features13 = {};
    features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    features13.shaderIntegerDotProduct = VK_TRUE;
#endif

#ifdef VK_EXT_host_image_copy
    VkPhysicalDeviceHostImageCopyFeatures hostImageCopyFeatures = {};
    hostImageCopyFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES_EXT;
    if (checkDeviceExtensionFeature(VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME)) {
        hostImageCopyFeatures.hostImageCopy = VK_TRUE;
        enabledExtensions.push_back(VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME);
        enabledFeatures.push_back(reinterpret_cast<uintptr_t>(&hostImageCopyFeatures));
    }
#endif
    if (devicefloat16Int8Features.shaderInt8) {
        float16Int8Features.shaderInt8 = VK_TRUE;
        if (checkDeviceExtensionFeature(VK_KHR_8BIT_STORAGE_EXTENSION_NAME)) {
            enabledExtensions.push_back(VK_KHR_8BIT_STORAGE_EXTENSION_NAME);
            enabledFeatures.push_back(reinterpret_cast<uintptr_t>(&storage8bitFeatures));
        }
    }
    if (devicefloat16Int8Features.shaderFloat16) {
        float16Int8Features.shaderFloat16 = VK_TRUE;
        if (checkDeviceExtensionFeature(VK_KHR_16BIT_STORAGE_EXTENSION_NAME)) {
            enabledExtensions.push_back(VK_KHR_16BIT_STORAGE_EXTENSION_NAME);
            if (deviceProperties.vendorID != 4318) {
                // tested on Nvidia A2000, it supports 16bit storage feature but did not need to enable it
                // enable it will cause validation error VK_ERROR_FEATURE_NOT_PRESENT
                enabledFeatures.push_back(reinterpret_cast<uintptr_t>(&storage16bitFeatures));
            }
        }
#ifdef VK_AMD_gpu_shader_half_float
        if (deviceProperties.vendorID == 4098) {
            // for AMD card, do we really need this ? over VK_KHR_shader_float16_int8
            if (checkDeviceExtensionFeature(VK_AMD_GPU_SHADER_HALF_FLOAT_EXTENSION_NAME)) {
                enabledExtensions.push_back(VK_AMD_GPU_SHADER_HALF_FLOAT_EXTENSION_NAME);
            }
        }
#endif
    }
    if (devicefloat16Int8Features.shaderFloat16 || devicefloat16Int8Features.shaderInt8) {
        if (checkDeviceExtensionFeature(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME)) {
            enabledFeatures.push_back(reinterpret_cast<uintptr_t>(&float16Int8Features));
            enabledExtensions.push_back(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
        }
    }
#ifdef VK_KHR_shader_integer_dot_product
    if (integerDotProductFeatures.shaderIntegerDotProduct) {
        if (checkDeviceExtensionFeature(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME)) {
            enabledExtensions.push_back(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);
            enabledFeatures.push_back(reinterpret_cast<uintptr_t>(&shaderIntegerDotProductFeatures));
        }
    }
#endif
#ifdef VK_KHR_bind_memory2
    if (checkDeviceExtensionFeature(VK_KHR_BIND_MEMORY_2_EXTENSION_NAME)) {
        enabledExtensions.push_back(VK_KHR_BIND_MEMORY_2_EXTENSION_NAME);
    }
#endif
#ifdef VK_KHR_shader_non_semantic_info
// #ifndef VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME
// #define VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME "VK_KHR_shader_non_semantic_info"
// #endif
    if (checkDeviceExtensionFeature(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME)) {
        enabledExtensions.push_back(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);
    }
#endif
#ifdef VK_KHR_get_physical_device_properties2
    if (checkDeviceExtensionFeature(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME)) {
        enabledExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    }
#endif
#ifdef VK_KHR_get_memory_requirements2
    if (checkDeviceExtensionFeature(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME)) {
        enabledExtensions.push_back(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
    }
#endif

#ifdef VK_KHR_format_feature_flags2
    if (checkDeviceExtensionFeature(VK_KHR_FORMAT_FEATURE_FLAGS_2_EXTENSION_NAME)) {
        enabledExtensions.push_back(VK_KHR_FORMAT_FEATURE_FLAGS_2_EXTENSION_NAME);
    }
#endif
#ifdef VK_KHR_copy_commands2
    if (checkDeviceExtensionFeature(VK_KHR_COPY_COMMANDS_2_EXTENSION_NAME)) {
        enabledExtensions.push_back(VK_KHR_COPY_COMMANDS_2_EXTENSION_NAME);
    }
#endif
#ifdef VK_EXT_tooling_info
    if (checkDeviceExtensionFeature(VK_EXT_TOOLING_INFO_EXTENSION_NAME)) {
        enabledExtensions.push_back(VK_EXT_TOOLING_INFO_EXTENSION_NAME);
    }
#endif
#ifdef VK_EXT_subgroup_size_control
    if (checkDeviceExtensionFeature(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME)) {
        enabledExtensions.push_back(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
    }
#endif
    struct GeneralFeature {
        VkStructureType sType;
        void*     pNext;
    };
    void* pFirst = nullptr;
    if (enabledFeatures.size() > 0) {        
        // std::cout << "enabledFeatures.size() = " << enabledFeatures.size() << std::endl;
        pFirst = reinterpret_cast<void *>(enabledFeatures[0]);
        struct GeneralFeature* ptr = reinterpret_cast<struct GeneralFeature*>(pFirst);
        for (size_t i = 1; i < enabledFeatures.size(); i++) {
            struct GeneralFeature* feat = reinterpret_cast<struct GeneralFeature*>(enabledFeatures[i]);
            ptr->pNext = feat;
            ptr = feat;
        }
    }


    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = computeQueueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.queueCreateInfoCount = 1;
    createInfo.pQueueCreateInfos = &queueCreateInfo;
    createInfo.enabledLayerCount = 0;
    createInfo.ppEnabledLayerNames = nullptr;
    createInfo.pEnabledFeatures = &features;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(enabledExtensions.size());
    createInfo.ppEnabledExtensionNames = enabledExtensions.data();
    createInfo.pNext = pFirst;
    std::cout << "enabled extensions count " << enabledExtensions.size() << std::endl;
    for (auto e: enabledExtensions) {
        std::cout << "enabled extension: " << e << std::endl;
    }

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &logicalDevice) != VK_SUCCESS) {
        std::cerr << "Failed to create logical device!" << std::endl;
        return false;
    }

    vkGetDeviceQueue(logicalDevice, computeQueueFamilyIndex, 0, &computeQueue);
    return true;
}

bool VulkanDevice::isDeviceSuitable(VkPhysicalDevice device) {
    int queueFamilyIndex = findComputeQueueFamily(device);
    return queueFamilyIndex != -1;
}

int VulkanDevice::findComputeQueueFamily(VkPhysicalDevice device) {
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    for (uint32_t i = 0; i < queueFamilies.size(); i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            return i;
        }
    }

    return -1;
}


bool VulkanDevice::checkDeviceExtensionFeature(const char *name) const 
{
    for (auto ext : this->ext_properties) {
        if (std::string(ext.extensionName).compare(name) == 0) {
            return true;
        }
    }
    return false;
}


} // namespace vkop