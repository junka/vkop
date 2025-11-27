// Copyright 2025 @junka
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanLib.hpp"
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "include/logger.hpp"

namespace vkop {

VulkanDevice::VulkanDevice(VkPhysicalDevice physicalDevice) {
    physicalDevice_ = physicalDevice;
    if (physicalDevice == VK_NULL_HANDLE) {
        throw std::runtime_error("Invalid Vulkan physical device handle.");
    }
    if (logicalDevice_ != VK_NULL_HANDLE) {
        throw std::runtime_error("Logical device already created.");
    }
    if (!computeQueues_.empty()) {
        throw std::runtime_error("Compute queue already created.");
    }
    auto props = getProperties();
    if (!createLogicalDevice(props)) {
        LOG_ERROR("Failed to create logical device!");
        return;
    }
    checkDeviceUnifiedMemoryAccess();
#ifdef USE_VMA
    m_vma_ = std::make_unique<VMA>(physicalDevice_, logicalDevice_);
#endif
}

VulkanDevice::~VulkanDevice() {
    for (auto &queue : computeQueues_) {
        queue.reset();
    }
#ifdef USE_VMA
    m_vma_.reset();
#endif
    if (logicalDevice_ != VK_NULL_HANDLE) {
        vkDestroyDevice(logicalDevice_, nullptr);
        logicalDevice_ = VK_NULL_HANDLE;
    }
}

VkPhysicalDeviceProperties VulkanDevice::getProperties() {
    VkPhysicalDeviceProperties2 properties2 = {};
    properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;

#ifdef VK_EXT_host_image_copy
    VkPhysicalDeviceHostImageCopyProperties hostimagecopyproperty = {};
    hostimagecopyproperty.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_PROPERTIES;
    hostimagecopyproperty.pNext = nullptr;

    properties2.pNext = &hostimagecopyproperty;

    vkGetPhysicalDeviceProperties2(physicalDevice_, &properties2);
    this->copySrcLayout_.resize(hostimagecopyproperty.copySrcLayoutCount);
    this->copyDstLayout_.resize(hostimagecopyproperty.copyDstLayoutCount);
    hostimagecopyproperty.pCopySrcLayouts = this->copySrcLayout_.data();
    hostimagecopyproperty.pCopyDstLayouts = this->copyDstLayout_.data();
#endif

    VkPhysicalDeviceSubgroupProperties subgroup_properties = {};
    subgroup_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
#ifdef VK_EXT_host_image_copy
    subgroup_properties.pNext = &hostimagecopyproperty;
#else
    subgroup_properties.pNext = nullptr;
#endif

    properties2.pNext = &subgroup_properties;

    vkGetPhysicalDeviceProperties2(physicalDevice_, &properties2);
    this->timestampPeriod_ = properties2.properties.limits.timestampPeriod;
    this->deviceName_ = properties2.properties.deviceName;
    LOG_INFO("GPU %s", this->deviceName_.c_str());
    return properties2.properties;
}

uint32_t VulkanDevice::findComputeQueueFamily() {
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_,
                                             &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(
        physicalDevice_, &queue_family_count, queue_families.data());

    const VkQueueFlags first = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
    const VkQueueFlags second =
        VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;

    for (uint32_t i = 0; i < queue_families.size(); i++) {
        if ((queue_families[i].queueFlags & (first)) == (first)) {
            std::tuple<uint32_t, uint32_t, VkQueueFlags> t = {
                i, queue_families[i].queueCount, queue_families[i].queueFlags};
            computeQueueIdxs_.emplace_back(t);
        } else if ((queue_families[i].queueFlags & (second)) == (second)) {
            std::tuple<uint32_t, uint32_t, VkQueueFlags> t = {
                i, queue_families[i].queueCount, queue_families[i].queueFlags};
            computeQueueIdxs_.emplace_back(t);
        }
    }

    return computeQueueIdxs_.size();
}

bool VulkanDevice::createLogicalDevice(
    const VkPhysicalDeviceProperties &deviceProperties) {
    auto num_queue = findComputeQueueFamily();
    if (num_queue == 0) {
        LOG_ERROR("Failed to find a suitable compute queue family!");
        return false;
    }
    std::vector<uintptr_t> enabled_features;
    std::vector<const char *> enabled_extensions;
    std::vector<VkExtensionProperties> ext_properties;

    uint32_t p_property_count = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr,
                                         &p_property_count, nullptr);
    ext_properties.resize(p_property_count);
    vkEnumerateDeviceExtensionProperties(
        physicalDevice_, nullptr, &p_property_count, ext_properties.data());
    // for (auto ext : ext_properties) {
    //     LOG_INFO("device extension %s", ext.extensionName);
    // }

    VkPhysicalDeviceFeatures device_features = {};
    vkGetPhysicalDeviceFeatures(physicalDevice_, &device_features);

    VkPhysicalDeviceShaderFloat16Int8Features devicefloat16_int8_features = {};
    devicefloat16_int8_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR;
#if VK_KHR_shader_integer_dot_product
    VkPhysicalDeviceShaderIntegerDotProductFeatures
        integer_dot_product_features = {};
    integer_dot_product_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR;
    devicefloat16_int8_features.pNext = &integer_dot_product_features;
#endif
#if VK_KHR_buffer_device_address
    VkPhysicalDeviceBufferDeviceAddressFeatures buffer_device_address_features =
        {};
    buffer_device_address_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    buffer_device_address_features.pNext = &devicefloat16_int8_features;
#endif
#if VK_KHR_timeline_semaphore
    VkPhysicalDeviceTimelineSemaphoreFeatures timeline_features{};
    timeline_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
#if VK_KHR_buffer_device_address
    timeline_features.pNext = &buffer_device_address_features;
#else
    timeline_features.pNext = &devicefloat16_int8_features;
#endif
#endif
    VkPhysicalDeviceFeatures2 features2 = {};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
#if VK_KHR_timeline_semaphore
    features2.pNext = &timeline_features;
#elif VK_KHR_buffer_device_address
    features2.pNext = &buffer_device_address_features;
#else
    features2.pNext = &devicefloat16_int8_features;
#endif

    vkGetPhysicalDeviceFeatures2(physicalDevice_, &features2);

    if (timeline_features.timelineSemaphore == VK_TRUE) {
        m_support_timeline_semaphore_ = true;
        LOG_INFO("Timeline semaphores supported");
    }

    VkPhysicalDeviceFeatures features = {};
    features.robustBufferAccess = VK_TRUE;
    if (device_features.shaderInt64)
        features.shaderInt64 = VK_TRUE;
    if (device_features.shaderFloat64)
        features.shaderFloat64 = VK_TRUE;
    if (device_features.shaderInt16)
        features.shaderInt16 = VK_TRUE;
    features.shaderStorageImageWriteWithoutFormat = VK_TRUE;

    VkPhysicalDeviceFloat16Int8FeaturesKHR float16_int8_features = {};
    float16_int8_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR;

    VkPhysicalDevice8BitStorageFeatures storage8bit_features = {};
    storage8bit_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
    storage8bit_features.uniformAndStorageBuffer8BitAccess = VK_TRUE;
    storage8bit_features.storageBuffer8BitAccess = VK_TRUE;

#ifdef VK_KHR_16bit_storage
    VkPhysicalDevice16BitStorageFeatures storage16bit_features = {};
    storage16bit_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    storage16bit_features.uniformAndStorageBuffer16BitAccess = VK_TRUE;
    storage16bit_features.storageBuffer16BitAccess = VK_TRUE;
    storage16bit_features.storageInputOutput16 = VK_TRUE;
#elif defined VK_VERSION_1_1
    VkPhysicalDeviceVulkan11Features storage16bitFeatures = {};
    storage16bitFeatures.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    storage16bitFeatures.storageBuffer16BitAccess = VK_TRUE;
    storage16bitFeatures.storageInputOutput16 = VK_TRUE;
    storage16bitFeatures.uniformAndStorageBuffer16BitAccess = VK_TRUE;
#endif

#ifdef VK_KHR_shader_integer_dot_product
    VkPhysicalDeviceShaderIntegerDotProductFeatures
        shader_integer_dot_product_features = {};
    shader_integer_dot_product_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR;
    shader_integer_dot_product_features.shaderIntegerDotProduct = VK_TRUE;
#elif defined VK_VERSION_1_3
    VkPhysicalDeviceVulkan13Features features13 = {};
    features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    features13.shaderIntegerDotProduct = VK_TRUE;
#endif

#if VK_EXT_host_image_copy
    VkPhysicalDeviceHostImageCopyFeatures host_image_copy_features = {};
    host_image_copy_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES_EXT;
    if (checkDeviceExtensionFeature(VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME)) {
        host_image_copy_features.hostImageCopy = VK_TRUE;
        enabled_extensions.push_back(VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME);
        enabled_features.push_back(
            reinterpret_cast<uintptr_t>(&host_image_copy_features));
        m_support_host_image_copy_ = true;
    }
#endif
    if (devicefloat16_int8_features.shaderInt8) {
        float16_int8_features.shaderInt8 = VK_TRUE;
        if (checkDeviceExtensionFeature(ext_properties,
                                        VK_KHR_8BIT_STORAGE_EXTENSION_NAME)) {
            enabled_extensions.push_back(VK_KHR_8BIT_STORAGE_EXTENSION_NAME);
            enabled_features.push_back(
                reinterpret_cast<uintptr_t>(&storage8bit_features));
        }
    }
    if (devicefloat16_int8_features.shaderFloat16) {
        float16_int8_features.shaderFloat16 = VK_TRUE;
        if (checkDeviceExtensionFeature(ext_properties,
                                        VK_KHR_16BIT_STORAGE_EXTENSION_NAME)) {
            enabled_extensions.push_back(VK_KHR_16BIT_STORAGE_EXTENSION_NAME);
            if (deviceProperties.vendorID != 4318) {
                // tested on Nvidia A2000, it supports 16bit storage feature but
                // did not need to enable it. enable will cause validation
                // error VK_ERROR_FEATURE_NOT_PRESENT
                enabled_features.push_back(
                    reinterpret_cast<uintptr_t>(&storage16bit_features));
            }
        }
#if VK_AMD_gpu_shader_half_float
        if (deviceProperties.vendorID == 4098) {
            // for AMD card, do we really need this ? over
            // VK_KHR_shader_float16_int8
            if (checkDeviceExtensionFeature(
                    ext_properties,
                    VK_AMD_GPU_SHADER_HALF_FLOAT_EXTENSION_NAME)) {
                enabled_extensions.push_back(
                    VK_AMD_GPU_SHADER_HALF_FLOAT_EXTENSION_NAME);
            }
        }
#endif
    }
    if (devicefloat16_int8_features.shaderFloat16 ||
        devicefloat16_int8_features.shaderInt8) {
        if (checkDeviceExtensionFeature(
                ext_properties, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME)) {
            enabled_features.push_back(
                reinterpret_cast<uintptr_t>(&float16_int8_features));
            enabled_extensions.push_back(
                VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
        }
    }
#ifdef VK_KHR_shader_integer_dot_product
    if (integer_dot_product_features.shaderIntegerDotProduct) {
        if (checkDeviceExtensionFeature(
                ext_properties,
                VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME)) {
            enabled_extensions.push_back(
                VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);
            enabled_features.push_back(reinterpret_cast<uintptr_t>(
                &shader_integer_dot_product_features));
        }
    }
#endif
#if VK_KHR_bind_memory2
    if (checkDeviceExtensionFeature(ext_properties,
                                    VK_KHR_BIND_MEMORY_2_EXTENSION_NAME)) {
        enabled_extensions.push_back(VK_KHR_BIND_MEMORY_2_EXTENSION_NAME);
    }
#endif
#if VK_KHR_shader_non_semantic_info
#ifndef VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME
#define VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME                         \
    "VK_KHR_shader_non_semantic_info"
#endif
    if (checkDeviceExtensionFeature(
            ext_properties, VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME)) {
        enabled_extensions.push_back(
            VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);
    }
#endif
#if VK_KHR_get_physical_device_properties2
    if (checkDeviceExtensionFeature(
            ext_properties,
            VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME)) {
        enabled_extensions.push_back(
            VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    }
#endif
#if VK_KHR_get_memory_requirements2
    if (checkDeviceExtensionFeature(
            ext_properties, VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME)) {
        enabled_extensions.push_back(
            VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
    }
#endif

#if VK_KHR_format_feature_flags2
    if (checkDeviceExtensionFeature(
            ext_properties, VK_KHR_FORMAT_FEATURE_FLAGS_2_EXTENSION_NAME)) {
        enabled_extensions.push_back(
            VK_KHR_FORMAT_FEATURE_FLAGS_2_EXTENSION_NAME);
    }
#endif
#if VK_KHR_copy_commands2
    if (checkDeviceExtensionFeature(ext_properties,
                                    VK_KHR_COPY_COMMANDS_2_EXTENSION_NAME)) {
        enabled_extensions.push_back(VK_KHR_COPY_COMMANDS_2_EXTENSION_NAME);
    }
#endif
#if VK_EXT_tooling_info
    if (checkDeviceExtensionFeature(ext_properties,
                                    VK_EXT_TOOLING_INFO_EXTENSION_NAME)) {
        enabled_extensions.push_back(VK_EXT_TOOLING_INFO_EXTENSION_NAME);
    }
#endif
#if VK_EXT_subgroup_size_control
    if (checkDeviceExtensionFeature(
            ext_properties, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME)) {
        enabled_extensions.push_back(
            VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
    }
#endif
#ifdef VK_EXT_image_robustness
    VkPhysicalDeviceImageRobustnessFeatures imagerobustfeature;
    imagerobustfeature.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES;
    if (checkDeviceExtensionFeature(ext_properties,
                                    VK_EXT_IMAGE_ROBUSTNESS_EXTENSION_NAME)) {
        imagerobustfeature.robustImageAccess = VK_TRUE;
        enabled_features.push_back(
            reinterpret_cast<uintptr_t>(&imagerobustfeature));
        enabled_extensions.push_back(VK_EXT_IMAGE_ROBUSTNESS_EXTENSION_NAME);
    }
#endif
#if VK_KHR_external_memory_fd
    if (checkDeviceExtensionFeature(ext_properties,
                                    VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME)) {
        enabled_extensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
    }
#endif
    // follow up extensions are for vma allocator
#if VK_KHR_dedicated_allocation
    if (checkDeviceExtensionFeature(
            ext_properties, VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME)) {
        enabled_extensions.push_back(
            VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME);
    }
#endif
#if VK_KHR_maintenance4
    if (checkDeviceExtensionFeature(ext_properties,
                                    VK_KHR_MAINTENANCE_4_EXTENSION_NAME)) {
        enabled_extensions.push_back(VK_KHR_MAINTENANCE_4_EXTENSION_NAME);
    }
#endif
#if VK_KHR_maintenance5
    if (checkDeviceExtensionFeature(ext_properties,
                                    VK_KHR_MAINTENANCE_5_EXTENSION_NAME)) {
        enabled_extensions.push_back(VK_KHR_MAINTENANCE_5_EXTENSION_NAME);
    }
#endif
#if VK_EXT_memory_budget
    if (checkDeviceExtensionFeature(ext_properties,
                                    VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) {
        enabled_extensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    }
#endif
#if VK_KHR_buffer_device_address
    VkPhysicalDeviceBufferDeviceAddressFeaturesKHR
        physical_device_buffer_device_address_features = {};
    physical_device_buffer_device_address_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR;
    physical_device_buffer_device_address_features.bufferDeviceAddress =
        VK_TRUE;

    if (checkDeviceExtensionFeature(
            ext_properties, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) &&
        buffer_device_address_features.bufferDeviceAddress) {
        enabled_extensions.push_back(
            VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
        m_support_buffer_device_address_ = true;
        enabled_features.push_back(reinterpret_cast<uintptr_t>(
            &physical_device_buffer_device_address_features));
    }
#endif
#if VK_EXT_memory_priority
    if (checkDeviceExtensionFeature(ext_properties,
                                    VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME)) {
        enabled_extensions.push_back(VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME);
    }
#endif
#if VK_AMD_device_coherent_memory
    if (checkDeviceExtensionFeature(
            ext_properties, VK_AMD_DEVICE_COHERENT_MEMORY_EXTENSION_NAME)) {
        enabled_extensions.push_back(
            VK_AMD_DEVICE_COHERENT_MEMORY_EXTENSION_NAME);
    }
#endif
#ifdef VK_KHR_timeline_semaphore
    VkPhysicalDeviceTimelineSemaphoreFeatures timeline_sem_features{};
    timeline_sem_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
    timeline_sem_features.timelineSemaphore = VK_TRUE;
    enabled_features.push_back(
        reinterpret_cast<uintptr_t>(&timeline_sem_features));
#endif
#ifdef VK_KHR_performance_query
    if (checkDeviceExtensionFeature(ext_properties,
                                    VK_KHR_PERFORMANCE_QUERY_EXTENSION_NAME)) {
        enabled_extensions.push_back(VK_KHR_PERFORMANCE_QUERY_EXTENSION_NAME);
    }
#endif
#if defined VK_KHR_cooperative_matrix || defined VK_NV_cooperative_matrix
#if defined VK_KHR_cooperative_matrix
    if (checkDeviceExtensionFeature(ext_properties,
                                    VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) {
        enabled_extensions.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    } else {
#endif
        if (checkDeviceExtensionFeature(
                ext_properties, VK_NV_COOPERATIVE_MATRIX_EXTENSION_NAME)) {
            enabled_extensions.push_back(
                VK_NV_COOPERATIVE_MATRIX_EXTENSION_NAME);
            uint32_t nv_cooperativematrix_cnt = 0;
            auto vkGetPhysicalDeviceCooperativeMatrixPropertiesNV =
                reinterpret_cast<
                    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesNV>(
                    vkGetInstanceProcAddr(
                        VulkanInstance::getVulkanInstance().getInstance(),
                        "vkGetPhysicalDeviceCooperativeMatrixPropertiesNV"));
            if (vkGetPhysicalDeviceCooperativeMatrixPropertiesNV) {
                vkGetPhysicalDeviceCooperativeMatrixPropertiesNV(
                    physicalDevice_, &nv_cooperativematrix_cnt, nullptr);
                std::vector<VkCooperativeMatrixPropertiesNV> cmprops(
                    nv_cooperativematrix_cnt);
                for (auto &p : cmprops) {
                    p.sType =
                        VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_NV;
                }
                vkGetPhysicalDeviceCooperativeMatrixPropertiesNV(
                    physicalDevice_, &nv_cooperativematrix_cnt, cmprops.data());
                const char *type_str[] = {"F16", "F32", "F64", "S8",
                                          "S16", "S32", "S64", "U8",
                                          "U16", "U32", "U64"};
                for (auto &p : cmprops) {
                    std::cout << "MSize " << p.MSize << " NSize " << p.NSize
                              << " KSize " << p.KSize << " A type "
                              << type_str[p.AType] << " B type "
                              << type_str[p.BType] << " C type "
                              << type_str[p.CType] << " D type "
                              << type_str[p.DType] << std::endl;
                }
            }
        }
#if defined VK_KHR_cooperative_matrix
    }
#endif
#endif
    struct GeneralFeature {
        VkStructureType sType;
        void *pNext;
    };
    void *p_first = nullptr;
    if (!enabled_features.empty()) {
        p_first = reinterpret_cast<void *>(enabled_features[0]);
        auto *ptr = reinterpret_cast<struct GeneralFeature *>(p_first);
        for (size_t i = 1; i < enabled_features.size(); i++) {
            auto *feat =
                reinterpret_cast<struct GeneralFeature *>(enabled_features[i]);
            ptr->pNext = feat;
            ptr = feat;
        }
    }
    std::vector<VkDeviceQueueCreateInfo> queue_create_info(
        computeQueueIdxs_.size());

    std::vector<std::vector<float>> queue_priority(computeQueueIdxs_.size());
    for (size_t i = 0; i < computeQueueIdxs_.size(); i++) {
        auto [qidx, queue_count, queueflags] = computeQueueIdxs_[i];
        LOG_INFO("queue count %d", queue_count);
        if (queue_count >= 4) {
            // reserve queues from one familiy
            queue_count = 4;
        }
        queue_priority[i].resize(queue_count);
        std::fill(queue_priority[i].begin(), queue_priority[i].end(), 1.0F);
        queue_create_info[i].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_create_info[i].queueFamilyIndex = qidx;
        // multiqueue for pipeline parallelism
        queue_create_info[i].queueCount = queue_count;
        queue_create_info[i].pQueuePriorities = queue_priority[i].data();
    }
    VkDeviceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.queueCreateInfoCount = queue_create_info.size();
    create_info.pQueueCreateInfos = queue_create_info.data();
    create_info.enabledLayerCount = 0;
    create_info.ppEnabledLayerNames = nullptr;
    create_info.pEnabledFeatures = &features;
    create_info.enabledExtensionCount =
        static_cast<uint32_t>(enabled_extensions.size());
    create_info.ppEnabledExtensionNames = enabled_extensions.data();
    create_info.pNext = p_first;
    // std::cout << "enabled extensions count " << enabled_extensions.size()
    //           << std::endl;
    // for (auto &e : enabled_extensions) {
    //     std::cout << "enabled extension: " << e << std::endl;
    // }

    if (vkCreateDevice(physicalDevice_, &create_info, nullptr,
                       &logicalDevice_) != VK_SUCCESS) {
        LOG_ERROR("Failed to create logical device!");
        return false;
    }

    for (auto [qidx, queue_count, queueflags] : computeQueueIdxs_) {
        if (queue_count >= 4) {
            queue_count = 4;
        }
        for (uint32_t i = 0; i < queue_count; i++) {
            VkQueue queue;
            vkGetDeviceQueue(logicalDevice_, qidx, i, &queue);
            computeQueues_.emplace_back(
                std::make_shared<VulkanQueue>(logicalDevice_, qidx, queue));
        }
    }
    return true;
}

bool VulkanDevice::checkDeviceExtensionFeature(
    const std::vector<VkExtensionProperties> &ext_properties,
    const char *name) const {
    for (auto ext : ext_properties) {
        if (std::string(ext.extensionName) == name) {
            return true;
        }
    }
    return false;
}

bool VulkanDevice::checkDeviceUnifiedMemoryAccess() {
    VkPhysicalDeviceMemoryProperties memory_properties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memory_properties);

    bool is_unified_memory = false;

    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
        const VkMemoryType &mem_type = memory_properties.memoryTypes[i];

        if ((mem_type.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (mem_type.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            is_unified_memory = true;
            LOG_INFO("Unified memory type found at index: %d", i);
        }
    }

    if (is_unified_memory) {
        LOG_INFO("This device supports Unified Memory Architecture (UMA).");
    } else {
        LOG_INFO("This device does not support Unified Memory Architecture.");
    }
    return is_unified_memory;
}

} // namespace vkop
