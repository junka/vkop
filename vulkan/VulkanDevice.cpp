// Copyright 2025 @junka
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanLib.hpp"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "include/logger.hpp"
#include "vulkan/vulkan_core.h"

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
    checkImageFormatSupport();
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

    VkPhysicalDeviceSubgroupProperties subgroup_properties = {};
    subgroup_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    subgroup_properties.pNext = nullptr;

#ifdef VK_NV_cuda_kernel_launch
    VkPhysicalDeviceCudaKernelLaunchPropertiesNV cuda_properties = {};
    cuda_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUDA_KERNEL_LAUNCH_PROPERTIES_NV;
    cuda_properties.pNext = &subgroup_properties;
    properties2.pNext = &cuda_properties;
#else
    properties2.pNext = &subgroup_properties;
#endif

#ifdef VK_EXT_host_image_copy
    VkPhysicalDeviceHostImageCopyProperties hostimagecopyproperty = {};
    hostimagecopyproperty.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_PROPERTIES;

#ifdef VK_NV_cuda_kernel_launch
    cuda_properties.pNext = &hostimagecopyproperty;
    hostimagecopyproperty.pNext = &subgroup_properties;
#else
    subgroup_properties.pNext = &hostimagecopyproperty;
    hostimagecopyproperty.pNext = nullptr;
#endif
#endif

#if VK_EXT_host_image_copy
    vkGetPhysicalDeviceProperties2(physicalDevice_, &properties2);
    if (hostimagecopyproperty.copySrcLayoutCount > 0) {
        this->copySrcLayout_.resize(hostimagecopyproperty.copySrcLayoutCount);
        this->copyDstLayout_.resize(hostimagecopyproperty.copyDstLayoutCount);
        hostimagecopyproperty.pCopySrcLayouts = this->copySrcLayout_.data();
        hostimagecopyproperty.pCopyDstLayouts = this->copyDstLayout_.data();
    }
#endif
    vkGetPhysicalDeviceProperties2(physicalDevice_, &properties2);
    this->timestampPeriod_ = properties2.properties.limits.timestampPeriod;
    this->maxImageArrayLayers_ =
        properties2.properties.limits.maxImageArrayLayers;
    this->deviceName_ = properties2.properties.deviceName;
    LOG_INFO("GPU %s", this->deviceName_.c_str());
    LOG_INFO("Max image array layers %d", this->maxImageArrayLayers_);
    LOG_INFO("Min TexelBuffer Alignment %llu",
             properties2.properties.limits.minTexelBufferOffsetAlignment);
    if (subgroup_properties.supportedOperations &
        VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) {
        LOG_INFO("Device support subgroup arithmetic");
        LOG_INFO("Subgroup size %d", subgroup_properties.subgroupSize);
    }
#if VK_EXT_host_image_copy
    for (uint32_t i = 0; i < hostimagecopyproperty.copySrcLayoutCount; i++) {
        LOG_INFO("HostImageCopy support src layout %d",
                 hostimagecopyproperty.pCopySrcLayouts[i]);
    }
    for (uint32_t i = 0; i < hostimagecopyproperty.copyDstLayoutCount; i++) {
        LOG_INFO("HostImageCopy support dst layout %d",
                 hostimagecopyproperty.pCopyDstLayouts[i]);
    }
#endif
#if VK_NV_cuda_kernel_launch
    LOG_INFO("CUDA kernel launch compute capability %u.%u",
             cuda_properties.computeCapabilityMajor,
             cuda_properties.computeCapabilityMinor);
#endif

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
        } else if constexpr (kInflight > 1) {
            if ((queue_families[i].queueFlags & (second)) == (second)) {
                std::tuple<uint32_t, uint32_t, VkQueueFlags> t = {
                    i, queue_families[i].queueCount,
                    queue_families[i].queueFlags};
                computeQueueIdxs_.emplace_back(t);
            }
        }
    }

    return static_cast<uint32_t>(computeQueueIdxs_.size());
}

void VulkanDevice::assertImageConfigurationSupported(VkFormat format) {

    VkImageType img_type = VK_IMAGE_TYPE_2D;
    VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL;
    VkImageUsageFlags usage =
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    VkImageCreateFlags flags = 0;

    VkFormatProperties format_props;
    vkGetPhysicalDeviceFormatProperties(physicalDevice_, format, &format_props);

    const VkFormatFeatureFlags *required_features_ptr = nullptr;
    if (tiling == VK_IMAGE_TILING_OPTIMAL) {
        required_features_ptr = &format_props.optimalTilingFeatures;
    } else if (tiling == VK_IMAGE_TILING_LINEAR) {
        required_features_ptr = &format_props.linearTilingFeatures;
    } else {
        assert(false && "Unsupported tiling mode");
        return;
    }

    VkFormatFeatureFlags required_features = 0;
    if (usage & VK_IMAGE_USAGE_SAMPLED_BIT) {
        required_features |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    }
    if (usage & VK_IMAGE_USAGE_STORAGE_BIT) {
        required_features |= VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
    }
    if (usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) {
        required_features |= VK_FORMAT_FEATURE_TRANSFER_SRC_BIT;
    }
    if (usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) {
        required_features |= VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    }

    assert((*required_features_ptr & required_features) == required_features &&
           "Format does not support required tiling features");

    VkImageFormatProperties props;
    VkResult result = vkGetPhysicalDeviceImageFormatProperties(
        physicalDevice_, format, img_type, tiling, usage, flags, &props);

    if (result == VK_SUCCESS) {
        LOG_INFO("format %d is supported", format);
    }
    assert(result == VK_SUCCESS &&
           "vkGetPhysicalDeviceImageFormatProperties failed: "
           "requested image configuration is not supported!");
    if (m_support_host_image_copy_) {
        usage |= VK_IMAGE_USAGE_HOST_TRANSFER_BIT;
    }
    result = vkGetPhysicalDeviceImageFormatProperties(
        physicalDevice_, format, img_type, tiling, usage, flags, &props);
    if (result == VK_SUCCESS) {
        LOG_INFO("format %d is supported for host image copy", format);
    } else {
        LOG_INFO("format %d is not supported for host image copy", format);
    }
}

EnableResult VulkanDevice::buildFeatureEnableChain(
    const std::vector<FeatureDescriptor> &descs,
    const VkPhysicalDeviceFeatures2 &queryFeatures2) {
    // Map sType → queried struct pointer
    std::unordered_map<VkStructureType, void *> queried_map;
    for (const void *p = queryFeatures2.pNext; p;) {
        auto *base = const_cast<VkBaseOutStructure *>(
            static_cast<const VkBaseOutStructure *>(p));
        queried_map[base->sType] = base;
        p = base->pNext;
    }

    EnableResult result;
    for (const auto &desc : descs) {
        auto it = queried_map.find(desc.sType);
        if (it == queried_map.end())
            continue;

        auto enabled_struct = desc.makeEnableStruct(it->second);
        if (!enabled_struct)
            continue;

        result.enableChain.push_back(std::move(enabled_struct));
        if (desc.extensionName) {
            result.enabledExtensions.push_back(desc.extensionName);
        }
    }

// Enable extensions without feature structrues
#ifdef VK_EXT_tooling_info
    result.enabledExtensions.push_back(VK_EXT_TOOLING_INFO_EXTENSION_NAME);
#endif
#ifdef VK_KHR_shader_non_semantic_info
    result.enabledExtensions.push_back(
        VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);
#endif
#ifdef VK_KHR_external_memory_fd
    result.enabledExtensions.push_back(
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
#endif
    return result;
}

QueryChainResult VulkanDevice::buildFeatureQueryChain(
    const std::vector<FeatureDescriptor> &descs) {
    QueryChainResult result;
    result.features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

    void *pnext = nullptr;
    // Build from back to front (Vulkan doesn't care, but consistent)
    for (auto it = descs.rbegin(); it != descs.rend(); ++it) {
        auto struct_ptr = it->makeQueryStruct();
        if (!struct_ptr)
            continue;
        struct_ptr->pNext = reinterpret_cast<VkBaseOutStructure *>(pnext);
        pnext = struct_ptr.get();
        result.ownedStructs.push_back(std::move(struct_ptr));
    }
    result.features2.pNext = pnext;
    return result;
}

void VulkanDevice::checkImageFormatSupport() {
    VkFormatProperties fmprops;
    vkGetPhysicalDeviceFormatProperties(
        physicalDevice_, VK_FORMAT_R32G32B32A32_SFLOAT, &fmprops);
    assert(fmprops.bufferFeatures &
           (VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
            VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT));
    vkGetPhysicalDeviceFormatProperties(
        physicalDevice_, VK_FORMAT_R16G16B16A16_SFLOAT, &fmprops);
    assert(fmprops.bufferFeatures &
           (VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
            VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT));
    // we would like SSCALE but not supported by most device, use SNORM instead
    vkGetPhysicalDeviceFormatProperties(physicalDevice_,
                                        VK_FORMAT_R8G8B8A8_SNORM, &fmprops);
    assert(fmprops.bufferFeatures &
           (VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
            VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT));
    vkGetPhysicalDeviceFormatProperties(physicalDevice_,
                                        VK_FORMAT_R8G8B8A8_UNORM, &fmprops);
    assert(fmprops.bufferFeatures &
           (VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
            VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT));

    assertImageConfigurationSupported(VK_FORMAT_R32G32B32A32_SFLOAT);
    assertImageConfigurationSupported(VK_FORMAT_R16G16B16A16_SFLOAT);
    assertImageConfigurationSupported(VK_FORMAT_R8G8B8A8_SNORM);
    assertImageConfigurationSupported(VK_FORMAT_R8G8B8A8_UNORM);
}

std::vector<FeatureDescriptor> VulkanDevice::createFeatureDescriptors(
    const VkPhysicalDeviceProperties &deviceProperties,
    const std::set<std::string> &supportedExtensions) {
    std::vector<FeatureDescriptor> descs;
#ifdef VK_VERSION_1_1
    if (deviceProperties.apiVersion >= VK_API_VERSION_1_1) {
        descs.push_back(FeatureDescriptor{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
            .extensionName = nullptr,
            .corePromotedVersion = VK_API_VERSION_1_1,
            .makeQueryStruct = [this]() -> std::unique_ptr<VkBaseOutStructure> {
                auto q = std::make_unique<VkPhysicalDeviceVulkan11Features>();
                *q = makeFeatureStruct<
                    VkPhysicalDeviceVulkan11Features,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES>();
                return std::unique_ptr<VkBaseOutStructure>(
                    reinterpret_cast<VkBaseOutStructure *>(q.release()));
            },
            .makeEnableStruct =
                [this](void *q) -> std::unique_ptr<VkBaseOutStructure> {
                auto *feat = static_cast<VkPhysicalDeviceVulkan11Features *>(q);
                if (!feat->storageBuffer16BitAccess &&
                    !feat->storageInputOutput16 &&
                    !feat->uniformAndStorageBuffer16BitAccess) {
                    return nullptr;
                }
                auto e = std::make_unique<VkPhysicalDeviceVulkan11Features>();
                *e = makeFeatureStruct<
                    VkPhysicalDeviceVulkan11Features,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES>();
                e->storageBuffer16BitAccess = feat->storageBuffer16BitAccess;
                e->storageInputOutput16 = feat->storageInputOutput16;
                e->uniformAndStorageBuffer16BitAccess =
                    feat->uniformAndStorageBuffer16BitAccess;
                return std::unique_ptr<VkBaseOutStructure>(
                    reinterpret_cast<VkBaseOutStructure *>(e.release()));
            }});
    }
#endif
#ifdef VK_VERSION_1_2
    if (deviceProperties.apiVersion >= VK_API_VERSION_1_2) {
        descs.push_back(FeatureDescriptor{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            .extensionName = nullptr,
            .corePromotedVersion = VK_API_VERSION_1_2,
            .makeQueryStruct = [this]() -> std::unique_ptr<VkBaseOutStructure> {
                auto q = std::make_unique<VkPhysicalDeviceVulkan12Features>();
                *q = makeFeatureStruct<
                    VkPhysicalDeviceVulkan12Features,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES>();
                return std::unique_ptr<VkBaseOutStructure>(
                    reinterpret_cast<VkBaseOutStructure *>(q.release()));
            },
            .makeEnableStruct =
                [this](void *q) -> std::unique_ptr<VkBaseOutStructure> {
                auto *feat = static_cast<VkPhysicalDeviceVulkan12Features *>(q);
                if (!feat->shaderFloat16 && !feat->shaderInt8 &&
                    !feat->timelineSemaphore && !feat->bufferDeviceAddress) {
                    return nullptr;
                }
                auto e = std::make_unique<VkPhysicalDeviceVulkan12Features>();
                *e = makeFeatureStruct<
                    VkPhysicalDeviceVulkan12Features,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES>();

                e->shaderFloat16 = feat->shaderFloat16;
                e->shaderInt8 = feat->shaderInt8;
                e->timelineSemaphore = feat->timelineSemaphore;
                e->bufferDeviceAddress = feat->bufferDeviceAddress;
                m_support_timeline_semaphore_ = feat->timelineSemaphore;
                return std::unique_ptr<VkBaseOutStructure>(
                    reinterpret_cast<VkBaseOutStructure *>(e.release()));
            }});
    }
#endif
#ifdef VK_KHR_16bit_storage
    if (supportedExtensions.count(VK_KHR_16BIT_STORAGE_EXTENSION_NAME) > 0 &&
        deviceProperties.apiVersion < VK_API_VERSION_1_1) {
        descs.push_back(FeatureDescriptor{
            .sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR,
            .extensionName = VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
            .corePromotedVersion = 0,
            .makeQueryStruct = [this]() -> std::unique_ptr<VkBaseOutStructure> {
                auto queried =
                    std::make_unique<VkPhysicalDevice16BitStorageFeaturesKHR>();
                *queried = makeFeatureStruct<
                    VkPhysicalDevice16BitStorageFeaturesKHR,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR>();
                return std::unique_ptr<VkBaseOutStructure>(
                    reinterpret_cast<VkBaseOutStructure *>(queried.release()));
            },
            .makeEnableStruct =
                [this](void *q) -> std::unique_ptr<VkBaseOutStructure> {
                auto *feat =
                    static_cast<VkPhysicalDevice16BitStorageFeaturesKHR *>(q);
                if (!feat->storageBuffer16BitAccess &&
                    !feat->storageInputOutput16 &&
                    !feat->uniformAndStorageBuffer16BitAccess) {
                    return nullptr;
                }
                auto e =
                    std::make_unique<VkPhysicalDevice16BitStorageFeaturesKHR>();
                *e = makeFeatureStruct<
                    VkPhysicalDevice16BitStorageFeaturesKHR,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR>();
                e->storageBuffer16BitAccess = feat->storageBuffer16BitAccess;
                e->storageInputOutput16 = feat->storageInputOutput16;
                e->uniformAndStorageBuffer16BitAccess =
                    feat->uniformAndStorageBuffer16BitAccess;
                return std::unique_ptr<VkBaseOutStructure>(
                    reinterpret_cast<VkBaseOutStructure *>(e.release()));
            }});
    }
#endif
#ifdef VK_KHR_shader_float16_int8
    if (supportedExtensions.count(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME) >
            0 &&
        deviceProperties.apiVersion < VK_API_VERSION_1_3) {
        descs.push_back(FeatureDescriptor{
            .sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR,
            .extensionName = VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
            .corePromotedVersion = 0,
            .makeQueryStruct = [this]() -> std::unique_ptr<VkBaseOutStructure> {
                auto queried = std::make_unique<
                    VkPhysicalDeviceShaderFloat16Int8FeaturesKHR>();
                *queried = makeFeatureStruct<
                    VkPhysicalDeviceShaderFloat16Int8FeaturesKHR,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR>();
                return std::unique_ptr<VkBaseOutStructure>(
                    reinterpret_cast<VkBaseOutStructure *>(queried.release()));
            },
            .makeEnableStruct =
                [this](void *q) -> std::unique_ptr<VkBaseOutStructure> {
                auto *queried =
                    static_cast<VkPhysicalDeviceShaderFloat16Int8FeaturesKHR *>(
                        q);
                if (!queried->shaderFloat16 && !queried->shaderInt8)
                    return nullptr;

                auto enabled = std::make_unique<
                    VkPhysicalDeviceShaderFloat16Int8FeaturesKHR>();
                *enabled = makeFeatureStruct<
                    VkPhysicalDeviceShaderFloat16Int8FeaturesKHR,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR>();
                enabled->shaderFloat16 = queried->shaderFloat16;
                enabled->shaderInt8 = queried->shaderInt8;
                return std::unique_ptr<VkBaseOutStructure>(
                    reinterpret_cast<VkBaseOutStructure *>(enabled.release()));
            }});
    }
#endif
#ifdef VK_KHR_shader_integer_dot_product
    if (supportedExtensions.count(
            VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME) > 0 &&
        deviceProperties.apiVersion < VK_API_VERSION_1_3) {
        descs.push_back(FeatureDescriptor{
            .sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR,
            .extensionName = VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME,
            .corePromotedVersion = 0,
            .makeQueryStruct = [this]() -> std::unique_ptr<VkBaseOutStructure> {
                auto queried = std::make_unique<
                    VkPhysicalDeviceShaderIntegerDotProductFeatures>();
                *queried = makeFeatureStruct<
                    VkPhysicalDeviceShaderIntegerDotProductFeatures,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR>();
                return std::unique_ptr<VkBaseOutStructure>(
                    reinterpret_cast<VkBaseOutStructure *>(queried.release()));
            },
            .makeEnableStruct =
                [this](void *q) -> std::unique_ptr<VkBaseOutStructure> {
                auto *queried = static_cast<
                    VkPhysicalDeviceShaderIntegerDotProductFeatures *>(q);
                if (!queried->shaderIntegerDotProduct)
                    return nullptr;

                auto enabled = std::make_unique<
                    VkPhysicalDeviceShaderIntegerDotProductFeatures>();
                *enabled = makeFeatureStruct<
                    VkPhysicalDeviceShaderIntegerDotProductFeatures,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR>();
                enabled->shaderIntegerDotProduct =
                    queried->shaderIntegerDotProduct;
                return std::unique_ptr<VkBaseOutStructure>(
                    reinterpret_cast<VkBaseOutStructure *>(enabled.release()));
            }});
    }
#endif

#if VK_VERSION_1_3
    if (deviceProperties.apiVersion >= VK_API_VERSION_1_3) {
        descs.push_back(FeatureDescriptor{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
            .extensionName = nullptr, // Core
            .corePromotedVersion = VK_API_VERSION_1_3,
            .makeQueryStruct = [this]() -> std::unique_ptr<VkBaseOutStructure> {
                auto q = std::make_unique<VkPhysicalDeviceVulkan13Features>();
                *q = makeFeatureStruct<
                    VkPhysicalDeviceVulkan13Features,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES>();
                return std::unique_ptr<VkBaseOutStructure>(
                    reinterpret_cast<VkBaseOutStructure *>(q.release()));
            },
            .makeEnableStruct =
                [this](void *q) -> std::unique_ptr<VkBaseOutStructure> {
                auto *feat = static_cast<VkPhysicalDeviceVulkan13Features *>(q);
                if (!feat->robustImageAccess && !feat->maintenance4 &&
                    !feat->subgroupSizeControl)
                    return nullptr;
                auto e = std::make_unique<VkPhysicalDeviceVulkan13Features>();
                *e = makeFeatureStruct<
                    VkPhysicalDeviceVulkan13Features,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES>();
                e->robustImageAccess = feat->robustImageAccess;
                e->subgroupSizeControl = feat->subgroupSizeControl;
                e->maintenance4 = feat->maintenance4;
                return std::unique_ptr<VkBaseOutStructure>(
                    reinterpret_cast<VkBaseOutStructure *>(e.release()));
            }});
    }
#endif
#ifdef VK_VERSION_1_4
    if (deviceProperties.apiVersion >= VK_API_VERSION_1_4) {
        descs.push_back(FeatureDescriptor{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES,
            .extensionName = nullptr,
            .corePromotedVersion = VK_API_VERSION_1_4,
            .makeQueryStruct = [this]() -> std::unique_ptr<VkBaseOutStructure> {
                auto q = std::make_unique<VkPhysicalDeviceVulkan14Features>();
                *q = makeFeatureStruct<
                    VkPhysicalDeviceVulkan14Features,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES>();
                return std::unique_ptr<VkBaseOutStructure>(
                    reinterpret_cast<VkBaseOutStructure *>(q.release()));
            },
            .makeEnableStruct =
                [this](void *q) -> std::unique_ptr<VkBaseOutStructure> {
                auto *feat = static_cast<VkPhysicalDeviceVulkan14Features *>(q);
                printf("feat14 host image copy %d\n", feat->hostImageCopy);
                if (!feat->hostImageCopy && !feat->maintenance5)
                    return nullptr;
                auto e = std::make_unique<VkPhysicalDeviceVulkan14Features>();
                *e = makeFeatureStruct<
                    VkPhysicalDeviceVulkan14Features,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES>();
                e->hostImageCopy = feat->hostImageCopy;
                m_support_host_image_copy_ = feat->hostImageCopy;
                e->maintenance5 = feat->maintenance5;
                return std::unique_ptr<VkBaseOutStructure>(
                    reinterpret_cast<VkBaseOutStructure *>(e.release()));
            }});
    }
#endif
#ifdef VK_EXT_host_image_copy
    if (deviceProperties.apiVersion < VK_API_VERSION_1_4 &&
        supportedExtensions.count(VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME) > 0) {
        descs.push_back(FeatureDescriptor{
            .sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES_EXT,
            .extensionName = VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME,
            .corePromotedVersion = 0,
            .makeQueryStruct = [this]() -> std::unique_ptr<VkBaseOutStructure> {
                auto q =
                    std::make_unique<VkPhysicalDeviceHostImageCopyFeatures>();
                *q = makeFeatureStruct<
                    VkPhysicalDeviceHostImageCopyFeatures,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES_EXT>();
                return std::unique_ptr<VkBaseOutStructure>(
                    reinterpret_cast<VkBaseOutStructure *>(q.release()));
            },
            .makeEnableStruct =
                [this](void *q) -> std::unique_ptr<VkBaseOutStructure> {
                auto *feat =
                    static_cast<VkPhysicalDeviceHostImageCopyFeatures *>(q);
                if (!feat->hostImageCopy)
                    return nullptr;
                auto e =
                    std::make_unique<VkPhysicalDeviceHostImageCopyFeatures>();
                *e = makeFeatureStruct<
                    VkPhysicalDeviceHostImageCopyFeatures,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES_EXT>();
                e->hostImageCopy = feat->hostImageCopy;
                m_support_host_image_copy_ = feat->hostImageCopy;
                return std::unique_ptr<VkBaseOutStructure>(
                    reinterpret_cast<VkBaseOutStructure *>(e.release()));
            }});
    }
#endif

#ifdef VK_EXT_image_robustness
    if (deviceProperties.apiVersion < VK_API_VERSION_1_3 &&
        supportedExtensions.count(VK_EXT_IMAGE_ROBUSTNESS_EXTENSION_NAME) > 0) {
        descs.push_back(FeatureDescriptor{
            .sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES,
            .extensionName = VK_EXT_IMAGE_ROBUSTNESS_EXTENSION_NAME,
            .corePromotedVersion = VK_API_VERSION_1_3,
            .makeQueryStruct = [this]() -> std::unique_ptr<VkBaseOutStructure> {
                auto q =
                    std::make_unique<VkPhysicalDeviceImageRobustnessFeatures>();
                *q = makeFeatureStruct<
                    VkPhysicalDeviceImageRobustnessFeatures,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES>();
                return std::unique_ptr<VkBaseOutStructure>(
                    reinterpret_cast<VkBaseOutStructure *>(q.release()));
            },
            .makeEnableStruct =
                [this](void *q) -> std::unique_ptr<VkBaseOutStructure> {
                auto *feat =
                    static_cast<VkPhysicalDeviceImageRobustnessFeatures *>(q);
                if (!feat->robustImageAccess)
                    return nullptr;
                auto e =
                    std::make_unique<VkPhysicalDeviceImageRobustnessFeatures>();
                *e = makeFeatureStruct<
                    VkPhysicalDeviceImageRobustnessFeatures,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES>();
                e->robustImageAccess = VK_TRUE;
                return std::unique_ptr<VkBaseOutStructure>(
                    reinterpret_cast<VkBaseOutStructure *>(e.release()));
            },
        });
    }
#endif

#ifdef VK_KHR_timeline_semaphore
    if (deviceProperties.apiVersion < VK_API_VERSION_1_2 &&
        supportedExtensions.count(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME) >
            0) {
        descs.push_back(FeatureDescriptor{
            .sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
            .extensionName = VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
            .corePromotedVersion = VK_API_VERSION_1_2,
            .makeQueryStruct = [this]() -> std::unique_ptr<VkBaseOutStructure> {
                auto q = std::make_unique<
                    VkPhysicalDeviceTimelineSemaphoreFeatures>();
                *q = makeFeatureStruct<
                    VkPhysicalDeviceTimelineSemaphoreFeatures,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES>();
                return std::unique_ptr<VkBaseOutStructure>(
                    reinterpret_cast<VkBaseOutStructure *>(q.release()));
            },
            .makeEnableStruct =
                [this](void *q) -> std::unique_ptr<VkBaseOutStructure> {
                auto *feat =
                    static_cast<VkPhysicalDeviceTimelineSemaphoreFeatures *>(q);
                if (!feat->timelineSemaphore)
                    return nullptr;
                auto e = std::make_unique<
                    VkPhysicalDeviceTimelineSemaphoreFeatures>();
                *e = makeFeatureStruct<
                    VkPhysicalDeviceTimelineSemaphoreFeatures,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES>();
                e->timelineSemaphore = VK_TRUE;
                m_support_timeline_semaphore_ = true;
                return std::unique_ptr<VkBaseOutStructure>(
                    reinterpret_cast<VkBaseOutStructure *>(e.release()));
            }});
    }
#endif

#ifdef VK_KHR_buffer_device_address
    if (deviceProperties.apiVersion < VK_API_VERSION_1_2 &&
        supportedExtensions.count(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) >
            0) {
        descs.push_back(FeatureDescriptor{
            .sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
            .extensionName = VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
            .corePromotedVersion = VK_API_VERSION_1_2,
            .makeQueryStruct = [this]() -> std::unique_ptr<VkBaseOutStructure> {
                auto q = std::make_unique<
                    VkPhysicalDeviceBufferDeviceAddressFeatures>();
                *q = makeFeatureStruct<
                    VkPhysicalDeviceBufferDeviceAddressFeatures,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES>();
                return std::unique_ptr<VkBaseOutStructure>(
                    reinterpret_cast<VkBaseOutStructure *>(q.release()));
            },
            .makeEnableStruct =
                [this](void *q) -> std::unique_ptr<VkBaseOutStructure> {
                auto *feat =
                    static_cast<VkPhysicalDeviceBufferDeviceAddressFeatures *>(
                        q);
                if (!feat->bufferDeviceAddress)
                    return nullptr;
                auto e = std::make_unique<
                    VkPhysicalDeviceBufferDeviceAddressFeatures>();
                *e = makeFeatureStruct<
                    VkPhysicalDeviceBufferDeviceAddressFeatures,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES>();
                e->bufferDeviceAddress = VK_TRUE;
                return std::unique_ptr<VkBaseOutStructure>(
                    reinterpret_cast<VkBaseOutStructure *>(e.release()));
            }});
    }
#endif
#ifdef VK_AMD_device_coherent_memory
    if (supportedExtensions.count(
            VK_AMD_DEVICE_COHERENT_MEMORY_EXTENSION_NAME) > 0) {
        descs.push_back(FeatureDescriptor{
            .sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COHERENT_MEMORY_FEATURES_AMD,
            .extensionName = VK_AMD_DEVICE_COHERENT_MEMORY_EXTENSION_NAME,
            .corePromotedVersion = 0,
            .makeQueryStruct = [this]() -> std::unique_ptr<VkBaseOutStructure> {
                auto q = std::make_unique<
                    VkPhysicalDeviceCoherentMemoryFeaturesAMD>();
                *q = makeFeatureStruct<
                    VkPhysicalDeviceCoherentMemoryFeaturesAMD,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COHERENT_MEMORY_FEATURES_AMD>();
                return std::unique_ptr<VkBaseOutStructure>(
                    reinterpret_cast<VkBaseOutStructure *>(q.release()));
            },
            .makeEnableStruct =
                [this](void *q) -> std::unique_ptr<VkBaseOutStructure> {
                auto *feat =
                    static_cast<VkPhysicalDeviceCoherentMemoryFeaturesAMD *>(q);
                if (!feat->deviceCoherentMemory)
                    return nullptr;
                auto e = std::make_unique<
                    VkPhysicalDeviceCoherentMemoryFeaturesAMD>();
                *e = makeFeatureStruct<
                    VkPhysicalDeviceCoherentMemoryFeaturesAMD,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COHERENT_MEMORY_FEATURES_AMD>();
                e->deviceCoherentMemory = feat->deviceCoherentMemory;
                return std::unique_ptr<VkBaseOutStructure>(
                    reinterpret_cast<VkBaseOutStructure *>(e.release()));
            }});
    }
#endif

#ifdef VK_KHR_performance_query
    if (supportedExtensions.count(VK_KHR_PERFORMANCE_QUERY_EXTENSION_NAME) >
        0) {
        descs.push_back(FeatureDescriptor{
            .sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PERFORMANCE_QUERY_FEATURES_KHR,
            .extensionName = VK_KHR_PERFORMANCE_QUERY_EXTENSION_NAME,
            .corePromotedVersion = 0,
            .makeQueryStruct = [this]() -> std::unique_ptr<VkBaseOutStructure> {
                auto q = std::make_unique<
                    VkPhysicalDevicePerformanceQueryFeaturesKHR>();
                *q = makeFeatureStruct<
                    VkPhysicalDevicePerformanceQueryFeaturesKHR,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PERFORMANCE_QUERY_FEATURES_KHR>();
                return std::unique_ptr<VkBaseOutStructure>(
                    reinterpret_cast<VkBaseOutStructure *>(q.release()));
            },
            .makeEnableStruct =
                [this](void *q) -> std::unique_ptr<VkBaseOutStructure> {
                auto *feat =
                    static_cast<VkPhysicalDevicePerformanceQueryFeaturesKHR *>(
                        q);
                if (!feat->performanceCounterQueryPools)
                    return nullptr;
                auto e = std::make_unique<
                    VkPhysicalDevicePerformanceQueryFeaturesKHR>();
                *e = makeFeatureStruct<
                    VkPhysicalDevicePerformanceQueryFeaturesKHR,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PERFORMANCE_QUERY_FEATURES_KHR>();
                e->performanceCounterQueryPools =
                    feat->performanceCounterQueryPools;
                return std::unique_ptr<VkBaseOutStructure>(
                    reinterpret_cast<VkBaseOutStructure *>(e.release()));
            }});
    }
#endif

#ifdef VK_EXT_subgroup_size_control
    if (deviceProperties.apiVersion < VK_API_VERSION_1_3 &&
        supportedExtensions.count(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME) >
            0) {
        descs.push_back(FeatureDescriptor{
            .sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT,
            .extensionName = VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME,
            .corePromotedVersion = 0,
            .makeQueryStruct = [this]() -> std::unique_ptr<VkBaseOutStructure> {
                auto q = std::make_unique<
                    VkPhysicalDeviceSubgroupSizeControlFeaturesEXT>();
                *q = makeFeatureStruct<
                    VkPhysicalDeviceSubgroupSizeControlFeaturesEXT,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT>();
                return std::unique_ptr<VkBaseOutStructure>(
                    reinterpret_cast<VkBaseOutStructure *>(q.release()));
            },
            .makeEnableStruct =
                [this](void *q) -> std::unique_ptr<VkBaseOutStructure> {
                auto *feat = static_cast<
                    VkPhysicalDeviceSubgroupSizeControlFeaturesEXT *>(q);
                if (!feat->subgroupSizeControl)
                    return nullptr;
                auto e = std::make_unique<
                    VkPhysicalDeviceSubgroupSizeControlFeaturesEXT>();
                *e = makeFeatureStruct<
                    VkPhysicalDeviceSubgroupSizeControlFeaturesEXT,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT>();
                e->subgroupSizeControl = feat->subgroupSizeControl;
                return std::unique_ptr<VkBaseOutStructure>(
                    reinterpret_cast<VkBaseOutStructure *>(e.release()));
            }});
    }
#endif

#ifdef VK_EXT_memory_priority
    if (supportedExtensions.count(VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME) > 0) {
        descs.push_back(FeatureDescriptor{
            .sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT,
            .extensionName = VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME,
            .corePromotedVersion = 0,
            .makeQueryStruct = [this]() -> std::unique_ptr<VkBaseOutStructure> {
                auto q = std::make_unique<
                    VkPhysicalDeviceMemoryPriorityFeaturesEXT>();
                *q = makeFeatureStruct<
                    VkPhysicalDeviceMemoryPriorityFeaturesEXT,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT>();
                return std::unique_ptr<VkBaseOutStructure>(
                    reinterpret_cast<VkBaseOutStructure *>(q.release()));
            },
            .makeEnableStruct =
                [this](void *q) -> std::unique_ptr<VkBaseOutStructure> {
                auto *feat =
                    static_cast<VkPhysicalDeviceMemoryPriorityFeaturesEXT *>(q);
                if (!feat->memoryPriority)
                    return nullptr;
                auto e = std::make_unique<
                    VkPhysicalDeviceMemoryPriorityFeaturesEXT>();
                *e = makeFeatureStruct<
                    VkPhysicalDeviceMemoryPriorityFeaturesEXT,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT>();
                e->memoryPriority = feat->memoryPriority;
                return std::unique_ptr<VkBaseOutStructure>(
                    reinterpret_cast<VkBaseOutStructure *>(e.release()));
            }});
    }
#endif
#ifdef VK_KHR_cooperative_matrix
    if (supportedExtensions.count(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) >
        0) {
        descs.push_back(FeatureDescriptor{
            .sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
            .extensionName = VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME,
            .corePromotedVersion = 0,
            .makeQueryStruct = [this]() -> std::unique_ptr<VkBaseOutStructure> {
                auto q = std::make_unique<
                    VkPhysicalDeviceCooperativeMatrixFeaturesKHR>();
                *q = makeFeatureStruct<
                    VkPhysicalDeviceCooperativeMatrixFeaturesKHR,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR>();
                return std::unique_ptr<VkBaseOutStructure>(
                    reinterpret_cast<VkBaseOutStructure *>(q.release()));
            },
            .makeEnableStruct =
                [this](void *q) -> std::unique_ptr<VkBaseOutStructure> {
                auto *feat =
                    static_cast<VkPhysicalDeviceCooperativeMatrixFeaturesKHR *>(
                        q);
                if (!feat->cooperativeMatrix)
                    return nullptr;
                auto e = std::make_unique<
                    VkPhysicalDeviceCooperativeMatrixFeaturesKHR>();
                *e = makeFeatureStruct<
                    VkPhysicalDeviceCooperativeMatrixFeaturesKHR,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR>();
                e->cooperativeMatrix = feat->cooperativeMatrix;
                return std::unique_ptr<VkBaseOutStructure>(
                    reinterpret_cast<VkBaseOutStructure *>(e.release()));
            }});
    }
#endif
#if defined VK_NV_cooperative_matrix
    if (supportedExtensions.count(VK_NV_COOPERATIVE_MATRIX_EXTENSION_NAME) >
        0) {
        descs.push_back(FeatureDescriptor{
            .sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_NV,
            .extensionName = VK_NV_COOPERATIVE_MATRIX_EXTENSION_NAME,
            .corePromotedVersion = 0,
            .makeQueryStruct = [this]() -> std::unique_ptr<VkBaseOutStructure> {
                auto q = std::make_unique<
                    VkPhysicalDeviceCooperativeMatrixFeaturesNV>();
                *q = makeFeatureStruct<
                    VkPhysicalDeviceCooperativeMatrixFeaturesNV,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_NV>();
                return std::unique_ptr<VkBaseOutStructure>(
                    reinterpret_cast<VkBaseOutStructure *>(q.release()));
            },
            .makeEnableStruct =
                [this](void *q) -> std::unique_ptr<VkBaseOutStructure> {
                auto *feat =
                    static_cast<VkPhysicalDeviceCooperativeMatrixFeaturesNV *>(
                        q);
                if (!feat->cooperativeMatrix)
                    return nullptr;
                auto e = std::make_unique<
                    VkPhysicalDeviceCooperativeMatrixFeaturesNV>();
                *e = makeFeatureStruct<
                    VkPhysicalDeviceCooperativeMatrixFeaturesNV,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_NV>();
                e->cooperativeMatrix = feat->cooperativeMatrix;
                return std::unique_ptr<VkBaseOutStructure>(
                    reinterpret_cast<VkBaseOutStructure *>(e.release()));
            }});
    }
#endif

#ifdef VK_NV_cuda_kernel_launch
    if (supportedExtensions.count(VK_NV_CUDA_KERNEL_LAUNCH_EXTENSION_NAME) >
        0) {
        descs.push_back(FeatureDescriptor{
            .sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUDA_KERNEL_LAUNCH_FEATURES_NV,
            .extensionName = VK_NV_CUDA_KERNEL_LAUNCH_EXTENSION_NAME,
            .corePromotedVersion = 0,
            .makeQueryStruct = [this]() -> std::unique_ptr<VkBaseOutStructure> {
                auto q = std::make_unique<
                    VkPhysicalDeviceCudaKernelLaunchFeaturesNV>();
                *q = makeFeatureStruct<
                    VkPhysicalDeviceCudaKernelLaunchFeaturesNV,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUDA_KERNEL_LAUNCH_FEATURES_NV>();
                return std::unique_ptr<VkBaseOutStructure>(
                    reinterpret_cast<VkBaseOutStructure *>(q.release()));
            },
            .makeEnableStruct =
                [this](void *q) -> std::unique_ptr<VkBaseOutStructure> {
                auto *feat =
                    static_cast<VkPhysicalDeviceCudaKernelLaunchFeaturesNV *>(
                        q);
                if (!feat->cudaKernelLaunchFeatures)
                    return nullptr;
                auto e = std::make_unique<
                    VkPhysicalDeviceCudaKernelLaunchFeaturesNV>();
                *e = makeFeatureStruct<
                    VkPhysicalDeviceCudaKernelLaunchFeaturesNV,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUDA_KERNEL_LAUNCH_FEATURES_NV>();
                e->cudaKernelLaunchFeatures = VK_TRUE;
                m_support_cuda_kernel_launch_ = true;
                return std::unique_ptr<VkBaseOutStructure>(
                    reinterpret_cast<VkBaseOutStructure *>(e.release()));
            }});
    }
#endif
    for (size_t i = 0; i < descs.size(); ++i) {
        const auto &desc = descs[i];
        printf("Descriptor %zu: sType=%d, extensionName=%s, coreVersion=%u\n",
               i, desc.sType,
               desc.extensionName ? desc.extensionName : "nullptr",
               desc.corePromotedVersion);
    }

    return descs;
}

bool VulkanDevice::createLogicalDevice(
    const VkPhysicalDeviceProperties &deviceProperties) {
    if (findComputeQueueFamily() == 0) {
        LOG_ERROR("Failed to find a suitable compute queue family!");
        return false;
    }

    std::vector<VkExtensionProperties> ext_properties;

    uint32_t p_property_count = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr,
                                         &p_property_count, nullptr);
    ext_properties.resize(p_property_count);
    vkEnumerateDeviceExtensionProperties(
        physicalDevice_, nullptr, &p_property_count, ext_properties.data());
    std::set<std::string> supported_extensions;
    for (auto ext : ext_properties) {
        supported_extensions.insert(ext.extensionName);
        LOG_INFO("device extension %s", ext.extensionName);
    }

    VkPhysicalDeviceFeatures device_features = {};
    vkGetPhysicalDeviceFeatures(physicalDevice_, &device_features);

    auto feature_descs =
        createFeatureDescriptors(deviceProperties, supported_extensions);

    auto query_result = buildFeatureQueryChain(feature_descs);
    vkGetPhysicalDeviceFeatures2(physicalDevice_, &query_result.features2);
    auto enable_result =
        buildFeatureEnableChain(feature_descs, query_result.features2);
    LOG_INFO("timeline semaphore %s supported",
             (m_support_timeline_semaphore_ ? "is" : "not"));
    LOG_INFO("host image copy %s supported",
             (m_support_host_image_copy_ ? "is" : "not"));

    std::vector<std::vector<float>> queue_priority(computeQueueIdxs_.size());
    auto queue_create_info = setupQueueCreateInfo(queue_priority);
    VkDeviceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.queueCreateInfoCount =
        static_cast<uint32_t>(queue_create_info.size());
    create_info.pQueueCreateInfos = queue_create_info.data();
    create_info.enabledLayerCount = 0;
    create_info.ppEnabledLayerNames = nullptr;
    // create_info.pEnabledFeatures = &features;
    create_info.enabledExtensionCount =
        static_cast<uint32_t>(enable_result.enabledExtensions.size());
    create_info.ppEnabledExtensionNames =
        enable_result.enabledExtensions.data();
    void *pnext = nullptr;
    for (auto it = enable_result.enableChain.rbegin();
         it != enable_result.enableChain.rend(); ++it) {
        (*it)->pNext = reinterpret_cast<VkBaseOutStructure *>(pnext);
        pnext = it->get();
    }
    create_info.pNext = pnext;
    std::cout << "enabled extensions count "
              << enable_result.enabledExtensions.size() << std::endl;
    for (auto &e : enable_result.enabledExtensions) {
        std::cout << "enabled extension: " << e << std::endl;
    }

    if (vkCreateDevice(physicalDevice_, &create_info, nullptr,
                       &logicalDevice_) != VK_SUCCESS) {
        LOG_ERROR("Failed to create logical device!");
        return false;
    }

    for (auto [qidx, queue_count, queueflags] : computeQueueIdxs_) {
        queue_count = std::min<
            std::tuple_element<1, class std::tuple<unsigned int, unsigned int,
                                                   unsigned int>>::type>(
            queue_count, (kInflight + 1) / 2);
        for (uint32_t i = 0; i < queue_count; i++) {
            VkQueue queue;
            vkGetDeviceQueue(logicalDevice_, qidx, i, &queue);
            computeQueues_.emplace_back(
                std::make_shared<VulkanQueue>(logicalDevice_, qidx, queue));
        }
    }
    return true;
}

std::vector<VkDeviceQueueCreateInfo> VulkanDevice::setupQueueCreateInfo(
    std::vector<std::vector<float>> &queue_priority) {
    std::vector<VkDeviceQueueCreateInfo> queue_create_info(
        computeQueueIdxs_.size());

    for (size_t i = 0; i < computeQueueIdxs_.size(); i++) {
        auto [qidx, queue_count, queueflags] = computeQueueIdxs_[i];
        LOG_INFO("queue count %d", queue_count);
        queue_count = std::min<std::tuple_element<
            1, std::tuple<unsigned int, unsigned int, unsigned int>>::type>(
            queue_count, (kInflight + 1) / 2);
        queue_priority[i].resize(queue_count);
        std::fill(queue_priority[i].begin(), queue_priority[i].end(), 1.0F);
        queue_create_info[i].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_create_info[i].queueFamilyIndex = qidx;
        // multiqueue for pipeline parallelism
        queue_create_info[i].queueCount = queue_count;
        queue_create_info[i].pQueuePriorities = queue_priority[i].data();
    }

    return queue_create_info;
}

bool VulkanDevice::checkDeviceExtensionFeature(
    const std::vector<VkExtensionProperties> &ext_properties,
    const char *name) {
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

void VulkanDevice::wait_all_done() { vkDeviceWaitIdle(logicalDevice_); }

} // namespace vkop
