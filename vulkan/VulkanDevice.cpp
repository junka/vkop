// Copyright 2025 @junka
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanDebug.hpp"
#include "vulkan/VulkanLib.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "include/logger.hpp"

namespace vkop {

namespace {

// The enable lambdas below copy only the bits the driver reported, so a
// capability that reaches vkCreateDevice is enabled exactly when it was
// supported. Recording both from the queried value keeps that invariant in one
// place; `available` is set where the descriptor is pushed, because that is the
// spot that knows whether the extension enumerated at all.
void recordCap(DeviceCap &cap, VkBool32 reported) {
    cap.supported = reported != VK_FALSE;
    cap.enabled = cap.supported;
}

} // namespace

VulkanDevice::VulkanDevice(VkPhysicalDevice physicalDevice) {
    physicalDevice_ = physicalDevice;
    if (physicalDevice_ == VK_NULL_HANDLE) {
        throw std::runtime_error("Invalid Vulkan physical device handle.");
    }
    if (logicalDevice_ != VK_NULL_HANDLE) {
        throw std::runtime_error("Logical device already created.");
    }
    if (!computeQueues_.empty()) {
        throw std::runtime_error("Compute queue already created.");
    }
    checkImageFormatSupport();
    // Before getProperties(): the UMA bit is part of the property pass.
    m_is_unified_memory_ = checkDeviceUnifiedMemoryAccess();
    auto props = getProperties();
    if (!createLogicalDevice(props)) {
        LOG_ERROR("Failed to create logical device!");
        return;
    }
    // Everything the table-driven selector keys on is final now: the coopmat
    // feature bool comes from createLogicalDevice()'s feature chain, and the
    // property list / UMA probe are already cached. One query per device.
    device_traits_ = core::device_traits_from(*this);
    const core::DeviceProfile &profile =
        core::device_profile_for(device_traits_);
    // NDEBUG compiles LOG_INFO away to nothing, which leaves `profile` unused
    // in a release build; the cast keeps the -Werror build clean without
    // changing the debug path.
    (void)profile;
    LOG_INFO("MatMul profile: %s (byte_min_m=%d fp16_min_m=%d wide_n=%d "
             "ks_target=%d)",
             profile.name, profile.byte_min_m, profile.fp16_min_m,
             profile.wide_n, profile.ks_target_blocks);
    // After device_traits_: the dump wants the resolved facts next to the
    // enabled feature bits. No-op unless VKOP_DEVCAPS=1.
    dumpShaderCapabilities();
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

    // Build the pNext chain by linking each extension struct to the current
    // head, then making it the new head. Order is irrelevant for property
    // queries — the driver walks the chain by sType.
    VkBaseOutStructure *head =
        reinterpret_cast<VkBaseOutStructure *>(&subgroup_properties);

    // Driver identification. The driverID is what separates MoltenVK's MSL
    // backend from a native driver on the same vendor ID; chaining it here
    // (before the extension structs below) keeps it populated in every build.
    VkPhysicalDeviceDriverProperties driver_properties = {};
    driver_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
    driver_properties.pNext = head;
    head = reinterpret_cast<VkBaseOutStructure *>(&driver_properties);

#ifdef VK_KHR_cooperative_matrix
    VkPhysicalDeviceCooperativeMatrixPropertiesKHR coopmat_properties = {};
    coopmat_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    coopmat_properties.pNext = head;
    head = reinterpret_cast<VkBaseOutStructure *>(&coopmat_properties);
#endif

#ifdef VK_NV_cuda_kernel_launch
    VkPhysicalDeviceCudaKernelLaunchPropertiesNV cuda_properties = {};
    cuda_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUDA_KERNEL_LAUNCH_PROPERTIES_NV;
    cuda_properties.pNext = head;
    head = reinterpret_cast<VkBaseOutStructure *>(&cuda_properties);
#endif

#ifdef VK_EXT_host_image_copy
    VkPhysicalDeviceHostImageCopyPropertiesEXT hostimagecopyproperty = {};
    hostimagecopyproperty.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_PROPERTIES_EXT;
    hostimagecopyproperty.pNext = head;
    head = reinterpret_cast<VkBaseOutStructure *>(&hostimagecopyproperty);
#endif

#ifdef VK_EXT_subgroup_size_control
    // Queried so the pipeline can request a fixed requiredSubgroupSize; the
    // Intel ANV driver is free to pick any subgroupSize in [min,max] when it
    // is not pinned, which breaks shaders that hard-assume a specific
    // numSubgroups (e.g. softmax.comp).
    VkPhysicalDeviceSubgroupSizeControlPropertiesEXT ssc_properties = {};
    ssc_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT;
    ssc_properties.pNext = head;
    head = reinterpret_cast<VkBaseOutStructure *>(&ssc_properties);
#endif

    properties2.pNext = head;
    vkGetPhysicalDeviceProperties2(physicalDevice_, &properties2);

#ifdef VK_EXT_host_image_copy
    // Layout arrays are queried in two passes: the driver reports counts
    // first, then we provide buffers and re-query to populate them.
    if (hostimagecopyproperty.copySrcLayoutCount > 0 ||
        hostimagecopyproperty.copyDstLayoutCount > 0) {
        this->copySrcLayout_.resize(hostimagecopyproperty.copySrcLayoutCount);
        this->copyDstLayout_.resize(hostimagecopyproperty.copyDstLayoutCount);
        hostimagecopyproperty.pCopySrcLayouts = this->copySrcLayout_.data();
        hostimagecopyproperty.pCopyDstLayouts = this->copyDstLayout_.data();
        vkGetPhysicalDeviceProperties2(physicalDevice_, &properties2);
    }
#endif

    this->timestampPeriod_ = properties2.properties.limits.timestampPeriod;
    this->maxImageArrayLayers_ =
        properties2.properties.limits.maxImageArrayLayers;
    this->maxImageDimension2D_ =
        properties2.properties.limits.maxImageDimension2D;
    this->deviceName_ = properties2.properties.deviceName;
    LOG_INFO("GPU %s", this->deviceName_.c_str());

    LOG_INFO("Driver ID %d (vendor 0x%x)", driver_properties.driverID,
             properties2.properties.vendorID);
    // The tile/GEMV crossovers this used to compute per driver now live in the
    // ordered table in core/KernelSelect.cpp; the resolved profile is logged
    // from the constructor once device_traits_ exists.
    LOG_INFO("Max image array layers %d", this->maxImageArrayLayers_);
    LOG_INFO("Max image dimension 2D %d", this->maxImageDimension2D_);
    LOG_INFO("Min TexelBuffer Alignment %llu",
             properties2.properties.limits.minTexelBufferOffsetAlignment);
#ifdef VK_EXT_subgroup_size_control
    minSubgroupSize_ = ssc_properties.minSubgroupSize;
    maxSubgroupSize_ = ssc_properties.maxSubgroupSize;
    LOG_INFO("Subgroup size control min %u max %u",
             ssc_properties.minSubgroupSize, ssc_properties.maxSubgroupSize);
#endif
    // Cached for the *_sg shader variant decision (Runtime::LoadModel and
    // Operator::resolve_primary_spv), so it must be set on every query.
    m_support_subgroup_arithmetic_ = (subgroup_properties.supportedOperations &
                                      VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
    if (m_support_subgroup_arithmetic_) {
        LOG_INFO("Device support subgroup arithmetic");
        LOG_INFO("Subgroup size %d", subgroup_properties.subgroupSize);
    }
#ifdef VK_KHR_cooperative_matrix
    m_coopmat_supported_stages_ =
        coopmat_properties.cooperativeMatrixSupportedStages;
    LOG_INFO("Cooperative matrix supported stages 0x%x",
             coopmat_properties.cooperativeMatrixSupportedStages);

    // Enumerate every (M, N, K, type, scope) combination the device supports.
    // RenderDoc's capture layer (up to and including 1.47) never installs a
    // device entry point for this query: the loader trampoline tail-jumps
    // into an empty dispatch-table slot and the process SIGSEGVs (#0 0x0,
    // call *%r13 with r13 == NULL). The list only feeds the coopmat
    // tile-shape selectors (supports_coopmat_*_tile), so skipping it under
    // capture degrades routing exactly like a non-coopmat device. The env
    // check keys on the same variable the layer's manifest enables; define
    // VKOP_DISABLE_COOPMAT_RDOC_GUARD to compile the guard out, or set
    // VKOP_RDOC_COOPMAT_ENUM=1 to force the enumeration at runtime (for
    // testing a renderer build where the entry point actually works).
#ifndef VKOP_DISABLE_COOPMAT_RDOC_GUARD
    static const bool coopmat_enum_skipped =
        std::getenv("ENABLE_VULKAN_RENDERDOC_CAPTURE") != nullptr &&
        std::getenv("VKOP_RDOC_COOPMAT_ENUM") == nullptr;
#else
    static constexpr bool coopmat_enum_skipped = false;
#endif
    if (coopmat_enum_skipped) {
        LOG_WARN("RenderDoc capture layer active: skipping "
                 "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR "
                 "(layer <= 1.47 lacks its device entry point); coopmat "
                 "tile probing disabled");
    }
    if (coopmat_properties.cooperativeMatrixSupportedStages != 0 &&
        !coopmat_enum_skipped) {
        uint32_t propCount = 0;
        auto vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR =
            reinterpret_cast<
                PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
                vkGetInstanceProcAddr(
                    VulkanInstance::getVulkanInstance().getInstance(),
                    "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
        if (vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR) {
            vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(
                physicalDevice_, &propCount, nullptr);
            if (propCount > 0) {
                this->coopmatProps_.resize(propCount);
                vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(
                    physicalDevice_, &propCount, this->coopmatProps_.data());
            }
            for (uint32_t i = 0; i < propCount; i++) {
                const auto &p = this->coopmatProps_[i];
                (void)p;
                LOG_INFO("CoopMat [%u]: %ux%ux%u AType=%d BType=%d CType=%d "
                         "ResultType=%d scope=%d",
                         i, p.MSize, p.NSize, p.KSize, p.AType, p.BType,
                         p.CType, p.ResultType, p.scope);
            }
        }
    }
#endif
#if VK_EXT_host_image_copy
    LOG_INFO("HostImageCopy identicalMemoryTypeRequirements %s",
             hostimagecopyproperty.identicalMemoryTypeRequirements == VK_TRUE
                 ? "True"
                 : "False");
    for (uint32_t i = 0; i < hostimagecopyproperty.copySrcLayoutCount; i++) {
        LOG_INFO("HostImageCopy support src layout %d",
                 hostimagecopyproperty.pCopySrcLayouts[i]);
    }
    for (uint32_t i = 0; i < hostimagecopyproperty.copyDstLayoutCount; i++) {
        LOG_INFO("HostImageCopy support dst layout %d",
                 hostimagecopyproperty.pCopyDstLayouts[i]);
    }
#endif
#ifdef VK_NV_cuda_kernel_launch
    LOG_INFO("CUDA kernel launch compute capability %u.%u",
             cuda_properties.computeCapabilityMajor,
             cuda_properties.computeCapabilityMinor);
#endif

    return properties2.properties;
}

VkDeviceSize VulkanDevice::minStorageBufferOffsetAlignment() {
    if (min_sb_offset_align_ == 0) {
        min_sb_offset_align_ =
            getProperties().limits.minStorageBufferOffsetAlignment;
        if (min_sb_offset_align_ == 0) {
            min_sb_offset_align_ = 1;
        }
    }
    return min_sb_offset_align_;
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
        if (kInflight > 1 &&
            (queue_families[i].queueFlags & (second)) == (second)) {
            std::tuple<uint32_t, uint32_t, VkQueueFlags> t = {
                i, queue_families[i].queueCount, queue_families[i].queueFlags};
            computeQueueIdxs_.emplace_back(t);
            graphicsQueueIdxs_.emplace_back(t);
            LOG_INFO("Found graphics+compute queue family %d with %d queues", i,
                     queue_families[i].queueCount);
        } else if ((queue_families[i].queueFlags & (first)) == (first)) {
            std::tuple<uint32_t, uint32_t, VkQueueFlags> t = {
                i, queue_families[i].queueCount, queue_families[i].queueFlags};
            computeQueueIdxs_.emplace_back(t);
            LOG_INFO("Found compute queue family %d with %d queues", i,
                     queue_families[i].queueCount);
        }
    }

    return static_cast<uint32_t>(computeQueueIdxs_.size());
}

void VulkanDevice::assertImageConfigurationSupported(VkFormat format) {
    const std::unordered_map<VkFormat, std::string> format_maps = {
        {VK_FORMAT_R32G32B32A32_SFLOAT, "RGBA32_SFLOT"},
        {VK_FORMAT_R16G16B16A16_SFLOAT, "RGBA16_SFLOAT"},
        {VK_FORMAT_R8G8B8A8_UNORM, "RGBA8_UNORM"},
        {VK_FORMAT_R8G8B8A8_SNORM, "RGBA8_SNORM"},
    };
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
    (void)required_features_ptr; // used only in the assert below (empty under
                                 // NDEBUG)

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
        LOG_INFO("format %s supports %u layers", format_maps.at(format).c_str(),
                 props.maxArrayLayers);
    }
    assert(result == VK_SUCCESS &&
           "vkGetPhysicalDeviceImageFormatProperties failed: "
           "requested image configuration is not supported!");
    if (m_support_host_image_copy_) {
        usage |= VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT;
    }
    result = vkGetPhysicalDeviceImageFormatProperties(
        physicalDevice_, format, img_type, tiling, usage, flags, &props);
    if (result == VK_SUCCESS) {
        LOG_INFO("format %s supports %u layers for hostImageCopy",
                 format_maps.at(format).c_str(), props.maxArrayLayers);
    } else {
        LOG_INFO("format %s not supports hostImageCopy",
                 format_maps.at(format).c_str());
    }
}

EnableResult VulkanDevice::buildFeatureEnableChain(
    const std::vector<FeatureDescriptor> &descs,
    const VkPhysicalDeviceFeatures2 &queryFeatures2,
    const std::set<std::string> &supportedExtensions) {
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
    if (supportedExtensions.count(VK_EXT_TOOLING_INFO_EXTENSION_NAME) > 0) {
        result.enabledExtensions.push_back(VK_EXT_TOOLING_INFO_EXTENSION_NAME);
    }
#endif
#ifdef VK_KHR_shader_non_semantic_info
    if (supportedExtensions.count(
            VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME) > 0) {
        result.enabledExtensions.push_back(
            VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);
    }
#endif
#ifdef VK_KHR_swapchain
    if (supportedExtensions.count(VK_KHR_SWAPCHAIN_EXTENSION_NAME) > 0) {
        // swapchain has no feature struct — old FeatureDescriptor path
        // incorrectly gated on geometryShader, excluding Intel ARL.
        result.enabledExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }
#endif
#ifdef VK_EXT_external_memory_host
    // No feature struct either — just the properties struct we query
    // separately in createLogicalDevice for the alignment. Enabled so
    // VulkanBuffer's host-import ctor is legal on this device.
    if (supportedExtensions.count(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME) >
        0) {
        result.enabledExtensions.push_back(
            VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
    }
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
        auto *base = static_cast<VkBaseOutStructure *>(struct_ptr.get());
        base->pNext = reinterpret_cast<VkBaseOutStructure *>(pnext);
        pnext = base;
        result.ownedStructs.push_back(std::move(struct_ptr));
    }
    result.features2.pNext = pnext;
    return result;
}

void VulkanDevice::checkImageFormatSupport() {
    VkFormatProperties3 formatProperties3{};
    formatProperties3.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3_KHR;
    VkFormatProperties2 formatProperties2{};
    formatProperties2.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2;
    formatProperties2.pNext = &formatProperties3;

    vkGetPhysicalDeviceFormatProperties2(
        physicalDevice_, VK_FORMAT_R32G32B32A32_SFLOAT, &formatProperties2);
    assert(formatProperties2.formatProperties.bufferFeatures &
           (VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT |
            VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_BIT));
    assert(formatProperties2.formatProperties.optimalTilingFeatures &
           (VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT |
            VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT |
            VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT_KHR |
            VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT_KHR |
            VK_FORMAT_FEATURE_2_HOST_IMAGE_TRANSFER_BIT_EXT));
    assert(formatProperties3.optimalTilingFeatures &
           VK_FORMAT_FEATURE_2_HOST_IMAGE_TRANSFER_BIT_EXT);
    vkGetPhysicalDeviceFormatProperties2(
        physicalDevice_, VK_FORMAT_R16G16B16A16_SFLOAT, &formatProperties2);
    assert(formatProperties2.formatProperties.bufferFeatures &
           (VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT |
            VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_BIT));
    assert(formatProperties2.formatProperties.optimalTilingFeatures &
           (VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT |
            VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT |
            VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT_KHR |
            VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT_KHR |
            VK_FORMAT_FEATURE_2_HOST_IMAGE_TRANSFER_BIT_EXT));
    assert(formatProperties3.optimalTilingFeatures &
           VK_FORMAT_FEATURE_2_HOST_IMAGE_TRANSFER_BIT_EXT);
    // we would like SSCALE but not supported by most device, use SNORM instead
    vkGetPhysicalDeviceFormatProperties2(
        physicalDevice_, VK_FORMAT_R8G8B8A8_SNORM, &formatProperties2);
    assert(formatProperties2.formatProperties.bufferFeatures &
           (VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT |
            VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_BIT));
    assert(formatProperties2.formatProperties.optimalTilingFeatures &
           (VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT |
            VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT |
            VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT_KHR |
            VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT_KHR |
            VK_FORMAT_FEATURE_2_HOST_IMAGE_TRANSFER_BIT_EXT));
    assert(formatProperties3.optimalTilingFeatures &
           VK_FORMAT_FEATURE_2_HOST_IMAGE_TRANSFER_BIT_EXT);
    vkGetPhysicalDeviceFormatProperties2(
        physicalDevice_, VK_FORMAT_R8G8B8A8_UNORM, &formatProperties2);
    assert(formatProperties2.formatProperties.bufferFeatures &
           (VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT |
            VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_BIT));
    assert(formatProperties2.formatProperties.optimalTilingFeatures &
           (VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT |
            VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT |
            VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT_KHR |
            VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT_KHR |
            VK_FORMAT_FEATURE_2_HOST_IMAGE_TRANSFER_BIT_EXT));
    assert(formatProperties3.optimalTilingFeatures &
           VK_FORMAT_FEATURE_2_HOST_IMAGE_TRANSFER_BIT_EXT);
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
        shader_caps_.storage_buffer_16bit.available = true;
        descs.push_back(FeatureDescriptor{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
            .extensionName = nullptr,
            .corePromotedVersion = VK_API_VERSION_1_1,
            .makeQueryStruct = [this]() -> std::shared_ptr<void> {
                auto q = std::make_unique<VkPhysicalDeviceVulkan11Features>();
                *q = makeFeatureStruct<
                    VkPhysicalDeviceVulkan11Features,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES>();
                return std::shared_ptr<void>(std::move(q));
            },
            .makeEnableStruct = [this](void *q) -> std::shared_ptr<void> {
                auto *feat = static_cast<VkPhysicalDeviceVulkan11Features *>(q);
                recordCap(shader_caps_.storage_buffer_16bit,
                          feat->storageBuffer16BitAccess);
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
                return std::shared_ptr<void>(std::move(e));
            }});
    }
#endif
#ifdef VK_VERSION_1_2
    if (deviceProperties.apiVersion >= VK_API_VERSION_1_2) {
        shader_caps_.shader_float16.available = true;
        shader_caps_.shader_int8.available = true;
        shader_caps_.storage_buffer_8bit.available = true;
        shader_caps_.uniform_storage_8bit.available = true;
        descs.push_back(FeatureDescriptor{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            .extensionName = nullptr,
            .corePromotedVersion = VK_API_VERSION_1_2,
            .makeQueryStruct = [this]() -> std::shared_ptr<void> {
                auto q = std::make_unique<VkPhysicalDeviceVulkan12Features>();
                *q = makeFeatureStruct<
                    VkPhysicalDeviceVulkan12Features,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES>();
                return std::shared_ptr<void>(std::move(q));
            },
            .makeEnableStruct = [this](void *q) -> std::shared_ptr<void> {
                auto *feat = static_cast<VkPhysicalDeviceVulkan12Features *>(q);
                recordCap(shader_caps_.shader_float16, feat->shaderFloat16);
                recordCap(shader_caps_.shader_int8, feat->shaderInt8);
                recordCap(shader_caps_.storage_buffer_8bit,
                          feat->storageBuffer8BitAccess);
                recordCap(shader_caps_.uniform_storage_8bit,
                          feat->uniformAndStorageBuffer8BitAccess);
                if (!feat->shaderFloat16 && !feat->shaderInt8 &&
                    !feat->timelineSemaphore && !feat->bufferDeviceAddress &&
                    !feat->descriptorBindingStorageBufferUpdateAfterBind &&
                    !feat->storageBuffer8BitAccess &&
                    !feat->uniformAndStorageBuffer8BitAccess) {
                    return nullptr;
                }
                auto e = std::make_unique<VkPhysicalDeviceVulkan12Features>();
                *e = makeFeatureStruct<
                    VkPhysicalDeviceVulkan12Features,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES>();

                e->shaderFloat16 = feat->shaderFloat16;
                e->shaderInt8 = feat->shaderInt8;
                // 8-bit loads for int8 weights/activations. Only the storage
                // buffer forms: storagePushConstant8 would widen the push
                // constant range nothing uses.
                e->storageBuffer8BitAccess = feat->storageBuffer8BitAccess;
                e->uniformAndStorageBuffer8BitAccess =
                    feat->uniformAndStorageBuffer8BitAccess;
                e->timelineSemaphore = feat->timelineSemaphore;
                e->bufferDeviceAddress = feat->bufferDeviceAddress;
                e->descriptorBindingStorageBufferUpdateAfterBind =
                    feat->descriptorBindingStorageBufferUpdateAfterBind;
                m_support_timeline_semaphore_ = feat->timelineSemaphore;
                m_support_descriptor_update_after_bind_ =
                    feat->descriptorBindingStorageBufferUpdateAfterBind;
                return std::shared_ptr<void>(std::move(e));
            }});
    }
#endif
#ifdef VK_KHR_16bit_storage
    if (supportedExtensions.count(VK_KHR_16BIT_STORAGE_EXTENSION_NAME) > 0 &&
        deviceProperties.apiVersion < VK_API_VERSION_1_1) {
        shader_caps_.storage_buffer_16bit.available = true;
        descs.push_back(FeatureDescriptor{
            .sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR,
            .extensionName = VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
            .corePromotedVersion = 0,
            .makeQueryStruct = [this]() -> std::shared_ptr<void> {
                auto queried =
                    std::make_unique<VkPhysicalDevice16BitStorageFeaturesKHR>();
                *queried = makeFeatureStruct<
                    VkPhysicalDevice16BitStorageFeaturesKHR,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR>();
                return std::shared_ptr<void>(std::move(queried));
            },
            .makeEnableStruct = [this](void *q) -> std::shared_ptr<void> {
                auto *feat =
                    static_cast<VkPhysicalDevice16BitStorageFeaturesKHR *>(q);
                recordCap(shader_caps_.storage_buffer_16bit,
                          feat->storageBuffer16BitAccess);
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
                return std::shared_ptr<void>(std::move(e));
            }});
    }
#endif
#ifdef VK_KHR_8bit_storage
    // Same promoted/legacy split as 16bit_storage: below 1.2 the bits come from
    // the extension struct instead of VkPhysicalDeviceVulkan12Features.
    if (supportedExtensions.count(VK_KHR_8BIT_STORAGE_EXTENSION_NAME) > 0 &&
        deviceProperties.apiVersion < VK_API_VERSION_1_2) {
        shader_caps_.storage_buffer_8bit.available = true;
        shader_caps_.uniform_storage_8bit.available = true;
        descs.push_back(FeatureDescriptor{
            .sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR,
            .extensionName = VK_KHR_8BIT_STORAGE_EXTENSION_NAME,
            .corePromotedVersion = VK_API_VERSION_1_2,
            .makeQueryStruct = [this]() -> std::shared_ptr<void> {
                auto queried =
                    std::make_unique<VkPhysicalDevice8BitStorageFeaturesKHR>();
                *queried = makeFeatureStruct<
                    VkPhysicalDevice8BitStorageFeaturesKHR,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR>();
                return std::shared_ptr<void>(std::move(queried));
            },
            .makeEnableStruct = [this](void *q) -> std::shared_ptr<void> {
                auto *feat =
                    static_cast<VkPhysicalDevice8BitStorageFeaturesKHR *>(q);
                recordCap(shader_caps_.storage_buffer_8bit,
                          feat->storageBuffer8BitAccess);
                recordCap(shader_caps_.uniform_storage_8bit,
                          feat->uniformAndStorageBuffer8BitAccess);
                if (!feat->storageBuffer8BitAccess &&
                    !feat->uniformAndStorageBuffer8BitAccess) {
                    return nullptr;
                }
                auto e =
                    std::make_unique<VkPhysicalDevice8BitStorageFeaturesKHR>();
                *e = makeFeatureStruct<
                    VkPhysicalDevice8BitStorageFeaturesKHR,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR>();
                e->storageBuffer8BitAccess = feat->storageBuffer8BitAccess;
                e->uniformAndStorageBuffer8BitAccess =
                    feat->uniformAndStorageBuffer8BitAccess;
                return std::shared_ptr<void>(std::move(e));
            }});
    }
#endif
#ifdef VK_KHR_shader_float16_int8
    if (supportedExtensions.count(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME) >
            0 &&
        deviceProperties.apiVersion < VK_API_VERSION_1_3) {
        shader_caps_.shader_float16.available = true;
        shader_caps_.shader_int8.available = true;
        descs.push_back(FeatureDescriptor{
            .sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR,
            .extensionName = VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
            .corePromotedVersion = 0,
            .makeQueryStruct = [this]() -> std::shared_ptr<void> {
                auto queried = std::make_unique<
                    VkPhysicalDeviceShaderFloat16Int8FeaturesKHR>();
                *queried = makeFeatureStruct<
                    VkPhysicalDeviceShaderFloat16Int8FeaturesKHR,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR>();
                return std::shared_ptr<void>(std::move(queried));
            },
            .makeEnableStruct = [this](void *q) -> std::shared_ptr<void> {
                auto *queried =
                    static_cast<VkPhysicalDeviceShaderFloat16Int8FeaturesKHR *>(
                        q);
                recordCap(shader_caps_.shader_float16, queried->shaderFloat16);
                recordCap(shader_caps_.shader_int8, queried->shaderInt8);
                if (!queried->shaderFloat16 && !queried->shaderInt8)
                    return nullptr;

                auto enabled = std::make_unique<
                    VkPhysicalDeviceShaderFloat16Int8FeaturesKHR>();
                *enabled = makeFeatureStruct<
                    VkPhysicalDeviceShaderFloat16Int8FeaturesKHR,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR>();
                enabled->shaderFloat16 = queried->shaderFloat16;
                enabled->shaderInt8 = queried->shaderInt8;
                return std::shared_ptr<void>(std::move(enabled));
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
            .makeQueryStruct = [this]() -> std::shared_ptr<void> {
                auto queried = std::make_unique<
                    VkPhysicalDeviceShaderIntegerDotProductFeatures>();
                *queried = makeFeatureStruct<
                    VkPhysicalDeviceShaderIntegerDotProductFeatures,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR>();
                return std::shared_ptr<void>(std::move(queried));
            },
            .makeEnableStruct = [this](void *q) -> std::shared_ptr<void> {
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
                return std::shared_ptr<void>(std::move(enabled));
            }});
    }
#endif

    // ---- shader-level capability probes -------------------------------------
    // Probed and enabled so the kernel selector has a truthful answer to ask
    // about later; nothing routes on them yet. Each entry is the same
    // query-then-copy-only-reported-bits shape as the ones above, so a driver
    // that lacks a bit costs nothing and one that has it cannot fail
    // vkCreateDevice.
#ifdef VK_EXT_shader_atomic_float
    // Never promoted to core. fp32 atomic add on a storage buffer is what would
    // collapse the two-pass split-K (deposit partials + fold) into one pass.
    if (supportedExtensions.count(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME) >
        0) {
        shader_caps_.f32_global_atomic_add.available = true;
        shader_caps_.f32_shared_atomic_add.available = true;
        descs.push_back(FeatureDescriptor{
            .sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
            .extensionName = VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME,
            .corePromotedVersion = 0,
            .makeQueryStruct = [this]() -> std::shared_ptr<void> {
                auto q = std::make_unique<
                    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT>();
                *q = makeFeatureStruct<
                    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT>();
                return std::shared_ptr<void>(std::move(q));
            },
            .makeEnableStruct = [this](void *q) -> std::shared_ptr<void> {
                auto *feat =
                    static_cast<VkPhysicalDeviceShaderAtomicFloatFeaturesEXT *>(
                        q);
                recordCap(shader_caps_.f32_global_atomic_add,
                          feat->shaderBufferFloat32AtomicAdd);
                recordCap(shader_caps_.f32_shared_atomic_add,
                          feat->shaderSharedFloat32AtomicAdd);
                // The float64 and image/sparse variants stay off: a bit the
                // engine never uses is a bit the driver can still refuse.
                if (!feat->shaderBufferFloat32AtomicAdd &&
                    !feat->shaderSharedFloat32AtomicAdd)
                    return nullptr;
                auto e = std::make_unique<
                    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT>();
                *e = makeFeatureStruct<
                    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT>();
                e->shaderBufferFloat32AtomicAdd =
                    feat->shaderBufferFloat32AtomicAdd;
                e->shaderSharedFloat32AtomicAdd =
                    feat->shaderSharedFloat32AtomicAdd;
                return std::shared_ptr<void>(std::move(e));
            }});
    }
#endif
#ifdef VK_NV_shader_atomic_float16_vector
    // Half2 global atomic add. The bit is reported on hardware that also has
    // shaderFloat16 clear (Pascal does), so the [devcaps] pair has to be read
    // together before a kernel assumes f16x2 atomics are usable.
    if (supportedExtensions.count(
            VK_NV_SHADER_ATOMIC_FLOAT16_VECTOR_EXTENSION_NAME) > 0) {
        shader_caps_.f16x2_global_atomic_add.available = true;
        descs.push_back(FeatureDescriptor{
            .sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT16_VECTOR_FEATURES_NV,
            .extensionName = VK_NV_SHADER_ATOMIC_FLOAT16_VECTOR_EXTENSION_NAME,
            .corePromotedVersion = 0,
            .makeQueryStruct = [this]() -> std::shared_ptr<void> {
                auto q = std::make_unique<
                    VkPhysicalDeviceShaderAtomicFloat16VectorFeaturesNV>();
                *q = makeFeatureStruct<
                    VkPhysicalDeviceShaderAtomicFloat16VectorFeaturesNV,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT16_VECTOR_FEATURES_NV>();
                return std::shared_ptr<void>(std::move(q));
            },
            .makeEnableStruct = [this](void *q) -> std::shared_ptr<void> {
                auto *feat = static_cast<
                    VkPhysicalDeviceShaderAtomicFloat16VectorFeaturesNV *>(q);
                recordCap(shader_caps_.f16x2_global_atomic_add,
                          feat->shaderFloat16VectorAtomics);
                if (!feat->shaderFloat16VectorAtomics)
                    return nullptr;
                auto e = std::make_unique<
                    VkPhysicalDeviceShaderAtomicFloat16VectorFeaturesNV>();
                *e = makeFeatureStruct<
                    VkPhysicalDeviceShaderAtomicFloat16VectorFeaturesNV,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT16_VECTOR_FEATURES_NV>();
                e->shaderFloat16VectorAtomics =
                    feat->shaderFloat16VectorAtomics;
                return std::shared_ptr<void>(std::move(e));
            }});
    }
#endif
#ifdef VK_KHR_workgroup_memory_explicit_layout
    // Lets one shared block be declared with an explicit layout, so the tile
    // buffers and the split-K accumulator can alias the same bytes and buy a
    // bigger tile. The 8/16-bit sub-features only mean something on top of the
    // matching storage features, which is why they are recorded separately.
    if (supportedExtensions.count(
            VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME) > 0) {
        shader_caps_.wmem_explicit_layout.available = true;
        shader_caps_.wmem_explicit_scalar_block.available = true;
        shader_caps_.wmem_explicit_8bit_access.available = true;
        shader_caps_.wmem_explicit_16bit_access.available = true;
        descs.push_back(FeatureDescriptor{
            .sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR,
            .extensionName =
                VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME,
            .corePromotedVersion = 0,
            .makeQueryStruct = [this]() -> std::shared_ptr<void> {
                auto q = std::make_unique<
                    VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR>();
                *q = makeFeatureStruct<
                    VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR>();
                return std::shared_ptr<void>(std::move(q));
            },
            .makeEnableStruct = [this](void *q) -> std::shared_ptr<void> {
                auto *feat = static_cast<
                    VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR *>(
                    q);
                recordCap(shader_caps_.wmem_explicit_layout,
                          feat->workgroupMemoryExplicitLayout);
                recordCap(shader_caps_.wmem_explicit_scalar_block,
                          feat->workgroupMemoryExplicitLayoutScalarBlockLayout);
                recordCap(shader_caps_.wmem_explicit_8bit_access,
                          feat->workgroupMemoryExplicitLayout8BitAccess);
                recordCap(shader_caps_.wmem_explicit_16bit_access,
                          feat->workgroupMemoryExplicitLayout16BitAccess);
                if (!feat->workgroupMemoryExplicitLayout)
                    return nullptr;
                auto e = std::make_unique<
                    VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR>();
                *e = makeFeatureStruct<
                    VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR>();
                e->workgroupMemoryExplicitLayout =
                    feat->workgroupMemoryExplicitLayout;
                e->workgroupMemoryExplicitLayoutScalarBlockLayout =
                    feat->workgroupMemoryExplicitLayoutScalarBlockLayout;
                e->workgroupMemoryExplicitLayout8BitAccess =
                    feat->workgroupMemoryExplicitLayout8BitAccess;
                e->workgroupMemoryExplicitLayout16BitAccess =
                    feat->workgroupMemoryExplicitLayout16BitAccess;
                return std::shared_ptr<void>(std::move(e));
            }});
    }
#endif
#ifdef VK_KHR_zero_initialize_workgroup_memory
    if (deviceProperties.apiVersion < VK_API_VERSION_1_3 &&
        supportedExtensions.count(
            VK_KHR_ZERO_INITIALIZE_WORKGROUP_MEMORY_EXTENSION_NAME) > 0) {
        shader_caps_.zero_init_wmem.available = true;
        descs.push_back(FeatureDescriptor{
            .sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ZERO_INITIALIZE_WORKGROUP_MEMORY_FEATURES_KHR,
            .extensionName =
                VK_KHR_ZERO_INITIALIZE_WORKGROUP_MEMORY_EXTENSION_NAME,
            .corePromotedVersion = VK_API_VERSION_1_3,
            .makeQueryStruct = [this]() -> std::shared_ptr<void> {
                auto q = std::make_unique<
                    VkPhysicalDeviceZeroInitializeWorkgroupMemoryFeaturesKHR>();
                *q = makeFeatureStruct<
                    VkPhysicalDeviceZeroInitializeWorkgroupMemoryFeaturesKHR,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ZERO_INITIALIZE_WORKGROUP_MEMORY_FEATURES_KHR>();
                return std::shared_ptr<void>(std::move(q));
            },
            .makeEnableStruct = [this](void *q) -> std::shared_ptr<void> {
                auto *feat = static_cast<
                    VkPhysicalDeviceZeroInitializeWorkgroupMemoryFeaturesKHR *>(
                    q);
                recordCap(shader_caps_.zero_init_wmem,
                          feat->shaderZeroInitializeWorkgroupMemory);
                if (!feat->shaderZeroInitializeWorkgroupMemory)
                    return nullptr;
                auto e = std::make_unique<
                    VkPhysicalDeviceZeroInitializeWorkgroupMemoryFeaturesKHR>();
                *e = makeFeatureStruct<
                    VkPhysicalDeviceZeroInitializeWorkgroupMemoryFeaturesKHR,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ZERO_INITIALIZE_WORKGROUP_MEMORY_FEATURES_KHR>();
                e->shaderZeroInitializeWorkgroupMemory =
                    feat->shaderZeroInitializeWorkgroupMemory;
                return std::shared_ptr<void>(std::move(e));
            }});
    }
#endif
#ifdef VK_KHR_shader_subgroup_rotate
    // VK_MAKE_API_VERSION instead of VK_API_VERSION_1_4: the promoted macro is
    // not defined by pre-1.4 headers, and this block sits above the host_image
    // copy block that patches it in.
    if (deviceProperties.apiVersion < VK_MAKE_API_VERSION(0, 1, 4, 0) &&
        supportedExtensions.count(
            VK_KHR_SHADER_SUBGROUP_ROTATE_EXTENSION_NAME) > 0) {
        shader_caps_.subgroup_rotate.available = true;
        shader_caps_.subgroup_rotate_clustered.available = true;
        descs.push_back(FeatureDescriptor{
            .sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_ROTATE_FEATURES_KHR,
            .extensionName = VK_KHR_SHADER_SUBGROUP_ROTATE_EXTENSION_NAME,
            .corePromotedVersion = VK_MAKE_API_VERSION(0, 1, 4, 0),
            .makeQueryStruct = [this]() -> std::shared_ptr<void> {
                auto q = std::make_unique<
                    VkPhysicalDeviceShaderSubgroupRotateFeaturesKHR>();
                *q = makeFeatureStruct<
                    VkPhysicalDeviceShaderSubgroupRotateFeaturesKHR,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_ROTATE_FEATURES_KHR>();
                return std::shared_ptr<void>(std::move(q));
            },
            .makeEnableStruct = [this](void *q) -> std::shared_ptr<void> {
                auto *feat = static_cast<
                    VkPhysicalDeviceShaderSubgroupRotateFeaturesKHR *>(q);
                recordCap(shader_caps_.subgroup_rotate,
                          feat->shaderSubgroupRotate);
                recordCap(shader_caps_.subgroup_rotate_clustered,
                          feat->shaderSubgroupRotateClustered);
                if (!feat->shaderSubgroupRotate)
                    return nullptr;
                auto e = std::make_unique<
                    VkPhysicalDeviceShaderSubgroupRotateFeaturesKHR>();
                *e = makeFeatureStruct<
                    VkPhysicalDeviceShaderSubgroupRotateFeaturesKHR,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_ROTATE_FEATURES_KHR>();
                e->shaderSubgroupRotate = feat->shaderSubgroupRotate;
                e->shaderSubgroupRotateClustered =
                    feat->shaderSubgroupRotateClustered;
                return std::shared_ptr<void>(std::move(e));
            }});
    }
#endif
    // ---- end shader-level capability probes ---------------------------------

#if VK_VERSION_1_3
    if (deviceProperties.apiVersion >= VK_API_VERSION_1_3) {
        shader_caps_.zero_init_wmem.available = true;
        descs.push_back(FeatureDescriptor{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
            .extensionName = nullptr, // Core
            .corePromotedVersion = VK_API_VERSION_1_3,
            .makeQueryStruct = [this]() -> std::shared_ptr<void> {
                auto q = std::make_unique<VkPhysicalDeviceVulkan13Features>();
                *q = makeFeatureStruct<
                    VkPhysicalDeviceVulkan13Features,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES>();
                return std::shared_ptr<void>(std::move(q));
            },
            .makeEnableStruct = [this](void *q) -> std::shared_ptr<void> {
                auto *feat = static_cast<VkPhysicalDeviceVulkan13Features *>(q);
                recordCap(shader_caps_.zero_init_wmem,
                          feat->shaderZeroInitializeWorkgroupMemory);
                if (!feat->robustImageAccess && !feat->maintenance4 &&
                    !feat->subgroupSizeControl &&
                    !feat->shaderZeroInitializeWorkgroupMemory)
                    return nullptr;
                auto e = std::make_unique<VkPhysicalDeviceVulkan13Features>();
                *e = makeFeatureStruct<
                    VkPhysicalDeviceVulkan13Features,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES>();
                e->robustImageAccess = feat->robustImageAccess;
                e->subgroupSizeControl = feat->subgroupSizeControl;
                e->maintenance4 = feat->maintenance4;
                // Lets a workgroup's shared accumulator start at zero without
                // the kernel clearing it first (VK_KHR_zero_initialize_
                // workgroup_memory, 1.3 core).
                e->shaderZeroInitializeWorkgroupMemory =
                    feat->shaderZeroInitializeWorkgroupMemory;
                return std::shared_ptr<void>(std::move(e));
            }});
    }
#endif
#ifdef VK_VERSION_1_4
    if (deviceProperties.apiVersion >= VK_API_VERSION_1_4) {
        shader_caps_.subgroup_rotate.available = true;
        shader_caps_.subgroup_rotate_clustered.available = true;
        descs.push_back(FeatureDescriptor{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES,
            .extensionName = nullptr,
            .corePromotedVersion = VK_API_VERSION_1_4,
            .makeQueryStruct = [this]() -> std::shared_ptr<void> {
                auto q = std::make_unique<VkPhysicalDeviceVulkan14Features>();
                *q = makeFeatureStruct<
                    VkPhysicalDeviceVulkan14Features,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES>();
                return std::shared_ptr<void>(std::move(q));
            },
            .makeEnableStruct = [this](void *q) -> std::shared_ptr<void> {
                auto *feat = static_cast<VkPhysicalDeviceVulkan14Features *>(q);
                recordCap(shader_caps_.subgroup_rotate,
                          feat->shaderSubgroupRotate);
                recordCap(shader_caps_.subgroup_rotate_clustered,
                          feat->shaderSubgroupRotateClustered);
                if (!feat->hostImageCopy && !feat->maintenance5 &&
                    !feat->shaderSubgroupRotate)
                    return nullptr;
                auto e = std::make_unique<VkPhysicalDeviceVulkan14Features>();
                *e = makeFeatureStruct<
                    VkPhysicalDeviceVulkan14Features,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES>();
                e->hostImageCopy = feat->hostImageCopy;
                m_support_host_image_copy_ = feat->hostImageCopy;
                e->maintenance5 = feat->maintenance5;
                // subgroupRotate for in-workgroup reduction/scan (the split-K
                // last-block fold). Deliberately NOT enabling pushDescriptor
                // here: measured host-side only, see the [dswrite] probe notes.
                e->shaderSubgroupRotate = feat->shaderSubgroupRotate;
                return std::shared_ptr<void>(std::move(e));
            }});
    }
#endif
#ifdef VK_EXT_host_image_copy

#ifndef VK_VERSION_1_4
#define VK_API_VERSION_1_4 VK_MAKE_API_VERSION(0, 1, 4, 0)
#endif
    if (deviceProperties.apiVersion < VK_API_VERSION_1_4 &&
        supportedExtensions.count(VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME) > 0) {
        descs.push_back(FeatureDescriptor{
            .sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES_EXT,
            .extensionName = VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME,
            .corePromotedVersion = 0,
            .makeQueryStruct = [this]() -> std::shared_ptr<void> {
                auto q = std::make_unique<
                    VkPhysicalDeviceHostImageCopyFeaturesEXT>();
                *q = makeFeatureStruct<
                    VkPhysicalDeviceHostImageCopyFeaturesEXT,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES_EXT>();
                return std::shared_ptr<void>(std::move(q));
            },
            .makeEnableStruct = [this](void *q) -> std::shared_ptr<void> {
                auto *feat =
                    static_cast<VkPhysicalDeviceHostImageCopyFeaturesEXT *>(q);
                if (!feat->hostImageCopy)
                    return nullptr;
                auto e = std::make_unique<
                    VkPhysicalDeviceHostImageCopyFeaturesEXT>();
                *e = makeFeatureStruct<
                    VkPhysicalDeviceHostImageCopyFeaturesEXT,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES_EXT>();
                e->hostImageCopy = feat->hostImageCopy;
                m_support_host_image_copy_ = feat->hostImageCopy;
                return std::shared_ptr<void>(std::move(e));
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
            .makeQueryStruct = [this]() -> std::shared_ptr<void> {
                auto q =
                    std::make_unique<VkPhysicalDeviceImageRobustnessFeatures>();
                *q = makeFeatureStruct<
                    VkPhysicalDeviceImageRobustnessFeatures,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES>();
                return std::shared_ptr<void>(std::move(q));
            },
            .makeEnableStruct = [this](void *q) -> std::shared_ptr<void> {
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
                return std::shared_ptr<void>(std::move(e));
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
            .makeQueryStruct = [this]() -> std::shared_ptr<void> {
                auto q = std::make_unique<
                    VkPhysicalDeviceTimelineSemaphoreFeatures>();
                *q = makeFeatureStruct<
                    VkPhysicalDeviceTimelineSemaphoreFeatures,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES>();
                return std::shared_ptr<void>(std::move(q));
            },
            .makeEnableStruct = [this](void *q) -> std::shared_ptr<void> {
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
                return std::shared_ptr<void>(std::move(e));
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
            .makeQueryStruct = [this]() -> std::shared_ptr<void> {
                auto q = std::make_unique<
                    VkPhysicalDeviceBufferDeviceAddressFeatures>();
                *q = makeFeatureStruct<
                    VkPhysicalDeviceBufferDeviceAddressFeatures,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES>();
                return std::shared_ptr<void>(std::move(q));
            },
            .makeEnableStruct = [this](void *q) -> std::shared_ptr<void> {
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
                return std::shared_ptr<void>(std::move(e));
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
            .makeQueryStruct = [this]() -> std::shared_ptr<void> {
                auto q = std::make_unique<
                    VkPhysicalDeviceCoherentMemoryFeaturesAMD>();
                *q = makeFeatureStruct<
                    VkPhysicalDeviceCoherentMemoryFeaturesAMD,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COHERENT_MEMORY_FEATURES_AMD>();
                return std::shared_ptr<void>(std::move(q));
            },
            .makeEnableStruct = [this](void *q) -> std::shared_ptr<void> {
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
                return std::shared_ptr<void>(std::move(e));
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
            .makeQueryStruct = [this]() -> std::shared_ptr<void> {
                auto q = std::make_unique<
                    VkPhysicalDevicePerformanceQueryFeaturesKHR>();
                *q = makeFeatureStruct<
                    VkPhysicalDevicePerformanceQueryFeaturesKHR,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PERFORMANCE_QUERY_FEATURES_KHR>();
                return std::shared_ptr<void>(std::move(q));
            },
            .makeEnableStruct = [this](void *q) -> std::shared_ptr<void> {
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
                return std::shared_ptr<void>(std::move(e));
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
            .makeQueryStruct = [this]() -> std::shared_ptr<void> {
                auto q = std::make_unique<
                    VkPhysicalDeviceSubgroupSizeControlFeaturesEXT>();
                *q = makeFeatureStruct<
                    VkPhysicalDeviceSubgroupSizeControlFeaturesEXT,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT>();
                return std::shared_ptr<void>(std::move(q));
            },
            .makeEnableStruct = [this](void *q) -> std::shared_ptr<void> {
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
                return std::shared_ptr<void>(std::move(e));
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
            .makeQueryStruct = [this]() -> std::shared_ptr<void> {
                auto q = std::make_unique<
                    VkPhysicalDeviceMemoryPriorityFeaturesEXT>();
                *q = makeFeatureStruct<
                    VkPhysicalDeviceMemoryPriorityFeaturesEXT,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT>();
                return std::shared_ptr<void>(std::move(q));
            },
            .makeEnableStruct = [this](void *q) -> std::shared_ptr<void> {
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
                return std::shared_ptr<void>(std::move(e));
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
            .makeQueryStruct = [this]() -> std::shared_ptr<void> {
                auto q = std::make_unique<
                    VkPhysicalDeviceCooperativeMatrixFeaturesKHR>();
                *q = makeFeatureStruct<
                    VkPhysicalDeviceCooperativeMatrixFeaturesKHR,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR>();
                return std::shared_ptr<void>(std::move(q));
            },
            .makeEnableStruct = [this](void *q) -> std::shared_ptr<void> {
                auto *feat =
                    static_cast<VkPhysicalDeviceCooperativeMatrixFeaturesKHR *>(
                        q);
                // The feature bool alone is not enough: MoltenVK advertises
                // the extension + feature but reports 0 supported stages, and
                // any coopmat pipeline then dies in SPIR-V -> MSL conversion.
                // Gate on the stages the device actually reported in
                // getProperties().
                if (!feat->cooperativeMatrix ||
                    m_coopmat_supported_stages_ == 0)
                    return nullptr;
                auto e = std::make_unique<
                    VkPhysicalDeviceCooperativeMatrixFeaturesKHR>();
                *e = makeFeatureStruct<
                    VkPhysicalDeviceCooperativeMatrixFeaturesKHR,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR>();
                e->cooperativeMatrix = feat->cooperativeMatrix;
                m_support_cooperate_matrix_ = feat->cooperativeMatrix;
                return std::shared_ptr<void>(std::move(e));
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
            .makeQueryStruct = [this]() -> std::shared_ptr<void> {
                auto q = std::make_unique<
                    VkPhysicalDeviceCooperativeMatrixFeaturesNV>();
                *q = makeFeatureStruct<
                    VkPhysicalDeviceCooperativeMatrixFeaturesNV,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_NV>();
                return std::shared_ptr<void>(std::move(q));
            },
            .makeEnableStruct = [this](void *q) -> std::shared_ptr<void> {
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
                m_support_nv_tensor_core_ = feat->cooperativeMatrix;
                return std::shared_ptr<void>(std::move(e));
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
            .makeQueryStruct = [this]() -> std::shared_ptr<void> {
                auto q = std::make_unique<
                    VkPhysicalDeviceCudaKernelLaunchFeaturesNV>();
                *q = makeFeatureStruct<
                    VkPhysicalDeviceCudaKernelLaunchFeaturesNV,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUDA_KERNEL_LAUNCH_FEATURES_NV>();
                return std::shared_ptr<void>(std::move(q));
            },
            .makeEnableStruct = [this](void *q) -> std::shared_ptr<void> {
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
                return std::shared_ptr<void>(std::move(e));
            }});
    }
#endif
#ifdef VK_KHR_external_memory_fd
    if (supportedExtensions.count(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME)) {
        descs.push_back(FeatureDescriptor{
            .sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT,
            .extensionName = VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
            .corePromotedVersion = 0,
            .makeQueryStruct = []() -> std::shared_ptr<void> {
                auto q = std::make_unique<
                    VkPhysicalDeviceExternalMemoryHostPropertiesEXT>();
                q->sType =
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT;
                q->pNext = nullptr;
                return std::shared_ptr<void>(std::move(q));
            },
            .makeEnableStruct = [](void *q) -> std::shared_ptr<void> {
                auto *feat = static_cast<
                    VkPhysicalDeviceExternalMemoryHostPropertiesEXT *>(q);
                if (!feat->minImportedHostPointerAlignment)
                    return nullptr;
                auto e = std::make_unique<
                    VkPhysicalDeviceExternalMemoryHostPropertiesEXT>();
                e->sType =
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT;
                e->pNext = nullptr;
                e->minImportedHostPointerAlignment =
                    feat->minImportedHostPointerAlignment;
                return std::shared_ptr<void>(std::move(e));
            }});
    }
#endif
#ifdef VK_EXT_descriptor_indexing
    if (deviceProperties.apiVersion < VK_API_VERSION_1_2 &&
        supportedExtensions.count(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME)) {
        descs.push_back(FeatureDescriptor{
            .sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES,
            .extensionName = VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
            .corePromotedVersion = VK_API_VERSION_1_2,
            .makeQueryStruct = [this]() -> std::shared_ptr<void> {
                auto q = std::make_unique<
                    VkPhysicalDeviceDescriptorIndexingFeaturesEXT>();
                q->sType =
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
                q->pNext = nullptr;
                return std::shared_ptr<void>(std::move(q));
            },
            .makeEnableStruct = [this](void *q) -> std::shared_ptr<void> {
                auto *feat = static_cast<
                    VkPhysicalDeviceDescriptorIndexingFeaturesEXT *>(q);
                if (!feat->descriptorBindingStorageBufferUpdateAfterBind)
                    return nullptr;
                auto e = std::make_unique<
                    VkPhysicalDeviceDescriptorIndexingFeaturesEXT>();
                e->sType =
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
                e->pNext = nullptr;
                e->descriptorBindingStorageBufferUpdateAfterBind =
                    feat->descriptorBindingStorageBufferUpdateAfterBind;
                m_support_descriptor_update_after_bind_ =
                    feat->descriptorBindingStorageBufferUpdateAfterBind;
                return std::shared_ptr<void>(std::move(e));
            },
        });
    }
#endif
#ifdef VK_KHR_maintenance5
    if (deviceProperties.apiVersion < VK_API_VERSION_1_4 &&
        supportedExtensions.count(VK_KHR_MAINTENANCE_5_EXTENSION_NAME) > 0) {
        descs.push_back(FeatureDescriptor{
            .sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR,
            .extensionName = VK_KHR_MAINTENANCE_5_EXTENSION_NAME,
            .corePromotedVersion = VK_API_VERSION_1_4,
            .makeQueryStruct = [this]() -> std::shared_ptr<void> {
                auto q =
                    std::make_unique<VkPhysicalDeviceMaintenance5FeaturesKHR>();
                *q = makeFeatureStruct<
                    VkPhysicalDeviceMaintenance5FeaturesKHR,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR>();
                return std::shared_ptr<void>(std::move(q));
            },
            .makeEnableStruct = [this](void *q) -> std::shared_ptr<void> {
                auto *feat =
                    static_cast<VkPhysicalDeviceMaintenance5FeaturesKHR *>(q);
                if (!feat->maintenance5)
                    return nullptr;
                auto e =
                    std::make_unique<VkPhysicalDeviceMaintenance5FeaturesKHR>();
                *e = makeFeatureStruct<
                    VkPhysicalDeviceMaintenance5FeaturesKHR,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR>();
                e->maintenance5 = feat->maintenance5;
                return std::shared_ptr<void>(std::move(e));
            }});
    }
#endif

    for (size_t i = 0; i < descs.size(); ++i) {
        const std::unordered_map<int, std::string> desc_names = {
            {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
             "VULKAN_1_1_FEATURES"},
            {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
             "VULKAN_1_2_FEATURES"},
            {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
             "VULKAN_1_3_FEATURES"},
#ifdef VK_VERSION_1_4
            {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES,
             "VULKAN_1_4_FEATURES"},
#endif
        };
        const auto &desc = descs[i];
        (void)desc;
        LOG_INFO("Descriptor %zu: sType=%s, extensionName=%s, coreVersion=%u",
                 i,
                 desc_names.find(desc.sType) != desc_names.end()
                     ? desc_names.find(desc.sType)->second.c_str()
                     : std::to_string(desc.sType).c_str(),
                 desc.extensionName ? desc.extensionName : "nullptr",
                 desc.corePromotedVersion);
    }

    return descs;
}

void VulkanDevice::dumpShaderCapabilities() const {
    const char *env = std::getenv("VKOP_DEVCAPS");
    if (!env || std::strcmp(env, "1") != 0) {
        return;
    }
    auto yes_no = [](bool b) { return b ? "yes" : "no"; };
    auto pr = [&](const char *name, const DeviceCap &c) {
        if (!c.available) {
            // Nothing was queried, so "no" would be a lie: this says the
            // extension did not enumerate (and the feature is not core here).
            std::fprintf(stderr, "[devcaps] %-30s %s\n", name, "not queried");
            return;
        }
        std::fprintf(stderr, "[devcaps] %-30s driver=%-4s device=%-4s\n", name,
                     yes_no(c.supported), yes_no(c.enabled));
    };
    const ShaderCapabilities &c = shader_caps_;
    std::fprintf(stderr, "[devcaps] device=%s\n", deviceName_.c_str());
    std::fprintf(stderr, "[devcaps] vendor=0x%x device=0x%x driver_id=%u\n",
                 device_traits_.vendor_id, device_traits_.device_id,
                 device_traits_.driver_id);
    std::fprintf(stderr,
                 "[devcaps] coopmat=%s fp16_8x8x16=%s fp16_16x16x16=%s "
                 "sint8_8x8x32=%s subgroup_arith=%s subgroup_size=%u\n",
                 yes_no(device_traits_.coopmat),
                 yes_no(device_traits_.coopmat_fp16_8x8x16),
                 yes_no(device_traits_.coopmat_fp16_16x16x16),
                 yes_no(device_traits_.coopmat_sint8_8x8x32),
                 yes_no(device_traits_.subgroup_arithmetic),
                 device_traits_.subgroup_size);
    pr("shader_float16", c.shader_float16);
    pr("shader_int8", c.shader_int8);
    pr("storage_buffer_16bit", c.storage_buffer_16bit);
    pr("storage_buffer_8bit", c.storage_buffer_8bit);
    pr("uniform_storage_buffer_8bit", c.uniform_storage_8bit);
    pr("f32_global_atomic_add", c.f32_global_atomic_add);
    pr("f32_shared_atomic_add", c.f32_shared_atomic_add);
    pr("f16x2_global_atomic_add", c.f16x2_global_atomic_add);
    pr("subgroup_rotate", c.subgroup_rotate);
    pr("subgroup_rotate_clustered", c.subgroup_rotate_clustered);
    pr("wmem_explicit_layout", c.wmem_explicit_layout);
    pr("wmem_explicit_scalar_block", c.wmem_explicit_scalar_block);
    pr("wmem_explicit_8bit_access", c.wmem_explicit_8bit_access);
    pr("wmem_explicit_16bit_access", c.wmem_explicit_16bit_access);
    pr("zero_init_wmem", c.zero_init_wmem);
    std::fflush(stderr);
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
    auto enable_result = buildFeatureEnableChain(
        feature_descs, query_result.features2, supported_extensions);
    LOG_INFO("timeline semaphore %s supported",
             (m_support_timeline_semaphore_ ? "is" : "not"));
    LOG_INFO("host image copy %s supported",
             (m_support_host_image_copy_ ? "is" : "not"));
    LOG_INFO("update after bind %s supported",
             (m_support_descriptor_update_after_bind_ ? "is" : "not"));
#ifdef VK_EXT_external_memory_host
    if (supported_extensions.count(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME) >
        0) {
        m_support_host_import_ = true;
        VkPhysicalDeviceExternalMemoryHostPropertiesEXT host_props = {};
        host_props.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT;
        VkPhysicalDeviceProperties2 props2 = {};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &host_props;
        vkGetPhysicalDeviceProperties2(physicalDevice_, &props2);
        if (host_props.minImportedHostPointerAlignment > 0) {
            m_host_import_alignment_ =
                host_props.minImportedHostPointerAlignment;
        }
        LOG_INFO("host import alignment %u", m_host_import_alignment_);
    }
#endif

    std::vector<std::vector<float>> queue_priority(computeQueueIdxs_.size());
    auto queue_create_info = setupQueueCreateInfo(queue_priority);
    VkDeviceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.queueCreateInfoCount =
        static_cast<uint32_t>(queue_create_info.size());
    create_info.pQueueCreateInfos = queue_create_info.data();
    create_info.enabledLayerCount = 0;
    create_info.ppEnabledLayerNames = nullptr;
    create_info.enabledExtensionCount =
        static_cast<uint32_t>(enable_result.enabledExtensions.size());
    create_info.ppEnabledExtensionNames =
        enable_result.enabledExtensions.data();
    void *pnext = nullptr;
    for (auto it = enable_result.enableChain.rbegin();
         it != enable_result.enableChain.rend(); ++it) {
        auto *base = static_cast<VkBaseOutStructure *>(it->get());
        base->pNext = reinterpret_cast<VkBaseOutStructure *>(pnext);
        pnext = base;
    }
    create_info.pNext = pnext;
    LOG_INFO("enabled extensions count %d",
             enable_result.enabledExtensions.size());
    for (const auto &e : enable_result.enabledExtensions) {
        (void)e;
        LOG_INFO("enabled extension: %s", e);
    }

    auto ret =
        vkCreateDevice(physicalDevice_, &create_info, nullptr, &logicalDevice_);
    if (ret != VK_SUCCESS) {
        LOG_ERROR("Failed to create logical device! ret=%d", ret);
        return false;
    }
    VulkanDebug::set_device(logicalDevice_);

    // Some drivers advertise host image copy yet report an empty memory-type
    // mask for OPTIMAL-tiling images created with HOST_TRANSFER usage (seen on
    // Windows with NVIDIA Pascal), which makes every such image unallocatable.
    // Probe one representative configuration and fall back to staged copies.
    if (m_support_host_image_copy_) {
        VkImageCreateInfo probe_info = {};
        probe_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        probe_info.imageType = VK_IMAGE_TYPE_2D;
        probe_info.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        probe_info.extent = {64, 640, 1};
        probe_info.mipLevels = 1;
        probe_info.arrayLayers = 2;
        probe_info.samples = VK_SAMPLE_COUNT_1_BIT;
        probe_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        probe_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                           VK_IMAGE_USAGE_STORAGE_BIT |
                           VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT;
        probe_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        probe_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImage probe_image = VK_NULL_HANDLE;
        VkResult probe_result =
            vkCreateImage(logicalDevice_, &probe_info, nullptr, &probe_image);
        bool host_copy_usable = false;
        if (probe_result == VK_SUCCESS) {
            VkImageMemoryRequirementsInfo2 req_info = {};
            req_info.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2;
            req_info.image = probe_image;
            VkMemoryRequirements2 req = {};
            req.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
            vkGetImageMemoryRequirements2(logicalDevice_, &req_info, &req);
            host_copy_usable = req.memoryRequirements.memoryTypeBits != 0;
            vkDestroyImage(logicalDevice_, probe_image, nullptr);
        }
        if (!host_copy_usable) {
            m_support_host_image_copy_ = false;
            LOG_WARN("host image copy unusable for OPTIMAL images on this "
                     "driver, falling back to staged copies");
        }
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
                std::make_shared<VulkanQueue>(qidx, queue));
        }
    }
    for (auto [qidx, queue_count, queueflags] : graphicsQueueIdxs_) {
        queue_count = std::min<
            std::tuple_element<1, class std::tuple<unsigned int, unsigned int,
                                                   unsigned int>>::type>(
            queue_count, (kInflight + 1) / 2);
        for (uint32_t i = 0; i < queue_count; i++) {
            VkQueue queue;
            vkGetDeviceQueue(logicalDevice_, qidx, i, &queue);
            graphicsQueues_.emplace_back(
                std::make_shared<VulkanQueue>(qidx, queue));
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

    // A HOST_VISIBLE|DEVICE_LOCAL memory type is NOT sufficient: discrete
    // NVIDIA/AMD expose their BAR as exactly that combination (a 256 MB window
    // of VRAM mapped to the host), which would tag every dGPU as UMA. The
    // reliable discriminator is the heap layout: an iGPU exposes a single
    // DEVICE_LOCAL heap, a dGPU always has a separate non-local system heap.
    bool is_unified_memory = memory_properties.memoryHeapCount == 1;

    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
        const VkMemoryType &mem_type = memory_properties.memoryTypes[i];

        if ((mem_type.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (mem_type.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            LOG_INFO("Host-visible device-local memory type at index: %d", i);
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
