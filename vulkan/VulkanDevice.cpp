// Copyright 2025 @junka
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/DescriptorBufferPool.hpp"
#include "vulkan/VulkanDebug.hpp"
#include "vulkan/VulkanLib.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "include/logger.hpp"

// Pre-1.4 headers do not define the promoted version macro; the extension
// fallbacks below still need to compare against it.
#ifndef VK_API_VERSION_1_4
#define VK_API_VERSION_1_4 VK_MAKE_API_VERSION(0, 1, 4, 0)
#endif

namespace vkop {

// Every Vulkan feature struct the device is asked about, in one pNext chain
// for vkGetPhysicalDeviceFeatures2, and then -- for the ones whose `pick`
// returns true -- in the pNext chain handed to vkCreateDevice.
//
// `pick(query, enable)` copies the bits the engine wants from the queried
// struct into a zeroed enable struct of the same type and returns whether any
// bit is set. Copying only reported bits is what keeps vkCreateDevice from
// failing on a driver that lacks one: a capability reaches the device exactly
// when it was supported. Picks run in insertion order, so a pick may read a
// flag an earlier one set (descriptor buffer reads the BDA verdict).
class FeatureChain {
  public:
    template <typename T, typename Pick>
    void add(VkStructureType stype, const char *extension, Pick pick) {
        auto query = std::make_shared<T>();
        auto enable = std::make_shared<T>();
        query->sType = stype;
        enable->sType = stype;
        entries_.push_back(Entry{extension, query, enable, [query, enable, pick] {
                                     return pick(*query, *enable);
                                 }});
    }

    void query(VkPhysicalDevice pd) {
        VkPhysicalDeviceFeatures2 features2{};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = link(&Entry::query, [](const Entry &) { return true; });
        vkGetPhysicalDeviceFeatures2(pd, &features2);
    }

    // Runs every pick (in order) and returns the head of the enable chain;
    // the extension of every picked entry is appended to `extensions`.
    void *enable(std::vector<const char *> &extensions) {
        for (auto &e : entries_) {
            e.picked = e.pick();
            if (e.picked && e.extension) {
                extensions.push_back(e.extension);
            }
        }
        return link(&Entry::enable, [](const Entry &e) { return e.picked; });
    }

  private:
    struct Entry {
        const char *extension; // nullptr for a core VkPhysicalDeviceVulkanXXFeatures
        // shared_ptr<void> keeps the concrete type's deleter.
        std::shared_ptr<void> query;
        std::shared_ptr<void> enable;
        std::function<bool()> pick;
        bool picked = false;
    };

    template <typename Filter>
    void *link(std::shared_ptr<void> Entry::*which, Filter keep) {
        void *head = nullptr;
        for (auto it = entries_.rbegin(); it != entries_.rend(); ++it) {
            if (!keep(*it)) {
                continue;
            }
            auto *base = static_cast<VkBaseOutStructure *>(((*it).*which).get());
            base->pNext = static_cast<VkBaseOutStructure *>(head);
            head = base;
        }
        return head;
    }

    std::vector<Entry> entries_;
};

namespace {

// Records a probed bit: `supported` = the driver reported it, `enabled` = it
// goes to vkCreateDevice. They are equal because every pick copies the
// reported bit; `available` is set where the entry is added, since that is
// the spot that knows whether the extension enumerated at all.
void recordCap(DeviceCap &cap, VkBool32 reported) {
    cap.supported = reported != VK_FALSE;
    cap.enabled = cap.supported;
}

template <typename... B> bool any(B... bits) {
    return ((bits != VK_FALSE) || ...);
}

// VKOP_NO_BDA is read once, before the feature chain is built, so the feature
// never reaches vkCreateDevice. That matters beyond the shader path: VMA's
// allocator flag and every buffer's SHADER_DEVICE_ADDRESS usage bit are gated
// on the same support flag, and both are invalid when the device feature is
// off.
bool buffer_device_address_disabled() {
    static const bool disabled = std::getenv("VKOP_NO_BDA") != nullptr;
    return disabled;
}

// VKOP_NO_DESCRIPTOR_BUFFER is read once, before the feature chain is built.
// The descriptor-buffer feature must never reach vkCreateDevice when this is
// set, because every descriptor buffer usage bit downstream is invalid then.
bool descriptor_buffer_disabled() {
    static const bool disabled =
        std::getenv("VKOP_NO_DESCRIPTOR_BUFFER") != nullptr;
    return disabled;
}

// VkDevice -> VulkanDevice registry. VulkanBasePipeline is constructed with a
// bare VkDevice handle (26 call sites); this lets it reach the device-scoped
// descriptor-buffer entry points without threading the owner through every
// constructor. Multiple devices can coexist in tests.
//
// Deliberately leaked (process lifetime): a VulkanDevice may itself be a
// function-local static destroyed after this registry in another TU, and
// ~VulkanDevice dereferences the map to unregister. Never destroyed => no
// static-destruction-order UB.
std::mutex &device_registry_mutex() {
    static std::mutex *m = new std::mutex();
    return *m;
}

std::unordered_map<VkDevice, VulkanDevice *> &device_registry() {
    static auto *r = new std::unordered_map<VkDevice, VulkanDevice *>();
    return *r;
}

// Prepends `s` (any Vulkan out-struct) to the chain starting at `head`.
template <typename T> void chain(void *&head, T &s) {
    s.pNext = head;
    head = &s;
}

} // namespace

bool DescriptorBufferInfo::valid() const {
#ifdef VK_EXT_descriptor_buffer
    return getLayoutSize != nullptr && getBindingOffset != nullptr &&
           getDescriptor != nullptr && cmdBindBuffers != nullptr &&
           cmdSetOffsets != nullptr && getBufferAddress != nullptr &&
           storageBufferSize > 0 && storageImageSize > 0 &&
           sampledImageSize > 0 && combinedImageSamplerSize > 0;
#else
    return false;
#endif
}

VkDeviceSize DescriptorBufferInfo::sizeFor(VkDescriptorType type) const {
    switch (type) {
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        return storageBufferSize;
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
        return uniformBufferSize;
    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
        return storageImageSize;
    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        return sampledImageSize;
    case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
        return combinedImageSamplerSize;
    case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
        return storageTexelBufferSize;
    case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
        return uniformTexelBufferSize;
    default:
        return 0;
    }
}

VulkanDevice *VulkanDevice::fromLogicalDevice(VkDevice device) {
    std::lock_guard<std::mutex> lock(device_registry_mutex());
    auto it = device_registry().find(device);
    return it == device_registry().end() ? nullptr : it->second;
}

bool VulkanDevice::supports_update_after_bind(VkDescriptorType type) const {
    switch (type) {
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
        return m_uab_storage_buffer_;
    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
        return m_uab_storage_image_;
    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
    case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
        return m_uab_sampled_image_;
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
        return m_uab_uniform_buffer_;
    default:
        return false;
    }
}

VulkanDevice::VulkanDevice(VkPhysicalDevice physicalDevice)
    : physicalDevice_(physicalDevice) {
    if (physicalDevice_ == VK_NULL_HANDLE) {
        throw std::runtime_error("Invalid Vulkan physical device handle.");
    }
    checkImageFormatSupport();
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
    // NDEBUG compiles LOG_INFO away, which leaves `profile` unused.
    (void)profile;
    LOG_INFO("MatMul profile: %s (byte_min_m=%d fp16_min_m=%d wide_n=%d "
             "ks_target=%d)",
             profile.name, profile.byte_min_m, profile.fp16_min_m,
             profile.wide_n, profile.ks_target_blocks);
    // After device_traits_: the dump wants the resolved facts next to the
    // enabled feature bits. No-op unless VKOP_DEVCAPS=1.
    dumpShaderCapabilities();
    // Resolve the entry point and prove the whole host-side path works before
    // anything (VMA especially) keys on the capability. A driver that reports
    // the feature but cannot allocate an addressable buffer is downgraded
    // here, so no caller ever needs its own deny list.
    initBufferDeviceAddress();
    LOG_INFO("buffer device address %s supported",
             (is_support_buffer_device_address() ? "is" : "not"));
    // After BDA: a descriptor buffer is bound by VkDeviceAddress, so the
    // extension is only usable once the address path proved itself.
    initDescriptorBuffer();
    LOG_INFO("descriptor buffer %s supported",
             (is_support_descriptor_buffer() ? "is" : "not"));
#ifdef USE_VMA
    // Conditional on purpose: VMA's BDA allocator flag makes every allocation
    // carry VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, which is invalid when the
    // device feature was not enabled.
    m_vma_ = std::make_unique<VMA>(physicalDevice_, logicalDevice_,
                                   is_support_buffer_device_address(),
                                   m_support_memory_budget_);
#endif
}

VulkanDevice::~VulkanDevice() {
    computeQueues_.clear();
    // Destroy the shared descriptor buffer while the device is still alive.
    // Pipelines normally hold the last reference; they are gone by now, so
    // this reset is what actually releases the VkBuffer.
    descriptor_buffer_pool_.reset();
#ifdef USE_VMA
    m_vma_.reset();
#endif
    if (logicalDevice_ != VK_NULL_HANDLE) {
        {
            std::lock_guard<std::mutex> lock(device_registry_mutex());
            device_registry().erase(logicalDevice_);
        }
        vkDestroyDevice(logicalDevice_, nullptr);
        logicalDevice_ = VK_NULL_HANDLE;
    }
}

// ---------------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------------

VkPhysicalDeviceProperties VulkanDevice::getProperties() {
    // The driver walks the chain by sType, so order is irrelevant; a struct
    // the driver does not know is left zeroed, which reads as "absent".
    void *head = nullptr;

    VkPhysicalDeviceSubgroupProperties subgroup_properties = {};
    subgroup_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    chain(head, subgroup_properties);

    // driverID is what separates MoltenVK's MSL backend from a native driver
    // on the same vendor ID.
    VkPhysicalDeviceDriverProperties driver_properties = {};
    driver_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
    chain(head, driver_properties);

#ifdef VK_KHR_cooperative_matrix
    VkPhysicalDeviceCooperativeMatrixPropertiesKHR coopmat_properties = {};
    coopmat_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    chain(head, coopmat_properties);
#endif
#ifdef VK_NV_cuda_kernel_launch
    VkPhysicalDeviceCudaKernelLaunchPropertiesNV cuda_properties = {};
    cuda_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUDA_KERNEL_LAUNCH_PROPERTIES_NV;
    chain(head, cuda_properties);
#endif
#ifdef VK_EXT_host_image_copy
    VkPhysicalDeviceHostImageCopyPropertiesEXT host_copy_properties = {};
    host_copy_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_PROPERTIES_EXT;
    chain(head, host_copy_properties);
#endif
#ifdef VK_EXT_subgroup_size_control
    // Queried so the pipeline can request a fixed requiredSubgroupSize; the
    // Intel ANV driver is free to pick any subgroupSize in [min,max] when it
    // is not pinned, which breaks shaders that hard-assume a specific
    // numSubgroups (e.g. softmax.comp).
    VkPhysicalDeviceSubgroupSizeControlPropertiesEXT ssc_properties = {};
    ssc_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT;
    chain(head, ssc_properties);
#endif
#ifdef VK_EXT_descriptor_buffer
    // Descriptor sizes and the set-offset alignment the descriptor-buffer
    // backend must honour.
    VkPhysicalDeviceDescriptorBufferPropertiesEXT db_properties = {};
    db_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT;
    chain(head, db_properties);
#endif
#ifdef VK_KHR_shader_integer_dot_product
    // Whether dot() on a packed 4x8 vector is a hardware instruction is a
    // PROPERTY, not a feature: the feature bit only grants permission to put
    // `OpCapability DotProduct` in a module. The sType is the same number in
    // its 1.3-core and KHR spellings, so one struct answers either way.
    VkPhysicalDeviceShaderIntegerDotProductProperties dot_properties = {};
    dot_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES;
    chain(head, dot_properties);
#endif
#ifdef VK_EXT_external_memory_host
    VkPhysicalDeviceExternalMemoryHostPropertiesEXT host_import_properties = {};
    host_import_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT;
    chain(head, host_import_properties);
#endif

    VkPhysicalDeviceProperties2 properties2 = {};
    properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties2.pNext = head;
    vkGetPhysicalDeviceProperties2(physicalDevice_, &properties2);
    const VkPhysicalDeviceLimits &limits = properties2.properties.limits;

#ifdef VK_EXT_host_image_copy
    // Layout arrays take a second pass: counts first, then the arrays.
    if (host_copy_properties.copySrcLayoutCount > 0 ||
        host_copy_properties.copyDstLayoutCount > 0) {
        copySrcLayout_.resize(host_copy_properties.copySrcLayoutCount);
        copyDstLayout_.resize(host_copy_properties.copyDstLayoutCount);
        host_copy_properties.pCopySrcLayouts = copySrcLayout_.data();
        host_copy_properties.pCopyDstLayouts = copyDstLayout_.data();
        vkGetPhysicalDeviceProperties2(physicalDevice_, &properties2);
    }
    LOG_INFO("HostImageCopy identicalMemoryTypeRequirements %s",
             host_copy_properties.identicalMemoryTypeRequirements ? "True"
                                                                  : "False");
    for (auto l : copySrcLayout_) {
        (void)l;
        LOG_INFO("HostImageCopy support src layout %d", l);
    }
    for (auto l : copyDstLayout_) {
        (void)l;
        LOG_INFO("HostImageCopy support dst layout %d", l);
    }
#endif
#ifdef VK_KHR_shader_integer_dot_product
    m_dot_product_props_ = dot_properties;
    m_dot_product_packed_signed_ =
        dot_properties.integerDotProduct4x8BitPackedSignedAccelerated !=
        VK_FALSE;
#endif
#ifdef VK_EXT_descriptor_buffer
    m_descriptor_buffer_props_ = db_properties;
#endif
#ifdef VK_EXT_external_memory_host
    // Only meaningful when the extension enumerates; createLogicalDevice()
    // decides that. minImportedHostPointerAlignment is 4096 on MoltenVK.
    if (host_import_properties.minImportedHostPointerAlignment > 0) {
        m_host_import_alignment_ = static_cast<uint32_t>(
            host_import_properties.minImportedHostPointerAlignment);
    }
#endif

    timestampPeriod_ = limits.timestampPeriod;
    maxImageArrayLayers_ = limits.maxImageArrayLayers;
    maxImageDimension2D_ = limits.maxImageDimension2D;
    min_sb_offset_align_ =
        std::max<VkDeviceSize>(1, limits.minStorageBufferOffsetAlignment);
    non_coherent_atom_size_ =
        std::max<VkDeviceSize>(1, limits.nonCoherentAtomSize);
    deviceName_ = properties2.properties.deviceName;
    LOG_INFO("GPU %s", deviceName_.c_str());
    LOG_INFO("Driver ID %d (vendor 0x%x)", driver_properties.driverID,
             properties2.properties.vendorID);
    LOG_INFO("Max image array layers %d", maxImageArrayLayers_);
    LOG_INFO("Max image dimension 2D %d", maxImageDimension2D_);
    LOG_INFO("Min TexelBuffer Alignment %llu",
             limits.minTexelBufferOffsetAlignment);
#ifdef VK_EXT_subgroup_size_control
    minSubgroupSize_ = ssc_properties.minSubgroupSize;
    maxSubgroupSize_ = ssc_properties.maxSubgroupSize;
    LOG_INFO("Subgroup size control min %u max %u", minSubgroupSize_,
             maxSubgroupSize_);
#endif
    // Feeds the *_sg shader variant decision (Runtime::LoadModel and
    // Operator::resolve_primary_spv).
    m_support_subgroup_arithmetic_ = (subgroup_properties.supportedOperations &
                                      VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
    if (m_support_subgroup_arithmetic_) {
        LOG_INFO("Device support subgroup arithmetic, subgroup size %d",
                 subgroup_properties.subgroupSize);
    }
#ifdef VK_KHR_cooperative_matrix
    m_coopmat_supported_stages_ =
        coopmat_properties.cooperativeMatrixSupportedStages;
    LOG_INFO("Cooperative matrix supported stages 0x%x",
             m_coopmat_supported_stages_);
    queryCooperativeMatrixProperties();
#endif
#ifdef VK_NV_cuda_kernel_launch
    LOG_INFO("CUDA kernel launch compute capability %u.%u",
             cuda_properties.computeCapabilityMajor,
             cuda_properties.computeCapabilityMinor);
#endif
    return properties2.properties;
}

#ifdef VK_KHR_cooperative_matrix
void VulkanDevice::queryCooperativeMatrixProperties() {
    // Enumerate every (M, N, K, type, scope) combination the device supports.
    // RenderDoc's capture layer (up to and including 1.47) never installs a
    // device entry point for this query: the loader trampoline tail-jumps
    // into an empty dispatch-table slot and the process SIGSEGVs. The list
    // only feeds the coopmat tile-shape selectors (supports_coopmat_*_tile),
    // so skipping it under capture degrades routing exactly like a
    // non-coopmat device. Define VKOP_DISABLE_COOPMAT_RDOC_GUARD to compile
    // the guard out, or set VKOP_RDOC_COOPMAT_ENUM=1 to force the enumeration
    // at runtime.
#ifndef VKOP_DISABLE_COOPMAT_RDOC_GUARD
    static const bool skipped =
        std::getenv("ENABLE_VULKAN_RENDERDOC_CAPTURE") != nullptr &&
        std::getenv("VKOP_RDOC_COOPMAT_ENUM") == nullptr;
#else
    static constexpr bool skipped = false;
#endif
    if (skipped) {
        LOG_WARN("RenderDoc capture layer active: skipping "
                 "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR "
                 "(layer <= 1.47 lacks its device entry point); coopmat "
                 "tile probing disabled");
        return;
    }
    if (m_coopmat_supported_stages_ == 0) {
        return;
    }
    auto fn = reinterpret_cast<
        PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
        vkGetInstanceProcAddr(
            VulkanInstance::getVulkanInstance().getInstance(),
            "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
    if (!fn) {
        return;
    }
    uint32_t count = 0;
    fn(physicalDevice_, &count, nullptr);
    VkCooperativeMatrixPropertiesKHR blank = {};
    blank.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    coopmatProps_.assign(count, blank);
    if (count > 0) {
        fn(physicalDevice_, &count, coopmatProps_.data());
    }
    for (uint32_t i = 0; i < count; i++) {
        const auto &p = coopmatProps_[i];
        (void)p;
        LOG_INFO("CoopMat [%u]: %ux%ux%u AType=%d BType=%d CType=%d "
                 "ResultType=%d scope=%d",
                 i, p.MSize, p.NSize, p.KSize, p.AType, p.BType, p.CType,
                 p.ResultType, p.scope);
    }
}
#endif

// ---------------------------------------------------------------------------
// Queues
// ---------------------------------------------------------------------------

uint32_t VulkanDevice::findComputeQueueFamily() {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &count,
                                             families.data());

    const VkQueueFlags compute = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
    const VkQueueFlags graphics = compute | VK_QUEUE_GRAPHICS_BIT;

    for (uint32_t i = 0; i < count; i++) {
        const VkQueueFlags flags = families[i].queueFlags;
        const QueueFamily f{i, families[i].queueCount, flags};
        if (kInflight > 1 && (flags & graphics) == graphics) {
            computeQueueIdxs_.emplace_back(f);
            graphicsQueueIdxs_.emplace_back(f);
            LOG_INFO("Found graphics+compute queue family %d with %d queues", i,
                     families[i].queueCount);
        } else if ((flags & compute) == compute) {
            computeQueueIdxs_.emplace_back(f);
            LOG_INFO("Found compute queue family %d with %d queues", i,
                     families[i].queueCount);
        }
    }
    return static_cast<uint32_t>(computeQueueIdxs_.size());
}

// ---------------------------------------------------------------------------
// Image backend probe
// ---------------------------------------------------------------------------

// The image backend packs tensors into four RGBA formats (Tensor::make_vkimg)
// and creates them as OPTIMAL-tiled sampled+storage images that are uploaded
// and read back through transfers; it also exposes them as texel buffers
// (VulkanBufferView). Any missing combination turns the image backend off
// (and records why) instead of asserting, so an image-less device (the
// Alibaba PPU) still runs the buffer backend.
//
// Deliberately the Vulkan 1.0 vkGetPhysicalDeviceFormatProperties and not the
// FormatProperties2/3 chain: the pNext chain is the part an unknown driver is
// most likely to mishandle, and nothing here needs a 64-bit feature bit. Host
// image transfer is NOT required: createLogicalDevice() probes it for real and
// the staged path covers its absence.
void VulkanDevice::checkImageFormatSupport() {
    struct FormatReq {
        VkFormat format;
        const char *name;
    };
    const FormatReq formats[] = {
        {VK_FORMAT_R32G32B32A32_SFLOAT, "RGBA32_SFLOAT"},
        {VK_FORMAT_R16G16B16A16_SFLOAT, "RGBA16_SFLOAT"},
        {VK_FORMAT_R8G8B8A8_SNORM, "RGBA8_SNORM"},
        {VK_FORMAT_R8G8B8A8_UNORM, "RGBA8_UNORM"},
    };
    const VkFormatFeatureFlags optimal_required =
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
        VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT |
        VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    const VkFormatFeatureFlags buffer_any =
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
        VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT;
    const VkImageUsageFlags usage =
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    auto fail = [this](const char *name, const std::string &why) {
        LOG_WARN("no image support: format %s: %s", name, why.c_str());
        image_unsupported_reason_ = std::string(name) + ": " + why;
    };

    for (const FormatReq &fr : formats) {
        VkFormatProperties props{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice_, fr.format,
                                            &props);
        if ((props.bufferFeatures & buffer_any) == 0) {
            fail(fr.name,
                 "bufferFeatures lack sampled-image|storage-texel-buffer");
            return;
        }
        if ((props.optimalTilingFeatures & optimal_required) !=
            optimal_required) {
            fail(fr.name, "optimalTilingFeatures 0x" +
                              std::to_string(props.optimalTilingFeatures) +
                              " lack sampled|storage|transfer-src|"
                              "transfer-dst");
            return;
        }
        VkImageFormatProperties image_props{};
        VkResult r = vkGetPhysicalDeviceImageFormatProperties(
            physicalDevice_, fr.format, VK_IMAGE_TYPE_2D,
            VK_IMAGE_TILING_OPTIMAL, usage, 0, &image_props);
        if (r != VK_SUCCESS) {
            fail(fr.name, "vkGetPhysicalDeviceImageFormatProperties returned " +
                              std::to_string(static_cast<int>(r)) +
                              " for a sampled|storage|transfer image");
            return;
        }
        LOG_INFO("format %s supports %u layers", fr.name,
                 image_props.maxArrayLayers);
    }
    m_support_images_ = true;
    // The probe passed, but VKOP_NO_IMAGES can still force the backend off.
    LOG_INFO("image backend %s", is_support_images()
                                     ? "supported"
                                     : "disabled by VKOP_NO_IMAGES");
}

// ---------------------------------------------------------------------------
// Feature probes
// ---------------------------------------------------------------------------

void VulkanDevice::addFeatureProbes(FeatureChain &fc, uint32_t api_version,
                                    const std::set<std::string> &extensions) {
    auto has = [&](const char *name) { return extensions.count(name) > 0; };
    ShaderCapabilities &sc = shader_caps_;

    // ---- core feature structs ----------------------------------------------
    // Each promoted feature is reached through exactly one struct: the core
    // VkPhysicalDeviceVulkanXXFeatures when api_version covers it, otherwise
    // the extension struct further down. Chaining both for one feature is
    // invalid (VUID-VkDeviceCreateInfo-pNext-02829/02830/06532/...), which
    // is why every extension fallback tests `api_version <` its promotion.
#ifdef VK_VERSION_1_1
    if (api_version >= VK_API_VERSION_1_1) {
        sc.storage_buffer_16bit.available = true;
        fc.add<VkPhysicalDeviceVulkan11Features>(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, nullptr,
            [this](const auto &q, auto &e) {
                recordCap(shader_caps_.storage_buffer_16bit,
                          q.storageBuffer16BitAccess);
                e.storageBuffer16BitAccess = q.storageBuffer16BitAccess;
                e.storageInputOutput16 = q.storageInputOutput16;
                e.uniformAndStorageBuffer16BitAccess =
                    q.uniformAndStorageBuffer16BitAccess;
                return any(e.storageBuffer16BitAccess, e.storageInputOutput16,
                           e.uniformAndStorageBuffer16BitAccess);
            });
    }
#endif
#ifdef VK_VERSION_1_2
    if (api_version >= VK_API_VERSION_1_2) {
        sc.shader_float16.available = true;
        sc.shader_int8.available = true;
        sc.storage_buffer_8bit.available = true;
        sc.uniform_storage_8bit.available = true;
        sc.descriptor_indexing.available = true;
        sc.descriptor_partially_bound.available = true;
        fc.add<VkPhysicalDeviceVulkan12Features>(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, nullptr,
            [this](const auto &q, auto &e) {
                auto &c = shader_caps_;
                recordCap(c.shader_float16, q.shaderFloat16);
                recordCap(c.shader_int8, q.shaderInt8);
                recordCap(c.storage_buffer_8bit, q.storageBuffer8BitAccess);
                recordCap(c.uniform_storage_8bit,
                          q.uniformAndStorageBuffer8BitAccess);
                e.shaderFloat16 = q.shaderFloat16;
                e.shaderInt8 = q.shaderInt8;
                // Only the storage-buffer 8-bit forms: storagePushConstant8
                // would widen the push constant range nothing uses.
                e.storageBuffer8BitAccess = q.storageBuffer8BitAccess;
                e.uniformAndStorageBuffer8BitAccess =
                    q.uniformAndStorageBuffer8BitAccess;
                e.timelineSemaphore = q.timelineSemaphore;
                m_support_timeline_semaphore_ = q.timelineSemaphore;
                // Everything downstream (VMA allocator flag, per-buffer usage
                // bit, BDA shader variants) keys on this enabled bit, so a
                // known-broken driver has one switch: VKOP_NO_BDA.
                e.bufferDeviceAddress =
                    q.bufferDeviceAddress && !buffer_device_address_disabled();
                m_support_buffer_device_address_ = e.bufferDeviceAddress;
                pickDescriptorIndexing(q, e);
                return any(e.shaderFloat16, e.shaderInt8,
                           e.storageBuffer8BitAccess,
                           e.uniformAndStorageBuffer8BitAccess,
                           e.timelineSemaphore, e.bufferDeviceAddress,
                           e.descriptorBindingStorageBufferUpdateAfterBind,
                           e.descriptorBindingStorageImageUpdateAfterBind,
                           e.descriptorBindingSampledImageUpdateAfterBind,
                           e.descriptorBindingUniformBufferUpdateAfterBind,
                           e.descriptorBindingPartiallyBound);
            });
    }
#endif
#ifdef VK_VERSION_1_3
    if (api_version >= VK_API_VERSION_1_3) {
        sc.zero_init_wmem.available = true;
        // VK_KHR_shader_integer_dot_product is core in 1.3, so on a 1.3+
        // device the bit lives here. Without it, any module carrying
        // OpCapability DotProduct is rejected at pipeline creation (b3:
        // instance 1.3.275, device 1.4.329 lands in this branch).
        sc.integer_dot_product.available = true;
        fc.add<VkPhysicalDeviceVulkan13Features>(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, nullptr,
            [this](const auto &q, auto &e) {
                recordCap(shader_caps_.zero_init_wmem,
                          q.shaderZeroInitializeWorkgroupMemory);
                recordCap(shader_caps_.integer_dot_product,
                          q.shaderIntegerDotProduct);
                e.robustImageAccess = q.robustImageAccess;
                e.subgroupSizeControl = q.subgroupSizeControl;
                e.maintenance4 = q.maintenance4;
                e.shaderZeroInitializeWorkgroupMemory =
                    q.shaderZeroInitializeWorkgroupMemory;
                e.shaderIntegerDotProduct = q.shaderIntegerDotProduct;
                return any(e.robustImageAccess, e.subgroupSizeControl,
                           e.maintenance4,
                           e.shaderZeroInitializeWorkgroupMemory,
                           e.shaderIntegerDotProduct);
            });
    }
#endif
#ifdef VK_VERSION_1_4
    if (api_version >= VK_API_VERSION_1_4) {
        sc.subgroup_rotate.available = true;
        sc.subgroup_rotate_clustered.available = true;
        fc.add<VkPhysicalDeviceVulkan14Features>(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES, nullptr,
            [this](const auto &q, auto &e) {
                recordCap(shader_caps_.subgroup_rotate, q.shaderSubgroupRotate);
                recordCap(shader_caps_.subgroup_rotate_clustered,
                          q.shaderSubgroupRotateClustered);
                e.hostImageCopy = q.hostImageCopy;
                m_support_host_image_copy_ = q.hostImageCopy;
                e.maintenance5 = q.maintenance5;
                // Deliberately NOT pushDescriptor: measured host-side only,
                // see the [dswrite] probe notes.
                e.shaderSubgroupRotate = q.shaderSubgroupRotate;
                // Same pair as the KHR fallback enables; [devcaps] already
                // reported it as enabled on this path.
                e.shaderSubgroupRotateClustered =
                    q.shaderSubgroupRotateClustered;
                return any(e.hostImageCopy, e.maintenance5,
                           e.shaderSubgroupRotate);
            });
    }
#endif

    // ---- extension fallbacks for promoted features -------------------------
#ifdef VK_KHR_16bit_storage
    if (api_version < VK_API_VERSION_1_1 &&
        has(VK_KHR_16BIT_STORAGE_EXTENSION_NAME)) {
        sc.storage_buffer_16bit.available = true;
        fc.add<VkPhysicalDevice16BitStorageFeaturesKHR>(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR,
            VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
            [this](const auto &q, auto &e) {
                recordCap(shader_caps_.storage_buffer_16bit,
                          q.storageBuffer16BitAccess);
                e.storageBuffer16BitAccess = q.storageBuffer16BitAccess;
                e.storageInputOutput16 = q.storageInputOutput16;
                e.uniformAndStorageBuffer16BitAccess =
                    q.uniformAndStorageBuffer16BitAccess;
                return any(e.storageBuffer16BitAccess, e.storageInputOutput16,
                           e.uniformAndStorageBuffer16BitAccess);
            });
    }
#endif
#ifdef VK_KHR_8bit_storage
    if (api_version < VK_API_VERSION_1_2 &&
        has(VK_KHR_8BIT_STORAGE_EXTENSION_NAME)) {
        sc.storage_buffer_8bit.available = true;
        sc.uniform_storage_8bit.available = true;
        fc.add<VkPhysicalDevice8BitStorageFeaturesKHR>(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR,
            VK_KHR_8BIT_STORAGE_EXTENSION_NAME,
            [this](const auto &q, auto &e) {
                recordCap(shader_caps_.storage_buffer_8bit,
                          q.storageBuffer8BitAccess);
                recordCap(shader_caps_.uniform_storage_8bit,
                          q.uniformAndStorageBuffer8BitAccess);
                e.storageBuffer8BitAccess = q.storageBuffer8BitAccess;
                e.uniformAndStorageBuffer8BitAccess =
                    q.uniformAndStorageBuffer8BitAccess;
                return any(e.storageBuffer8BitAccess,
                           e.uniformAndStorageBuffer8BitAccess);
            });
    }
#endif
#ifdef VK_KHR_shader_float16_int8
    // Promoted in 1.2 (shaderFloat16/shaderInt8 live in Vulkan12Features).
    // This used to be gated `< 1.3`, which on a 1.2 device chained this
    // struct next to Vulkan12Features -- invalid per VUID 02830.
    if (api_version < VK_API_VERSION_1_2 &&
        has(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME)) {
        sc.shader_float16.available = true;
        sc.shader_int8.available = true;
        fc.add<VkPhysicalDeviceShaderFloat16Int8FeaturesKHR>(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR,
            VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
            [this](const auto &q, auto &e) {
                recordCap(shader_caps_.shader_float16, q.shaderFloat16);
                recordCap(shader_caps_.shader_int8, q.shaderInt8);
                e.shaderFloat16 = q.shaderFloat16;
                e.shaderInt8 = q.shaderInt8;
                return any(e.shaderFloat16, e.shaderInt8);
            });
    }
#endif
#ifdef VK_KHR_timeline_semaphore
    if (api_version < VK_API_VERSION_1_2 &&
        has(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME)) {
        fc.add<VkPhysicalDeviceTimelineSemaphoreFeatures>(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
            VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
            [this](const auto &q, auto &e) {
                e.timelineSemaphore = q.timelineSemaphore;
                m_support_timeline_semaphore_ = q.timelineSemaphore;
                return any(e.timelineSemaphore);
            });
    }
#endif
#ifdef VK_KHR_buffer_device_address
    if (api_version < VK_API_VERSION_1_2 &&
        has(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)) {
        fc.add<VkPhysicalDeviceBufferDeviceAddressFeatures>(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
            VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
            [this](const auto &q, auto &e) {
                e.bufferDeviceAddress =
                    q.bufferDeviceAddress && !buffer_device_address_disabled();
                m_support_buffer_device_address_ = e.bufferDeviceAddress;
                return any(e.bufferDeviceAddress);
            });
    }
#endif
#ifdef VK_EXT_descriptor_indexing
    if (api_version < VK_API_VERSION_1_2 &&
        has(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME)) {
        sc.descriptor_indexing.available = true;
        sc.descriptor_partially_bound.available = true;
        fc.add<VkPhysicalDeviceDescriptorIndexingFeaturesEXT>(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES,
            VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
            [this](const auto &q, auto &e) {
                pickDescriptorIndexing(q, e);
                return any(e.descriptorBindingStorageBufferUpdateAfterBind,
                           e.descriptorBindingStorageImageUpdateAfterBind,
                           e.descriptorBindingSampledImageUpdateAfterBind,
                           e.descriptorBindingUniformBufferUpdateAfterBind,
                           e.descriptorBindingPartiallyBound);
            });
    }
#endif
#ifdef VK_KHR_shader_integer_dot_product
    if (api_version < VK_API_VERSION_1_3 &&
        has(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME)) {
        sc.integer_dot_product.available = true;
        fc.add<VkPhysicalDeviceShaderIntegerDotProductFeatures>(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR,
            VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME,
            [this](const auto &q, auto &e) {
                recordCap(shader_caps_.integer_dot_product,
                          q.shaderIntegerDotProduct);
                e.shaderIntegerDotProduct = q.shaderIntegerDotProduct;
                return any(e.shaderIntegerDotProduct);
            });
    }
#endif
#ifdef VK_KHR_zero_initialize_workgroup_memory
    if (api_version < VK_API_VERSION_1_3 &&
        has(VK_KHR_ZERO_INITIALIZE_WORKGROUP_MEMORY_EXTENSION_NAME)) {
        sc.zero_init_wmem.available = true;
        fc.add<VkPhysicalDeviceZeroInitializeWorkgroupMemoryFeaturesKHR>(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ZERO_INITIALIZE_WORKGROUP_MEMORY_FEATURES_KHR,
            VK_KHR_ZERO_INITIALIZE_WORKGROUP_MEMORY_EXTENSION_NAME,
            [this](const auto &q, auto &e) {
                recordCap(shader_caps_.zero_init_wmem,
                          q.shaderZeroInitializeWorkgroupMemory);
                e.shaderZeroInitializeWorkgroupMemory =
                    q.shaderZeroInitializeWorkgroupMemory;
                return any(e.shaderZeroInitializeWorkgroupMemory);
            });
    }
#endif
#ifdef VK_EXT_image_robustness
    if (api_version < VK_API_VERSION_1_3 &&
        has(VK_EXT_IMAGE_ROBUSTNESS_EXTENSION_NAME)) {
        fc.add<VkPhysicalDeviceImageRobustnessFeatures>(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES,
            VK_EXT_IMAGE_ROBUSTNESS_EXTENSION_NAME,
            [](const auto &q, auto &e) {
                e.robustImageAccess = q.robustImageAccess;
                return any(e.robustImageAccess);
            });
    }
#endif
#ifdef VK_EXT_subgroup_size_control
    if (api_version < VK_API_VERSION_1_3 &&
        has(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME)) {
        fc.add<VkPhysicalDeviceSubgroupSizeControlFeaturesEXT>(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT,
            VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME,
            [](const auto &q, auto &e) {
                e.subgroupSizeControl = q.subgroupSizeControl;
                return any(e.subgroupSizeControl);
            });
    }
#endif
#ifdef VK_KHR_shader_subgroup_rotate
    if (api_version < VK_API_VERSION_1_4 &&
        has(VK_KHR_SHADER_SUBGROUP_ROTATE_EXTENSION_NAME)) {
        sc.subgroup_rotate.available = true;
        sc.subgroup_rotate_clustered.available = true;
        fc.add<VkPhysicalDeviceShaderSubgroupRotateFeaturesKHR>(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_ROTATE_FEATURES_KHR,
            VK_KHR_SHADER_SUBGROUP_ROTATE_EXTENSION_NAME,
            [this](const auto &q, auto &e) {
                recordCap(shader_caps_.subgroup_rotate, q.shaderSubgroupRotate);
                recordCap(shader_caps_.subgroup_rotate_clustered,
                          q.shaderSubgroupRotateClustered);
                e.shaderSubgroupRotate = q.shaderSubgroupRotate;
                e.shaderSubgroupRotateClustered =
                    q.shaderSubgroupRotateClustered;
                return any(e.shaderSubgroupRotate);
            });
    }
#endif
#ifdef VK_EXT_host_image_copy
    if (api_version < VK_API_VERSION_1_4 &&
        has(VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME)) {
        fc.add<VkPhysicalDeviceHostImageCopyFeaturesEXT>(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES_EXT,
            VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME,
            [this](const auto &q, auto &e) {
                e.hostImageCopy = q.hostImageCopy;
                m_support_host_image_copy_ = q.hostImageCopy;
                return any(e.hostImageCopy);
            });
    }
#endif
#ifdef VK_KHR_maintenance5
    if (api_version < VK_API_VERSION_1_4 &&
        has(VK_KHR_MAINTENANCE_5_EXTENSION_NAME)) {
        fc.add<VkPhysicalDeviceMaintenance5FeaturesKHR>(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR,
            VK_KHR_MAINTENANCE_5_EXTENSION_NAME, [](const auto &q, auto &e) {
                e.maintenance5 = q.maintenance5;
                return any(e.maintenance5);
            });
    }
#endif

    // ---- extensions never promoted -----------------------------------------
#ifdef VK_EXT_shader_atomic_float
    // fp32 atomic add on a storage buffer is what would collapse the two-pass
    // split-K (deposit partials + fold) into one pass. The float64 and
    // image/sparse variants stay off.
    if (has(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME)) {
        sc.f32_global_atomic_add.available = true;
        sc.f32_shared_atomic_add.available = true;
        fc.add<VkPhysicalDeviceShaderAtomicFloatFeaturesEXT>(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
            VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME,
            [this](const auto &q, auto &e) {
                recordCap(shader_caps_.f32_global_atomic_add,
                          q.shaderBufferFloat32AtomicAdd);
                recordCap(shader_caps_.f32_shared_atomic_add,
                          q.shaderSharedFloat32AtomicAdd);
                e.shaderBufferFloat32AtomicAdd = q.shaderBufferFloat32AtomicAdd;
                e.shaderSharedFloat32AtomicAdd = q.shaderSharedFloat32AtomicAdd;
                return any(e.shaderBufferFloat32AtomicAdd,
                           e.shaderSharedFloat32AtomicAdd);
            });
    }
#endif
#ifdef VK_NV_shader_atomic_float16_vector
    // Half2 global atomic add. Reported on hardware that also has
    // shaderFloat16 clear (Pascal does), so read the [devcaps] pair together.
    if (has(VK_NV_SHADER_ATOMIC_FLOAT16_VECTOR_EXTENSION_NAME)) {
        sc.f16x2_global_atomic_add.available = true;
        fc.add<VkPhysicalDeviceShaderAtomicFloat16VectorFeaturesNV>(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT16_VECTOR_FEATURES_NV,
            VK_NV_SHADER_ATOMIC_FLOAT16_VECTOR_EXTENSION_NAME,
            [this](const auto &q, auto &e) {
                recordCap(shader_caps_.f16x2_global_atomic_add,
                          q.shaderFloat16VectorAtomics);
                e.shaderFloat16VectorAtomics = q.shaderFloat16VectorAtomics;
                return any(e.shaderFloat16VectorAtomics);
            });
    }
#endif
#ifdef VK_KHR_workgroup_memory_explicit_layout
    // One shared block with an explicit layout lets the tile buffers and the
    // split-K accumulator alias the same bytes. The 8/16-bit sub-features
    // only mean something on top of the matching storage features.
    if (has(VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME)) {
        sc.wmem_explicit_layout.available = true;
        sc.wmem_explicit_scalar_block.available = true;
        sc.wmem_explicit_8bit_access.available = true;
        sc.wmem_explicit_16bit_access.available = true;
        fc.add<VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR>(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR,
            VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME,
            [this](const auto &q, auto &e) {
                auto &c = shader_caps_;
                recordCap(c.wmem_explicit_layout,
                          q.workgroupMemoryExplicitLayout);
                recordCap(c.wmem_explicit_scalar_block,
                          q.workgroupMemoryExplicitLayoutScalarBlockLayout);
                recordCap(c.wmem_explicit_8bit_access,
                          q.workgroupMemoryExplicitLayout8BitAccess);
                recordCap(c.wmem_explicit_16bit_access,
                          q.workgroupMemoryExplicitLayout16BitAccess);
                e.workgroupMemoryExplicitLayout =
                    q.workgroupMemoryExplicitLayout;
                e.workgroupMemoryExplicitLayoutScalarBlockLayout =
                    q.workgroupMemoryExplicitLayoutScalarBlockLayout;
                e.workgroupMemoryExplicitLayout8BitAccess =
                    q.workgroupMemoryExplicitLayout8BitAccess;
                e.workgroupMemoryExplicitLayout16BitAccess =
                    q.workgroupMemoryExplicitLayout16BitAccess;
                return any(e.workgroupMemoryExplicitLayout);
            });
    }
#endif
#ifdef VK_EXT_shader_float8
    // Not promoted into Vulkan 1.4, so there is no core branch; gated on the
    // header (1.3.275 headers lack the struct) and on the extension. Float8EXT
    // is a module-wide capability, so the hardware fp8 kernel is a separate
    // module (-DFP8_HW on buffer/matmul.comp). The coopmat bit is separate
    // because a device can decode fp8 without an fp8 MMA.
    if (has(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME)) {
        sc.shader_float8.available = true;
        sc.shader_float8_coopmat.available = true;
        fc.add<VkPhysicalDeviceShaderFloat8FeaturesEXT>(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT,
            VK_EXT_SHADER_FLOAT8_EXTENSION_NAME,
            [this](const auto &q, auto &e) {
                recordCap(shader_caps_.shader_float8, q.shaderFloat8);
                recordCap(shader_caps_.shader_float8_coopmat,
                          q.shaderFloat8CooperativeMatrix);
                e.shaderFloat8 = q.shaderFloat8;
                e.shaderFloat8CooperativeMatrix =
                    q.shaderFloat8CooperativeMatrix;
                return any(e.shaderFloat8, e.shaderFloat8CooperativeMatrix);
            });
    }
#endif
#ifdef VK_AMD_device_coherent_memory
    if (has(VK_AMD_DEVICE_COHERENT_MEMORY_EXTENSION_NAME)) {
        fc.add<VkPhysicalDeviceCoherentMemoryFeaturesAMD>(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COHERENT_MEMORY_FEATURES_AMD,
            VK_AMD_DEVICE_COHERENT_MEMORY_EXTENSION_NAME,
            [](const auto &q, auto &e) {
                e.deviceCoherentMemory = q.deviceCoherentMemory;
                return any(e.deviceCoherentMemory);
            });
    }
#endif
#ifdef VK_KHR_performance_query
    if (has(VK_KHR_PERFORMANCE_QUERY_EXTENSION_NAME)) {
        fc.add<VkPhysicalDevicePerformanceQueryFeaturesKHR>(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PERFORMANCE_QUERY_FEATURES_KHR,
            VK_KHR_PERFORMANCE_QUERY_EXTENSION_NAME,
            [](const auto &q, auto &e) {
                e.performanceCounterQueryPools = q.performanceCounterQueryPools;
                return any(e.performanceCounterQueryPools);
            });
    }
#endif
#ifdef VK_EXT_memory_priority
    if (has(VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME)) {
        fc.add<VkPhysicalDeviceMemoryPriorityFeaturesEXT>(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT,
            VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME, [](const auto &q, auto &e) {
                e.memoryPriority = q.memoryPriority;
                return any(e.memoryPriority);
            });
    }
#endif
#ifdef VK_KHR_cooperative_matrix
    if (has(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) {
        fc.add<VkPhysicalDeviceCooperativeMatrixFeaturesKHR>(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
            VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME,
            [this](const auto &q, auto &e) {
                // MoltenVK advertises the extension + feature but reports 0
                // supported stages, and any coopmat pipeline then dies in
                // SPIR-V -> MSL conversion; gate on the reported stages too.
                e.cooperativeMatrix =
                    q.cooperativeMatrix && m_coopmat_supported_stages_ != 0;
                m_support_cooperate_matrix_ = e.cooperativeMatrix;
                return any(e.cooperativeMatrix);
            });
    }
#endif
#ifdef VK_NV_cooperative_matrix
    if (has(VK_NV_COOPERATIVE_MATRIX_EXTENSION_NAME)) {
        fc.add<VkPhysicalDeviceCooperativeMatrixFeaturesNV>(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_NV,
            VK_NV_COOPERATIVE_MATRIX_EXTENSION_NAME,
            [this](const auto &q, auto &e) {
                e.cooperativeMatrix = q.cooperativeMatrix;
                m_support_nv_tensor_core_ = q.cooperativeMatrix;
                return any(e.cooperativeMatrix);
            });
    }
#endif
#ifdef VK_NV_cuda_kernel_launch
    if (has(VK_NV_CUDA_KERNEL_LAUNCH_EXTENSION_NAME)) {
        fc.add<VkPhysicalDeviceCudaKernelLaunchFeaturesNV>(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUDA_KERNEL_LAUNCH_FEATURES_NV,
            VK_NV_CUDA_KERNEL_LAUNCH_EXTENSION_NAME,
            [this](const auto &q, auto &e) {
                e.cudaKernelLaunchFeatures = q.cudaKernelLaunchFeatures;
                m_support_cuda_kernel_launch_ = q.cudaKernelLaunchFeatures;
                return any(e.cudaKernelLaunchFeatures);
            });
    }
#endif
#ifdef VK_EXT_descriptor_buffer
    // Descriptor buffers remove descriptor pools, vkAllocateDescriptorSets and
    // vkUpdateDescriptorSets from the per-dispatch path. Added LAST: its pick
    // reads the BDA verdict the 1.2 / KHR-BDA picks above set, because the
    // binding info addresses the buffer by VkDeviceAddress.
    // VKOP_NO_DESCRIPTOR_BUFFER keeps the feature out of vkCreateDevice.
    if (has(VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME)) {
        sc.descriptor_buffer.available = true;
        if (!descriptor_buffer_disabled()) {
            fc.add<VkPhysicalDeviceDescriptorBufferFeaturesEXT>(
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT,
                VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME,
                [this](const auto &q, auto &e) {
                    shader_caps_.descriptor_buffer.supported =
                        q.descriptorBuffer != VK_FALSE;
                    e.descriptorBuffer =
                        q.descriptorBuffer && m_support_buffer_device_address_;
                    shader_caps_.descriptor_buffer.enabled = e.descriptorBuffer;
                    m_support_descriptor_buffer_ = e.descriptorBuffer;
                    return any(e.descriptorBuffer);
                });
        }
    }
#endif
}

// Shared by the 1.2 core struct and the pre-1.2 EXT struct: same field names.
// The engine's UAB layouts put the binding flag on every descriptor in the
// set, including the image bindings SplitImage/Conv2dImage use, so the bit is
// enabled per type the driver reports; flagging a type whose feature was not
// enabled is invalid usage.
template <typename T>
void VulkanDevice::pickDescriptorIndexing(const T &q, T &e) {
    recordCap(shader_caps_.descriptor_indexing,
              q.descriptorBindingStorageBufferUpdateAfterBind);
    recordCap(shader_caps_.descriptor_partially_bound,
              q.descriptorBindingPartiallyBound);
    e.descriptorBindingStorageBufferUpdateAfterBind =
        q.descriptorBindingStorageBufferUpdateAfterBind;
    e.descriptorBindingStorageImageUpdateAfterBind =
        q.descriptorBindingStorageImageUpdateAfterBind;
    e.descriptorBindingSampledImageUpdateAfterBind =
        q.descriptorBindingSampledImageUpdateAfterBind;
    e.descriptorBindingUniformBufferUpdateAfterBind =
        q.descriptorBindingUniformBufferUpdateAfterBind;
    e.descriptorBindingPartiallyBound = q.descriptorBindingPartiallyBound;
    m_uab_storage_buffer_ = q.descriptorBindingStorageBufferUpdateAfterBind;
    m_uab_storage_image_ = q.descriptorBindingStorageImageUpdateAfterBind;
    m_uab_sampled_image_ = q.descriptorBindingSampledImageUpdateAfterBind;
    m_uab_uniform_buffer_ = q.descriptorBindingUniformBufferUpdateAfterBind;
    m_support_descriptor_update_after_bind_ = m_uab_storage_buffer_;
    m_support_descriptor_partially_bound_ = q.descriptorBindingPartiallyBound;
}

void VulkanDevice::dumpShaderCapabilities() const {
    const char *env = std::getenv("VKOP_DEVCAPS");
    if (!env || std::strcmp(env, "1") != 0) {
        return;
    }
    auto yes_no = [](bool b) { return b ? "yes" : "no"; };
    auto pr = [&](const char *name, const DeviceCap &c) {
        if (!c.available) {
            // Nothing was queried: the extension did not enumerate and the
            // feature is not core here, so "no" would be a lie.
            std::fprintf(stderr, "[devcaps] %-30s %s\n", name, "not queried");
            return;
        }
        std::fprintf(stderr, "[devcaps] %-30s driver=%-4s device=%-4s\n", name,
                     yes_no(c.supported), yes_no(c.enabled));
    };
    const ShaderCapabilities &c = shader_caps_;
    const core::DeviceTraits &t = device_traits_;
    std::fprintf(stderr, "[devcaps] device=%s\n", deviceName_.c_str());
    std::fprintf(stderr, "[devcaps] vendor=0x%x device=0x%x driver_id=%u\n",
                 t.vendor_id, t.device_id, t.driver_id);
    std::fprintf(stderr,
                 "[devcaps] coopmat=%s fp16_8x8x16=%s fp16_16x16x16=%s "
                 "sint8_8x8x32=%s subgroup_arith=%s subgroup_size=%u\n",
                 yes_no(t.coopmat), yes_no(t.coopmat_fp16_8x8x16),
                 yes_no(t.coopmat_fp16_16x16x16),
                 yes_no(t.coopmat_sint8_8x8x32), yes_no(t.subgroup_arithmetic),
                 t.subgroup_size);
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
    pr("descriptor_indexing", c.descriptor_indexing);
    pr("descriptor_partially_bound", c.descriptor_partially_bound);
    pr("descriptor_buffer", c.descriptor_buffer);
    pr("shader_float8", c.shader_float8);
    pr("shader_float8_coopmat", c.shader_float8_coopmat);
    pr("integer_dot_product", c.integer_dot_product);
    // The packed 4x8 answer is a property, and it is the one that decides
    // whether a dot() route is worth taking -- a feature bit alone would say
    // "yes" to a kernel the driver expands into shifts.
#ifdef VK_KHR_shader_integer_dot_product
    std::fprintf(
        stderr, "[devcaps] %-30s packed_signed=%s (8bit_signed=%s) feature=%s\n",
        "dot_product_accel", yes_no(m_dot_product_packed_signed_),
        yes_no(m_dot_product_props_.integerDotProduct8BitSignedAccelerated !=
               VK_FALSE),
        yes_no(c.integer_dot_product.enabled));
#else
    std::fprintf(stderr, "[devcaps] %-30s packed_signed=no (headers lack "
                         "VK_KHR_shader_integer_dot_product)\n",
                 "dot_product_accel");
#endif
    const bool images = is_support_images();
    if (images || image_unsupported_reason_.empty()) {
        std::fprintf(stderr, "[devcaps] %-30s %s\n", "images", yes_no(images));
    } else {
        std::fprintf(stderr, "[devcaps] %-30s no (%s)\n", "images",
                     image_unsupported_reason_.c_str());
    }
    std::fflush(stderr);
}

// ---------------------------------------------------------------------------
// Logical device
// ---------------------------------------------------------------------------

bool VulkanDevice::createLogicalDevice(
    const VkPhysicalDeviceProperties &deviceProperties) {
    if (findComputeQueueFamily() == 0) {
        LOG_ERROR("Failed to find a suitable compute queue family!");
        return false;
    }

    uint32_t ext_count = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr, &ext_count,
                                         nullptr);
    std::vector<VkExtensionProperties> ext_properties(ext_count);
    vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr, &ext_count,
                                         ext_properties.data());
    std::set<std::string> extensions;
    for (const auto &ext : ext_properties) {
        extensions.insert(ext.extensionName);
        LOG_INFO("device extension %s", ext.extensionName);
    }

    // shaderInt64 lives in the Vulkan 1.0 core feature set, not in any
    // feature2 struct, so it goes through pEnabledFeatures. Needed by kernels
    // that do 64-bit integer ARITHMETIC; byte movers over ivec2 do not.
    VkPhysicalDeviceFeatures device_features = {};
    vkGetPhysicalDeviceFeatures(physicalDevice_, &device_features);
    VkPhysicalDeviceFeatures enabled_core_features = {};
    enabled_core_features.shaderInt64 = device_features.shaderInt64;
    shader_caps_.shader_int64.available = true;
    recordCap(shader_caps_.shader_int64, device_features.shaderInt64);

    // The version every gate compares against is the one the application may
    // legally use, min(instance apiVersion, device apiVersion). A driver can
    // advertise a newer apiVersion than the VkInstance was created with
    // (NVIDIA 595 reports 1.4.329 through an instance created at 1.3.275);
    // gating on the device value there would pick a core struct that cannot
    // legally be used and drop the EXT path that can.
    const uint32_t api_version =
        std::min(deviceProperties.apiVersion,
                 VulkanInstance::getVulkanInstance().apiVersion());

    FeatureChain features;
    addFeatureProbes(features, api_version, extensions);
    features.query(physicalDevice_);
    std::vector<const char *> enabled_extensions;
    void *enable_chain = features.enable(enabled_extensions);

    // Extensions with no feature struct of their own.
    // - swapchain: an old path gated it on geometryShader, excluding Intel ARL.
    // - external_memory_host: makes VulkanBuffer's host-import ctor legal;
    //   the alignment comes from getProperties().
    const char *plain_extensions[] = {
#ifdef VK_EXT_tooling_info
        VK_EXT_TOOLING_INFO_EXTENSION_NAME,
#endif
#ifdef VK_KHR_shader_non_semantic_info
        VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME,
#endif
#ifdef VK_KHR_swapchain
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
#endif
#ifdef VK_EXT_external_memory_host
        VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME,
#endif
#ifdef VK_EXT_memory_budget
        // Lets VMA track real heap usage/budget (it is only legal to ask VMA
        // for that once the extension is enabled on the device).
        VK_EXT_MEMORY_BUDGET_EXTENSION_NAME,
#endif
    };
    for (const char *name : plain_extensions) {
        if (extensions.count(name) > 0) {
            enabled_extensions.push_back(name);
        }
    }
#ifdef VK_EXT_memory_budget
    m_support_memory_budget_ =
        extensions.count(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) > 0;
#endif
#ifdef VK_EXT_external_memory_host
    m_support_host_import_ =
        extensions.count(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME) > 0;
    if (m_support_host_import_) {
        LOG_INFO("host import alignment %u", m_host_import_alignment_);
    }
#endif
    LOG_INFO("timeline semaphore %s supported",
             (m_support_timeline_semaphore_ ? "is" : "not"));
    LOG_INFO("host image copy %s supported",
             (m_support_host_image_copy_ ? "is" : "not"));
    LOG_INFO("update after bind %s supported",
             (m_support_descriptor_update_after_bind_ ? "is" : "not"));
    for (const char *e : enabled_extensions) {
        (void)e;
        LOG_INFO("enabled extension: %s", e);
    }

    std::vector<std::vector<float>> queue_priority;
    auto queue_create_info = setupQueueCreateInfo(queue_priority);
    VkDeviceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.pNext = enable_chain;
    create_info.pEnabledFeatures = &enabled_core_features;
    create_info.queueCreateInfoCount =
        static_cast<uint32_t>(queue_create_info.size());
    create_info.pQueueCreateInfos = queue_create_info.data();
    create_info.enabledExtensionCount =
        static_cast<uint32_t>(enabled_extensions.size());
    create_info.ppEnabledExtensionNames = enabled_extensions.data();

    VkResult ret =
        vkCreateDevice(physicalDevice_, &create_info, nullptr, &logicalDevice_);
    if (ret != VK_SUCCESS) {
        LOG_ERROR("Failed to create logical device! ret=%d", ret);
        return false;
    }
    VulkanDebug::set_device(logicalDevice_);

    if (m_support_host_image_copy_ && !probeHostImageCopy()) {
        m_support_host_image_copy_ = false;
        LOG_WARN("host image copy unusable for OPTIMAL images on this "
                 "driver, falling back to staged copies");
    }

    auto get_queues = [this](const std::vector<QueueFamily> &families,
                             std::vector<std::shared_ptr<VulkanQueue>> &out) {
        for (const auto &[qidx, count, flags] : families) {
            (void)flags;
            for (uint32_t i = 0; i < queuesPerFamily(count); i++) {
                VkQueue queue;
                vkGetDeviceQueue(logicalDevice_, qidx, i, &queue);
                out.emplace_back(std::make_shared<VulkanQueue>(qidx, queue));
            }
        }
    };
    get_queues(computeQueueIdxs_, computeQueues_);
    get_queues(graphicsQueueIdxs_, graphicsQueues_);

    std::lock_guard<std::mutex> lock(device_registry_mutex());
    device_registry()[logicalDevice_] = this;
    return true;
}

// Some drivers advertise host image copy yet cannot use it:
// - an empty memory-type mask for OPTIMAL images with HOST_TRANSFER usage
//   (Windows, NVIDIA Pascal), which makes every such image unallocatable;
// - NVIDIA 595 on Ada answers VK_ERROR_EXTENSION_NOT_PRESENT to
//   vkTransitionImageLayout on a device created through this engine, though
//   the extension and feature are both enabled.
// Probe one representative image through both steps instead of trusting the
// advertised bit; VulkanImage only asserts on a failing transition.
bool VulkanDevice::probeHostImageCopy() {
    VkImageCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    info.extent = {64, 640, 1};
    info.mipLevels = 1;
    info.arrayLayers = 2;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                 VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImage image = VK_NULL_HANDLE;
    if (vkCreateImage(logicalDevice_, &info, nullptr, &image) != VK_SUCCESS) {
        return false;
    }

    VkImageMemoryRequirementsInfo2 req_info = {};
    req_info.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2;
    req_info.image = image;
    VkMemoryRequirements2 req = {};
    req.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
    vkGetImageMemoryRequirements2(logicalDevice_, &req_info, &req);
    bool usable = req.memoryRequirements.memoryTypeBits != 0;

#ifdef VK_EXT_host_image_copy
    if (usable) {
        auto instance = VulkanInstance::getVulkanInstance().getInstance();
        auto transition = reinterpret_cast<PFN_vkTransitionImageLayoutEXT>(
            vkGetInstanceProcAddr(instance, "vkTransitionImageLayout"));
        if (!transition) {
            transition = reinterpret_cast<PFN_vkTransitionImageLayoutEXT>(
                vkGetInstanceProcAddr(instance, "vkTransitionImageLayoutEXT"));
        }
        VkHostImageLayoutTransitionInfoEXT t = {};
        t.sType = VK_STRUCTURE_TYPE_HOST_IMAGE_LAYOUT_TRANSITION_INFO_EXT;
        t.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        t.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        t.image = image;
        t.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        t.subresourceRange.levelCount = 1;
        t.subresourceRange.layerCount = 2;
        usable = transition != nullptr &&
                 transition(logicalDevice_, 1, &t) == VK_SUCCESS;
    }
#endif
    vkDestroyImage(logicalDevice_, image, nullptr);
    return usable;
}

std::vector<VkDeviceQueueCreateInfo> VulkanDevice::setupQueueCreateInfo(
    std::vector<std::vector<float>> &queue_priority) {
    std::vector<VkDeviceQueueCreateInfo> infos(computeQueueIdxs_.size());
    queue_priority.resize(computeQueueIdxs_.size());
    for (size_t i = 0; i < computeQueueIdxs_.size(); i++) {
        const auto &[qidx, count, flags] = computeQueueIdxs_[i];
        (void)flags;
        LOG_INFO("queue count %d", count);
        // Multiple queues per family for pipeline parallelism.
        queue_priority[i].assign(queuesPerFamily(count), 1.0F);
        infos[i].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        infos[i].queueFamilyIndex = qidx;
        infos[i].queueCount = static_cast<uint32_t>(queue_priority[i].size());
        infos[i].pQueuePriorities = queue_priority[i].data();
    }
    return infos;
}

// ---------------------------------------------------------------------------
// Buffer device address / descriptor buffer
// ---------------------------------------------------------------------------

void VulkanDevice::initBufferDeviceAddress() {
    if (!m_support_buffer_device_address_) {
        return;
    }
    // The core symbol is the 1.2 entry point; a 1.1 driver with only
    // VK_KHR_buffer_device_address exports the KHR alias instead. Neither is
    // in VulkanLib's assert-all list, so a missing one is a normal fallback.
    auto &lib = VulkanLib::getVulkanLib();
    auto fn = reinterpret_cast<PFN_vkGetBufferDeviceAddress>(
        lib.get_proc_address("vkGetBufferDeviceAddress"));
    if (!fn) {
        fn = reinterpret_cast<PFN_vkGetBufferDeviceAddress>(
            lib.get_proc_address("vkGetBufferDeviceAddressKHR"));
    }
    if (!fn) {
        m_support_buffer_device_address_ = false;
        LOG_WARN("buffer device address enabled but no entry point resolved; "
                 "falling back to descriptors");
        return;
    }
    get_buffer_device_address_fn_ = fn;
    if (!probeBufferDeviceAddress()) {
        get_buffer_device_address_fn_ = nullptr;
        m_support_buffer_device_address_ = false;
        LOG_WARN("buffer device address probe failed; falling back to "
                 "descriptors");
    }
}

bool VulkanDevice::probeBufferDeviceAddress() {
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = 64;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buffer = VK_NULL_HANDLE;
    if (vkCreateBuffer(logicalDevice_, &buffer_info, nullptr, &buffer) !=
        VK_SUCCESS) {
        return false;
    }

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(logicalDevice_, buffer, &requirements);
    // Prefer DEVICE_LOCAL, but any type that accepts the allocation proves the
    // flag/usage combination is valid on this driver.
    int32_t type = findMemoryType(requirements.memoryTypeBits,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type < 0) {
        type = findMemoryType(requirements.memoryTypeBits, 0);
    }
    if (type < 0) {
        vkDestroyBuffer(logicalDevice_, buffer, nullptr);
        return false;
    }

    VkMemoryAllocateFlagsInfo flags_info{};
    flags_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flags_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo allocate_info{};
    allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate_info.pNext = &flags_info;
    allocate_info.allocationSize = requirements.size;
    allocate_info.memoryTypeIndex = static_cast<uint32_t>(type);

    VkDeviceMemory memory = VK_NULL_HANDLE;
    bool ok = vkAllocateMemory(logicalDevice_, &allocate_info, nullptr,
                               &memory) == VK_SUCCESS &&
              vkBindBufferMemory(logicalDevice_, buffer, memory, 0) ==
                  VK_SUCCESS;
    if (ok) {
        VkBufferDeviceAddressInfo address_info{};
        address_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        address_info.buffer = buffer;
        // Nonzero is the compatibility gate. Concrete alignment is a
        // per-shader buffer_reference_align contract, checked where a BDA
        // variant is selected.
        ok = get_buffer_device_address_fn_(logicalDevice_, &address_info) != 0;
    }
    if (memory != VK_NULL_HANDLE) {
        vkFreeMemory(logicalDevice_, memory, nullptr);
    }
    vkDestroyBuffer(logicalDevice_, buffer, nullptr);
    return ok;
}

VkDeviceAddress VulkanDevice::getBufferDeviceAddress(VkBuffer buffer) const {
    if (!m_support_buffer_device_address_ ||
        get_buffer_device_address_fn_ == nullptr) {
        throw std::runtime_error(
            "vkop: buffer device address requested but the device does not "
            "support it (feature disabled, probe failed, or VKOP_NO_BDA)");
    }
    VkBufferDeviceAddressInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    info.buffer = buffer;
    return get_buffer_device_address_fn_(logicalDevice_, &info);
}

void VulkanDevice::initDescriptorBuffer() {
#ifdef VK_EXT_descriptor_buffer
    if (!m_support_descriptor_buffer_) {
        return;
    }
    auto disable = [this](const char *why) {
        m_support_descriptor_buffer_ = false;
        shader_caps_.descriptor_buffer.enabled = false;
        LOG_WARN("VK_EXT_descriptor_buffer %s; falling back to descriptor "
                 "pools",
                 why);
    };
    if (descriptor_buffer_disabled() || !is_support_buffer_device_address()) {
        disable("enabled but BDA is unusable");
        return;
    }
    auto load = [this](const char *name) {
        return vkGetDeviceProcAddr(logicalDevice_, name);
    };
    auto &info = m_descriptor_buffer_info_;
    info.getLayoutSize = reinterpret_cast<PFN_vkGetDescriptorSetLayoutSizeEXT>(
        load("vkGetDescriptorSetLayoutSizeEXT"));
    info.getBindingOffset =
        reinterpret_cast<PFN_vkGetDescriptorSetLayoutBindingOffsetEXT>(
            load("vkGetDescriptorSetLayoutBindingOffsetEXT"));
    info.getDescriptor =
        reinterpret_cast<PFN_vkGetDescriptorEXT>(load("vkGetDescriptorEXT"));
    info.cmdBindBuffers = reinterpret_cast<PFN_vkCmdBindDescriptorBuffersEXT>(
        load("vkCmdBindDescriptorBuffersEXT"));
    info.cmdSetOffsets =
        reinterpret_cast<PFN_vkCmdSetDescriptorBufferOffsetsEXT>(
            load("vkCmdSetDescriptorBufferOffsetsEXT"));
    info.getBufferAddress = get_buffer_device_address_fn_;

    const auto &props = m_descriptor_buffer_props_;
    info.offsetAlignment =
        std::max<VkDeviceSize>(1, props.descriptorBufferOffsetAlignment);
    info.storageBufferSize = props.storageBufferDescriptorSize;
    info.uniformBufferSize = props.uniformBufferDescriptorSize;
    info.storageImageSize = props.storageImageDescriptorSize;
    info.sampledImageSize = props.sampledImageDescriptorSize;
    info.combinedImageSamplerSize = props.combinedImageSamplerDescriptorSize;
    info.storageTexelBufferSize = props.storageTexelBufferDescriptorSize;
    info.uniformTexelBufferSize = props.uniformTexelBufferDescriptorSize;

    if (!info.valid()) {
        disable("entry points or descriptor sizes missing");
        return;
    }
    // Whether the shared descriptor buffer can live in device-local memory
    // (UMA on an iGPU, BAR on a discrete GPU). This decides both the memory
    // flags and whether discrete devices enable the backend by default.
    m_descriptor_buffer_device_local_memory_ =
        findMemoryType(~0U, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) >= 0;
    LOG_INFO("descriptor buffer: offsetAlign %llu sb %llu ubo %llu img %llu "
             "sampled %llu cimg %llu",
             static_cast<unsigned long long>(info.offsetAlignment),
             static_cast<unsigned long long>(info.storageBufferSize),
             static_cast<unsigned long long>(info.uniformBufferSize),
             static_cast<unsigned long long>(info.storageImageSize),
             static_cast<unsigned long long>(info.sampledImageSize),
             static_cast<unsigned long long>(info.combinedImageSamplerSize));
#else
    m_support_descriptor_buffer_ = false;
#endif
}

std::shared_ptr<DescriptorBufferPool> VulkanDevice::descriptor_buffer_pool() {
    if (!m_support_descriptor_buffer_) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(descriptor_buffer_pool_mutex_);
    if (descriptor_buffer_pool_) {
        return descriptor_buffer_pool_;
    }

    // Descriptor memory is host-written but GPU-fetched. Prefer a
    // DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT type (the UMA memory type on an
    // iGPU, BAR on a discrete GPU); b3's decode lost ~2.4 ms/round (~8.6%)
    // when the descriptors lived in system RAM and were fetched over PCIe.
    VkMemoryPropertyFlags memory_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (m_descriptor_buffer_device_local_memory_) {
        memory_flags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    }

    size_t pool_mb = 32;
    if (const char *env = std::getenv("VKOP_DESCRIPTOR_BUFFER_POOL_MB")) {
        const int v = std::atoi(env);
        if (v > 0) {
            pool_mb = static_cast<size_t>(v);
        }
    }
    // VulkanBuffer needs a shared_ptr to the device; the real owner (Runtime /
    // the test fixture) outlives every pipeline, exactly like the raw VkDevice
    // handle VulkanBasePipeline already holds.
    std::shared_ptr<VulkanDevice> self(this, [](VulkanDevice *) {});
    auto pool = DescriptorBufferPool::create(
        self, &m_descriptor_buffer_info_, memory_flags,
        static_cast<VkDeviceSize>(pool_mb) << 20);
    if (!pool) {
        LOG_WARN("descriptor buffer pool allocation failed (%zu MiB, flags "
                 "0x%x); pipelines keep descriptor sets",
                 pool_mb, static_cast<unsigned>(memory_flags));
        return nullptr;
    }
    LOG_INFO("descriptor buffer pool: %zu MiB, memory flags 0x%x%s", pool_mb,
             static_cast<unsigned>(memory_flags),
             m_descriptor_buffer_device_local_memory_ ? " (device-local)"
                                                      : " (system)");
    descriptor_buffer_pool_ = std::move(pool);
    return descriptor_buffer_pool_;
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

int32_t VulkanDevice::findMemoryType(uint32_t type_bits,
                                     VkMemoryPropertyFlags want) const {
    VkPhysicalDeviceMemoryProperties mem{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &mem);
    for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
        if ((type_bits & (1U << i)) &&
            (mem.memoryTypes[i].propertyFlags & want) == want) {
            return static_cast<int32_t>(i);
        }
    }
    return -1;
}

bool VulkanDevice::checkDeviceUnifiedMemoryAccess() {
    VkPhysicalDeviceMemoryProperties mem{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &mem);
    // A HOST_VISIBLE|DEVICE_LOCAL memory type is NOT sufficient: discrete
    // NVIDIA/AMD expose their BAR as exactly that combination (a 256 MB window
    // of VRAM mapped to the host), which would tag every dGPU as UMA. The
    // reliable discriminator is the heap layout: an iGPU exposes a single
    // DEVICE_LOCAL heap, a dGPU always has a separate non-local system heap.
    const bool uma = mem.memoryHeapCount == 1;
    LOG_INFO("This device %s Unified Memory Architecture (UMA).",
             uma ? "supports" : "does not support");
    return uma;
}

void VulkanDevice::wait_all_done() { vkDeviceWaitIdle(logicalDevice_); }

} // namespace vkop
