// Copyright 2025 @junka
#ifndef SRC_VULKANDEVICE_HPP_
#define SRC_VULKANDEVICE_HPP_

#include <functional>
#include <memory>
#include <set>

#include "vulkan/VMA.hpp"

namespace vkop {

constexpr int kInflight = 2;
class VulkanQueue {
  public:
    VulkanQueue(uint32_t familyIdx, VkQueue queue)
        : familyIdx_(familyIdx), queue_(queue) {}
    ~VulkanQueue() = default;

    VkQueue getQueue() const { return queue_; }
    uint32_t getFamilyIdx() const { return familyIdx_; }
    // Monotonic submit counter: incremented on every vkQueueSubmit through
    // this queue. The runtime's level-batching uses it to DETECT synchronous
    // readbacks (copyToCPU does its own cmd.submit+wait on this same queue)
    // — if the counter advanced during a level's onExecute recording, that
    // level contained a sync readback, so the batch MUST be flushed before
    // it (the readback relied on the producer already being queued, which
    // single-queue FIFO ordering only guarantees if prior levels submitted).
    uint64_t submitCount() const { return submit_count_; }
    void bumpSubmitCount() { submit_count_++; }

  private:
    uint32_t familyIdx_;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint64_t submit_count_ = 0;
};

struct FeatureDescriptor {
    VkStructureType sType = {};
    const char *extensionName = nullptr;
    uint32_t corePromotedVersion = 0;

    std::function<std::shared_ptr<void>()> makeQueryStruct;

    std::function<std::shared_ptr<void>(void *queried)> makeEnableStruct;
};
struct QueryChainResult {
    VkPhysicalDeviceFeatures2 features2{};
    // shared_ptr<void> (not unique_ptr<VkBaseOutStructure>): the type-erased
    // deleter remembers the concrete struct type allocated by the
    // makeQueryStruct/makeEnableStruct lambdas, so destruction frees the
    // derived object correctly. unique_ptr<VkBaseOutStructure> would delete a
    // plain C struct through the base pointer (new/delete type mismatch —
    // caught by ASan, UB regardless).
    std::vector<std::shared_ptr<void>> ownedStructs;
};
struct EnableResult {
    std::vector<std::shared_ptr<void>> enableChain;
    std::vector<const char *> enabledExtensions;
};
class VulkanDevice {

  private:
    // Helper functions
    VkPhysicalDeviceProperties getProperties();

    bool
    createLogicalDevice(const VkPhysicalDeviceProperties &deviceProperties);

    uint32_t findComputeQueueFamily();

    static bool checkDeviceExtensionFeature(
        const std::vector<VkExtensionProperties> &properties, const char *name);

    void assertImageConfigurationSupported(VkFormat format);

    std::vector<VkDeviceQueueCreateInfo>
    setupQueueCreateInfo(std::vector<std::vector<float>> &queue_priority);

    template <typename T, VkStructureType ST>
    static constexpr T makeFeatureStruct() {
        T f{};
        f.sType = ST;
        f.pNext = nullptr;
        return f;
    }

    static QueryChainResult
    buildFeatureQueryChain(const std::vector<FeatureDescriptor> &descs);

    static EnableResult
    buildFeatureEnableChain(const std::vector<FeatureDescriptor> &descs,
                            const VkPhysicalDeviceFeatures2 &queryFeatures2,
                            const std::set<std::string> &supportedExtensions);

    void checkImageFormatSupport();

    std::vector<FeatureDescriptor>
    createFeatureDescriptors(const VkPhysicalDeviceProperties &deviceProperties,
                             const std::set<std::string> &supportedExtensions);

  public:
    explicit VulkanDevice(VkPhysicalDevice physicalDevice_);
    ~VulkanDevice();

    // Getters
    VkDevice getLogicalDevice() const { return logicalDevice_; }
    VkPhysicalDevice getPhysicalDevice() const { return physicalDevice_; }
    size_t getNumComputeQueues() const { return computeQueues_.size(); }

    // minStorageBufferOffsetAlignment, cached on first use. A binding that
    // points an SSBO at a non-zero byte offset (the Slice/Split zero-copy
    // range-view fast path) must keep that offset a multiple of this, so the
    // op gates on it before aliasing.
    VkDeviceSize minStorageBufferOffsetAlignment();

    std::shared_ptr<VulkanQueue> getComputeQueue(uint32_t idx = 0) const {
        int offset = idx / 2;
        int cat = idx % 2;
        cat = cat * kInflight / 2;
        cat += offset;
        return computeQueues_[(cat) % computeQueues_.size()];
    }

    std::string getDeviceName() const { return deviceName_; }

    std::vector<std::tuple<uint32_t, uint32_t, VkQueueFlags>>
    getComputeQueueFamilyIndex() const {
        return computeQueueIdxs_;
    }
    std::vector<std::tuple<uint32_t, uint32_t, VkQueueFlags>>
    getGraphicsQueueFamilyIndex() const {
        return graphicsQueueIdxs_;
    }

    std::shared_ptr<VulkanQueue> getGraphicsQueues(uint32_t idx = 0) const {
        int offset = idx / 2;
        int cat = idx % 2;
        cat = cat * kInflight / 2;
        cat += offset;
        return graphicsQueues_[(cat) % graphicsQueues_.size()];
    }

    float getTimestampPeriod() const { return timestampPeriod_; }
    uint32_t getMaxImageArrayLayers() const { return maxImageArrayLayers_; }
    uint32_t getMaxImageDimension2D() const { return maxImageDimension2D_; }

    bool checkHostImageCopyDstLayoutSupport(VkImageLayout layout) {
        return std::any_of(copyDstLayout_.begin(), copyDstLayout_.end(),
                           [layout](VkImageLayout i) { return i == layout; });
    }
    void wait_all_done();
    bool is_support_host_image_copy() const {
        return m_support_host_image_copy_;
    }
    // UMA: a memory type that is both HOST_VISIBLE and DEVICE_LOCAL. iGPU
    // (Intel/AMD APU) report it; discrete cards do not. Ops use this to pick
    // routing thresholds measured on the matching device class.
    bool is_unified_memory() const { return m_is_unified_memory_; }
    bool is_support_buffer_device_address() const {
        return m_support_buffer_device_address_;
    }
    bool is_support_timeline_semaphore() const {
        return m_support_timeline_semaphore_;
    }
    bool is_support_cuda_kernel_launch() const {
        return m_support_cuda_kernel_launch_;
    }
    bool is_support_nv_tensor_core() const { return m_support_nv_tensor_core_; }
    bool is_support_cooperate_matrix() const {
        return m_support_cooperate_matrix_;
    }
    // cooperativeMatrixSupportedStages as reported by the physical device.
    // MoltenVK advertises the KHR extension and the feature bool, but reports
    // 0 supported stages — a combo that fails at pipeline creation, so the
    // feature gate below must treat 0 stages as "not usable".
    uint32_t coopmat_supported_stages() const {
        return m_coopmat_supported_stages_;
    }
    // MatMul tile-vs-GEMV crossover, per device. The shared-memory tile runs a
    // fixed TILE_M = 64 row block, so its cost is flat in m while the split-K
    // GEMV pays one full weight pass per output row; the m at which they cross
    // depends on the shader compiler and memory system, not on the shape.
    // byte_min_m / fp16_min_m default to the ARL iGPU / RTX 4000 Ada
    // calibration. wide_n selects a second band for weights whose row pass is
    // DRAM-bound (large N), where the GEMV's per-row re-read degrades and the
    // tile wins at a larger m; 0 disables the band. The power-of-two >= 64 KiB
    // row-stride override in Matmul still takes precedence over both.
    struct MatmulTileProfile {
        int byte_min_m = 2;
        int fp16_min_m = 3;
        int wide_n = 0;
        int byte_wide_min_m = 2;
        int fp16_wide_min_m = 3;
        // Workgroups at which the tiled fp16 GEMM saturates this device; the
        // tile-split-K route targets this many blocks. Measured, not the
        // theoretical resident-workgroup limit: a bandwidth-bound part
        // saturates early, and splitting past that only adds scratch traffic.
        int ks_target_blocks = 128;
    };
    const MatmulTileProfile &matmul_tile_profile() const {
        return matmul_tile_profile_;
    }
#ifdef VK_KHR_cooperative_matrix
    const std::vector<VkCooperativeMatrixPropertiesKHR> &
    getCooperativeMatrixProperties() const {
        return coopmatProps_;
    }
    // SINT8 -> SINT32 subgroup-scope cooperative matrix (W8A8) with an
    // explicit tile shape. There is no boolean feature for "this device does
    // integer MMA"; the KHR properties list is the only source of truth, so
    // scan it for the exact combination the int8 MatMul kernel declares.
    // Intel ARL reports 8x8x32 A=SINT8 B=SINT8 C=SINT32 R=SINT32; NVIDIA Ada
    // lists 16x16x32 instead and silently produces zeros for 8x8x32, so the
    // kernel shape has to be matched, not just the component types.
    bool supports_coopmat_sint8_tile(uint32_t M, uint32_t N, uint32_t K) const {
        for (const auto &p : coopmatProps_) {
            if (p.AType == VK_COMPONENT_TYPE_SINT8_KHR &&
                p.BType == VK_COMPONENT_TYPE_SINT8_KHR &&
                p.ResultType == VK_COMPONENT_TYPE_SINT32_KHR &&
                p.scope == VK_SCOPE_SUBGROUP_KHR && p.MSize == M &&
                p.NSize == N && p.KSize == K) {
                return true;
            }
        }
        return false;
    }
    // fp16 -> fp32 subgroup-scope cooperative matrix with an explicit tile
    // shape. The 8x8x16 combo is the only fp16 shape Intel ARL exposes; it
    // is what matmul_coop.comp declares. NVIDIA (Ada Lovelace) does NOT list
    // 8x8x16 -- it lists 16x16x16 / 16x8x16 / 16x8x8 -- so the 8x8x16 shader
    // silently miscomputes there. matmul_coop16.comp declares 16x16x16; this
    // query picks the shader whose tile the device actually supports.
    bool supports_coopmat_fp16_tile(uint32_t M, uint32_t N, uint32_t K) const {
        for (const auto &p : coopmatProps_) {
            if (p.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                p.BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                p.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                p.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                p.scope == VK_SCOPE_SUBGROUP_KHR && p.MSize == M &&
                p.NSize == N && p.KSize == K) {
                return true;
            }
        }
        return false;
    }
#else
    bool supports_coopmat_sint8_tile(uint32_t, uint32_t, uint32_t) const {
        return false;
    }
    bool supports_coopmat_fp16_tile(uint32_t, uint32_t, uint32_t) const {
        return false;
    }
#endif
    bool is_support_descriptor_update_after_bind() const {
        return m_support_descriptor_update_after_bind_;
    }
#ifdef VK_EXT_subgroup_size_control
    // Range of subgroup sizes the driver will allow a compute pipeline to
    // pin via requiredSubgroupSize. 0 when the extension/feature is absent.
    uint32_t getMinSubgroupSize() const { return minSubgroupSize_; }
    uint32_t getMaxSubgroupSize() const { return maxSubgroupSize_; }
#else
    uint32_t getMinSubgroupSize() const { return 0; }
    uint32_t getMaxSubgroupSize() const { return 0; }
#endif
#ifdef USE_VMA
    vkop::VMA *getVMA() const { return m_vma_.get(); }
#endif

    bool checkDeviceUnifiedMemoryAccess();

  private:
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice logicalDevice_ = VK_NULL_HANDLE;
    std::vector<std::shared_ptr<VulkanQueue>> computeQueues_;
    std::vector<std::shared_ptr<VulkanQueue>> graphicsQueues_;

    bool m_support_host_image_copy_ = false;
    bool m_is_unified_memory_ = false;
    bool m_support_buffer_device_address_ = false;
    bool m_support_timeline_semaphore_ = false;
    bool m_support_cuda_kernel_launch_ = false;
    bool m_support_nv_tensor_core_ = false;
    bool m_support_cooperate_matrix_ = false;
    uint32_t m_coopmat_supported_stages_ = 0;
    // Filled by getProperties() from driverID/vendorID; see MatmulTileProfile.
    MatmulTileProfile matmul_tile_profile_;
    bool m_support_descriptor_update_after_bind_ = false;

#ifdef VK_EXT_subgroup_size_control
    uint32_t minSubgroupSize_ = 0;
    uint32_t maxSubgroupSize_ = 0;
#endif

    std::vector<std::tuple<uint32_t, uint32_t, VkQueueFlags>> computeQueueIdxs_;
    std::vector<std::tuple<uint32_t, uint32_t, VkQueueFlags>>
        graphicsQueueIdxs_;

    float timestampPeriod_;
    uint32_t maxImageArrayLayers_;
    // Cached minStorageBufferOffsetAlignment (0 = not yet queried).
    VkDeviceSize min_sb_offset_align_ = 0;
    // Packed NCHW images address along width, so a tensor whose W*C4 exceeds
    // this cannot be represented as one image at all.
    uint32_t maxImageDimension2D_ = 0;

    std::vector<VkImageLayout> copySrcLayout_;
    std::vector<VkImageLayout> copyDstLayout_;

#ifdef VK_KHR_cooperative_matrix
    std::vector<VkCooperativeMatrixPropertiesKHR> coopmatProps_;
#endif

    std::string deviceName_;

#ifdef USE_VMA
    std::unique_ptr<vkop::VMA> m_vma_;
#endif
};

} // namespace vkop

#endif // SRC_VULKANDEVICE_HPP_
