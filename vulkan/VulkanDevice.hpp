// Copyright 2025 @junka
#ifndef SRC_VULKANDEVICE_HPP_
#define SRC_VULKANDEVICE_HPP_

#include <functional>
#include <memory>
#include <set>

#include "vulkan/VMA.hpp"

namespace vkop {

constexpr int kInflight = 1;
class VulkanQueue {
  public:
    VulkanQueue(VkDevice dev, uint32_t familyIdx, VkQueue queue)
        : logicalDevice_(dev), familyIdx_(familyIdx), queue_(queue) {
        createSemaphore();
    }
    ~VulkanQueue() { destroySemaphore(); }

    VkQueue getQueue() const { return queue_; }
    VkSemaphore getSemaphore() const { return m_semaphore_; }
    uint32_t getFamilyIdx() const { return familyIdx_; }

  private:
    VkDevice logicalDevice_ = VK_NULL_HANDLE;
    uint32_t familyIdx_;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkSemaphore m_semaphore_ = VK_NULL_HANDLE;

    void createSemaphore() {
        VkSemaphoreTypeCreateInfo timeline_info{};
        timeline_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        timeline_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        timeline_info.initialValue = 0;

        VkSemaphoreCreateInfo sem_info{};
        sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        sem_info.pNext = &timeline_info;
        if (vkCreateSemaphore(logicalDevice_, &sem_info, nullptr,
                              &m_semaphore_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create timeline semaphore");
        }
    }
    void destroySemaphore() {
        if (m_semaphore_) {
            vkDestroySemaphore(logicalDevice_, m_semaphore_, nullptr);
        }
    }
};

struct FeatureDescriptor {
    VkStructureType sType = {};
    const char *extensionName = nullptr;
    uint32_t corePromotedVersion = 0;

    std::function<std::unique_ptr<VkBaseOutStructure>()> makeQueryStruct;

    std::function<std::unique_ptr<VkBaseOutStructure>(void *queried)>
        makeEnableStruct;
};
struct QueryChainResult {
    VkPhysicalDeviceFeatures2 features2{};
    std::vector<std::unique_ptr<VkBaseOutStructure>> ownedStructs;
};
struct EnableResult {
    std::vector<std::unique_ptr<VkBaseOutStructure>> enableChain;
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

    template <typename T, VkStructureType ST> T makeFeatureStruct() {
        T f{};
        f.sType = ST;
        return f;
    }
    static QueryChainResult
    buildFeatureQueryChain(const std::vector<FeatureDescriptor> &descs);

    static EnableResult
    buildFeatureEnableChain(const std::vector<FeatureDescriptor> &descs,
                            const VkPhysicalDeviceFeatures2 &queryFeatures2);

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

    float getTimestampPeriod() const { return timestampPeriod_; }
    uint32_t getMaxImageArrayLayers() const { return maxImageArrayLayers_; }

    bool checkHostImageCopyDstLayoutSupport(VkImageLayout layout) {
        return std::any_of(copyDstLayout_.begin(), copyDstLayout_.end(),
                           [layout](VkImageLayout i) { return i == layout; });
        return false;
    }
    void wait_all_done();
    bool is_support_host_image_copy() const {
        return m_support_host_image_copy_;
    }
    bool is_support_buffer_device_address() const {
        return m_support_buffer_device_address_;
    }
    bool is_support_timeline_semaphore() const {
        return m_support_timeline_semaphore_;
    }
    bool is_support_cuda_kernel_launch() const {
        return m_support_cuda_kernel_launch_;
    }
#ifdef USE_VMA
    vkop::VMA *getVMA() const { return m_vma_.get(); }
#endif

    bool checkDeviceUnifiedMemoryAccess();

  private:
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice logicalDevice_ = VK_NULL_HANDLE;
    std::vector<std::shared_ptr<VulkanQueue>> computeQueues_;

    bool m_support_host_image_copy_ = false;
    bool m_support_buffer_device_address_ = false;
    bool m_support_timeline_semaphore_ = false;
    bool m_support_cuda_kernel_launch_ = false;

    std::vector<std::tuple<uint32_t, uint32_t, VkQueueFlags>> computeQueueIdxs_;

    float timestampPeriod_;
    uint32_t maxImageArrayLayers_;

    std::vector<VkImageLayout> copySrcLayout_;
    std::vector<VkImageLayout> copyDstLayout_;

    std::string deviceName_;

#ifdef USE_VMA
    std::unique_ptr<vkop::VMA> m_vma_;
#endif
};

} // namespace vkop

#endif // SRC_VULKANDEVICE_HPP_
