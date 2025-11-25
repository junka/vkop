// Copyright 2025 @junka
#ifndef SRC_VULKANDEVICE_HPP_
#define SRC_VULKANDEVICE_HPP_

#include <vector>

#include "vulkan/VMA.hpp"

namespace vkop {

class VulkanDevice {

  private:
    // Helper functions
    VkPhysicalDeviceProperties getProperties();

    bool
    createLogicalDevice(const VkPhysicalDeviceProperties &deviceProperties);

    uint32_t findComputeQueueFamily();

    bool checkDeviceExtensionFeature(
        const std::vector<VkExtensionProperties> &properties,
        const char *name) const;

  public:
    explicit VulkanDevice(VkPhysicalDevice physicalDevice_);
    ~VulkanDevice();

    // Getters
    VkDevice getLogicalDevice() const { return logicalDevice_; }
    VkPhysicalDevice getPhysicalDevice() const { return physicalDevice_; }
    VkQueue getComputeQueue(uint32_t idx = 0) const {
        return computeQueues_[idx % computeQueues_.size()];
    }

    std::string getDeviceName() const { return deviceName_; }

    int getComputeQueueFamilyIndex() const {
        return std::get<0>(computeQueueIdxs_[0]);
    }

    float getTimestampPeriod() const { return timestampPeriod_; }

    bool checkHostImageCopyDstLayoutSupport(VkImageLayout layout) {
        return std::any_of(copyDstLayout_.begin(), copyDstLayout_.end(),
                           [layout](VkImageLayout i) { return i == layout; });
        return false;
    }
    bool is_support_host_image_copy() const {
        return m_support_host_image_copy_;
    }
    bool is_support_buffer_device_address() const {
        return m_support_buffer_device_address_;
    }
#ifdef USE_VMA
    vkop::VMA *getVMA() const { return m_vma_.get(); }
#endif

    bool checkDeviceUnifiedMemoryAccess();

  private:
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice logicalDevice_ = VK_NULL_HANDLE;
    std::vector<VkQueue> computeQueues_;

    bool m_support_host_image_copy_ = false;
    bool m_support_buffer_device_address_ = false;

    std::vector<std::tuple<uint32_t, uint32_t, VkQueueFlags>> computeQueueIdxs_;

    float timestampPeriod_;

    std::vector<VkImageLayout> copySrcLayout_;
    std::vector<VkImageLayout> copyDstLayout_;

    std::string deviceName_;

#ifdef USE_VMA
    std::unique_ptr<vkop::VMA> m_vma_;
#endif
};

} // namespace vkop

#endif // SRC_VULKANDEVICE_HPP_
