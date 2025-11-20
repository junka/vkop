// Copyright 2025 @junka
#ifndef SRC_VULKANDEVICE_HPP_
#define SRC_VULKANDEVICE_HPP_

#include <vector>

#include "vulkan/VMA.hpp"

namespace vkop {

class VulkanDevice {
  public:
    explicit VulkanDevice(VkPhysicalDevice physicalDevice_);
    ~VulkanDevice();

    // Initialization and cleanup
    void create();

    // Getters
    VkDevice getLogicalDevice() const { return logicalDevice_; }
    VkPhysicalDevice getPhysicalDevice() const { return physicalDevice_; }
    VkQueue getComputeQueue() const { return computeQueue_; }

    std::string getDeviceName() const { return deviceName_; }

    int getComputeQueueFamilyIndex() const { return computeQueueFamilyIndex_; }

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
    VkQueue computeQueue_ = VK_NULL_HANDLE;

    int computeQueueFamilyIndex_ = -1;
    float timestampPeriod_;

    bool m_support_host_image_copy_ = false;
    bool m_support_buffer_device_address_ = false;

    std::vector<VkImageLayout> copySrcLayout_;
    std::vector<VkImageLayout> copyDstLayout_;

    std::vector<uintptr_t> enabledFeatures_;
    std::vector<const char *> enabledExtensions_;

    std::vector<VkExtensionProperties> ext_properties_;
    std::string deviceName_;

#ifdef USE_VMA
    std::unique_ptr<vkop::VMA> m_vma_;
#endif
    // Helper functions
    VkPhysicalDeviceProperties getProperties();

    bool
    createLogicalDevice(const VkPhysicalDeviceProperties &deviceProperties);

    // Device suitability checks
    bool isDeviceSuitable();

    int findComputeQueueFamily();

    bool checkDeviceExtensionFeature(const char *name) const;
};

} // namespace vkop

#endif // SRC_VULKANDEVICE_HPP_
