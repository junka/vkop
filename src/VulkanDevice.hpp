// Copyright 2025 @junka
#ifndef SRC_VULKANDEVICE_HPP_
#define SRC_VULKANDEVICE_HPP_

#include <vector>
#include <vulkan/vulkan.hpp>

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

    std::string getDeviceName() const { return deviceProperties_.deviceName; }

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

    bool checkDeviceUnifiedMemoryAccess();

  private:
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice logicalDevice_ = VK_NULL_HANDLE;
    VkQueue computeQueue_ = VK_NULL_HANDLE;
    // Queue family index
    int computeQueueFamilyIndex_ = -1;

    std::vector<VkExtensionProperties> ext_properties_;
    VkPhysicalDeviceProperties deviceProperties_;
    float timestampPeriod_;

    std::vector<VkImageLayout> copySrcLayout_;
    std::vector<VkImageLayout> copyDstLayout_;

    std::vector<uintptr_t> enabledFeatures_;
    std::vector<const char *> enabledExtensions_;

    // Helper functions
    void getProperties();

    bool createLogicalDevice();

    // Device suitability checks
    bool isDeviceSuitable();

    int findComputeQueueFamily();

    bool checkDeviceExtensionFeature(const char *name) const;

    bool m_support_host_image_copy_ = false;
};

} // namespace vkop

#endif // SRC_VULKANDEVICE_HPP_
