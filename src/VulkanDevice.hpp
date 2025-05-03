#ifndef VULKAN_DEVICE_HPP
#define VULKAN_DEVICE_HPP


#include <vulkan/vulkan.hpp>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace vkop {

class VulkanDevice {
public:
    VulkanDevice(VkPhysicalDevice physicalDevice);
    ~VulkanDevice();

    // Initialization and cleanup
    void create();
    void destroy();

    // Getters
    VkDevice getLogicalDevice() const { return logicalDevice; }
    VkPhysicalDevice getPhysicalDevice() const { return physicalDevice; }
    VkQueue getComputeQueue() const { return computeQueue; }

    std::string getDeviceName() const { return deviceProperties.deviceName; }

    int getComputeQueueFamilyIndex() const { return computeQueueFamilyIndex; }

    float getTimestampPeriod() const { return timestampPeriod; }

    
    bool checkHostImageCopyDstLayoutSupport(VkImageLayout layout)
    {
        for (auto i : this->copyDstLayout) {
            if (i == layout) {
                return true;
            }
        }
        return false;
    }
    bool is_support_host_image_copy() const {
        return m_support_host_image_copy;
    }

    bool checkDeviceUnifiedMemoryAccess();
private:
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice logicalDevice = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
    // Queue family index
    int computeQueueFamilyIndex = -1;

    std::vector<VkExtensionProperties> ext_properties;
    VkPhysicalDeviceProperties deviceProperties;
    float timestampPeriod;

    std::vector<VkImageLayout> copySrcLayout;
    std::vector<VkImageLayout> copyDstLayout;

    std::vector<uintptr_t> enabledFeatures;
    std::vector<const char *> enabledExtensions;

    // Helper functions
    void getProperties();

    bool createLogicalDevice();

    // Device suitability checks
    bool isDeviceSuitable(VkPhysicalDevice device);

    int findComputeQueueFamily(VkPhysicalDevice device);

    bool checkDeviceExtensionFeature(const char *name) const;

    bool m_support_host_image_copy;
};

} // namespace vkop

#endif // VULKAN_DEVICE_HPP