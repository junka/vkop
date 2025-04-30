#ifndef VULKAN_RESOURCE_HPP
#define VULKAN_RESOURCE_HPP

#include "VulkanLib.hpp"
#include <vulkan/vulkan.h>
#include <variant>

namespace vkop {
// Enum to represent the type of resource
enum class ResourceType {
    Image,
    Buffer
};

// Abstract base class for Vulkan resources
class VulkanResource {
public:
    explicit VulkanResource(VkPhysicalDevice physicalDevice, const uint32_t queueFamilyIndex, VkDevice device) : m_physicalDevice(physicalDevice), m_queueFamilyIndex(queueFamilyIndex), m_device(device) {}
    ~VulkanResource() {
        if (m_memory)
            vkFreeMemory(m_device, m_memory, nullptr);
    };

    // Method to get the resource type
    virtual ResourceType getResourceType() const = 0;

    // Method to get descriptor info
    virtual std::variant<VkDescriptorImageInfo, VkDescriptorBufferInfo> getDescriptorInfo() const = 0;

    int32_t findMemoryTypeFromProperties(uint32_t memoryTypeBits,
            VkPhysicalDeviceMemoryProperties properties,
            VkMemoryPropertyFlags requiredProperties)
    {
        for (uint32_t index = 0; index < properties.memoryTypeCount; ++index) {
            if (((memoryTypeBits & (1 << index))) &&
                ((properties.memoryTypes[index].propertyFlags & requiredProperties) == requiredProperties)) {
                return (int32_t)index;
            }
        }
        return -1;
    }
    bool allocMemory(VkMemoryRequirements memoryRequirements, VkMemoryPropertyFlags requiredProperties) {

        VkPhysicalDeviceMemoryProperties memoryProperties;
        vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memoryProperties);
        
        auto memoryTypeIndex = findMemoryTypeFromProperties(
            memoryRequirements.memoryTypeBits,
            memoryProperties,
            requiredProperties
        );
        VkMemoryAllocateInfo allocateInfo = {};
        allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.allocationSize = memoryRequirements.size;
        allocateInfo.memoryTypeIndex = memoryTypeIndex;
        return vkAllocateMemory(m_device, &allocateInfo, nullptr, &m_memory) == VK_SUCCESS;
    };

    VkDeviceMemory getMemory() const {
        return m_memory;
    }

    void *getMappedMemory() const {
        void *data;
        vkMapMemory(m_device, m_memory, 0, VK_WHOLE_SIZE, 0, &data);
        return data;
    }
    void unmapMemory() {
        vkUnmapMemory(m_device, m_memory);
    }

    VulkanResource() = delete;
    VulkanResource(const VulkanResource& buff)  = delete;
    VulkanResource(const VulkanResource&& buff) = delete;
    VulkanResource& operator=(const VulkanResource&) = delete;
    VulkanResource& operator=(const VulkanResource&&) = delete;
    // Pure virtual method to bind memory, must be implemented in subclasses
    // virtual void bindMemory(VkDeviceSize memoryOffset = 0) = 0;
protected:
    VkPhysicalDevice m_physicalDevice;
    const uint32_t m_queueFamilyIndex;
    VkDevice m_device;
private:
    VkDeviceMemory m_memory;
};

}
#endif // VULKAN_RESOURCE_HPP