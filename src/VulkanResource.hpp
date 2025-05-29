// Copyright 2025 @junka
#ifndef SRC_VULKANRESOURCE_HPP_
#define SRC_VULKANRESOURCE_HPP_

#include "VulkanLib.hpp"

#include <variant>
#include <vulkan/vulkan.h>

namespace vkop {
// Enum to represent the type of resource
enum class ResourceType { VK_IMAGE, VK_BUFFER };

// Abstract base class for Vulkan resources
class VulkanResource {
  public:
    explicit VulkanResource(VkPhysicalDevice physicalDevice,
                            const uint32_t queueFamilyIndex, VkDevice device)
        : m_physicalDevice_(physicalDevice),
          m_queueFamilyIndex_(queueFamilyIndex), m_device_(device) {}

    virtual ~VulkanResource() {
        if (m_memory_)
            vkFreeMemory(m_device_, m_memory_, nullptr);
    }

    // Method to get the resource type
    virtual ResourceType getResourceType() const = 0;

    // Method to get descriptor info
    virtual std::variant<VkDescriptorImageInfo, VkDescriptorBufferInfo>
    getDescriptorInfo() const = 0;

    static int32_t
    findMemoryTypeFromProperties(uint32_t memoryTypeBits,
                                 VkPhysicalDeviceMemoryProperties properties,
                                 VkMemoryPropertyFlags requiredProperties) {
        for (uint32_t index = 0; index < properties.memoryTypeCount; ++index) {
            if (((memoryTypeBits & (1 << index))) &&
                ((properties.memoryTypes[index].propertyFlags &
                  requiredProperties) == requiredProperties)) {
                return static_cast<int32_t>(index);
            }
        }
        return -1;
    }
    bool allocMemory(VkMemoryRequirements memoryRequirements,
                     VkMemoryPropertyFlags requiredProperties) {
        VkPhysicalDeviceMemoryProperties memory_properties;
        vkGetPhysicalDeviceMemoryProperties(m_physicalDevice_,
                                            &memory_properties);

        auto memory_type_index =
            findMemoryTypeFromProperties(memoryRequirements.memoryTypeBits,
                                         memory_properties, requiredProperties);
        VkMemoryAllocateInfo allocate_info = {};
        allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocate_info.allocationSize = memoryRequirements.size;
        allocate_info.memoryTypeIndex = memory_type_index;
        return vkAllocateMemory(m_device_, &allocate_info, nullptr,
                                &m_memory_) == VK_SUCCESS;
    }

    VkDeviceMemory getMemory() const { return m_memory_; }

    void *getMappedMemory() const {
        void *data;
        vkMapMemory(m_device_, m_memory_, 0, VK_WHOLE_SIZE, 0, &data);
        return data;
    }
    void unmapMemory() { vkUnmapMemory(m_device_, m_memory_); }

    VulkanResource() = delete;
    VulkanResource(const VulkanResource &buff) = delete;
    VulkanResource(const VulkanResource &&buff) = delete;
    VulkanResource &operator=(const VulkanResource &) = delete;
    VulkanResource &operator=(const VulkanResource &&) = delete;
    // Pure virtual method to bind memory, must be implemented in subclasses
    // virtual void bindMemory(VkDeviceSize memoryOffset = 0) = 0;
  protected:
    VkPhysicalDevice m_physicalDevice_;
    const uint32_t m_queueFamilyIndex_;
    VkDevice m_device_;

  private:
    VkDeviceMemory m_memory_;
};

} // namespace vkop
#endif // SRC_VULKANRESOURCE_HPP_
