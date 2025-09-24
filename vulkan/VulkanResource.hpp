// Copyright 2025 @junka
#ifndef SRC_VULKANRESOURCE_HPP_
#define SRC_VULKANRESOURCE_HPP_

#include "vulkan/VulkanDevice.hpp"

#include <variant>

namespace vkop {
// Enum to represent the type of resource
enum class ResourceType { VK_IMAGE, VK_BUFFER };

// Abstract base class for Vulkan resources
class VulkanResource {
  public:
    explicit VulkanResource(std::shared_ptr<VulkanDevice> &vdev,
                            const uint32_t queueFamilyIndex)
        : m_vdev_(vdev), m_queueFamilyIndex_(queueFamilyIndex) {}

    virtual ~VulkanResource() {
#ifndef USE_VMA
        if (m_memory_)
            vkFreeMemory(m_vdev_->getLogicalDevice(), m_memory_, nullptr);
#endif
    }

    // Method to get the resource type
    virtual ResourceType getResourceType() const = 0;

    // Method to get descriptor info
    virtual std::variant<VkDescriptorImageInfo, VkDescriptorBufferInfo>
    getDescriptorInfo() const = 0;

#ifndef USE_VMA
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
                     VkMemoryPropertyFlags requiredProperties,
                     int ext_fd = -1) {
        VkPhysicalDeviceMemoryProperties memory_properties;
        vkGetPhysicalDeviceMemoryProperties(m_vdev_->getPhysicalDevice(),
                                            &memory_properties);

        auto memory_type_index =
            findMemoryTypeFromProperties(memoryRequirements.memoryTypeBits,
                                         memory_properties, requiredProperties);
        VkMemoryAllocateInfo allocate_info = {};
        allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocate_info.allocationSize = memoryRequirements.size;
        allocate_info.memoryTypeIndex = memory_type_index;
#ifdef VK_KHR_external_memory_fd
        VkImportMemoryFdInfoKHR importfdinfo = {};
        importfdinfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
        importfdinfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        importfdinfo.fd = ext_fd;
        if (ext_fd != -1)
            allocate_info.pNext = &importfdinfo;
#endif
        return vkAllocateMemory(m_vdev_->getLogicalDevice(), &allocate_info,
                                nullptr, &m_memory_) == VK_SUCCESS;
    }

    VkDeviceMemory getMemory() const { return m_memory_; }

#endif
    virtual void *getMappedMemory() {
        void *data = nullptr;
#ifndef USE_VMA
        vkMapMemory(m_vdev_->getLogicalDevice(), m_memory_, offset_,
                    VK_WHOLE_SIZE, 0, &data);
#endif
        return data;
    }
    void unmapMemory() {
#ifndef USE_VMA
        vkUnmapMemory(m_vdev_->getLogicalDevice(), m_memory_);
#endif
    }

    VulkanResource() = delete;
    VulkanResource(const VulkanResource &buff) = delete;
    VulkanResource(const VulkanResource &&buff) = delete;
    VulkanResource &operator=(const VulkanResource &) = delete;
    VulkanResource &operator=(const VulkanResource &&) = delete;

  protected:
    std::shared_ptr<VulkanDevice> m_vdev_;
    const uint32_t m_queueFamilyIndex_;

  private:
#ifndef USE_VMA
    VkDeviceMemory m_memory_;
    uint64_t offset_ = 0;
#endif
};

} // namespace vkop
#endif // SRC_VULKANRESOURCE_HPP_
