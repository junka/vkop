// Copyright 2025 @junka
#ifndef SRC_VULKANRESOURCE_HPP_
#define SRC_VULKANRESOURCE_HPP_

#include "vulkan/VulkanDevice.hpp"
#include <memory>
#include <variant>

#define STORAGE VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
#define UNIFORM VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT
#define DESCRIPTOR_TYPE_STORAGE VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
#define DESCRIPTOR_TYPE_UNIFORM VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER

namespace vkop {
// Enum to represent the type of resource
enum class ResourceType { VK_IMAGE, VK_BUFFER, VK_BUFFER_VIEW };

// Abstract base class for Vulkan resources
class VulkanResource {
  public:
    explicit VulkanResource(std::shared_ptr<VulkanDevice> &vdev)
        : m_vdev_(vdev) {}

    virtual ~VulkanResource() {
#ifndef USE_VMA
        if (m_memory_)
            vkFreeMemory(m_vdev_->getLogicalDevice(), m_memory_, nullptr);
#endif
    }

    // Method to get the resource type
    virtual ResourceType getResourceType() const = 0;

    // Method to get descriptor info
    virtual std::variant<VkDescriptorImageInfo *, VkDescriptorBufferInfo *,
                         VkBufferView *>
    getDescriptorInfo() = 0;

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

    VkDescriptorType getDescriptorType() const { return m_desc_type_; }

    VulkanResource() = delete;
    VulkanResource(const VulkanResource &buff) = delete;
    VulkanResource(const VulkanResource &&buff) = delete;
    VulkanResource &operator=(const VulkanResource &) = delete;
    VulkanResource &operator=(const VulkanResource &&) = delete;

  protected:
    std::shared_ptr<VulkanDevice> m_vdev_;
    VkDescriptorType m_desc_type_;

  private:
#ifndef USE_VMA
    VkDeviceMemory m_memory_;
    uint64_t offset_ = 0;
#endif
};

} // namespace vkop
#endif // SRC_VULKANRESOURCE_HPP_
