#ifndef VULKAN_BUFFER_HPP
#define VULKAN_BUFFER_HPP

#include <variant>
#include <vulkan/vulkan.hpp>

#include "VulkanResource.hpp"

namespace vkop {

class VulkanBuffer : public VulkanResource {
public:
    VulkanBuffer(VkPhysicalDevice physicalDevice, const uint32_t queueFamilyIndex, VkDevice device, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties);
    ~VulkanBuffer();

    VkBuffer getBuffer() const;
    ResourceType getResourceType() const override {
        return ResourceType::Buffer;
    }
    std::variant<VkDescriptorImageInfo, VkDescriptorBufferInfo> getDescriptorInfo() const override;

private:
    VkBuffer m_buffer;

    VkDeviceSize m_size;
    // VkDeviceSize m_offset;

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage);
    void cleanup();
};
}
#endif // VULKAN_BUFFER_HPP
