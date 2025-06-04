// Copyright 2025 @junka
#ifndef SRC_VULKANBUFFER_HPP_
#define SRC_VULKANBUFFER_HPP_

#include <variant>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>

#include "VulkanResource.hpp"

namespace vkop {

class VulkanBuffer : public VulkanResource {
  public:
    VulkanBuffer(VkPhysicalDevice &physicalDevice, uint32_t queueFamilyIndex,
                 VkDevice &device, VkDeviceSize size, VkBufferUsageFlags usage,
                 VkMemoryPropertyFlags properties);
    ~VulkanBuffer() override;

    VkBuffer getBuffer() const;
    ResourceType getResourceType() const override {
        return ResourceType::VK_BUFFER;
    }
    std::variant<VkDescriptorImageInfo, VkDescriptorBufferInfo>
    getDescriptorInfo() const override;

    void transferWriteBarrier(VkCommandBuffer commandBuffer);
    void transferReadBarrier(VkCommandBuffer commandBuffer);

  private:
    VkBuffer m_buffer_ = VK_NULL_HANDLE;

    VkDeviceSize m_size_;
    // VkDeviceSize m_offset;

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage);
};
} // namespace vkop
#endif // SRC_VULKANBUFFER_HPP_
