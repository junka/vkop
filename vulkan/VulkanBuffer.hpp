// Copyright 2025 @junka
#ifndef SRC_VULKANBUFFER_HPP_
#define SRC_VULKANBUFFER_HPP_

#include <variant>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>

#include "vulkan/VulkanResource.hpp"

namespace vkop {

class VulkanBuffer : public VulkanResource {
  public:
    VulkanBuffer(std::shared_ptr<VulkanDevice> &vdev, VkDeviceSize size,
                 VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                 VkFormat format = VK_FORMAT_UNDEFINED, int ext_fd = -1);
    ~VulkanBuffer() override;

    VkBuffer getBuffer() const;
    VkBufferView getBufferView() const { return m_buffer_view_; };
    ResourceType getResourceType() const override {
        return ResourceType::VK_BUFFER;
    }
    std::variant<VkDescriptorImageInfo *, VkDescriptorBufferInfo *>
    getDescriptorInfo() override;

    void transferBarrier(VkCommandBuffer commandBuffer,
                         VkAccessFlags dstAccessMask);
    void transferWriteBarrier(VkCommandBuffer commandBuffer);
    void transferReadBarrier(VkCommandBuffer commandBuffer);
    void readBarrier(VkCommandBuffer commandBuffer);
    void writeBarrier(VkCommandBuffer commandBuffer);

    void copyBufferToStageBuffer(VkCommandBuffer commandBuffer,
                                 VkBuffer dstbuffer, VkDeviceSize dstoffset);

    void copyStageBufferToBuffer(VkCommandBuffer commandBuffer,
                                 VkBuffer srcbuffer, VkDeviceSize srcoffset);

#ifdef USE_VMA
    void *getMappedMemory() override {
        return VMA::getMappedMemory(&m_vma_buffer_);
    };
#endif

  private:
#ifndef USE_VMA
    VkBuffer m_buffer_ = VK_NULL_HANDLE;
#else
    VMA::VmaBuffer m_vma_buffer_;
#endif
    VkDeviceSize m_size_;
    VkAccessFlags m_access_ = 0;
    // VkFormat m_format_;
    VkBufferView m_buffer_view_ = VK_NULL_HANDLE;
    VkDescriptorBufferInfo buffer_info_;

    void transitionBuffer(VkCommandBuffer commandBuffer,
                          VkAccessFlags dstAccessMask,
                          VkPipelineStageFlags src_stage,
                          VkPipelineStageFlags dst_stage, VkDeviceSize offset);
    void createBuffer(VkBufferUsageFlags usage, bool device_local);
    void createBufferView(VkFormat format);
};
} // namespace vkop
#endif // SRC_VULKANBUFFER_HPP_
