// Copyright 2025 @junka
#ifndef SRC_VULKANBUFFER_HPP_
#define SRC_VULKANBUFFER_HPP_

#include <variant>
#include <vulkan/vulkan.hpp>

#include "vulkan/VulkanResource.hpp"

namespace vkop {

class VulkanBuffer : public VulkanResource {
  public:
    VulkanBuffer(std::shared_ptr<VulkanDevice> &vdev, VkDeviceSize size,
                 VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                 int ext_fd = -1);
    ~VulkanBuffer() override;

    VkBuffer getBuffer() const;
    ResourceType getResourceType() const override {
        return ResourceType::VK_BUFFER;
    }
    std::variant<VkDescriptorImageInfo *, VkDescriptorBufferInfo *,
                 VkBufferView *>
    getDescriptorInfo() override;

    VkDeviceSize getSize() const { return m_size_; }

    void transferBarrier(VkCommandBuffer commandBuffer,
                         VkAccessFlags dstAccessMask,
                         VkDeviceSize size = VK_WHOLE_SIZE,
                         VkDeviceSize offset = 0);
    void transferWriteBarrier(VkCommandBuffer commandBuffer,
                              VkDeviceSize size = VK_WHOLE_SIZE,
                              VkDeviceSize offset = 0);
    void transferReadBarrier(VkCommandBuffer commandBuffer,
                             VkDeviceSize size = VK_WHOLE_SIZE,
                             VkDeviceSize offset = 0);
    void readBarrier(VkCommandBuffer commandBuffer,
                     VkDeviceSize size = VK_WHOLE_SIZE,
                     VkDeviceSize offset = 0);
    void writeBarrier(VkCommandBuffer commandBuffer,
                      VkDeviceSize size = VK_WHOLE_SIZE,
                      VkDeviceSize offset = 0);

    void copyBufferToStageBuffer(VkCommandBuffer commandBuffer,
                                 VkBuffer dstbuffer, VkDeviceSize dstoffset,
                                 VkDeviceSize size, VkDeviceSize offset = 0);

    void copyStageBufferToBuffer(VkCommandBuffer commandBuffer,
                                 VkBuffer srcbuffer, VkDeviceSize srcoffset,
                                 VkDeviceSize size, VkDeviceSize offset = 0);

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

    VkDescriptorBufferInfo buffer_info_;

    void transitionBuffer(VkCommandBuffer commandBuffer,
                          VkAccessFlags dstAccessMask,
                          VkPipelineStageFlags src_stage,
                          VkPipelineStageFlags dst_stage,
                          VkDeviceSize size = VK_WHOLE_SIZE,
                          VkDeviceSize offset = 0);
    void createBuffer(VkBufferUsageFlags usage, bool device_local);
    void createBufferView(VkFormat format, VkDeviceSize offset);
};
} // namespace vkop
#endif // SRC_VULKANBUFFER_HPP_
