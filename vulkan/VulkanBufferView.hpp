// Copyright 2025 @junka
#ifndef SRC_VULKANBUFFERVIEW_HPP_
#define SRC_VULKANBUFFERVIEW_HPP_

#include "vulkan/VulkanBuffer.hpp"

namespace vkop {

class VulkanBufferView : public VulkanResource {
  public:
    VulkanBufferView(std::shared_ptr<VulkanDevice> &vdev,
                     const std::shared_ptr<VulkanBuffer> &vkbuf,
                     VkFormat format, VkDeviceSize size, VkDeviceSize offset);
    ~VulkanBufferView() override;
    VkBufferView getBufferView() const { return m_buffer_view_; };

    std::variant<VkDescriptorImageInfo *, VkDescriptorBufferInfo *,
                 VkBufferView *>
    getDescriptorInfo() override;

    ResourceType getResourceType() const override {
        return ResourceType::VK_BUFFER_VIEW;
    }

    std::shared_ptr<VulkanBuffer> getBuffer() const { return m_vkbuf_; };

  private:
    std::shared_ptr<VulkanBuffer> m_vkbuf_;
    VkDeviceSize m_size_;
    VkDeviceSize m_offset_ = 0;
    VkBufferView m_buffer_view_ = VK_NULL_HANDLE;

    void createBufferView(VkFormat format, VkDeviceSize offset);
};
} // namespace vkop
#endif // SRC_VULKANBUFFERVIEW_HPP_
