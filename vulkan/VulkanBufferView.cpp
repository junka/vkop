// Copyright 2025 @junka
#include "VulkanBufferView.hpp"
#include "VulkanLib.hpp"
#include <stdexcept>
#include <vulkan/vulkan_core.h>

namespace vkop {

VulkanBufferView::VulkanBufferView(std::shared_ptr<VulkanDevice> &vdev,
                                   const std::shared_ptr<VulkanBuffer> &vkbuf,
                                   VkFormat format, VkDeviceSize size,
                                   VkDeviceSize offset)
    : VulkanResource(vdev), m_vkbuf_(vkbuf), m_size_(size), m_offset_(offset) {
    VkBufferViewCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO;
    buffer_info.flags = 0;
    buffer_info.buffer = m_vkbuf_->getBuffer();
    buffer_info.format = format;
    buffer_info.range = m_size_;
    buffer_info.offset = offset;

    auto ret = vkCreateBufferView(m_vdev_->getLogicalDevice(), &buffer_info,
                                  nullptr, &m_buffer_view_);
    if (ret != VK_SUCCESS) {
        throw std::runtime_error("Failed to create buffer view");
    }
}

VulkanBufferView::~VulkanBufferView() {

    vkDestroyBufferView(m_vdev_->getLogicalDevice(), m_buffer_view_, nullptr);
}

std::variant<VkDescriptorImageInfo *, VkDescriptorBufferInfo *, VkBufferView *>
VulkanBufferView::getDescriptorInfo() {
    return &m_buffer_view_;
}

} // namespace vkop
