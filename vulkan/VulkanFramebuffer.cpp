#include "vulkan/VulkanFrameBuffer.hpp"

namespace vkop {

VulkanFrameBuffer::VulkanFrameBuffer(const std::shared_ptr<VulkanDevice> &vdev,
                                     VulkanRenderPass &render_pass,
                                     VkImageView &attachments, int width,
                                     int height)
    : vdev_(vdev) {
    VkFramebufferCreateInfo framebuffer_info = {};
    framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebuffer_info.renderPass = render_pass.getRenderPass();
    framebuffer_info.attachmentCount = 1;
    framebuffer_info.pAttachments = &attachments;
    framebuffer_info.width = width;
    framebuffer_info.height = height;
    framebuffer_info.layers = 1;
    auto ret = vkCreateFramebuffer(vdev_->getLogicalDevice(), &framebuffer_info,
                                   nullptr, &frame_buffer_);
    if (ret != VK_SUCCESS) {
        throw std::runtime_error("Failed to create framebuffer");
    }
}

VulkanFrameBuffer::~VulkanFrameBuffer() {
    if (frame_buffer_ == VK_NULL_HANDLE)
        return;
    vkDestroyFramebuffer(vdev_->getLogicalDevice(), frame_buffer_, nullptr);
}

} // namespace vkop