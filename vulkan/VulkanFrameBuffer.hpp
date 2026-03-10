#ifndef VULKAN_FRAMEBUFFER_HPP_
#define VULKAN_FRAMEBUFFER_HPP_

#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanRenderPass.hpp"

namespace vkop {

class VulkanFrameBuffer {
  public:
    VulkanFrameBuffer(const std::shared_ptr<VulkanDevice> &vdev,
                      VulkanRenderPass &render_pass, VkImageView &attachments,
                      int width, int height);
    ~VulkanFrameBuffer();

    VkFramebuffer getFrameBuffer() const { return frame_buffer_; }

  private:
    std::shared_ptr<VulkanDevice> vdev_;
    VkFramebuffer frame_buffer_;
};

} // namespace vkop

#endif // VULKAN_FRAMEBUFFER_HPP_