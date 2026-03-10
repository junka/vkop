#ifndef VULKAN_RENDERPASS_HPP_
#define VULKAN_RENDERPASS_HPP_

#include "vulkan/VulkanLib.hpp"

namespace vkop {

class VulkanRenderPass {
  public:
    VulkanRenderPass(VkDevice device, VkFormat format);
    ~VulkanRenderPass();

    VkRenderPass getRenderPass() const { return renderPass_; };

    void begin(VkCommandBuffer commandBuffer, VkFramebuffer framebuffer,
               uint32_t width, uint32_t height);
    void end(VkCommandBuffer commandBuffer);

  private:
    VkDevice device_;
    VkFormat format_;
    VkRenderPass renderPass_;
};

} // namespace vkop

#endif // VULKAN_RENDERPASS_HPP_