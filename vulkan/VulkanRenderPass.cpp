#include "vulkan/VulkanRenderPass.hpp"

namespace vkop {
VulkanRenderPass::VulkanRenderPass(VkDevice device, VkFormat format)
    : device_(device), format_(format) {
    VkAttachmentDescription color_attachment{};
    color_attachment.format = format;
    color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference color_attachment_ref{};
    color_attachment_ref.attachment = 0;
    color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_attachment_ref;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderpass_info{};
    renderpass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderpass_info.attachmentCount = 1;
    renderpass_info.pAttachments = &color_attachment;
    renderpass_info.subpassCount = 1;
    renderpass_info.pSubpasses = &subpass;
    renderpass_info.dependencyCount = 1;
    renderpass_info.pDependencies = &dependency;

    auto ret =
        vkCreateRenderPass(device, &renderpass_info, nullptr, &renderPass_);
    if (ret != VK_SUCCESS) {
        throw std::runtime_error("Failed to create RenderPass");
    }
}

VulkanRenderPass::~VulkanRenderPass() {
    if (renderPass_ == VK_NULL_HANDLE) {
        return;
    }
    vkDestroyRenderPass(device_, renderPass_, nullptr);
}

void VulkanRenderPass::begin(VkCommandBuffer commandBuffer,
                             VkFramebuffer framebuffer, uint32_t width,
                             uint32_t height) {

    VkRenderPassBeginInfo renderpass_begininfo = {};
    VkClearValue clear_values = {};
    clear_values.color = {{0.0F, 0.0F, 1.0F, 1.0F}};

    renderpass_begininfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderpass_begininfo.renderPass = renderPass_;
    renderpass_begininfo.framebuffer = framebuffer;
    renderpass_begininfo.renderArea.offset = {0, 0};
    renderpass_begininfo.renderArea.extent = {width, height};
    renderpass_begininfo.clearValueCount = 1;
    renderpass_begininfo.pClearValues = &clear_values;

    vkCmdBeginRenderPass(commandBuffer, &renderpass_begininfo,
                         VK_SUBPASS_CONTENTS_INLINE);

    // VkViewport viewport = {};
    // viewport.x = 0.0F;
    // viewport.y = 0.0F;
    // viewport.width = static_cast<float>(width);
    // viewport.height = static_cast<float>(height);
    // viewport.minDepth = 0.0F;
    // viewport.maxDepth = 1.0F;
    // vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    // VkRect2D scissor = {};
    // scissor.offset.x = 0;
    // scissor.offset.y = 0;
    // scissor.extent.width = width;
    // scissor.extent.height = height;
    // vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
}

void VulkanRenderPass::end(VkCommandBuffer commandBuffer) {
    vkCmdEndRenderPass(commandBuffer);
}

} // namespace vkop