// vulkan/VulkanGraphicsPipeline.hpp
#pragma once

#include "VulkanBasePipeline.hpp"
#include <vector>

namespace vkop {

class VulkanGraphicsPipeline : public VulkanBasePipeline {
  public:
    VulkanGraphicsPipeline(VkDevice device, VkRenderPass render_pass,
                           std::vector<VkDescriptorType> types,
                           VkExtent2D extent, const uint32_t *vert_spv,
                           unsigned int vert_spv_len, const uint32_t *frag_spv,
                           unsigned int frag_spv_len);

  private:
    VkExtent2D m_extent_;

    void createGraphicsPipeline(VkRenderPass render_pass,
                                const uint32_t *vert_spv,
                                unsigned int vert_spv_len,
                                const uint32_t *frag_spv,
                                unsigned int frag_spv_len);
};

} // namespace vkop