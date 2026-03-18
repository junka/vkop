// vulkan/VulkanGraphicsPipeline.hpp
#pragma once

#include "vulkan/vulkan_core.h"
#include <memory>
#include <vector>
#include <vulkan/vulkan.h>

namespace vkop {

class VulkanGraphicsPipeline {
  public:
    VulkanGraphicsPipeline(VkDevice device, VkRenderPass render_pass,
                           std::vector<VkDescriptorType> types,
                           VkExtent2D extent, const uint32_t *vert_spv,
                           unsigned int vert_spv_len, const uint32_t *frag_spv,
                           unsigned int frag_spv_len);

    ~VulkanGraphicsPipeline();

    VkPipeline getPipeline() const { return m_pipeline_; }
    VkPipelineLayout getPipelineLayout() const { return m_pipelineLayout_; }

    VkDescriptorSet allocDescriptorSets();
    void freeDescriptorSets(VkDescriptorSet ds);

    void updateDescriptorSets(const std::vector<VkWriteDescriptorSet> &writes);

  private:
    VkDevice m_device_;
    std::vector<VkDescriptorType> m_types_;
    VkExtent2D m_extent_;

    VkDescriptorSetLayout m_descriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipeline m_pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout_ = VK_NULL_HANDLE;

    VkDescriptorPool m_descriptorPool_ = VK_NULL_HANDLE;
    void createDescriptorSetLayout(VkShaderStageFlags flags);

    void createPipeline(VkRenderPass render_pass, const uint32_t *vert_spv,
                        unsigned int vert_spv_len, const uint32_t *frag_spv,
                        unsigned int frag_spv_len);

    void createDescriptorPool();

    std::vector<VkDescriptorSetLayoutBinding>
    allocDescriptorSetLaoutBindings(VkShaderStageFlags flags);
};

} // namespace vkop