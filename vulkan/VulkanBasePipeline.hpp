// vulkan/VulkanBasePipeline.hpp
#ifndef __VULKANBASEPIPELINE_HPP__
#define __VULKANBASEPIPELINE_HPP__

#include "vulkan/VulkanDevice.hpp"
#include <vector>

namespace vkop {

class VulkanBasePipeline {
  public:
    ~VulkanBasePipeline();

  protected:
    VulkanBasePipeline(VkDevice device, std::vector<VkDescriptorType> types);

    // Common members
    VkDevice m_device_;
    std::vector<VkDescriptorType> m_types_;

    VkDescriptorSetLayout m_descriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline m_pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptorPool_ = VK_NULL_HANDLE;

    // Common methods
    void createDescriptorSetLayout(VkShaderStageFlags flags);
    void createDescriptorPool();
    void createPipelineLayout(VkShaderStageFlags flags,
                              size_t pushconstant_size = 0);

    std::vector<VkDescriptorSetLayoutBinding>
    allocDescriptorSetLayoutBindings(VkShaderStageFlags flags);

  public:
    // Public interface
    VkPipeline getPipeline() const { return m_pipeline_; }
    VkPipelineLayout getPipelineLayout() const { return m_pipelineLayout_; }

    VkDescriptorSet allocDescriptorSets();
    void freeDescriptorSets(VkDescriptorSet ds);
    void updateDescriptorSets(const std::vector<VkWriteDescriptorSet> &writes);
};

} // namespace vkop

#endif // __VULKANBASEPIPELINE_HPP__