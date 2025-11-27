// Copyright 2025 @junka
#ifndef SRC_VULKANPIPELINE_HPP_
#define SRC_VULKANPIPELINE_HPP_

#include <memory>
#include <vector>

#include "vulkan/VulkanResource.hpp"

namespace vkop {
constexpr int kInflight = 2;
class VulkanPipeline {
  public:
    VulkanPipeline(VkDevice device, std::vector<VkDescriptorType> types,
                   size_t pushconstant_size, const uint32_t *spirv,
                   int codesize);
    ~VulkanPipeline();

    VkPipeline getComputePipeline() const { return m_pipeline_; }
    VkPipelineLayout getPipelineLayout() const { return m_pipelineLayout_; }

    VkDescriptorSet allocDescriptorSets();
    void freeDescriptorSets(VkDescriptorSet ds);

    void updateDescriptorSets(
        VkDescriptorSet ds,
        const std::vector<std::shared_ptr<VulkanResource>> &m_objs, int n_img);

  private:
    VkDevice m_device_;
    std::vector<VkDescriptorType> m_types_;
    size_t m_pushconstant_size_ = 0;

    VkDescriptorSetLayout m_descriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipeline m_pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout_ = VK_NULL_HANDLE;

    VkDescriptorPool m_descriptorPool_ = VK_NULL_HANDLE;

    void createPipelineLayout();
    void createDescriptorSetLayout();
    VkDescriptorSetLayout getDescriptorSetLayout() const {
        return m_descriptorSetLayout_;
    }

    void createComputePipeline(VkPipelineLayout pipelineLayout,
                               VkShaderModule shaderModule);

    void createDescriptorPool();
    // void createDescriptorUpdataTemplate();

    std::vector<VkDescriptorSetLayoutBinding> allocDescriptorSetLaoutBindings();
    void cleanup();
};
} // namespace vkop

#endif // SRC_VULKANPIPELINE_HPP_
