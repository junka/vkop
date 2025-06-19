// Copyright 2025 @junka
#ifndef SRC_VULKANPIPELINE_HPP_
#define SRC_VULKANPIPELINE_HPP_

#include <memory>
#include <vector>
#include <vulkan/vulkan.h>

#include "vulkan/VulkanResource.hpp"

namespace vkop {
class VulkanPipeline {
  public:
    VulkanPipeline(VkDevice device, std::vector<VkDescriptorType> types,
                   std::vector<std::shared_ptr<VulkanResource>> objs,
                   const uint32_t *spirv, int codesize);
    ~VulkanPipeline();

    VkPipeline getComputePipeline() const { return m_pipeline_; }
    VkPipelineLayout getPipelineLayout() const { return m_pipelineLayout_; }
    VkDescriptorSet getDescriptorSet() const { return m_descriptorSet_; }

  private:
    VkDevice m_device_;
    std::vector<VkDescriptorType> m_types_;
    std::vector<std::shared_ptr<VulkanResource>> m_objs_;

    VkDescriptorSetLayout m_descriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipeline m_pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout_ = VK_NULL_HANDLE;

    VkDescriptorSet m_descriptorSet_ = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptorPool_ = VK_NULL_HANDLE;

    void createPipelineLayout();
    void createDescriptorSetLayout();
    VkDescriptorSetLayout getDescriptorSetLayout() const {
        return m_descriptorSetLayout_;
    }

    void createComputePipeline(VkPipelineLayout pipelineLayout,
                               VkShaderModule shaderModule);

    void createDescriptorPool();
    void allocDescriptorSets();
    void updateDescriptorSets();
    void createDescriptorUpdataTemplate();

    std::vector<VkDescriptorSetLayoutBinding> allocDescriptorSetLaoutBindings();
    void cleanup();
};
} // namespace vkop

#endif // SRC_VULKANPIPELINE_HPP_
