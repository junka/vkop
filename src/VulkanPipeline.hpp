#ifndef VULKAN_PIPELINE_HPP
#define VULKAN_PIPELINE_HPP

#include <memory>
#include <vulkan/vulkan.h>
#include <vector>

#include "VulkanResource.hpp"


namespace vkop {
class VulkanPipeline {
public:
    VulkanPipeline(VkDevice device, std::vector<VkDescriptorType> types, std::vector<std::shared_ptr<VulkanResource>> objs, const uint32_t *spirv, int codesize);
    ~VulkanPipeline();

    VkPipeline getComputePipeline() const { return m_pipeline; }
    VkPipelineLayout getPipelineLayout() const { return m_pipelineLayout; }
    VkDescriptorSet getDescriptorSet() const { return m_descriptorSet; }

private:
    VkDevice m_device;
    std::vector<VkDescriptorType> m_types;
    std::vector<std::shared_ptr<VulkanResource>> m_objs;

    VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;

    VkDescriptorSet m_descriptorSet = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;


    void createPipelineLayout();
    void createDescriptorSetLayout();
    VkDescriptorSetLayout getDescriptorSetLayout() const { return m_descriptorSetLayout; }

    void createComputePipeline(VkPipelineLayout pipelineLayout, VkShaderModule shaderModule);

    void createDescriptorPool();
    void allocDescriptorSets();
    void updateDescriptorSets();
    void createDescriptorUpdataTemplate();

    std::vector<VkDescriptorSetLayoutBinding> allocDescriptorSetLaoutBindings(void);
    void cleanup();
};
}

#endif // VULKAN_PIPELINE_HPP