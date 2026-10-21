// Copyright 2025 @junka
#ifndef SRC_VULKANPIPELINE_HPP_
#define SRC_VULKANPIPELINE_HPP_

#include <vector>

#include "vulkan/VulkanBasePipeline.hpp"

namespace vkop {
class VulkanPipeline : public VulkanBasePipeline {
  public:
    VulkanPipeline(VkDevice device, std::vector<VkDescriptorType> types,
                   size_t pushconstant_size, const uint32_t *spirv,
                   int codesize, bool update_after_bind,
                   uint32_t required_subgroup_size = 0,
                   VkPipelineCache cache = VK_NULL_HANDLE);

  private:
    size_t m_pushconstant_size_;
    uint32_t required_subgroup_size_;

    void createComputePipeline(VkPipelineLayout pipelineLayout,
                               VkShaderModule shaderModule,
                               VkPipelineCache cache, int spv_size = 0);
};
} // namespace vkop

#endif // SRC_VULKANPIPELINE_HPP_
