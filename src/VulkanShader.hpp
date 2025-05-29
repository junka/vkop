// Copyright 2025 @junka
#ifndef SRC_VULKANSHADER_HPP_
#define SRC_VULKANSHADER_HPP_

#include <vulkan/vulkan.hpp>

namespace vkop {

class VulkanShader {
  public:
    VulkanShader(VkDevice device, const uint32_t *spirvCode, size_t codeSize);
    ~VulkanShader();

    VkShaderModule getShaderModule() const { return m_shaderModule_; }

  private:
    // Load shader module from embedded SPIR-V array
    VkShaderModule loadShaderModule(const uint32_t *spirvCode, size_t codeSize);

    VkDevice m_device_;
    VkShaderModule m_shaderModule_;
};

} // namespace vkop

#endif // SRC_VULKANSHADER_HPP_
