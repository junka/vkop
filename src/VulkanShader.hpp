#ifndef VULKAN_SHADER_HPP
#define VULKAN_SHADER_HPP

#include <vulkan/vulkan.hpp>

namespace vkop {

class VulkanShader {
public:
    VulkanShader(VkDevice device, const uint32_t* spirvCode, size_t codeSize);
    ~VulkanShader();

    VkShaderModule getShaderModule() const { return m_shaderModule; }

private:
    // Load shader module from embedded SPIR-V array
    VkShaderModule loadShaderModule(const uint32_t* spirvCode, size_t codeSize);

    VkDevice m_device;
    VkShaderModule m_shaderModule;
};

} // namespace vkop

#endif // VULKAN_SHADER_HPP