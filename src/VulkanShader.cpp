#include "VulkanLib.hpp"
#include "VulkanShader.hpp"

#include <stdexcept>

namespace vkop {

VulkanShader::VulkanShader(VkDevice device, const uint32_t* spirvCode, size_t codeSize)
    : m_device(device) {
    if (m_device == VK_NULL_HANDLE) {
        throw std::runtime_error("Invalid Vulkan device handle.");
    }
    m_shaderModule = loadShaderModule(spirvCode, codeSize);
}

VulkanShader::~VulkanShader() {
    if (m_shaderModule != VK_NULL_HANDLE)
        vkDestroyShaderModule(m_device, m_shaderModule, nullptr);
}

VkShaderModule VulkanShader::loadShaderModule(const uint32_t* spirvCode, size_t codeSize) {
    if (!spirvCode || codeSize == 0) {
        throw std::runtime_error("Invalid SPIR-V code or size.");
    }

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = codeSize;
    createInfo.pCode = spirvCode;

    VkShaderModule shaderModule;
    VkResult result = vkCreateShaderModule(m_device, &createInfo, nullptr, &shaderModule);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan shader module.");
    }

    return shaderModule;
}

} // namespace vkop