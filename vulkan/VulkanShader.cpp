// Copyright 2025 @junka
#include "vulkan/VulkanShader.hpp"
#include "vulkan/VulkanLib.hpp"

#include <stdexcept>

namespace vkop {

VulkanShader::VulkanShader(VkDevice device, const uint32_t *spirvCode,
                           size_t codeSize)
    : m_device_(device) {
    if (m_device_ == VK_NULL_HANDLE) {
        throw std::runtime_error("Invalid Vulkan device handle.");
    }
    m_shaderModule_ = loadShaderModule(spirvCode, codeSize);
}

VulkanShader::~VulkanShader() {
    if (m_shaderModule_ != VK_NULL_HANDLE)
        vkDestroyShaderModule(m_device_, m_shaderModule_, nullptr);
}

VkShaderModule VulkanShader::loadShaderModule(const uint32_t *spirvCode,
                                              size_t codeSize) {
    if (!spirvCode || codeSize == 0) {
        throw std::runtime_error("Invalid SPIR-V code or size.");
    }

    VkShaderModuleCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = codeSize;
    create_info.pCode = spirvCode;

    VkShaderModule shader_module;
    VkResult result =
        vkCreateShaderModule(m_device_, &create_info, nullptr, &shader_module);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan shader module.");
    }

    return shader_module;
}

} // namespace vkop
