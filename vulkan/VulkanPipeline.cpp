// Copyright 2025 @junka
#include "VulkanLib.hpp"

#include <map>
#include <stdexcept>
#include <vector>

#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanPipeline.hpp"
#include "vulkan/VulkanShader.hpp"

namespace vkop {
VulkanPipeline::VulkanPipeline(VkDevice device,
                               std::vector<VkDescriptorType> types,
                               size_t pushconstant_size, const uint32_t *spirv,
                               int codesize)
    : VulkanBasePipeline(device, std::move(types)),
      m_pushconstant_size_(pushconstant_size) {
    VulkanShader shader(device, spirv, codesize);
    createDescriptorSetLayout(VK_SHADER_STAGE_COMPUTE_BIT);
    createPipelineLayout(VK_SHADER_STAGE_COMPUTE_BIT, pushconstant_size);
    createDescriptorPool();

    createComputePipeline(m_pipelineLayout_, shader.getShaderModule());
    // could destroy shader here
}

void VulkanPipeline::createComputePipeline(VkPipelineLayout pipelineLayout,
                                           VkShaderModule shaderModule) {
    VkPipelineShaderStageCreateInfo shader_stage_create_info = {};
    shader_stage_create_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stage_create_info.pNext = nullptr;
    shader_stage_create_info.flags = 0;
    shader_stage_create_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shader_stage_create_info.module = shaderModule;
    shader_stage_create_info.pName = "main";
    shader_stage_create_info.pSpecializationInfo = nullptr;

    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = shader_stage_create_info;
    pipeline_info.layout = pipelineLayout;

    auto ret = vkCreateComputePipelines(m_device_, VK_NULL_HANDLE, 1,
                                        &pipeline_info, nullptr, &m_pipeline_);
    if (ret != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute pipeline!" +
                                 std::to_string(ret));
    }
}

// void VulkanPipeline::createDescriptorUpdataTemplate() {
//     std::vector<VkDescriptorUpdateTemplateEntry> entries(m_types_.size());
//     for (size_t i = 0; i < m_types_.size(); i++) {
//         entries[i].dstBinding = static_cast<uint32_t>(i);
//         entries[i].dstArrayElement = 0;
//         entries[i].descriptorCount = 1;
//         entries[i].descriptorType = m_types_[i];
//         entries[i].offset = sizeof(VkDescriptorSetLayoutBinding) * i;
//         if (m_objs_[i]->getResourceType() == ResourceType::VK_BUFFER) {
//             entries[i].stride = sizeof(VkDescriptorBufferInfo);
//         } else {
//             entries[i].stride = sizeof(VkDescriptorImageInfo);
//         }
//     }

//     VkDescriptorUpdateTemplateCreateInfo tempcreateinfo = {};
//     tempcreateinfo.sType =
//         VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO;
//     tempcreateinfo.templateType =
//         VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET;
//     tempcreateinfo.descriptorSetLayout = m_descriptorSetLayout_;
//     tempcreateinfo.pipelineLayout = m_pipelineLayout_;
//     tempcreateinfo.pipelineBindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;
//     tempcreateinfo.descriptorUpdateEntryCount =
//         static_cast<uint32_t>(m_types_.size());
//     tempcreateinfo.pDescriptorUpdateEntries = entries.data();
//     tempcreateinfo.set = 0;

//     VkDescriptorUpdateTemplate temp;
//     vkCreateDescriptorUpdateTemplate(m_device_, &tempcreateinfo, nullptr,
//                                      &temp);
// }

} // namespace vkop
