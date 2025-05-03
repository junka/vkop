#include "VulkanLib.hpp"

#include <cstddef>
#include <stdexcept>
#include <sys/syslog.h>
#include <vector>
#include <map>

#include "VulkanShader.hpp"
#include "VulkanPipeline.hpp"

namespace vkop {
VulkanPipeline::VulkanPipeline(VkDevice device, std::vector<VkDescriptorType> types, std::vector<std::shared_ptr<VulkanResource>> objs, const uint32_t *spirv, int codesize) :
 m_device(device), m_types(types), m_objs(objs)
{
    VulkanShader shader(device, spirv, codesize);
    createDescriptorSetLayout();
    createPipelineLayout();

    createComputePipeline(m_pipelineLayout, shader.getShaderModule());
    //could destroy shader here

    createDescriptorPool();
    allocDescriptorSets();

    updateDescriptorSets();
}

VulkanPipeline::~VulkanPipeline() {
    cleanup();
}

std::vector<VkDescriptorSetLayoutBinding> VulkanPipeline::allocDescriptorSetLaoutBindings()
{
    std::vector<VkDescriptorSetLayoutBinding> bindings(m_types.size());
    for (int i = 0; i < (int)m_types.size(); i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = m_types[i];
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags =VK_SHADER_STAGE_COMPUTE_BIT;
    }
    return bindings;
}

void VulkanPipeline::createDescriptorSetLayout() {
    auto bindings = allocDescriptorSetLaoutBindings();
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout!");
    }
}

void VulkanPipeline::createPipelineLayout()
{
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = &m_descriptorSetLayout;

    if(vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout!");
    }
}

void VulkanPipeline::createDescriptorPool()
{
    std::map<VkDescriptorType, int> typeCounts;
    for (auto t: m_types) {
        if (typeCounts.find(t) == typeCounts.end()) {
            typeCounts[t] = 1;
        } else {
            typeCounts[t]++;
        }
    }
    std::vector<VkDescriptorPoolSize> poolSizes;
    for (auto &t: typeCounts) {
        VkDescriptorPoolSize ps;
        ps.type = t.first;
        ps.descriptorCount = t.second;
        poolSizes.push_back(ps);
    }

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
    descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolCreateInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    descriptorPoolCreateInfo.maxSets = 1;
    descriptorPoolCreateInfo.poolSizeCount = poolSizes.size();
    descriptorPoolCreateInfo.pPoolSizes = poolSizes.data();
    if (vkCreateDescriptorPool(m_device, &descriptorPoolCreateInfo, nullptr, &m_descriptorPool)!= VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor pool!");
    }
}

void VulkanPipeline::createComputePipeline(VkPipelineLayout pipelineLayout, VkShaderModule shaderModule) {
    VkPipelineShaderStageCreateInfo shader_stage_create_info = {};
    shader_stage_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stage_create_info.pNext = nullptr;
    shader_stage_create_info.flags = 0;
    shader_stage_create_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shader_stage_create_info.module = shaderModule;
    shader_stage_create_info.pName = "main";
    shader_stage_create_info.pSpecializationInfo = nullptr;


    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = shader_stage_create_info;
    pipelineInfo.layout = pipelineLayout;

    if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute pipeline!");
    }
}

void VulkanPipeline::allocDescriptorSets()
{
    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.descriptorPool = m_descriptorPool;
    descriptorSetAllocateInfo.descriptorSetCount = 1;
    descriptorSetAllocateInfo.pSetLayouts = &m_descriptorSetLayout;

    if(vkAllocateDescriptorSets(m_device, &descriptorSetAllocateInfo, &m_descriptorSet) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to allocate descriptor set!");
    }
}

void VulkanPipeline::createDescriptorUpdataTemplate()
{
    std::vector<VkDescriptorUpdateTemplateEntry> entries(m_types.size());
    for (size_t i = 0; i < m_types.size(); i++) {
        entries[i].dstBinding = static_cast<uint32_t>(i);
        entries[i].dstArrayElement = 0;
        entries[i].descriptorCount = 1;
        entries[i].descriptorType = m_types[i];
        entries[i].offset = sizeof(VkDescriptorSetLayoutBinding) * i;
        if (m_objs[i]->getResourceType() == ResourceType::Buffer) {
            entries[i].stride = sizeof(VkDescriptorBufferInfo);
        } else {
            entries[i].stride = sizeof(VkDescriptorImageInfo);
        }
    }

    VkDescriptorUpdateTemplateCreateInfo tempcreateinfo ={};
    tempcreateinfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO;
    tempcreateinfo.templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET;
    tempcreateinfo.descriptorSetLayout = m_descriptorSetLayout;
    tempcreateinfo.pipelineLayout = m_pipelineLayout;
    tempcreateinfo.pipelineBindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;
    tempcreateinfo.descriptorUpdateEntryCount = static_cast<uint32_t>(m_types.size());
    tempcreateinfo.pDescriptorUpdateEntries = entries.data();
    tempcreateinfo.set = 0;

    VkDescriptorUpdateTemplate temp;
    vkCreateDescriptorUpdateTemplate(m_device, &tempcreateinfo, nullptr, &temp);
}

void VulkanPipeline::updateDescriptorSets()
{
    std::vector<VkWriteDescriptorSet> writeDescriptorSets(m_types.size());
    std::vector<VkDescriptorBufferInfo> bufferInfos;
    std::vector<VkDescriptorImageInfo> imageInfos;
    int n_img = 0, n_buf = 0;
    
    for (size_t i = 0; i < m_types.size(); i++) {
        if (m_objs[i]->getResourceType() == ResourceType::Buffer) {
            bufferInfos.push_back(std::get<VkDescriptorBufferInfo>(m_objs[i]->getDescriptorInfo()));
        } else {
            imageInfos.push_back(std::get<VkDescriptorImageInfo>(m_objs[i]->getDescriptorInfo()));
        }
    }

    for (size_t i = 0; i < m_types.size(); i++) {
        VkWriteDescriptorSet &writeDescriptorSet = writeDescriptorSets[i];
        writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSet.dstSet = m_descriptorSet;
        writeDescriptorSet.dstBinding = static_cast<uint32_t>(i);
        writeDescriptorSet.dstArrayElement = 0;
        writeDescriptorSet.descriptorCount = 1;
        writeDescriptorSet.descriptorType = m_types[i];
        if (m_objs[i]->getResourceType() == ResourceType::Buffer) {
            writeDescriptorSet.pBufferInfo = &bufferInfos[n_buf++];
        } else {
            writeDescriptorSet.pImageInfo = &imageInfos[n_img++];
        }
    }
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
}


void VulkanPipeline::cleanup() {
    if (m_descriptorSet != VK_NULL_HANDLE) {
        vkFreeDescriptorSets(m_device, m_descriptorPool, 1, &m_descriptorSet);
        m_descriptorSet = VK_NULL_HANDLE;
    }
    if (m_descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
        m_descriptorPool = VK_NULL_HANDLE;
    }
    if (m_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_pipeline, nullptr);
        m_pipeline = VK_NULL_HANDLE;
    }
    if (m_descriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
        m_descriptorSetLayout = VK_NULL_HANDLE;
    }
    if (m_pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
        m_pipelineLayout = VK_NULL_HANDLE;
    }
}


}