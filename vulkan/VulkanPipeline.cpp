// Copyright 2025 @junka
#include "VulkanLib.hpp"

#include <cstddef>
#include <map>
#include <stdexcept>
#include <sys/syslog.h>
#include <vector>

#include "vulkan/VulkanPipeline.hpp"
#include "vulkan/VulkanShader.hpp"

namespace vkop {
VulkanPipeline::VulkanPipeline(
    VkDevice device, std::vector<VkDescriptorType> types,
    std::vector<std::shared_ptr<VulkanResource>> objs, size_t pushconstant_size,
    const uint32_t *spirv, int codesize)
    : m_device_(device), m_types_(std::move(types)), m_objs_(std::move(objs)),
      m_pushconstant_size_(pushconstant_size) {
    VulkanShader shader(device, spirv, codesize);
    createDescriptorSetLayout();
    createPipelineLayout();

    createComputePipeline(m_pipelineLayout_, shader.getShaderModule());
    // could destroy shader here

    createDescriptorPool();
    allocDescriptorSets();

    updateDescriptorSets();
}

VulkanPipeline::~VulkanPipeline() { cleanup(); }

std::vector<VkDescriptorSetLayoutBinding>
VulkanPipeline::allocDescriptorSetLaoutBindings() {
    std::vector<VkDescriptorSetLayoutBinding> bindings(m_types_.size());
    for (int i = 0; i < static_cast<int>(m_types_.size()); i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = m_types_[i];
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    return bindings;
}

void VulkanPipeline::createDescriptorSetLayout() {
    auto bindings = allocDescriptorSetLaoutBindings();
    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
    layout_info.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(m_device_, &layout_info, nullptr,
                                    &m_descriptorSetLayout_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout!");
    }
}

void VulkanPipeline::createPipelineLayout() {
    VkPushConstantRange push_constant_range{};
    push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant_range.offset = 0;
    push_constant_range.size = m_pushconstant_size_;

    VkPipelineLayoutCreateInfo pipeline_layout_create_info = {};
    pipeline_layout_create_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_create_info.setLayoutCount = 1;
    pipeline_layout_create_info.pSetLayouts = &m_descriptorSetLayout_;
    if (m_pushconstant_size_ > 0) {
        pipeline_layout_create_info.pushConstantRangeCount = 1;
        pipeline_layout_create_info.pPushConstantRanges = &push_constant_range;
    }
    if (vkCreatePipelineLayout(m_device_, &pipeline_layout_create_info, nullptr,
                               &m_pipelineLayout_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout!");
    }
}

void VulkanPipeline::createDescriptorPool() {
    std::map<VkDescriptorType, int> type_counts;
    for (auto t : m_types_) {
        if (type_counts.find(t) == type_counts.end()) {
            type_counts[t] = 1;
        } else {
            type_counts[t]++;
        }
    }
    std::vector<VkDescriptorPoolSize> pool_sizes;
    for (auto &t : type_counts) {
        VkDescriptorPoolSize ps;
        ps.type = t.first;
        ps.descriptorCount = t.second;
        pool_sizes.push_back(ps);
    }

    VkDescriptorPoolCreateInfo descriptor_pool_create_info = {};
    descriptor_pool_create_info.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptor_pool_create_info.flags =
        VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    descriptor_pool_create_info.maxSets = 1;
    descriptor_pool_create_info.poolSizeCount = pool_sizes.size();
    descriptor_pool_create_info.pPoolSizes = pool_sizes.data();
    if (vkCreateDescriptorPool(m_device_, &descriptor_pool_create_info, nullptr,
                               &m_descriptorPool_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor pool!");
    }
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

    if (vkCreateComputePipelines(m_device_, VK_NULL_HANDLE, 1, &pipeline_info,
                                 nullptr, &m_pipeline_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute pipeline!");
    }
}

void VulkanPipeline::allocDescriptorSets() {
    VkDescriptorSetAllocateInfo descriptor_set_allocate_info = {};
    descriptor_set_allocate_info.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptor_set_allocate_info.descriptorPool = m_descriptorPool_;
    descriptor_set_allocate_info.descriptorSetCount = 1;
    descriptor_set_allocate_info.pSetLayouts = &m_descriptorSetLayout_;

    if (vkAllocateDescriptorSets(m_device_, &descriptor_set_allocate_info,
                                 &m_descriptorSet_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor set!");
    }
}

void VulkanPipeline::createDescriptorUpdataTemplate() {
    std::vector<VkDescriptorUpdateTemplateEntry> entries(m_types_.size());
    for (size_t i = 0; i < m_types_.size(); i++) {
        entries[i].dstBinding = static_cast<uint32_t>(i);
        entries[i].dstArrayElement = 0;
        entries[i].descriptorCount = 1;
        entries[i].descriptorType = m_types_[i];
        entries[i].offset = sizeof(VkDescriptorSetLayoutBinding) * i;
        if (m_objs_[i]->getResourceType() == ResourceType::VK_BUFFER) {
            entries[i].stride = sizeof(VkDescriptorBufferInfo);
        } else {
            entries[i].stride = sizeof(VkDescriptorImageInfo);
        }
    }

    VkDescriptorUpdateTemplateCreateInfo tempcreateinfo = {};
    tempcreateinfo.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO;
    tempcreateinfo.templateType =
        VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET;
    tempcreateinfo.descriptorSetLayout = m_descriptorSetLayout_;
    tempcreateinfo.pipelineLayout = m_pipelineLayout_;
    tempcreateinfo.pipelineBindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;
    tempcreateinfo.descriptorUpdateEntryCount =
        static_cast<uint32_t>(m_types_.size());
    tempcreateinfo.pDescriptorUpdateEntries = entries.data();
    tempcreateinfo.set = 0;

    VkDescriptorUpdateTemplate temp;
    vkCreateDescriptorUpdateTemplate(m_device_, &tempcreateinfo, nullptr,
                                     &temp);
}

void VulkanPipeline::updateDescriptorSets() {
    std::vector<VkWriteDescriptorSet> write_descriptor_sets(m_types_.size());
    std::vector<VkDescriptorBufferInfo> buffer_infos;
    std::vector<VkDescriptorImageInfo> image_infos;
    int n_img = 0;
    int n_buf = 0;

    for (size_t i = 0; i < m_types_.size(); i++) {
        if (m_objs_[i]->getResourceType() == ResourceType::VK_BUFFER) {
            buffer_infos.push_back(std::get<VkDescriptorBufferInfo>(
                m_objs_[i]->getDescriptorInfo()));
        } else {
            image_infos.push_back(std::get<VkDescriptorImageInfo>(
                m_objs_[i]->getDescriptorInfo()));
        }
    }

    for (size_t i = 0; i < m_types_.size(); i++) {
        VkWriteDescriptorSet &write_descriptor_set = write_descriptor_sets[i];
        write_descriptor_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_descriptor_set.dstSet = m_descriptorSet_;
        write_descriptor_set.dstBinding = static_cast<uint32_t>(i);
        write_descriptor_set.dstArrayElement = 0;
        write_descriptor_set.descriptorCount = 1;
        write_descriptor_set.descriptorType = m_types_[i];
        if (m_objs_[i]->getResourceType() == ResourceType::VK_BUFFER) {
            write_descriptor_set.pBufferInfo = &buffer_infos[n_buf++];
        } else {
            write_descriptor_set.pImageInfo = &image_infos[n_img++];
        }
    }
    vkUpdateDescriptorSets(m_device_,
                           static_cast<uint32_t>(write_descriptor_sets.size()),
                           write_descriptor_sets.data(), 0, nullptr);
}

void VulkanPipeline::cleanup() {
    if (m_descriptorSet_ != VK_NULL_HANDLE) {
        vkFreeDescriptorSets(m_device_, m_descriptorPool_, 1,
                             &m_descriptorSet_);
        m_descriptorSet_ = VK_NULL_HANDLE;
    }
    if (m_descriptorPool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_device_, m_descriptorPool_, nullptr);
        m_descriptorPool_ = VK_NULL_HANDLE;
    }
    if (m_pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device_, m_pipeline_, nullptr);
        m_pipeline_ = VK_NULL_HANDLE;
    }
    if (m_descriptorSetLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_device_, m_descriptorSetLayout_,
                                     nullptr);
        m_descriptorSetLayout_ = VK_NULL_HANDLE;
    }
    if (m_pipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device_, m_pipelineLayout_, nullptr);
        m_pipelineLayout_ = VK_NULL_HANDLE;
    }
}

} // namespace vkop
