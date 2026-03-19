// vulkan/VulkanBasePipeline.cpp
#include "VulkanBasePipeline.hpp"
#include <map>
#include <stdexcept>
#include <vector>

namespace vkop {

VulkanBasePipeline::VulkanBasePipeline(VkDevice device,
                                       std::vector<VkDescriptorType> types)
    : m_device_(device), m_types_(std::move(types)) {}

VulkanBasePipeline::~VulkanBasePipeline() {
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

    if (m_descriptorPool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_device_, m_descriptorPool_, nullptr);
        m_descriptorPool_ = VK_NULL_HANDLE;
    }
}

void VulkanBasePipeline::createDescriptorSetLayout(VkShaderStageFlags flags) {
    auto bindings = allocDescriptorSetLayoutBindings(flags);
    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
    layout_info.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(m_device_, &layout_info, nullptr,
                                    &m_descriptorSetLayout_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout!");
    }
}

std::vector<VkDescriptorSetLayoutBinding>
VulkanBasePipeline::allocDescriptorSetLayoutBindings(VkShaderStageFlags flags) {
    std::vector<VkDescriptorSetLayoutBinding> bindings(m_types_.size());
    for (int i = 0; i < static_cast<int>(m_types_.size()); i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = m_types_[i];
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = flags;
    }
    return bindings;
}

void VulkanBasePipeline::createDescriptorPool() {
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
    descriptor_pool_create_info.maxSets = kInflight;
    descriptor_pool_create_info.poolSizeCount =
        static_cast<uint32_t>(pool_sizes.size());
    descriptor_pool_create_info.pPoolSizes = pool_sizes.data();
    if (vkCreateDescriptorPool(m_device_, &descriptor_pool_create_info, nullptr,
                               &m_descriptorPool_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor pool!");
    }
}

void VulkanBasePipeline::createPipelineLayout(VkShaderStageFlags flags,
                                              size_t pushconstant_size) {
    std::vector<VkPushConstantRange> push_constant_ranges;
    if (pushconstant_size > 0) {
        VkPushConstantRange push_constant_range{};
        push_constant_range.stageFlags = flags;
        push_constant_range.offset = 0;
        push_constant_range.size = static_cast<uint32_t>(pushconstant_size);
        push_constant_ranges.push_back(push_constant_range);
    }

    VkPipelineLayoutCreateInfo pipeline_layout_create_info = {};
    pipeline_layout_create_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_create_info.setLayoutCount = 1;
    pipeline_layout_create_info.pSetLayouts = &m_descriptorSetLayout_;
    pipeline_layout_create_info.pushConstantRangeCount =
        static_cast<uint32_t>(push_constant_ranges.size());
    pipeline_layout_create_info.pPushConstantRanges =
        push_constant_ranges.empty() ? nullptr : push_constant_ranges.data();

    if (vkCreatePipelineLayout(m_device_, &pipeline_layout_create_info, nullptr,
                               &m_pipelineLayout_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout!");
    }
}

VkDescriptorSet VulkanBasePipeline::allocDescriptorSets() {
    VkDescriptorSet descriptor_set;
    VkDescriptorSetAllocateInfo descriptor_set_allocate_info = {};
    descriptor_set_allocate_info.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptor_set_allocate_info.descriptorPool = m_descriptorPool_;
    descriptor_set_allocate_info.descriptorSetCount = 1;
    descriptor_set_allocate_info.pSetLayouts = &m_descriptorSetLayout_;

    auto ret = vkAllocateDescriptorSets(
        m_device_, &descriptor_set_allocate_info, &descriptor_set);
    if (ret != VK_SUCCESS) {
        printf("allocate descriptor set fail %d\n", ret);
        throw std::runtime_error("Failed to allocate descriptor set!");
    }
    return descriptor_set;
}

void VulkanBasePipeline::freeDescriptorSets(VkDescriptorSet ds) {
    if (ds != VK_NULL_HANDLE) {
        vkFreeDescriptorSets(m_device_, m_descriptorPool_, 1, &ds);
    }
}

void VulkanBasePipeline::updateDescriptorSets(
    const std::vector<VkWriteDescriptorSet> &writes) {
    vkUpdateDescriptorSets(m_device_, static_cast<uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);
}

} // namespace vkop