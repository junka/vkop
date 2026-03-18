// vulkan/VulkanGraphicsPipeline.cpp
#include "VulkanLib.hpp"

#include <map>
#include <stdexcept>
#include <vector>

#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanGraphicsPipeline.hpp"
#include "vulkan/VulkanShader.hpp"
#include "vulkan/vulkan_core.h"

namespace vkop {

VulkanGraphicsPipeline::VulkanGraphicsPipeline(
    VkDevice device, VkRenderPass render_pass,
    std::vector<VkDescriptorType> types, VkExtent2D extent,
    const uint32_t *vert_spv, unsigned int vert_spv_len,
    const uint32_t *frag_spv, unsigned int frag_spv_len)
    : m_device_(device), m_types_(std::move(types)), m_extent_(extent) {

    createPipeline(render_pass, vert_spv, vert_spv_len, frag_spv, frag_spv_len);
    createDescriptorPool();
}

VulkanGraphicsPipeline::~VulkanGraphicsPipeline() {
    if (m_pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device_, m_pipeline_, nullptr);
        m_pipeline_ = VK_NULL_HANDLE;
    }

    if (m_pipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device_, m_pipelineLayout_, nullptr);
        m_pipelineLayout_ = VK_NULL_HANDLE;
    }
}

void VulkanGraphicsPipeline::createPipeline(VkRenderPass render_pass,
                                            const uint32_t *vert_spv,
                                            unsigned int vert_spv_len,
                                            const uint32_t *frag_spv,
                                            unsigned int frag_spv_len) {
    // Create shader modules
    VulkanShader vert_shader(m_device_, vert_spv,
                             static_cast<int>(vert_spv_len));
    VulkanShader frag_shader(m_device_, frag_spv,
                             static_cast<int>(frag_spv_len));

    // Shader stages
    VkPipelineShaderStageCreateInfo vert_stage{};
    vert_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vert_stage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vert_stage.module = vert_shader.getShaderModule();
    vert_stage.pName = "main";

    VkPipelineShaderStageCreateInfo frag_stage{};
    frag_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    frag_stage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    frag_stage.module = frag_shader.getShaderModule();
    frag_stage.pName = "main";

    std::vector<VkPipelineShaderStageCreateInfo> shader_stages = {vert_stage,
                                                                  frag_stage};

    // Vertex input state (empty for now - adjust based on your vertex format)
    VkPipelineVertexInputStateCreateInfo vertex_input_info{};
    vertex_input_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input_info.vertexBindingDescriptionCount = 0;
    vertex_input_info.pVertexBindingDescriptions = nullptr;
    vertex_input_info.vertexAttributeDescriptionCount = 0;
    vertex_input_info.pVertexAttributeDescriptions = nullptr;

    // Input assembly
    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    input_assembly.sType =
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    input_assembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport = {};
    viewport.x = 0.0F;
    viewport.y = 0.0F;
    viewport.width = static_cast<float>(m_extent_.width);
    viewport.height = static_cast<float>(m_extent_.height);
    viewport.minDepth = 0.0F;
    viewport.maxDepth = 1.0F;

    VkRect2D scissor = {};
    scissor.offset.x = 0;
    scissor.offset.y = 0;
    scissor.extent.width = m_extent_.width;
    scissor.extent.height = m_extent_.height;

    // Viewport state (dynamic)
    VkPipelineViewportStateCreateInfo viewport_state{};
    viewport_state.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.pViewports = &viewport;
    viewport_state.scissorCount = 1;
    viewport_state.pScissors = &scissor;

    // Rasterizer
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0F;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.depthBiasConstantFactor = 0.0F;
    rasterizer.depthBiasClamp = 0.0F;
    rasterizer.depthBiasSlopeFactor = 0.0F;

    // Multisampling
    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType =
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 1.0F;
    multisampling.pSampleMask = nullptr;
    multisampling.alphaToCoverageEnable = VK_FALSE;
    multisampling.alphaToOneEnable = VK_FALSE;

    // Color blending
    VkPipelineColorBlendAttachmentState color_blend_attachment{};
    color_blend_attachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    color_blend_attachment.blendEnable = VK_TRUE;
    color_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    color_blend_attachment.dstColorBlendFactor =
        VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    color_blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
    color_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    color_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    color_blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo color_blending{};
    color_blending.sType =
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    color_blending.logicOpEnable = VK_FALSE;
    color_blending.logicOp = VK_LOGIC_OP_COPY;
    color_blending.attachmentCount = 1;
    color_blending.pAttachments = &color_blend_attachment;
    color_blending.blendConstants[0] = 0.0F;
    color_blending.blendConstants[1] = 0.0F;
    color_blending.blendConstants[2] = 0.0F;
    color_blending.blendConstants[3] = 0.0F;

    // Dynamic states
    std::vector<VkDynamicState> dynamic_states = {VK_DYNAMIC_STATE_VIEWPORT,
                                                  VK_DYNAMIC_STATE_SCISSOR};

    VkPipelineDynamicStateCreateInfo dynamic_state{};
    dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic_state.dynamicStateCount =
        static_cast<uint32_t>(dynamic_states.size());
    dynamic_state.pDynamicStates = dynamic_states.data();

    // Pipeline layout (no descriptor sets or push constants for now)
    createDescriptorSetLayout(VK_SHADER_STAGE_FRAGMENT_BIT);

    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &m_descriptorSetLayout_;
    pipeline_layout_info.pushConstantRangeCount = 0;
    pipeline_layout_info.pPushConstantRanges = nullptr;

    if (vkCreatePipelineLayout(m_device_, &pipeline_layout_info, nullptr,
                               &m_pipelineLayout_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout!");
    }

    // Graphics pipeline
    VkGraphicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_info.stageCount = 2;
    pipeline_info.pStages = shader_stages.data();
    pipeline_info.pVertexInputState = &vertex_input_info;
    pipeline_info.pInputAssemblyState = &input_assembly;
    pipeline_info.pViewportState = &viewport_state;
    pipeline_info.pRasterizationState = &rasterizer;
    pipeline_info.pMultisampleState = &multisampling;
    pipeline_info.pColorBlendState = &color_blending;
    pipeline_info.pDynamicState = &dynamic_state;
    pipeline_info.layout = m_pipelineLayout_;
    pipeline_info.renderPass = render_pass;
    pipeline_info.subpass = 0;
    pipeline_info.basePipelineHandle = VK_NULL_HANDLE;
    pipeline_info.basePipelineIndex = -1;

    VkResult result = vkCreateGraphicsPipelines(
        m_device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &m_pipeline_);

    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            "Failed to create graphics pipeline! Error code: " +
            std::to_string(result));
    }
}

void VulkanGraphicsPipeline::createDescriptorSetLayout(
    VkShaderStageFlags flags) {
    auto bindings = allocDescriptorSetLaoutBindings(flags);
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
VulkanGraphicsPipeline::allocDescriptorSetLaoutBindings(
    VkShaderStageFlags flags) {
    std::vector<VkDescriptorSetLayoutBinding> bindings(m_types_.size());
    for (int i = 0; i < static_cast<int>(m_types_.size()); i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = m_types_[i];
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = flags;
    }
    return bindings;
}

void VulkanGraphicsPipeline::createDescriptorPool() {
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

VkDescriptorSet VulkanGraphicsPipeline::allocDescriptorSets() {
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

void VulkanGraphicsPipeline::freeDescriptorSets(VkDescriptorSet ds) {
    if (ds != VK_NULL_HANDLE) {
        vkFreeDescriptorSets(m_device_, m_descriptorPool_, 1, &ds);
    }
}

void VulkanGraphicsPipeline::updateDescriptorSets(
    const std::vector<VkWriteDescriptorSet> &writes) {
    vkUpdateDescriptorSets(m_device_, static_cast<uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);
}

} // namespace vkop