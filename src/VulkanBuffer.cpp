// Copyright 2025 @junka
#include "VulkanBuffer.hpp"
#include "VulkanLib.hpp"
#include "VulkanResource.hpp"
#include <stdexcept>

namespace vkop {
VulkanBuffer::VulkanBuffer(VkPhysicalDevice physicalDevice,
                           const uint32_t queueFamilyIndex, VkDevice device,
                           VkDeviceSize size, VkBufferUsageFlags usage,
                           VkMemoryPropertyFlags requireProperties)
    : VulkanResource(physicalDevice, queueFamilyIndex, device), m_size_(size) {
    createBuffer(size, usage);
#ifdef VK_KHR_get_memory_requirements2
    VkMemoryRequirements2 mem_requirements2 = {};
    mem_requirements2.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
    VkBufferMemoryRequirementsInfo2 buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2;
    buffer_info.buffer = m_buffer_;
    vkGetBufferMemoryRequirements2(m_device_, &buffer_info, &mem_requirements2);
    VkMemoryRequirements mem_requirements =
        mem_requirements2.memoryRequirements;
#else
    VkMemoryRequirements mem_requirements;
    vkGetBufferMemoryRequirements(m_device_, m_buffer_, &mem_requirements);
#endif
    allocMemory(mem_requirements, requireProperties);

    vkBindBufferMemory(m_device_, m_buffer_, getMemory(), 0);
}

VulkanBuffer::~VulkanBuffer() { cleanup(); }

VkBuffer VulkanBuffer::getBuffer() const { return m_buffer_; }

void VulkanBuffer::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage) {
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    buffer_info.queueFamilyIndexCount = 1;
    buffer_info.pQueueFamilyIndices = &m_queueFamilyIndex_;

    if (vkCreateBuffer(m_device_, &buffer_info, nullptr, &m_buffer_) !=
        VK_SUCCESS) {
        throw std::runtime_error("Failed to create buffer!");
    }
}

std::variant<VkDescriptorImageInfo, VkDescriptorBufferInfo>
VulkanBuffer::getDescriptorInfo() const {
    VkDescriptorBufferInfo buffer_info{};
    buffer_info.buffer = m_buffer_;
    buffer_info.offset = 0;
    buffer_info.range = m_size_;
    return buffer_info;
}

void VulkanBuffer::cleanup() {
    if (m_buffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(m_device_, m_buffer_, nullptr);
    }
}

void VulkanBuffer::transferReadBarrier(VkCommandBuffer commandBuffer) {
    VkBufferMemoryBarrier barrier;
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.pNext = nullptr;
    barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = m_buffer_;
    barrier.offset = 0;
    barrier.size = m_size_;

    VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_HOST_BIT;
    VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

    vkCmdPipelineBarrier(commandBuffer, src_stage, dst_stage, 0, 0, nullptr, 1,
                         &barrier, 0, nullptr);
}

void VulkanBuffer::transferWriteBarrier(VkCommandBuffer commandBuffer) {
    VkBufferMemoryBarrier barrier;
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.pNext = nullptr;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = m_buffer_;
    barrier.offset = 0;
    barrier.size = m_size_;

    VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

    vkCmdPipelineBarrier(commandBuffer, src_stage, dst_stage, 0, 0, nullptr, 1,
                         &barrier, 0, nullptr);
}

} // namespace vkop
