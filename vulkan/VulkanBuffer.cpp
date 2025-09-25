// Copyright 2025 @junka
#include "VulkanBuffer.hpp"
#include "VulkanLib.hpp"
#include "VulkanResource.hpp"
#include <stdexcept>

namespace vkop {
VulkanBuffer::VulkanBuffer(std::shared_ptr<VulkanDevice> &vdev,
                           VkDeviceSize size, VkBufferUsageFlags usage,
                           VkMemoryPropertyFlags requireProperties, int ext_fd)
    : VulkanResource(vdev), m_size_(size) {
    createBuffer(size, usage);
#ifndef USE_VMA
#ifdef VK_KHR_get_memory_requirements2
    VkMemoryRequirements2 mem_requirements2 = {};
    mem_requirements2.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
    VkBufferMemoryRequirementsInfo2 buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2;
    buffer_info.buffer = m_buffer_;
    vkGetBufferMemoryRequirements2(m_vdev_->getLogicalDevice(), &buffer_info,
                                   &mem_requirements2);
    VkMemoryRequirements mem_requirements =
        mem_requirements2.memoryRequirements;
#else
    VkMemoryRequirements mem_requirements;
    vkGetBufferMemoryRequirements(m_vdev_->getLogicalDevice(), m_buffer_,
                                  &mem_requirements);
#endif
    allocMemory(mem_requirements, requireProperties, ext_fd);
    vkBindBufferMemory(m_vdev_->getLogicalDevice(), m_buffer_, getMemory(), 0);
#else
    (void)ext_fd;
    (void)requireProperties;
#endif
}

VulkanBuffer::~VulkanBuffer() {
#ifndef USE_VMA
    if (m_buffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(m_vdev_->getLogicalDevice(), m_buffer_, nullptr);
    }
#else
    m_vdev_->getVMA()->destroyBuffer(&m_vma_buffer_);
#endif
}

VkBuffer VulkanBuffer::getBuffer() const {
#ifndef USE_VMA
    return m_buffer_;
#else
    return m_vma_buffer_.buffer;
#endif
}

void VulkanBuffer::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage) {
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    buffer_info.queueFamilyIndexCount = 1;
    uint32_t qidx = m_vdev_->getComputeQueueFamilyIndex();
    buffer_info.pQueueFamilyIndices = &qidx;
#ifdef USE_VMA
    auto ret = m_vdev_->getVMA()->createBuffer(&buffer_info, &m_vma_buffer_);
#else
    auto ret = vkCreateBuffer(m_vdev_->getLogicalDevice(), &buffer_info,
                              nullptr, &m_buffer_);
#endif
    if (ret != VK_SUCCESS) {
        throw std::runtime_error("Failed to create buffer!");
    }
}

std::variant<VkDescriptorImageInfo, VkDescriptorBufferInfo>
VulkanBuffer::getDescriptorInfo() const {
    VkDescriptorBufferInfo buffer_info{};
#ifndef USE_VMA
    buffer_info.buffer = m_buffer_;
#else
    buffer_info.buffer = m_vma_buffer_.buffer;
#endif
    buffer_info.offset = 0;
    buffer_info.range = m_size_;
    return buffer_info;
}

void VulkanBuffer::transferReadBarrier(VkCommandBuffer commandBuffer) {
    VkBufferMemoryBarrier barrier;
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.pNext = nullptr;
    barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
#ifndef USE_VMA
    barrier.buffer = m_buffer_;
#else
    barrier.buffer = m_vma_buffer_.buffer;
#endif
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
#ifndef USE_VMA
    barrier.buffer = m_buffer_;
#else
    barrier.buffer = m_vma_buffer_.buffer;
#endif
    barrier.offset = 0;
    barrier.size = m_size_;

    VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

    vkCmdPipelineBarrier(commandBuffer, src_stage, dst_stage, 0, 0, nullptr, 1,
                         &barrier, 0, nullptr);
}

} // namespace vkop
