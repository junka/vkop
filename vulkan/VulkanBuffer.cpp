// Copyright 2025 @junka
#include "VulkanBuffer.hpp"
#include "VulkanLib.hpp"
#include "VulkanResource.hpp"
#include <stdexcept>
#include <vulkan/vulkan_core.h>

namespace vkop {
VulkanBuffer::VulkanBuffer(std::shared_ptr<VulkanDevice> &vdev,
                           VkDeviceSize size, VkBufferUsageFlags usage,
                           VkMemoryPropertyFlags requireProperties,
                           VkFormat format, int ext_fd)
    : VulkanResource(vdev), m_size_(size) {
    createBuffer(usage, ((requireProperties &
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0));
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
#endif
    if (format != VK_FORMAT_UNDEFINED) {
        createBufferView(format);
    }
}

VulkanBuffer::~VulkanBuffer() {
    if (m_buffer_view_ != VK_NULL_HANDLE) {
        vkDestroyBufferView(m_vdev_->getLogicalDevice(), m_buffer_view_,
                            nullptr);
    }
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

void VulkanBuffer::createBuffer(VkBufferUsageFlags usage, bool device_local) {
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = m_size_;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    buffer_info.queueFamilyIndexCount = 0;
    buffer_info.pQueueFamilyIndices = nullptr;
#ifdef USE_VMA
    auto ret = m_vdev_->getVMA()->createBuffer(&buffer_info, &m_vma_buffer_,
                                               device_local);
#else
    auto ret = vkCreateBuffer(m_vdev_->getLogicalDevice(), &buffer_info,
                              nullptr, &m_buffer_);
#endif
    if (ret != VK_SUCCESS) {
        throw std::runtime_error("Failed to create buffer!");
    }
    if (usage & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) {
        m_desc_type_ = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    } else if (usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) {
        m_desc_type_ = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    } else if (usage & VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT) {
        m_desc_type_ = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
    } else if (usage & VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT) {
        m_desc_type_ = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
    }
}

void VulkanBuffer::createBufferView(VkFormat format) {
    VkBuffer buff = getBuffer();
    VkBufferViewCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO;
    buffer_info.flags = 0;
    buffer_info.buffer = buff;
    buffer_info.format = format;
    buffer_info.range = m_size_;
    buffer_info.offset = 0;

    auto ret = vkCreateBufferView(m_vdev_->getLogicalDevice(), &buffer_info,
                                  nullptr, &m_buffer_view_);
    if (ret != VK_SUCCESS) {
        throw std::runtime_error("Failed to create buffer view");
    }
    printf("Create buffer view %p\n", m_buffer_view_);
}

std::variant<VkDescriptorImageInfo, VkDescriptorBufferInfo>
VulkanBuffer::getDescriptorInfo() const {
    VkBuffer buff = getBuffer();
    VkDescriptorBufferInfo buffer_info{};
    buffer_info.buffer = buff;
    buffer_info.offset = 0;
    buffer_info.range = m_size_;
    return buffer_info;
}

void VulkanBuffer::transitionBuffer(VkCommandBuffer commandBuffer,
                                    VkAccessFlags dstAccessMask,
                                    VkPipelineStageFlags src_stage,
                                    VkPipelineStageFlags dst_stage,
                                    VkDeviceSize offset) {
    VkBuffer buff = getBuffer();
    VkBufferMemoryBarrier barrier;
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.pNext = nullptr;
    barrier.srcAccessMask = m_access_;
    barrier.dstAccessMask = dstAccessMask;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buff;
    barrier.offset = offset;
    barrier.size = m_size_;

    m_access_ = dstAccessMask;

    vkCmdPipelineBarrier(commandBuffer, src_stage, dst_stage, 0, 0, nullptr, 1,
                         &barrier, 0, nullptr);
}

void VulkanBuffer::transferBarrier(VkCommandBuffer commandBuffer,
                                   VkAccessFlags dstAccessMask) {
    VkPipelineStageFlags source_stage =
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkPipelineStageFlags destination_stage =
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    transitionBuffer(commandBuffer, dstAccessMask, source_stage,
                     destination_stage, 0);
}

void VulkanBuffer::transferReadBarrier(VkCommandBuffer commandBuffer) {
    transitionBuffer(commandBuffer, VK_ACCESS_TRANSFER_READ_BIT,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                     0);
}

void VulkanBuffer::transferWriteBarrier(VkCommandBuffer commandBuffer) {
    transitionBuffer(commandBuffer, VK_ACCESS_TRANSFER_WRITE_BIT,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0);
}

void VulkanBuffer::readBarrier(VkCommandBuffer commandBuffer) {
    transitionBuffer(commandBuffer, VK_ACCESS_SHADER_READ_BIT,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0);
}

void VulkanBuffer::writeBarrier(VkCommandBuffer commandBuffer) {
    transitionBuffer(commandBuffer, VK_ACCESS_SHADER_WRITE_BIT,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0);
}

void VulkanBuffer::copyBufferToStageBuffer(VkCommandBuffer commandBuffer,
                                           VkBuffer dstbuffer,
                                           VkDeviceSize dstoffset) {
    VkAccessFlags old_access = m_access_;
    if (m_access_ != VK_ACCESS_TRANSFER_READ_BIT) {
        transferReadBarrier(commandBuffer);
    }
    VkBuffer buffer = getBuffer();

    VkBufferCopy copy_region = {};
    copy_region.srcOffset = 0;
    copy_region.dstOffset = dstoffset;
    copy_region.size = m_size_;

    vkCmdCopyBuffer(commandBuffer, buffer, dstbuffer, 1, &copy_region);

    transferBarrier(commandBuffer, old_access);
}

void VulkanBuffer::copyStageBufferToBuffer(VkCommandBuffer commandBuffer,
                                           VkBuffer srcbuffer,
                                           VkDeviceSize srcoffset) {
    VkAccessFlags old_access = m_access_;
    if (m_access_ != VK_ACCESS_TRANSFER_WRITE_BIT) {
        transferWriteBarrier(commandBuffer);
    }
    VkBuffer buffer = getBuffer();

    VkBufferCopy copy_region = {};
    copy_region.srcOffset = srcoffset;
    copy_region.dstOffset = 0;
    copy_region.size = m_size_;

    vkCmdCopyBuffer(commandBuffer, srcbuffer, buffer, 1, &copy_region);

    transferBarrier(commandBuffer, old_access);
}
} // namespace vkop
