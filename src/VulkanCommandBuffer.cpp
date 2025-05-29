// Copyright 2025 @junka
#include "VulkanCommandBuffer.hpp"
#include "VulkanLib.hpp"

namespace vkop {

#define UP_DIV(x, y) (((x) + (y)-1) / (y))

VulkanCommandBuffer::VulkanCommandBuffer(VkDevice device,
                                         VkCommandPool commandPool, int count)
    : m_device_(device), m_commandPool_(commandPool) {
    allocate(count);
}

void VulkanCommandBuffer::bind(VulkanPipeline &pipeline) {
    vkCmdBindPipeline(m_commandBuffers_[m_avail_],
                      VK_PIPELINE_BIND_POINT_COMPUTE,
                      pipeline.getComputePipeline());
    VkDescriptorSet descriptor_set = pipeline.getDescriptorSet();
    vkCmdBindDescriptorSets(
        m_commandBuffers_[m_avail_], VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline.getPipelineLayout(), 0, 1, &descriptor_set, 0, nullptr);
}

VulkanCommandBuffer::~VulkanCommandBuffer() {
    if (!m_commandBuffers_.empty()) {
        vkFreeCommandBuffers(m_device_, m_commandPool_,
                             m_commandBuffers_.size(),
                             m_commandBuffers_.data());
    }
}

void VulkanCommandBuffer::allocate(int count) {
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = m_commandPool_;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = count;

    m_commandBuffers_.resize(count);
    if (vkAllocateCommandBuffers(m_device_, &alloc_info,
                                 m_commandBuffers_.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate command buffer!");
    }
}

void VulkanCommandBuffer::begin() {
    if (m_avail_ >= static_cast<int>(m_commandBuffers_.size())) {
        throw std::runtime_error("Command buffer is full!");
    }
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(m_commandBuffers_[m_avail_], &begin_info) !=
        VK_SUCCESS) {
        throw std::runtime_error("Failed to begin recording command buffer!");
    }
}

void VulkanCommandBuffer::begin(int idx) {
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(m_commandBuffers_[idx], &begin_info) !=
        VK_SUCCESS) {
        throw std::runtime_error("Failed to begin recording command buffer!");
    }
}

void VulkanCommandBuffer::end() {
    if (vkEndCommandBuffer(m_commandBuffers_[m_avail_]) != VK_SUCCESS) {
        throw std::runtime_error("Failed to record command buffer!");
    }
    m_avail_++;
}

void VulkanCommandBuffer::end(int idx) {
    if (vkEndCommandBuffer(m_commandBuffers_[idx]) != VK_SUCCESS) {
        throw std::runtime_error("Failed to record command buffer!");
    }
}

int VulkanCommandBuffer::submit(VkQueue queue, VkFence fence) {
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = m_commandBuffers_.size();
    submit_info.pCommandBuffers = m_commandBuffers_.data();

    if (vkQueueSubmit(queue, 1, &submit_info, fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit command buffer!");
    }
    return vkQueueWaitIdle(queue);
}

void VulkanCommandBuffer::reset(int idx) {
    if (vkResetCommandBuffer(m_commandBuffers_[idx], 0) != VK_SUCCESS) {
        throw std::runtime_error("Failed to reset command buffer!");
    }
}

void VulkanCommandBuffer::dispatch(int w, int h, int z) {
    vkCmdDispatch(m_commandBuffers_[m_avail_], UP_DIV(w, 16), UP_DIV(h, 16),
                  UP_DIV(z, 16));
}

} // namespace vkop
