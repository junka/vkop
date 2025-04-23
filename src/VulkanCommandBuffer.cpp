#include "VulkanLib.hpp"
#include <vulkan/vulkan_core.h>
#include "VulkanCommandBuffer.hpp"

namespace vkop {

#define UP_DIV(x, y) (((x) + (y) - 1) / (y))

VulkanCommandBuffer::VulkanCommandBuffer(VkDevice device, VkCommandPool commandPool, int count)
    : m_device(device), m_commandPool(commandPool)
{
    allocate(count);
}

void VulkanCommandBuffer::bind(VulkanPipeline &pipeline)
{
    vkCmdBindPipeline(m_commandBuffers[m_avail], VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.getComputePipeline());
    VkDescriptorSet descriptorSet = pipeline.getDescriptorSet();
    vkCmdBindDescriptorSets(m_commandBuffers[m_avail], VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.getPipelineLayout(), 0, 1, &descriptorSet, 0, nullptr);
}

VulkanCommandBuffer::~VulkanCommandBuffer() {
    if (m_commandBuffers.size() > 0) {
        vkFreeCommandBuffers(m_device, m_commandPool, m_commandBuffers.size(), m_commandBuffers.data());
    }
}

void VulkanCommandBuffer::allocate(int count) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = m_commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = count;

    m_commandBuffers.resize(count);
    if (vkAllocateCommandBuffers(m_device, &allocInfo, m_commandBuffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate command buffer!");
    }
}

void VulkanCommandBuffer::begin() {
    if (m_avail >= (int)m_commandBuffers.size()) {
        throw std::runtime_error("Command buffer is full!");
    }
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(m_commandBuffers[m_avail], &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("Failed to begin recording command buffer!");
    }
}

void VulkanCommandBuffer::begin(int idx) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(m_commandBuffers[idx], &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("Failed to begin recording command buffer!");
    }
}

void VulkanCommandBuffer::end() {
    if (vkEndCommandBuffer(m_commandBuffers[m_avail]) != VK_SUCCESS) {
        throw std::runtime_error("Failed to record command buffer!");
    }
    m_avail ++;
}

void VulkanCommandBuffer::end(int idx) {
    if (vkEndCommandBuffer(m_commandBuffers[idx]) != VK_SUCCESS) {
        throw std::runtime_error("Failed to record command buffer!");
    }
}

void VulkanCommandBuffer::submit(VkQueue queue, VkFence fence)
{
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = m_commandBuffers.size();
    submitInfo.pCommandBuffers = m_commandBuffers.data();

    if (vkQueueSubmit(queue, 1, &submitInfo, fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit command buffer!");
    }
    vkQueueWaitIdle(queue);
}

void VulkanCommandBuffer::reset(int idx) {
    if (vkResetCommandBuffer(m_commandBuffers[idx], 0) != VK_SUCCESS) {
        throw std::runtime_error("Failed to reset command buffer!");
    }
}

void VulkanCommandBuffer::dispatch(int w, int h, int z) {
    vkCmdDispatch(m_commandBuffers[m_avail], UP_DIV(w, 16), UP_DIV(h, 16), UP_DIV(z, 16));
}


} // namespace vkop