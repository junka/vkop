// Copyright 2025 @junka
#include "vulkan/VulkanCommandBuffer.hpp"
#include "vulkan/VulkanLib.hpp"

namespace vkop {

#define UP_DIV(x, y) (((x) + (y)-1) / (y))

VulkanCommandBuffer::VulkanCommandBuffer(VkDevice device,
                                         VkCommandPool commandPool,
                                         VkSemaphore semaphore)
    : m_device_(device), m_commandPool_(commandPool), m_semaphore_(semaphore) {
    allocate();
}

void VulkanCommandBuffer::bind(VulkanPipeline &pipeline) {
    vkCmdBindPipeline(m_commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pipeline.getComputePipeline());
    VkDescriptorSet descriptor_set = pipeline.getDescriptorSet();
    vkCmdBindDescriptorSets(m_commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline.getPipelineLayout(), 0, 1, &descriptor_set,
                            0, nullptr);
}

VulkanCommandBuffer::~VulkanCommandBuffer() {
    if (m_commandBuffer_) {
        vkFreeCommandBuffers(m_device_, m_commandPool_, 1, &m_commandBuffer_);
    }
    if (m_primaryBuffer_) {
        vkFreeCommandBuffers(m_device_, m_commandPool_, 1, &m_primaryBuffer_);
    }
}

void VulkanCommandBuffer::allocate() {
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = m_commandPool_;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(m_device_, &alloc_info, &m_commandBuffer_) !=
        VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate command buffer!");
    }

    // VkCommandBufferAllocateInfo alloc_pri_info{};
    // alloc_pri_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    // alloc_pri_info.commandPool = m_commandPool_;
    // alloc_pri_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    // alloc_pri_info.commandBufferCount = 1;

    // if (vkAllocateCommandBuffers(m_device_, &alloc_pri_info,
    //                              &m_primaryBuffer_) != VK_SUCCESS) {
    //     throw std::runtime_error("Failed to allocate primary command
    //     buffer!");
    // }
}

void VulkanCommandBuffer::begin() {
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(m_commandBuffer_, &begin_info) != VK_SUCCESS) {
        throw std::runtime_error("Failed to begin recording command buffer!");
    }
}

void VulkanCommandBuffer::end() {
    if (vkEndCommandBuffer(m_commandBuffer_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to record command buffer!");
    }
    m_avail_++;
}

int VulkanCommandBuffer::submit(VkQueue queue, VkFence fence) {
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &m_commandBuffer_;
    wait(fence);
    if (vkResetFences(m_device_, 1, &fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to reset fence!");
    }
    if (vkQueueSubmit(queue, 1, &submit_info, fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit command buffer!");
    }
    return 0;
}

int VulkanCommandBuffer::submit(VkQueue queue, uint64_t submitValue) {

    VkTimelineSemaphoreSubmitInfo timeline_submit_info{};
    timeline_submit_info.sType =
        VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timeline_submit_info.signalSemaphoreValueCount = 1;
    timeline_submit_info.pSignalSemaphoreValues = &submitValue;

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.pNext = &timeline_submit_info;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = &m_semaphore_;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &m_commandBuffer_;

    if (vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit command buffer!");
    }
    return 0;
}

int VulkanCommandBuffer::wait(VkFence fence) {
    return vkWaitForFences(m_device_, 1, &fence, VK_TRUE, UINT64_MAX);
}
int VulkanCommandBuffer::wait(uint64_t waitValue) {
    VkSemaphoreWaitInfo wait_info{};
    wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    wait_info.semaphoreCount = 1;
    wait_info.pSemaphores = &m_semaphore_;
    wait_info.pValues = &waitValue;
    if (VK_SUCCESS != vkWaitSemaphores(m_device_, &wait_info, UINT64_MAX)) {
        throw std::runtime_error("Failed to wait for semaphore!");
    }

    return 0;
}

void VulkanCommandBuffer::reset() {
    if (vkResetCommandBuffer(m_commandBuffer_, 0) != VK_SUCCESS) {
        throw std::runtime_error("Failed to reset command buffer!");
    }
}

void VulkanCommandBuffer::push_constants(VulkanPipeline &pipeline,
                                         uint32_t size, const void *ptr) {
    vkCmdPushConstants(m_commandBuffer_, pipeline.getPipelineLayout(),
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, size, ptr);
}

void VulkanCommandBuffer::dispatch(int w, int h, int z) {
    vkCmdDispatch(m_commandBuffer_, w, h, z);
}

void VulkanCommandBuffer::exec(VkQueue queue, VkFence fence) {
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(m_primaryBuffer_, &begin_info) != VK_SUCCESS) {
        throw std::runtime_error("Failed to begin recording command buffer!");
    }
    vkCmdExecuteCommands(m_primaryBuffer_, 1, &m_commandBuffer_);
    if (vkEndCommandBuffer(m_primaryBuffer_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to record primary command buffer!");
    }
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &m_primaryBuffer_;
    wait(fence);
    if (vkResetFences(m_device_, 1, &fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to reset fence!");
    }
    if (vkQueueSubmit(queue, 1, &submit_info, fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit command buffer!");
    }
}

} // namespace vkop
