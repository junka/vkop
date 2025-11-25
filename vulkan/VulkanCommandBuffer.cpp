// Copyright 2025 @junka
#include "vulkan/VulkanCommandBuffer.hpp"
#include "vulkan/VulkanLib.hpp"

namespace vkop {

#define UP_DIV(x, y) (((x) + (y)-1) / (y))

VulkanCommandBuffer::VulkanCommandBuffer(VkDevice device,
                                         VkCommandPool commandPool,
                                         VkSemaphore semaphore, int count)
    : m_device_(device), m_commandPool_(commandPool), m_semaphore_(semaphore) {
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
    submit_info.commandBufferCount = m_avail_;
    submit_info.pCommandBuffers = m_commandBuffers_.data();
    wait(fence);

    if (vkResetFences(m_device_, 1, &fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to reset fence!");
    }
    assert(queue != nullptr);
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
    submit_info.commandBufferCount = m_commandBuffers_.size();
    submit_info.pCommandBuffers = m_commandBuffers_.data();

    if (vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit command buffer!");
    }
    return 0;
}

int VulkanCommandBuffer::wait(VkFence fence) {
    return vkWaitForFences(m_device_, 1, &fence, VK_TRUE, UINT64_MAX);
}
int VulkanCommandBuffer::wait(uint64_t waitValue) {
    // uint64_t currentValue;
    // vkGetSemaphoreCounterValue(m_device_, m_semaphore_, &currentValue);
    // if (currentValue >= waitValue) {
    //     // Work is complete
    // }

    VkSemaphoreWaitInfo wait_info{};
    wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    wait_info.semaphoreCount = 1;
    wait_info.pSemaphores = &m_semaphore_;
    wait_info.pValues = &waitValue;
    return vkWaitSemaphores(m_device_, &wait_info, UINT64_MAX);
}

void VulkanCommandBuffer::reset(int idx) {
    if (vkResetCommandBuffer(m_commandBuffers_[idx], 0) != VK_SUCCESS) {
        throw std::runtime_error("Failed to reset command buffer!");
    }
}

void VulkanCommandBuffer::reset() {
    for (int i = 0; i < m_avail_; i++) {
        reset(i);
    }
    m_avail_ = 0;
}
void VulkanCommandBuffer::push_constants(VulkanPipeline &pipeline,
                                         uint32_t size, const void *ptr) {
    vkCmdPushConstants(m_commandBuffers_[m_avail_],
                       pipeline.getPipelineLayout(),
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, size, ptr);
}

void VulkanCommandBuffer::dispatch(int w, int h, int z) {
    vkCmdDispatch(m_commandBuffers_[m_avail_], w, h, z);
}

} // namespace vkop
