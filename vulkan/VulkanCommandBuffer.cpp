// Copyright 2025 @junka
#include "vulkan/VulkanCommandBuffer.hpp"

#include "vulkan/VulkanLib.hpp"

namespace vkop {

#define UP_DIV(x, y) (((x) + (y)-1) / (y))

VulkanCommandBuffer::VulkanCommandBuffer(
    std::shared_ptr<VulkanDevice> device,
    std::shared_ptr<VulkanCommandPool> cmdpool, bool signaled)
    : m_device_(std::move(device)), m_cmdpool_(std::move(cmdpool)) {
    allocate();
    m_usefence_ =
#ifndef VK_KHR_timeline_semaphore
        true;
#else
        !m_device_->is_support_timeline_semaphore();
#endif

    if (m_usefence_) {
        createFence(signaled);
    }
}

void VulkanCommandBuffer::bind(VulkanPipeline &pipeline,
                               VkDescriptorSet descriptor_set) {
    vkCmdBindPipeline(m_commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pipeline.getComputePipeline());
    vkCmdBindDescriptorSets(m_commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline.getPipelineLayout(), 0, 1, &descriptor_set,
                            0, nullptr);
}

VulkanCommandBuffer::~VulkanCommandBuffer() {
    if (m_fence_ != VK_NULL_HANDLE) {
        vkDestroyFence(m_device_->getLogicalDevice(), m_fence_, nullptr);
        m_fence_ = VK_NULL_HANDLE;
    }
    if (m_commandBuffer_) {
        vkFreeCommandBuffers(m_device_->getLogicalDevice(),
                             m_cmdpool_->getCommandPool(), 1,
                             &m_commandBuffer_);
    }
    if (m_primaryBuffer_) {
        vkFreeCommandBuffers(m_device_->getLogicalDevice(),
                             m_cmdpool_->getCommandPool(), 1,
                             &m_primaryBuffer_);
    }
}

void VulkanCommandBuffer::allocate() {
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = m_cmdpool_->getCommandPool();
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(m_device_->getLogicalDevice(), &alloc_info,
                                 &m_commandBuffer_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate command buffer!");
    }
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
}

int VulkanCommandBuffer::submit(VkQueue queue) {
    if (!m_usefence_) {
        m_timelineValue_ = m_cmdpool_->getNextSubmitValue();
        VkTimelineSemaphoreSubmitInfo timeline_submit_info{};
        timeline_submit_info.sType =
            VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timeline_submit_info.signalSemaphoreValueCount = 1;
        timeline_submit_info.pSignalSemaphoreValues = &m_timelineValue_;

        auto *semaphore = m_cmdpool_->getSemaphore();
        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.pNext = &timeline_submit_info;
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = &semaphore;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &m_commandBuffer_;
        auto ret = vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
        if (ret != VK_SUCCESS) {
            printf("ret %d\n", ret);
            throw std::runtime_error("Failed to submit sem command buffer!");
        }
        return m_timelineValue_;
    }
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &m_commandBuffer_;
    auto ret = vkQueueSubmit(queue, 1, &submit_info, m_fence_);
    if (ret != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit command buffer!");
    }

    return m_cmdpool_->getCompletedTimelineValue();
}

int VulkanCommandBuffer::wait() {
    if (!m_usefence_) {
        uint64_t cur_time;
        auto *semaphore = m_cmdpool_->getSemaphore();
        vkGetSemaphoreCounterValue(m_device_->getLogicalDevice(), semaphore,
                                   &cur_time);
        if (cur_time > m_timelineValue_) {
            return 0;
        }
        VkSemaphoreWaitInfo wait_info{};
        wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        wait_info.semaphoreCount = 1;
        wait_info.pSemaphores = &semaphore;
        wait_info.pValues = &m_timelineValue_;
        if (VK_SUCCESS != vkWaitSemaphores(m_device_->getLogicalDevice(),
                                           &wait_info, UINT64_MAX)) {
            throw std::runtime_error("Failed to wait for semaphore!");
        }
    } else {
        vkWaitForFences(m_device_->getLogicalDevice(), 1, &m_fence_, VK_TRUE,
                        UINT64_MAX);
        if (vkResetFences(m_device_->getLogicalDevice(), 1, &m_fence_) !=
            VK_SUCCESS) {
            throw std::runtime_error("Failed to reset fence!");
        }
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

void VulkanCommandBuffer::exec(VkQueue queue) {
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
    if (m_usefence_) {
        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &m_primaryBuffer_;
        wait();
        if (vkResetFences(m_device_->getLogicalDevice(), 1, &m_fence_) !=
            VK_SUCCESS) {
            throw std::runtime_error("Failed to reset fence!");
        }
        auto ret = vkQueueSubmit(queue, 1, &submit_info, m_fence_);
        if (ret != VK_SUCCESS) {
            throw std::runtime_error("Failed to submit command buffer!");
        }
    }
}

void VulkanCommandBuffer::createFence(bool signaled) {
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = (signaled ? VK_FENCE_CREATE_SIGNALED_BIT : 0);
    if (vkCreateFence(m_device_->getLogicalDevice(), &fence_info, nullptr,
                      &m_fence_) != VK_SUCCESS) {
    }
}

} // namespace vkop
