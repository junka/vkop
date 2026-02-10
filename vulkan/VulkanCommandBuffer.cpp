// Copyright 2025 - 2026 @junka
#include "vulkan/VulkanCommandBuffer.hpp"
#include "vulkan/VulkanSemaphore.hpp"

#include "vulkan/VulkanLib.hpp"
#include <memory>

namespace vkop {

#define UP_DIV(x, y) (((x) + (y) - 1) / (y))

VulkanCommandBuffer::VulkanCommandBuffer(
    std::shared_ptr<VulkanCommandPool> cmdpool, int id)
    : id_(id), m_cmdpool_(std::move(cmdpool)) {
    allocate();
    std::shared_ptr<VulkanDevice> device = m_cmdpool_->getVulkanDevice();
    m_support_timeline_ = device->is_support_timeline_semaphore();

    m_signalsem_ = std::make_unique<VulkanSemaphore>(
        device->getLogicalDevice(), device->is_support_timeline_semaphore());
    m_signalsem_value_ = m_signalsem_->getSemaphore();
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
    std::shared_ptr<VulkanDevice> device = m_cmdpool_->getVulkanDevice();
    if (m_commandBuffer_) {
        vkFreeCommandBuffers(device->getLogicalDevice(),
                             m_cmdpool_->getCommandPool(id_), 1,
                             &m_commandBuffer_);
    }
    if (m_primaryBuffer_) {
        vkFreeCommandBuffers(device->getLogicalDevice(),
                             m_cmdpool_->getCommandPool(id_), 1,
                             &m_primaryBuffer_);
    }
}

void VulkanCommandBuffer::allocate() {
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = m_cmdpool_->getCommandPool(id_);
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    std::shared_ptr<VulkanDevice> device = m_cmdpool_->getVulkanDevice();
    auto ret = vkAllocateCommandBuffers(device->getLogicalDevice(), &alloc_info,
                                        &m_commandBuffer_);
    if (ret != VK_SUCCESS) {
        printf("ret %d\n", ret);
        throw std::runtime_error("Failed to allocate command buffer");
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

uint64_t
VulkanCommandBuffer::submit(const std::shared_ptr<VulkanQueue> &queue) {

    if (m_support_timeline_) {
        m_signalValue_ = m_cmdpool_->getNextSubmitValue();
    } else {
        m_signalValue_ = 0;
    }

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = &m_signalsem_value_;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &m_commandBuffer_;

    submit_info.waitSemaphoreCount = static_cast<uint32_t>(m_waitsems_.size());
    submit_info.pWaitSemaphores =
        m_waitsems_.empty() ? nullptr : m_waitsems_.data();
    submit_info.pWaitDstStageMask =
        m_waitstages_.empty() ? nullptr : m_waitstages_.data();
    VkTimelineSemaphoreSubmitInfo timeline_submit_info{};
    if (m_support_timeline_) {
        timeline_submit_info.sType =
            VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timeline_submit_info.waitSemaphoreValueCount =
            static_cast<uint32_t>(m_waitvalues_.size());
        timeline_submit_info.pWaitSemaphoreValues =
            m_waitvalues_.empty() ? nullptr : m_waitvalues_.data();
        timeline_submit_info.signalSemaphoreValueCount = 1;
        timeline_submit_info.pSignalSemaphoreValues = &m_signalValue_;

        submit_info.pNext = &timeline_submit_info;
    }

    auto ret =
        vkQueueSubmit(queue->getQueue(), 1, &submit_info, VK_NULL_HANDLE);
    if (ret != VK_SUCCESS) {
        printf("ret %d\n", ret);
        throw std::runtime_error("Failed to submit sem command buffer!");
    }
    clearWaits();
    return m_signalValue_;
}

void VulkanCommandBuffer::submit(const std::shared_ptr<VulkanQueue> &queue,
                                 std::vector<VkSubmitInfo> &submit_infos) {
    auto ret = vkQueueSubmit(queue->getQueue(),
                             static_cast<uint32_t>(submit_infos.size()),
                             submit_infos.data(), VK_NULL_HANDLE);
    if (ret != VK_SUCCESS) {
        printf("ret %d\n", ret);
        throw std::runtime_error("Failed to submit sem command buffer!");
    }
}

VkSubmitInfo VulkanCommandBuffer::buildSubmitInfo() {
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    // signal 1 sem, or 0 if same queue
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = &m_signalsem_value_;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &m_commandBuffer_;

    submit_info.waitSemaphoreCount = static_cast<uint32_t>(m_waitsems_.size());
    submit_info.pWaitSemaphores =
        m_waitsems_.empty() ? nullptr : m_waitsems_.data();
    submit_info.pWaitDstStageMask =
        m_waitstages_.empty() ? nullptr : m_waitstages_.data();

    if (m_support_timeline_) {
        m_signalValue_ = m_cmdpool_->getNextSubmitValue();
        m_timeline_submit_info_.sType =
            VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        m_timeline_submit_info_.pNext = nullptr;
        m_timeline_submit_info_.waitSemaphoreValueCount =
            static_cast<uint32_t>(m_waitvalues_.size());
        m_timeline_submit_info_.pWaitSemaphoreValues =
            m_waitvalues_.empty() ? nullptr : m_waitvalues_.data();
        m_timeline_submit_info_.signalSemaphoreValueCount = 1;
        m_timeline_submit_info_.pSignalSemaphoreValues = &m_signalValue_;
        submit_info.pNext = &m_timeline_submit_info_;
    }

    return submit_info;
}

int VulkanCommandBuffer::wait() {
    std::shared_ptr<VulkanDevice> device = m_cmdpool_->getVulkanDevice();

    VkSemaphoreWaitInfo wait_info{};
    wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    wait_info.semaphoreCount = 1;
    wait_info.pSemaphores = &m_signalsem_value_;
    if (m_support_timeline_) {
        uint64_t cur_time;
        vkGetSemaphoreCounterValue(device->getLogicalDevice(),
                                   m_signalsem_value_, &cur_time);
        if (cur_time > m_signalValue_) {
            return 0;
        }

        wait_info.pValues = &m_signalValue_;
    } else {
        wait_info.pValues = nullptr;
    }

    if (VK_SUCCESS !=
        vkWaitSemaphores(device->getLogicalDevice(), &wait_info, UINT64_MAX)) {
        throw std::runtime_error("Failed to wait for binary semaphore!");
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

void VulkanCommandBuffer::exec(const std::shared_ptr<VulkanQueue> &queue) {
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
    wait();
    std::shared_ptr<VulkanDevice> device = m_cmdpool_->getVulkanDevice();
    auto ret = vkQueueSubmit(queue->getQueue(), 1, &submit_info, nullptr);
    if (ret != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit command buffer!");
    }
}

} // namespace vkop
