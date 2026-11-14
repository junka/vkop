// Copyright 2025 - 2026 @junka
#include "vulkan/VulkanCommandBuffer.hpp"
#include "vulkan/VulkanSemaphore.hpp"

#include "vulkan/DescriptorWriteProf.hpp"
#include "vulkan/VulkanLib.hpp"
#include <memory>
#include <string>

namespace vkop {

#define UP_DIV(x, y) (((x) + (y) - 1) / (y))

namespace {
// A negative VkResult out of a submit or a wait is usually
// VK_ERROR_DEVICE_LOST (-4): the driver killed the GPU mid-submission and
// every later submit from this process fails too. The bare number reads like
// a timeout, so name it -- the recovery is a process restart, not a retry,
// and llm_serve keys off the "VK_ERROR_DEVICE_LOST" text.
std::string vk_result_label(VkResult r) {
    if (r == VK_ERROR_DEVICE_LOST)
        return "[VK_ERROR_DEVICE_LOST: GPU is gone, restart required]";
    return "[VkResult=" + std::to_string(static_cast<int>(r)) + "]";
}
}  // namespace

std::function<void(core::ITensor &)> VulkanCommandBuffer::pre_readback_hook =
    nullptr;

void VulkanCommandBuffer::pipelineBarrier() {
    // Narrow barrier: the graph segment only ever issues compute shader
    // dispatches + vkCmdCopyBuffer/vkCmdUpdateBuffer/image transfer. We do
    // NOT emit graphics, raytrace, host-copy or indirect-dispatch reads
    // between levels (indirect dispatch dimensions are baked at build time).
    // COMPUTE_SHADER|TRANSFER + SHADER|TRANSFER access is the tightest safe
    // scope and cuts ~0.2 ms/barrier from the 980-level decode loop.
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.pNext = nullptr;
    barrier.srcAccessMask =
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                            VK_ACCESS_SHADER_WRITE_BIT |
                            VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(
        m_commandBuffer_,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);
}

void VulkanCommandBuffer::execute_commands(
    const std::vector<VkCommandBuffer> &secondaries) {
    if (secondaries.empty())
        return;
    // ONE call for the whole round; the memory dependency between the chained
    // secondaries is the barrier each one records as its last command (see
    // Runtime's VKOP_SEC_CMD branch of graph_submit).
    vkCmdExecuteCommands(m_commandBuffer_,
                         static_cast<uint32_t>(secondaries.size()),
                         secondaries.data());
}

VulkanCommandBuffer::VulkanCommandBuffer(
    std::shared_ptr<VulkanCommandPool> cmdpool, int id, bool secondary)
    : id_(id), m_cmdpool_(std::move(cmdpool)) {
    secondary_ = secondary;
    m_inheritance_.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
    allocate();
    std::shared_ptr<VulkanDevice> device = m_cmdpool_->getVulkanDevice();
    m_support_timeline_ = device->is_support_timeline_semaphore();

    m_signalsem_ = std::make_unique<VulkanSemaphore>(
        device->getLogicalDevice(), device->is_support_timeline_semaphore());
}

void VulkanCommandBuffer::bind(VulkanPipeline &pipeline,
                               VkDescriptorSet descriptor_set) {
    // Split-timed per command: under VK_KHR_push_descriptor
    // vkCmdBindDescriptorSets goes away (the bindings are pushed instead) while
    // vkCmdBindPipeline stays, so only the "bind_ds" column is in play.
    const uint64_t t_prof0 = ds_prof_now();
    vkCmdBindPipeline(m_commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pipeline.getPipeline());
    const uint64_t t_prof1 = ds_prof_now();
    if (pipeline.usesDescriptorBuffer()) {
        // The descriptor bytes live in the pipeline's own buffer; only the
        // buffer binding is stateful. Re-bind it when switching pipelines and
        // always set this dispatch's set offset (the token encodes the slot).
        const VkBuffer buffer = pipeline.getDescriptorBuffer();
        if (buffer != m_descriptor_buffer_bound_) {
            pipeline.bindDescriptorBuffer(m_commandBuffer_);
            m_descriptor_buffer_bound_ = buffer;
        }
        pipeline.setDescriptorBufferOffsets(m_commandBuffer_, descriptor_set);
    } else {
        vkCmdBindDescriptorSets(
            m_commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.getPipelineLayout(), 0, 1, &descriptor_set, 0, nullptr);
    }
    ds_prof_record("bind_pipeline", t_prof1 - t_prof0, 1);
    ds_prof_record("bind_ds", ds_prof_now() - t_prof1, 1);
}

void VulkanCommandBuffer::bind_pipeline(VulkanPipeline &pipeline) {
    const uint64_t t_prof0 = ds_prof_now();
    vkCmdBindPipeline(m_commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pipeline.getPipeline());
    ds_prof_record("bind_pipeline", ds_prof_now() - t_prof0, 1);
}

void VulkanCommandBuffer::bindGraphics(VulkanGraphicsPipeline &pipeline,
                                       VkDescriptorSet descriptor_set,
                                       VkBuffer vertex_buffer,
                                       VkBuffer indexbuffer) {
    vkCmdBindPipeline(m_commandBuffer_, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipeline.getPipeline());
    if (vertex_buffer) {
        VkBuffer vertex_buffers[] = {vertex_buffer};
        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(m_commandBuffer_, 0, 1, vertex_buffers, offsets);
    }
    if (indexbuffer) {
        vkCmdBindIndexBuffer(m_commandBuffer_, indexbuffer, 0,
                             VK_INDEX_TYPE_UINT32);
    }
    vkCmdBindDescriptorSets(m_commandBuffer_, VK_PIPELINE_BIND_POINT_GRAPHICS,
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
    alloc_info.level = secondary_ ? VK_COMMAND_BUFFER_LEVEL_SECONDARY
                                  : VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    std::shared_ptr<VulkanDevice> device = m_cmdpool_->getVulkanDevice();
    auto ret = vkAllocateCommandBuffers(device->getLogicalDevice(), &alloc_info,
                                        &m_commandBuffer_);
    if (ret != VK_SUCCESS) {
        fprintf(stderr, "ret %d\n", ret);
        throw std::runtime_error("Failed to allocate command buffer");
    }
}

void VulkanCommandBuffer::begin() {
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    // ONE_TIME_SUBMIT is the default (cheapest on most drivers) but prevents
    // re-submission. Replayable buffers (cached for cuda-graph-style replay)
    // use SIMULTANEOUS_USE so the same recording can be submitted every round.
    begin_info.flags = replayable_
                           ? VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT
                           : VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    // A compute-only secondary inherits nothing (no render pass), but the
    // inheritance info struct itself is required at SECONDARY level.
    if (secondary_)
        begin_info.pInheritanceInfo = &m_inheritance_;

    // replayable_ 的 buffer 可能已经处于 Executable 状态 (上轮 end + submit
    // 过, reset 循环不再主动 reset), 必须先 vkResetCommandBuffer 让它回到
    // Initial 再 begin. 非 replayable buffer 依赖外部 reset 循环, 不做隐式 reset.
    if (replayable_) {
        vkResetCommandBuffer(m_commandBuffer_, 0);
    }
    // begin() always starts a fresh recording (non-replayable buffers were
    // reset externally); descriptor-buffer binding is recording state.
    m_descriptor_buffer_bound_ = VK_NULL_HANDLE;

    // Defensive: a NULL handle means the buffer was never allocated (or was
    // lost); re-allocate rather than crashing inside vkBeginCommandBuffer.
    if (m_commandBuffer_ == VK_NULL_HANDLE) {
        allocate();
    }
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

    auto submit_info = buildSubmitInfo();

    auto ret =
        vkQueueSubmit(queue->getQueue(), 1, &submit_info, VK_NULL_HANDLE);
    if (ret != VK_SUCCESS) {
        fprintf(stderr, "ret %d\n", ret);
        throw std::runtime_error("Failed to submit sem command buffer! " +
                                 vk_result_label(ret));
    }
    queue->bumpSubmitCount();
    clearWaits();
    return m_signalValue_;
}

void VulkanCommandBuffer::submit(const std::shared_ptr<VulkanQueue> &queue,
                                 std::vector<VkSubmitInfo> &submit_infos) {
    auto ret = vkQueueSubmit(queue->getQueue(),
                             static_cast<uint32_t>(submit_infos.size()),
                             submit_infos.data(), VK_NULL_HANDLE);
    if (ret != VK_SUCCESS) {
        fprintf(stderr, "ret %d\n", ret);
        throw std::runtime_error("Failed to submit sem command buffer! " +
                                 vk_result_label(ret));
    }
    queue->bumpSubmitCount();
}

VkSubmitInfo VulkanCommandBuffer::buildSubmitInfo() {
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    auto *signalsem_value = m_signalsem_->getSemaphore();
    m_sigsems_.push_back(signalsem_value);
    submit_info.signalSemaphoreCount = static_cast<uint32_t>(m_sigsems_.size());
    submit_info.pSignalSemaphores = m_sigsems_.data();

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
        m_timeline_submit_info_.signalSemaphoreValueCount =
            static_cast<uint32_t>(m_sigsems_.size());
        m_sigvalues_.resize(m_sigsems_.size(), m_signalValue_);
        m_timeline_submit_info_.pSignalSemaphoreValues = m_sigvalues_.data();
        submit_info.pNext = &m_timeline_submit_info_;
    }

    return submit_info;
}

int VulkanCommandBuffer::wait() {
    std::shared_ptr<VulkanDevice> device = m_cmdpool_->getVulkanDevice();

    if (!m_waitsems_.empty())
        m_waitsems_.pop_back();

    // this structure can not accept binary semaphore

    auto *signalsem = m_signalsem_->getSemaphore();
    VkSemaphoreWaitInfo wait_info{};
    wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    wait_info.semaphoreCount = 1;
    wait_info.pSemaphores = &signalsem;

    uint64_t cur_time = 0;
    if (m_support_timeline_) {
        vkGetSemaphoreCounterValue(device->getLogicalDevice(), signalsem,
                                   &cur_time);
        if (cur_time > m_signalValue_) {
            return 0;
        }

        wait_info.pValues = &m_signalValue_;
    } else {
        wait_info.pValues = nullptr;
    }

    VkResult wres = vkWaitSemaphores(device->getLogicalDevice(), &wait_info,
                                     UINT64_MAX);
    if (VK_SUCCESS != wres) {
        std::fprintf(stderr,
                     "[sem-wait] res=%d timeline=%d want=%llu cur=%llu "
                     "this=%p\n",
                     static_cast<int>(wres),
                     static_cast<int>(m_support_timeline_),
                     static_cast<unsigned long long>(m_signalValue_),
                     static_cast<unsigned long long>(cur_time),
                     static_cast<void *>(this));
        throw std::runtime_error("Failed to wait for binary semaphore! " +
                                 vk_result_label(wres));
    }

    return 0;
}

void VulkanCommandBuffer::reset() {
    if (vkResetCommandBuffer(m_commandBuffer_, 0) != VK_SUCCESS) {
        throw std::runtime_error("Failed to reset command buffer!");
    }
    m_descriptor_buffer_bound_ = VK_NULL_HANDLE;
}

void VulkanCommandBuffer::push_constants(VulkanPipeline &pipeline,
                                         uint32_t size, const void *ptr) {
    // Timed as the anchor for what an inline recorded command costs on this
    // driver: vkCmdPushDescriptorSetKHR, which is what VK_KHR_push_descriptor
    // would swap in for vkUpdateDescriptorSets, is the same kind of call.
    const uint64_t t_prof0 = ds_prof_now();
    vkCmdPushConstants(m_commandBuffer_, pipeline.getPipelineLayout(),
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, size, ptr);
    ds_prof_record("push_const", ds_prof_now() - t_prof0, 1);
}

void VulkanCommandBuffer::dispatch(int w, int h, int z) {
    VulkanDebug::insert_label(m_commandBuffer_, debug_label_.c_str());
    vkCmdDispatch(m_commandBuffer_, w, h, z);
}

void VulkanCommandBuffer::dispatch_indirect(VkBuffer buffer,
                                            VkDeviceSize offset) {
    VulkanDebug::insert_label(m_commandBuffer_, debug_label_.c_str());
    vkCmdDispatchIndirect(m_commandBuffer_, buffer, offset);
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
