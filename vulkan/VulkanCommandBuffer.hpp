// Copyright 2025 @junka
#ifndef SRC_VULKANCOMMANDBUFFER_HPP_
#define SRC_VULKANCOMMANDBUFFER_HPP_

#include "vulkan/VulkanCommandPool.hpp"
#include "vulkan/VulkanGraphicsPipeline.hpp"
#include "vulkan/VulkanPipeline.hpp"
#include "vulkan/VulkanSemaphore.hpp"

#include <memory>
#include <vulkan/vulkan.hpp>

namespace vkop {
class VulkanCommandBuffer {

  public:
    explicit VulkanCommandBuffer(std::shared_ptr<VulkanCommandPool> cmdpool,
                                 int id = 0);
    ~VulkanCommandBuffer();

    VulkanCommandBuffer() = delete;

    // Begin recording commands
    void begin();

    // End recording commands
    void end();

    void bind(VulkanPipeline &pipeline, VkDescriptorSet descriptor_set);
    void bindGraphics(VulkanGraphicsPipeline &pipeline,
                      VkDescriptorSet descriptor_set, VkBuffer vertex_buffer,
                      VkBuffer indexbuffer);

    // Submit the command buffer to a queue
    uint64_t submit(const std::shared_ptr<VulkanQueue> &queue);
    static void submit(const std::shared_ptr<VulkanQueue> &queue,
                       std::vector<VkSubmitInfo> &submit_infos);

    VkSubmitInfo buildSubmitInfo();

    // Reset the command buffer
    void reset();

    void clearWaits() {
        m_waitsems_.clear();
        m_waitstages_.clear();
        m_waitvalues_.clear();
        m_sigsems_.clear();
        m_sigvalues_.clear();
    }
    void
    addWait(VkSemaphore sem, uint64_t value,
            VkPipelineStageFlags stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT) {
        m_waitsems_.emplace_back(sem);
        if (m_support_timeline_) {
            m_waitvalues_.emplace_back(value);
        }
        m_waitstages_.emplace_back(stage);
    }

    void
    addWait(VkSemaphore sem, uint64_t value, VkSemaphore sigsem,
            VkPipelineStageFlags stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT) {
        m_waitsems_.emplace_back(sem);
        m_sigsems_.emplace_back(sigsem);
        if (m_support_timeline_) {
            m_waitvalues_.emplace_back(value);
        }
        m_waitstages_.emplace_back(stage);
    }

    int wait();

    // Get the Vulkan command buffer handle
    VkCommandBuffer get() const { return m_commandBuffer_; }

    // Get the command pool this buffer was allocated from (used by tensor
    // upload fallbacks that need a VulkanCommandPool for a fresh staging copy).
    std::shared_ptr<VulkanCommandPool> getCommandPool() const {
        return m_cmdpool_;
    }

    // Mark this command buffer as replayable: begin() will use
    // SIMULTANEOUS_USE_BIT instead of ONE_TIME_SUBMIT_BIT so the recorded
    // buffer can be submitted more than once (cuda-graph-style replay). The
    // caller must NOT call reset() between replays.
    void set_replayable(bool v) { replayable_ = v; }
    bool is_replayable() const { return replayable_; }

    void push_constants(VulkanPipeline &pipeline, uint32_t size,
                        const void *ptr);
    void dispatch(int w = 1, int h = 1, int z = 1);

    // Dispatch with dimensions read from a GPU buffer (vkCmdDispatchIndirect).
    // `buffer` holds a VkDispatchIndirectCommand{uint32 x,y,z} at `offset`.
    // Enables data-driven dispatch where the thread count depends on a shape
    // value computed on the GPU (e.g. kv_len-dependent attention), avoiding a
    // GPU->CPU readback just to know the dispatch dims. The buffer must have
    // been written (e.g. by a shape->dispatch shader) and barriered to
    // INDIRECT_READ before this call.
    void dispatch_indirect(VkBuffer buffer, VkDeviceSize offset = 0);

    void exec(const std::shared_ptr<VulkanQueue> &queue);

    // GPU timestamp profiling (submit-side attribution). Writes a timestamp
    // into this command buffer's recording at the given query index. Call
    // writeTimestampBegin AFTER begin() (TOP_OF_PIPE = when the cmd reaches
    // the front of the queue) and writeTimestampEnd BEFORE end()
    // (BOTTOM_OF_PIPE = after all prior work in this cmd completes). The host
    // reads results via vkGetQueryPoolResults after the round's final wait.
    // Used by VKOP_SUBMIT_PROF to attribute the ~515ms submit floor per op-type
    // (opprof only measures CPU record time, not GPU execution).
    void writeTimestampBegin(VkQueryPool pool, uint32_t query) {
        vkCmdWriteTimestamp(m_commandBuffer_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            pool, query);
    }
    void writeTimestampEnd(VkQueryPool pool, uint32_t query) {
        vkCmdWriteTimestamp(m_commandBuffer_,
                            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, query);
    }
    // Reset a range of timestamp queries at the start of the frame. Must be
    // recorded into a command buffer that runs before any query writes.
    static void resetQueryRange(VkCommandBuffer cmd, VkQueryPool pool,
                                uint32_t first, uint32_t count) {
        vkCmdResetQueryPool(cmd, pool, first, count);
    }

    VkSemaphore getSignalSemaphore() const {
        return m_signalsem_->getSemaphore();
    }
    uint64_t getSignalValue() const { return m_signalValue_; }

  private:
    int id_ = 0;
    std::shared_ptr<VulkanCommandPool> m_cmdpool_;
    bool m_support_timeline_ = true;
    uint64_t m_signalValue_ = 0;

    VkCommandBuffer m_primaryBuffer_ = VK_NULL_HANDLE;
    VkCommandBuffer m_commandBuffer_ = VK_NULL_HANDLE;

    std::unique_ptr<VulkanSemaphore> m_signalsem_ = nullptr;
    std::vector<VkSemaphore> m_waitsems_;
    std::vector<VkSemaphore> m_sigsems_;
    std::vector<uint64_t> m_waitvalues_;
    std::vector<uint64_t> m_sigvalues_;
    std::vector<VkPipelineStageFlags> m_waitstages_;
    VkTimelineSemaphoreSubmitInfo m_timeline_submit_info_ = {};
    bool replayable_ = false;

    // Allocate command buffers
    void allocate();
};

} // namespace vkop

#endif // SRC_VULKANCOMMANDBUFFER_HPP_
