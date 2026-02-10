// Copyright 2025 @junka
#ifndef SRC_VULKANCOMMANDBUFFER_HPP_
#define SRC_VULKANCOMMANDBUFFER_HPP_

#include "vulkan/VulkanCommandPool.hpp"
#include "vulkan/VulkanPipeline.hpp"
#include "vulkan/VulkanSemaphore.hpp"
#include "vulkan/vulkan_core.h"

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

    int wait();

    // Get the Vulkan command buffer handle
    VkCommandBuffer get() const { return m_commandBuffer_; }

    void push_constants(VulkanPipeline &pipeline, uint32_t size,
                        const void *ptr);
    void dispatch(int w = 1, int h = 1, int z = 1);

    void exec(const std::shared_ptr<VulkanQueue> &queue);

    VkSemaphore getSignalSemaphore() const { return m_signalsem_value_; }
    uint64_t getSignalValue() const { return m_signalValue_; }

  private:
    int id_ = 0;
    std::shared_ptr<VulkanCommandPool> m_cmdpool_;
    bool m_support_timeline_ = true;
    uint64_t m_signalValue_ = 0;

    VkCommandBuffer m_primaryBuffer_ = VK_NULL_HANDLE;
    VkCommandBuffer m_commandBuffer_ = VK_NULL_HANDLE;

    std::unique_ptr<VulkanSemaphore> m_signalsem_ = nullptr;
    VkSemaphore m_signalsem_value_ = VK_NULL_HANDLE;
    std::vector<VkSemaphore> m_waitsems_;
    std::vector<uint64_t> m_waitvalues_;
    std::vector<VkPipelineStageFlags> m_waitstages_;
    VkTimelineSemaphoreSubmitInfo m_timeline_submit_info_ = {};

    // Allocate command buffers
    void allocate();
};

} // namespace vkop

#endif // SRC_VULKANCOMMANDBUFFER_HPP_
