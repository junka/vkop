// Copyright 2025 @junka
#ifndef SRC_VULKANCOMMANDBUFFER_HPP_
#define SRC_VULKANCOMMANDBUFFER_HPP_

#include "vulkan/VulkanCommandPool.hpp"
#include "vulkan/VulkanPipeline.hpp"

#include "vulkan/vulkan_core.h"
#include <vulkan/vulkan.hpp>

namespace vkop {
class VulkanCommandBuffer {
  public:
    VulkanCommandBuffer(std::shared_ptr<VulkanCommandPool> cmdpool,
                        bool signaled = true, int id = 0);
    ~VulkanCommandBuffer();

    VulkanCommandBuffer() = delete;

    // Begin recording commands
    void begin();

    // End recording commands
    void end();

    void bind(VulkanPipeline &pipeline, VkDescriptorSet descriptor_set);

    // Submit the command buffer to a queue
    int submit(const std::shared_ptr<VulkanQueue> &queue);

    // Reset the command buffer
    void reset();

    int wait(const std::shared_ptr<VulkanQueue> &queue);

    // Get the Vulkan command buffer handle
    VkCommandBuffer get() const { return m_commandBuffer_; }

    void push_constants(VulkanPipeline &pipeline, uint32_t size,
                        const void *ptr);
    void dispatch(int w = 1, int h = 1, int z = 1);

    void exec(const std::shared_ptr<VulkanQueue> &queue);

  private:
    int id_ = 0;
    std::shared_ptr<VulkanCommandPool> m_cmdpool_;
    bool m_usefence_ = true;
    uint64_t m_timelineValue_ = 0;
    VkFence m_fence_ = VK_NULL_HANDLE;
    VkCommandBuffer m_primaryBuffer_ = VK_NULL_HANDLE;
    VkCommandBuffer m_commandBuffer_ = VK_NULL_HANDLE;

    // Allocate command buffers
    void allocate();
    void createFence(bool signaled);
};

} // namespace vkop

#endif // SRC_VULKANCOMMANDBUFFER_HPP_
