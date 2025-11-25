// Copyright 2025 @junka
#ifndef SRC_VULKANCOMMANDBUFFER_HPP_
#define SRC_VULKANCOMMANDBUFFER_HPP_

#include "vulkan/VulkanPipeline.hpp"
#include "vulkan/vulkan_core.h"
#include <vulkan/vulkan.hpp>

namespace vkop {
class VulkanCommandBuffer {
  public:
    VulkanCommandBuffer(VkDevice device, VkCommandPool commandPool,
                        VkSemaphore semaphore = VK_NULL_HANDLE, int count = 1);
    ~VulkanCommandBuffer();

    VulkanCommandBuffer() = delete;
    // Allocate command buffers
    void allocate(int count);

    // Begin recording commands
    void begin();
    void begin(int idx);

    // End recording commands
    void end();
    void end(int idx);

    void bind(VulkanPipeline &pipeline);

    // Submit the command buffer to a queue
    int submit(VkQueue queue, VkFence fence);
    // Submit the command buffer to a queue with timeline semaphore
    int submit(VkQueue queue, uint64_t submitValue);

    // Reset the command buffer
    void reset(int idx);
    void reset();

    int wait(VkFence fence);
    int wait(uint64_t waitValue);

    // Get the Vulkan command buffer handle
    VkCommandBuffer get() const { return m_commandBuffers_[m_avail_]; }

    void push_constants(VulkanPipeline &pipeline, uint32_t size,
                        const void *ptr);
    void dispatch(int w = 1, int h = 1, int z = 1);

  private:
    VkDevice m_device_;
    VkCommandPool m_commandPool_;
    VkSemaphore m_semaphore_;
    std::vector<VkCommandBuffer> m_commandBuffers_;
    int m_avail_ = 0;
};

} // namespace vkop

#endif // SRC_VULKANCOMMANDBUFFER_HPP_
