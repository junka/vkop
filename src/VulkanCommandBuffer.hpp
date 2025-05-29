// Copyright 2025 @junka
#ifndef SRC_VULKANCOMMANDBUFFER_HPP_
#define SRC_VULKANCOMMANDBUFFER_HPP_

#include "VulkanPipeline.hpp"
#include <vulkan/vulkan.hpp>

namespace vkop {
class VulkanCommandBuffer {
  public:
    VulkanCommandBuffer(VkDevice device, VkCommandPool commandPool,
                        int count = 1);
    ~VulkanCommandBuffer();

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
    int submit(VkQueue queue, VkFence fence = VK_NULL_HANDLE);

    // Reset the command buffer
    void reset(int idx = 0);

    // Get the Vulkan command buffer handle
    VkCommandBuffer get() const { return m_commandBuffers_[m_avail_]; }

    void dispatch(int w, int h, int z);

  private:
    VkDevice m_device_;
    VkCommandPool m_commandPool_;
    // VkCommandBuffer m_commandBuffer;
    std::vector<VkCommandBuffer> m_commandBuffers_;
    int m_avail_ = 0;
};

} // namespace vkop

#endif // SRC_VULKANCOMMANDBUFFER_HPP_
