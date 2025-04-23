#ifndef VULKAN_COMMAND_BUFFER_HPP
#define VULKAN_COMMAND_BUFFER_HPP

#include "VulkanPipeline.hpp"
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>

namespace vkop {
class VulkanCommandBuffer {
public:
    VulkanCommandBuffer(VkDevice device, VkCommandPool commandPool, int count = 1);
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
    void submit(VkQueue queue, VkFence fence = VK_NULL_HANDLE);

    // Reset the command buffer
    void reset(int idx = 0);

    // Get the Vulkan command buffer handle
    VkCommandBuffer get() const { return m_commandBuffers[m_avail]; };

    void dispatch(int w, int h, int z);

private:
    VkDevice m_device;
    VkCommandPool m_commandPool;
    // VkCommandBuffer m_commandBuffer;
    std::vector<VkCommandBuffer> m_commandBuffers;
    int m_avail = 0;

};

}

#endif // VULKAN_COMMAND_BUFFER_HPP