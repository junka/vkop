#ifndef VULKAN_QUERY_POOL_HPP
#define VULKAN_QUERY_POOL_HPP

#include <vulkan/vulkan.hpp>
#include <vector>

namespace vkop {

class VulkanQueryPool {
public:
    VulkanQueryPool(VkDevice device, uint32_t queryCount, VkQueryType queryType);

    ~VulkanQueryPool();

    VulkanQueryPool(const VulkanQueryPool&) = delete;
    VulkanQueryPool& operator=(const VulkanQueryPool&) = delete;
    VulkanQueryPool& operator=(VulkanQueryPool&& other) = delete;

    VkQueryPool get() const { return m_queryPool; }

    uint64_t getOneResult(int index);

    std::vector<uint64_t> getResults();

    void reset();

    void begin(VkCommandBuffer cmd);

    void end(VkCommandBuffer cmd);

private:
    VkDevice m_device;
    uint32_t m_queryCount;
    VkQueryPool m_queryPool;
};

} // namespace vkop

#endif