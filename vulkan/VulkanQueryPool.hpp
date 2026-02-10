// Copyright 2025 @junka
#ifndef SRC_VULKANQUERYPOOL_HPP_
#define SRC_VULKANQUERYPOOL_HPP_

#include <vector>
#include <vulkan/vulkan.hpp>

namespace vkop {

class VulkanQueryPool {
  public:
    VulkanQueryPool(VkDevice device, uint32_t queryCount,
                    VkQueryType queryType = VK_QUERY_TYPE_TIMESTAMP);

    ~VulkanQueryPool();

    VulkanQueryPool(const VulkanQueryPool &) = delete;
    VulkanQueryPool &operator=(const VulkanQueryPool &) = delete;
    VulkanQueryPool &operator=(VulkanQueryPool &&other) = delete;

    uint64_t getOneResult(int index);

    std::vector<uint64_t> getResults();

    void reset();

    void begin(VkCommandBuffer cmd);

    void end(VkCommandBuffer cmd);

  private:
    VkDevice m_device_;
    uint32_t m_queryCount_;
    VkQueryPool m_queryPool_;
};

} // namespace vkop

#endif // SRC_VULKANQUERYPOOL_HPP_
