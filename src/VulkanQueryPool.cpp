
#include "VulkanLib.hpp"
#include "VulkanQueryPool.hpp"

#include <cstdint>
#include <stdexcept>

namespace vkop {
VulkanQueryPool::VulkanQueryPool(VkDevice device, uint32_t queryCount, VkQueryType queryType)
    : m_device(device), m_queryCount(queryCount)
{
    if (m_queryCount < 2) {
        m_queryCount = 2;
    }
    VkQueryPoolCreateInfo queryPoolInfo{};
    queryPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    queryPoolInfo.queryType = (VkQueryType)(VK_QUERY_TYPE_TIMESTAMP|queryType);
    queryPoolInfo.queryCount = queryCount;

    if (vkCreateQueryPool(m_device, &queryPoolInfo, nullptr, &m_queryPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create query pool");
    }
}
VulkanQueryPool::~VulkanQueryPool() {
    if (m_queryPool != VK_NULL_HANDLE) {
        vkDestroyQueryPool(m_device, m_queryPool, nullptr);
    }
}

uint64_t VulkanQueryPool::getOneResult(int index)
{
    VkQueryResultFlags flags = VK_QUERY_RESULT_WAIT_BIT | VK_QUERY_RESULT_64_BIT;
    uint64_t ret;
    vkGetQueryPoolResults(m_device, m_queryPool, index, 1, sizeof(uint64_t), &ret, sizeof(uint64_t), flags);
    return ret;
}
std::vector<uint64_t> VulkanQueryPool::getResults()
{
    VkQueryResultFlags flags = VK_QUERY_RESULT_WAIT_BIT | VK_QUERY_RESULT_64_BIT;
    std::vector<uint64_t> ret(m_queryCount);
    vkGetQueryPoolResults(m_device, m_queryPool, 0, m_queryCount, m_queryCount *sizeof(uint64_t), ret.data(), sizeof(uint64_t), flags);
    return ret;
}
void VulkanQueryPool::reset()
{
    vkResetQueryPool(m_device, m_queryPool, 0, m_queryCount);
}

void VulkanQueryPool::begin(VkCommandBuffer cmd)
{
    vkCmdResetQueryPool(cmd, m_queryPool, 0, m_queryCount);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_queryPool, 0);
}

void VulkanQueryPool::end(VkCommandBuffer cmd)
{
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_queryPool, 1);
}

}