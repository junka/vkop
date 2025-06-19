// Copyright 2025 @junka
#include "vulkan/VulkanQueryPool.hpp"
#include "vulkan/VulkanLib.hpp"

#include <cstdint>
#include <stdexcept>

namespace vkop {
VulkanQueryPool::VulkanQueryPool(VkDevice device, uint32_t queryCount,
                                 VkQueryType queryType)
    : m_device_(device), m_queryCount_(queryCount) {
    if (m_queryCount_ < 2) {
        m_queryCount_ = 2;
    }
    VkQueryPoolCreateInfo query_pool_info{};
    query_pool_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    query_pool_info.queryType =
        static_cast<VkQueryType>(VK_QUERY_TYPE_TIMESTAMP | queryType);
    query_pool_info.queryCount = queryCount;

    if (vkCreateQueryPool(m_device_, &query_pool_info, nullptr,
                          &m_queryPool_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create query pool");
    }
}
VulkanQueryPool::~VulkanQueryPool() {
    if (m_queryPool_ != VK_NULL_HANDLE) {
        vkDestroyQueryPool(m_device_, m_queryPool_, nullptr);
    }
}

uint64_t VulkanQueryPool::getOneResult(int index) {
    VkQueryResultFlags flags =
        VK_QUERY_RESULT_WAIT_BIT | VK_QUERY_RESULT_64_BIT;
    uint64_t ret;
    vkGetQueryPoolResults(m_device_, m_queryPool_, index, 1, sizeof(uint64_t),
                          &ret, sizeof(uint64_t), flags);
    return ret;
}
std::vector<uint64_t> VulkanQueryPool::getResults() {
    VkQueryResultFlags flags =
        VK_QUERY_RESULT_WAIT_BIT | VK_QUERY_RESULT_64_BIT;
    std::vector<uint64_t> ret(m_queryCount_);
    vkGetQueryPoolResults(m_device_, m_queryPool_, 0, m_queryCount_,
                          m_queryCount_ * sizeof(uint64_t), ret.data(),
                          sizeof(uint64_t), flags);
    return ret;
}
void VulkanQueryPool::reset() {
    vkResetQueryPool(m_device_, m_queryPool_, 0, m_queryCount_);
}

void VulkanQueryPool::begin(VkCommandBuffer cmd) {
    vkCmdResetQueryPool(cmd, m_queryPool_, 0, m_queryCount_);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_queryPool_,
                        0);
}

void VulkanQueryPool::end(VkCommandBuffer cmd) {
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_queryPool_,
                        1);
}

} // namespace vkop
