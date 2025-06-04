// Copyright 2025 @junka
#include "Conv2d.hpp"

#include <cmath>
#include <cstdint>
#include <memory>

#include "logger.hpp"

extern unsigned char conv2d_spv[];
extern unsigned int conv2d_spv_len;

namespace vkop {
namespace ops {

void Conv2d::submit(int out_width, int out_height) {
    std::vector<VkDescriptorType> types = {
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
    std::vector<std::shared_ptr<VulkanResource>> objs = {
        outputImage_, inputImage_, weightImage_, biasImage_, paramBuffer_};
    VkDevice device = m_dev_->getLogicalDevice();
    VulkanPipeline pipeline(device, types, objs,
                            reinterpret_cast<const uint32_t *>(conv2d_spv),
                            conv2d_spv_len);

    VulkanCommandBuffer cmd2(device, m_cmdpool_->getCommandPool());
    VulkanQueryPool query_pool(device, 2, VK_QUERY_TYPE_TIMESTAMP);
    cmd2.begin();
    cmd2.bind(pipeline);
    query_pool.begin(cmd2.get());
    cmd2.dispatch(out_width, out_height, 1);
    query_pool.end(cmd2.get());
    cmd2.end();
    cmd2.submit(m_dev_->getComputeQueue());
    auto r = query_pool.getResults();
    double ts = static_cast<double>(r[1] - r[0]) * (1e-9) *
                m_dev_->getTimestampPeriod();
    LOG_INFO("Time: %f s", ts);
}

} // namespace ops
} // namespace vkop