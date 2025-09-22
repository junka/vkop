// Copyright 2025 @junka
#include "Softmax.hpp"

#include <cmath>
#include <cstdint>
#include <memory>

#include "OperatorFactory.hpp"
#include "Ops.hpp"
#include "include/logger.hpp"

namespace vkop {
namespace ops {

void Softmax::submit(const unsigned char *spv, unsigned int spv_len,
                     int out_width, int out_height) {
    std::vector<VkDescriptorType> types = {
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
    std::vector<std::shared_ptr<VulkanResource>> objs = {
        outputImage_, inputImage_, paramBuffer_};
    VkDevice device = m_dev_->getLogicalDevice();
    VulkanPipeline pipeline(device, types, objs,
                            reinterpret_cast<const uint32_t *>(spv), spv_len);

    VulkanCommandBuffer cmd2(device, m_cmdpool_->getCommandPool());
    VulkanQueryPool query_pool(device, 2, VK_QUERY_TYPE_TIMESTAMP);
    cmd2.begin();
    cmd2.bind(pipeline);
    query_pool.begin(cmd2.get());
    cmd2.dispatch(out_width, out_height);
    query_pool.end(cmd2.get());
    cmd2.end();
    cmd2.submit(m_dev_->getComputeQueue());
    auto r = query_pool.getResults();
    double ts = static_cast<double>(r[1] - r[0]) * (1e-9) *
                m_dev_->getTimestampPeriod();
    LOG_INFO("Time: %f s", ts);
}

void Softmax::execute(
    std::vector<std::shared_ptr<core::Tensor<float>>> inputs,
    std::vector<std::shared_ptr<core::Tensor<float>>> outputs) {
    apply<float>(inputs, outputs);
}

void Softmax::execute(std::vector<std::shared_ptr<core::Tensor<int>>> inputs,
                      std::vector<std::shared_ptr<core::Tensor<int>>> outputs) {
    apply<int>(inputs, outputs);
}
void Softmax::execute(
    std::vector<std::shared_ptr<core::Tensor<uint16_t>>> inputs,
    std::vector<std::shared_ptr<core::Tensor<uint16_t>>> outputs) {
    apply<uint16_t>(inputs, outputs);
}

namespace {
REGISTER_OPERATOR(OpType::SOFTMAX, Softmax);
} // namespace

} // namespace ops
} // namespace vkop
