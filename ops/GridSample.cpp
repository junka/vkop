// Copyright 2025 @junka
#include "logger.hpp"
#include <cmath>
#include <cstdint>
#include <memory>

#include "GridSample.hpp"

/* definition in spriv generate source file to avoid violate ODR */
extern unsigned char grid_sample_spv[];
extern unsigned int grid_sample_spv_len;

namespace vkop {

namespace ops {

void GridSample::prepare(const std::vector<int> &input_shape,
                         const std::vector<int> &grid_shape) {
    int batch = input_shape[0];
    int depth = input_shape[1];
    int gridbatch = grid_shape[0];
    int in_height = input_shape[2];
    int in_width = input_shape[3];
    int out_height = grid_shape[1];
    int out_width = grid_shape[2];

    VkDevice device = m_dev_->getLogicalDevice();
    int exflags = 0;
    if (m_dev_->is_support_host_image_copy()) {
#ifdef VK_EXT_host_image_copy
        exflags |= VK_IMAGE_USAGE_HOST_TRANSFER_BIT;
#endif
    }
    outputImage_ = std::make_shared<VulkanImage>(
        m_phydev_, m_dev_->getComputeQueueFamilyIndex(), device,
        VkExtent3D{static_cast<uint32_t>(out_width) * UP_DIV(depth, 4),
                   static_cast<uint32_t>(out_height) * batch, 1},
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | exflags,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    inputImage_ = std::make_shared<VulkanImage>(
        m_phydev_, m_dev_->getComputeQueueFamilyIndex(), device,
        VkExtent3D{static_cast<uint32_t>(in_width) * UP_DIV(depth, 4),
                   static_cast<uint32_t>(in_height) * batch, 1},
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | exflags,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    gridImage_ = std::make_shared<VulkanImage>(
        m_phydev_, m_dev_->getComputeQueueFamilyIndex(), device,
        VkExtent3D{2 * UP_DIV((uint32_t)out_height, 4),
                   static_cast<uint32_t>(out_width) * gridbatch, 1},
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | exflags,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    paramBuffer_ = std::make_shared<VulkanBuffer>(
        m_phydev_, m_dev_->getComputeQueueFamilyIndex(), device,
        sizeof(GpuGridSampleParam), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
#ifdef VK_EXT_host_image_copy
    if (m_dev->is_support_host_image_copy()) {
        if (m_dev->checkHostImageCopyDstLayoutSupport(
                VK_IMAGE_LAYOUT_GENERAL)) {
            outputImage->hostImaggeTransition(VK_IMAGE_LAYOUT_GENERAL);
        } else {
            outputImage->hostImaggeTransition(
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        }
        inputImage->hostImaggeTransition(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        gridImage->hostImaggeTransition(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    } else
#endif
    {
        VulkanCommandBuffer cmd(device, m_cmdpool_->getCommandPool());
        cmd.begin();
        outputImage_->writeBarrier(cmd.get());
        inputImage_->readBarrier(cmd.get());
        gridImage_->readBarrier(cmd.get());
        cmd.end();
        cmd.submit(m_dev_->getComputeQueue());
    }
}

void GridSample::submit(int out_width, int out_height) {
    std::vector<VkDescriptorType> types = {
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
    std::vector<std::shared_ptr<VulkanResource>> objs = {
        outputImage_, inputImage_, gridImage_, paramBuffer_};
    VkDevice device = m_dev_->getLogicalDevice();
    VulkanPipeline pipeline(device, types, objs,
                            reinterpret_cast<const uint32_t *>(grid_sample_spv),
                            grid_sample_spv_len);

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
