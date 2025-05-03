
#include <cstdint>
#include <memory>
#include <cmath>
#include "logger.hpp"

#include "GridSample.hpp"

#include "grid_sample.h"

namespace vkop {

namespace ops {

void GridSample::prepare(const std::vector<int>& input_shape, const std::vector<int>& grid_shape)
{
    int batch = input_shape[0];
    int depth = input_shape[1];
    int gridbatch = grid_shape[0];
    int inHeight = input_shape[2];
    int inWidth = input_shape[3];
    int outHeight = grid_shape[1];
    int outWidth = grid_shape[2];

    VkDevice device = m_dev->getLogicalDevice();
    int exflags = 0;
    if (m_dev->is_support_host_image_copy()) {
#ifdef VK_EXT_host_image_copy
        exflags |= VK_IMAGE_USAGE_HOST_TRANSFER_BIT;
#endif
    }
    outputImage = std::make_shared<VulkanImage>(m_phydev, m_dev->getComputeQueueFamilyIndex(), device, VkExtent3D {
        (uint32_t)outWidth * UP_DIV(depth, 4),
        (uint32_t)outHeight * batch,
        1
    }, VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT|exflags, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    inputImage = std::make_shared<VulkanImage>(m_phydev, m_dev->getComputeQueueFamilyIndex(), device, VkExtent3D{
        (uint32_t)inWidth * UP_DIV(depth, 4),
        (uint32_t)inHeight * batch,
        1
    }, VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT|exflags, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    gridImage = std::make_shared<VulkanImage>(m_phydev, m_dev->getComputeQueueFamilyIndex(), device, VkExtent3D{
        2 * UP_DIV((uint32_t)outHeight, 4),
        (uint32_t)outWidth * gridbatch,
        1
    }, VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT|exflags, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    paramBuffer = std::make_shared<VulkanBuffer>(m_phydev, m_dev->getComputeQueueFamilyIndex(), device, sizeof(GpuGridSampleParam),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
#ifdef VK_EXT_host_image_copy
    if (m_dev->is_support_host_image_copy()) {
        if (m_dev->checkHostImageCopyDstLayoutSupport(VK_IMAGE_LAYOUT_GENERAL)) {
            outputImage->hostImaggeTransition(VK_IMAGE_LAYOUT_GENERAL);
        } else {
            outputImage->hostImaggeTransition(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        }
        inputImage->hostImaggeTransition(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        gridImage->hostImaggeTransition(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    } else 
#endif
    {
        VulkanCommandBuffer cmd(device, m_cmdpool->getCommandPool());
        cmd.begin();
        outputImage->writeBarrier(cmd.get());
        inputImage->readBarrier(cmd.get());
        gridImage->readBarrier(cmd.get());
        cmd.end();
        cmd.submit(m_dev->getComputeQueue());
    }
}

void GridSample::submit(int outWidth, int outHeight)
{
    std::vector<VkDescriptorType> types = {
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
    };
    std::vector<std::shared_ptr<VulkanResource>> objs = {
        outputImage,
        inputImage,
        gridImage,
        paramBuffer
    };
    VkDevice device = m_dev->getLogicalDevice();
    VulkanPipeline pipeline(device, types, objs, reinterpret_cast<const uint32_t *>(grid_sample_spv), grid_sample_spv_len);

    VulkanCommandBuffer cmd2(device, m_cmdpool->getCommandPool());
    VulkanQueryPool queryPool(device, 2, VK_QUERY_TYPE_TIMESTAMP);
    cmd2.begin();
    cmd2.bind(pipeline);
    queryPool.begin(cmd2.get());
    cmd2.dispatch(outWidth, outHeight, 1);
    queryPool.end(cmd2.get());
    cmd2.end();
    cmd2.submit(m_dev->getComputeQueue());
    auto r = queryPool.getResults();
    double ts = double(r[1]-r[0])* double(1e-9) * m_dev->getTimestampPeriod();
    LOG_INFO("Time: %f s", ts);
}

}
}