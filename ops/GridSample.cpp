
#include <cstdint>
#include <iostream>
#include <memory>
#include <cmath>

#include "GridSample.hpp"

#include "grid_sample.h"

namespace vkop {

namespace ops {

void GridSample::prepare(const std::vector<int>& input_shape, const std::vector<int>& grid_shape)
{
    int inHeight = input_shape[2];
    int inWidth = input_shape[3];
    int outHeight = grid_shape[1];
    int outWidth = grid_shape[2];
    int depth = input_shape[1];

    VkDevice device = m_dev->getLogicalDevice();
    outputImage = std::make_shared<VulkanImage>(m_phydev, m_dev->getComputeQueueFamilyIndex(), device, VkExtent3D {
        (uint32_t)outWidth * UP_DIV(depth, 4),
        (uint32_t)outHeight,
        1
    }, VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_HOST_TRANSFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        VK_IMAGE_TYPE_2D);

    inputImage = std::make_shared<VulkanImage>(m_phydev, m_dev->getComputeQueueFamilyIndex(), device, VkExtent3D{
        (uint32_t)inWidth * UP_DIV(depth, 4),
        (uint32_t)inHeight,
        1
    }, VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_HOST_TRANSFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        VK_IMAGE_TYPE_2D);

    gridImage = std::make_shared<VulkanImage>(m_phydev, m_dev->getComputeQueueFamilyIndex(), device, VkExtent3D{
        2 * UP_DIV((uint32_t)outHeight, 4),
        (uint32_t)outWidth,
        1
    }, VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_HOST_TRANSFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        VK_IMAGE_TYPE_2D);

    paramBuffer = std::make_shared<VulkanBuffer>(m_phydev, m_dev->getComputeQueueFamilyIndex(), device, sizeof(GpuGridSampleParam),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (m_dev->checkHostImageCopyDstLayoutSupport(VK_IMAGE_LAYOUT_GENERAL)) {
        outputImage->hostImaggeTransition(VK_IMAGE_LAYOUT_GENERAL);
    } else {
        outputImage->hostImaggeTransition(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    }
    inputImage->hostImaggeTransition(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    gridImage->hostImaggeTransition(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

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
    std::cout << "Time: " << ts  << " s" << std::endl;
}

}
}