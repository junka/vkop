#include "VulkanLib.hpp"
#include "VulkanInstance.hpp"
#include "VulkanImage.hpp"

#include <iostream>

namespace vkop {

VulkanImage::VulkanImage(VkPhysicalDevice physicalDevice, const uint32_t queueFamilyIndex, VkDevice device, VkExtent3D dim,
        VkFormat format, VkImageUsageFlags usage, VkMemoryPropertyFlags requireProperties,
        VkImageType imagetype)
: VulkanResource(physicalDevice, queueFamilyIndex, device), m_dim(dim), m_format(format),
  m_usage(usage), m_imagetype(imagetype), m_layout(VK_IMAGE_LAYOUT_UNDEFINED), m_access(0)
{
    if (m_device == VK_NULL_HANDLE) {
        throw std::runtime_error("Invalid Vulkan device handle.");
    }
    if (m_format == VK_FORMAT_UNDEFINED) {
        throw std::runtime_error("Invalid Vulkan image format.");
    }
    if (m_usage == 0) {
        throw std::runtime_error("Invalid Vulkan image usage.");
    }
    createImage();
#ifdef VK_KHR_get_memory_requirements2
    VkMemoryRequirements2 memRequirements2 = {};
    memRequirements2.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
    VkImageMemoryRequirementsInfo2 imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2;
    imageInfo.image = m_image;
    vkGetImageMemoryRequirements2(m_device, &imageInfo, &memRequirements2);
    VkMemoryRequirements memoryRequirements = memRequirements2.memoryRequirements;
#else
    VkMemoryRequirements memoryRequirements;
    vkGetImageMemoryRequirements(m_device, m_image, &memoryRequirements);
#endif
    if (allocMemory(memoryRequirements, requireProperties)!=true) {
        destroyImage();
        throw std::runtime_error("failed to allocate image memory!");
    }
    if (vkBindImageMemory(m_device, m_image, getMemory(), 0)) {
        destroyImage();
        throw std::runtime_error("failed to bind image memory!");
    }

    createImageView();
    createSampler();

}

VulkanImage::~VulkanImage() {
    destroyImage();
}

void VulkanImage::createImage()
{
    VkImageCreateInfo imageCreateInfo = {};
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.imageType = m_imagetype;
    imageCreateInfo.format = m_format;
    imageCreateInfo.extent = m_dim;
    imageCreateInfo.mipLevels = 1;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.usage = m_usage;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.flags = VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageCreateInfo.queueFamilyIndexCount = 1;
    imageCreateInfo.pQueueFamilyIndices = &m_queueFamilyIndex;
    if (vkCreateImage(m_device, &imageCreateInfo, nullptr, &m_image) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image!");
    }
}

void VulkanImage::destroyImage()
{
    if (m_sampler)
        vkDestroySampler(m_device, m_sampler, nullptr);
    if (m_imageView != VK_NULL_HANDLE)
        vkDestroyImageView(m_device, m_imageView, nullptr);
    if (m_image)
        vkDestroyImage(m_device, m_image, nullptr);
}

void VulkanImage::createImageView()
{
    VkImageViewCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    info.image = m_image;
    info.viewType = VK_IMAGE_VIEW_TYPE_1D;
    if (m_imagetype == VK_IMAGE_TYPE_3D) {
        info.viewType = VK_IMAGE_VIEW_TYPE_3D;
    } else if (m_imagetype == VK_IMAGE_TYPE_2D) {
        info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    }
    info.format = m_format;
    info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    info.subresourceRange.baseMipLevel = 0;
    info.subresourceRange.levelCount = 1;
    info.subresourceRange.baseArrayLayer = 0;
    info.subresourceRange.layerCount = 1;

    if (vkCreateImageView(m_device, &info, nullptr, &m_imageView) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image view!");
    }
}

void VulkanImage::createSampler()
{
    VkFilter filter = VK_FILTER_NEAREST;
    VkSamplerAddressMode mode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VkSamplerCreateInfo samplerInfo = {};
    samplerInfo.sType            = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter        = filter;
    samplerInfo.minFilter        = filter;
    samplerInfo.mipmapMode       = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU     = mode;
    samplerInfo.addressModeV     = mode;
    samplerInfo.addressModeW     = mode;
    samplerInfo.mipLodBias       = 0.0f;
    samplerInfo.borderColor      = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy    = 1.0f;
    samplerInfo.compareEnable    = VK_FALSE;
    samplerInfo.minLod           = 0.0f;
    samplerInfo.maxLod           = 0.0f;
    if (vkCreateSampler(m_device, &samplerInfo, nullptr, &m_sampler)!= VK_SUCCESS) {
        throw std::runtime_error("failed to create texture sampler!");
    }
}

std::variant<VkDescriptorImageInfo, VkDescriptorBufferInfo> VulkanImage::getDescriptorInfo() const {
    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageLayout = m_layout;
    imageInfo.imageView = m_imageView;
    imageInfo.sampler = m_sampler;
    return imageInfo;
}

void VulkanImage::transitionImageLayout(VkCommandBuffer commandBuffer, VkImageLayout newLayout,
                VkAccessFlags dstAccessMask) {
    if (m_layout == newLayout) {
        return;
    }

    VkImageSubresourceRange subrange = {};
    subrange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subrange.baseMipLevel = 0;
    subrange.baseArrayLayer = 0;
    subrange.levelCount = 1;
    subrange.layerCount = 1;

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = m_layout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = m_image;
    barrier.subresourceRange = subrange;
    barrier.srcAccessMask = m_access;
    barrier.dstAccessMask = dstAccessMask;

    VkPipelineStageFlags sourceStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkPipelineStageFlags destinationStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    vkCmdPipelineBarrier(
        commandBuffer, sourceStage, destinationStage,
        0, 0, nullptr, 0, nullptr,
        1, &barrier
    );

    m_layout = newLayout;
    m_access = dstAccessMask;
}

void VulkanImage::copyImageToBuffer(VkCommandBuffer commandBuffer, VulkanBuffer& buffer)
{
    VkBuffer bufferHandle = buffer.getBuffer();
    VkImageLayout old_layout = m_layout;
    VkAccessFlags old_access = m_access;
    // Ensure the image is in a readable layout
    if (m_layout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
        transitionImageLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                              VK_ACCESS_TRANSFER_READ_BIT);
    }

    // Define the region to copy
    VkBufferImageCopy region = {};
    region.bufferOffset = 0;
    region.bufferRowLength = 0; // Tightly packed
    region.bufferImageHeight = 0; // Tightly packed
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = m_dim;

    // Perform the copy operation
    vkCmdCopyImageToBuffer(commandBuffer, m_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, bufferHandle, 1, &region);

    // Optionally transition the image back to its original layout
    transitionImageLayout(commandBuffer, old_layout, old_access);
}

uint32_t VulkanImage::getRowPitch() const
{
    VkSubresourceLayout layout;
    VkImageSubresource subresource = {};
    subresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresource.mipLevel = 0;
    subresource.arrayLayer = 0;

    vkGetImageSubresourceLayout(m_device, m_image, &subresource, &layout);
    const uint32_t rowPitch = layout.rowPitch;
    return rowPitch;
}

/*
* For host image copy image layout transition
* This is done on host
*/
void VulkanImage::hostImaggeTransition(VkImageLayout newLayout) {
#ifdef VK_EXT_host_image_copy

    VkImageSubresourceRange subrange = {};
    subrange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subrange.baseMipLevel = 0;
    subrange.baseArrayLayer = 0;
    subrange.levelCount = 1;
    subrange.layerCount = 1;

    VkHostImageLayoutTransitionInfo transinfo = {};
    transinfo.sType = VK_STRUCTURE_TYPE_HOST_IMAGE_LAYOUT_TRANSITION_INFO;
    transinfo.oldLayout = m_layout;
    transinfo.newLayout = newLayout;
    transinfo.image = m_image;
    transinfo.subresourceRange = subrange;
    auto vkTransitionImageLayoutEXT = reinterpret_cast<PFN_vkTransitionImageLayoutEXT>(
                vkGetInstanceProcAddr(VulkanInstance::getVulkanInstance().getInstance(), "vkTransitionImageLayoutEXT"));
    if (vkTransitionImageLayoutEXT) {
        vkTransitionImageLayoutEXT(m_device, 1, &transinfo);
        m_layout = newLayout;
    }
#else
    (void)newLayout;
#endif
}

void VulkanImage::hostImageCopyFrom(void *ptr) {
#ifdef VK_EXT_host_image_copy
    VkMemoryToImageCopy region = {};
    region.sType = VK_STRUCTURE_TYPE_MEMORY_TO_IMAGE_COPY_EXT;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = m_dim;
    region.pHostPointer = ptr;

    VkCopyMemoryToImageInfo copyinfo = {};
    copyinfo.sType = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_IMAGE_INFO_EXT;
    // copyinfo.flags = VK_HOST_IMAGE_COPY_MEMCPY;
    copyinfo.dstImage = m_image;
    copyinfo.dstImageLayout = m_layout;
    copyinfo.regionCount = 1;
    copyinfo.pRegions = &region;
    auto vkCopyMemoryToImageEXT = reinterpret_cast<PFN_vkCopyMemoryToImageEXT>(
        vkGetInstanceProcAddr(VulkanInstance::getVulkanInstance().getInstance(), "vkCopyMemoryToImageEXT"));
    if (vkCopyMemoryToImageEXT)
        vkCopyMemoryToImageEXT(m_device, &copyinfo);
#else
    (void)(ptr);
#endif
}



void VulkanImage::hostImageCopyTo(void *ptr) {
#ifdef VK_EXT_host_image_copy
    VkImageToMemoryCopy region = {};
    region.sType = VK_STRUCTURE_TYPE_IMAGE_TO_MEMORY_COPY_EXT;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = m_dim;
    region.pHostPointer = ptr;
    region.memoryRowLength = m_dim.width;
    region.memoryImageHeight = m_dim.height;

    VkCopyImageToMemoryInfo copyinfo = {};
    copyinfo.sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_MEMORY_INFO_EXT;
    // copyinfo.flags = VK_HOST_IMAGE_COPY_MEMCPY;
    copyinfo.srcImage = m_image;
    copyinfo.srcImageLayout = m_layout;
    copyinfo.regionCount = 1;
    copyinfo.pRegions = &region;
    auto vkCopyImageToMemoryEXT = reinterpret_cast<PFN_vkCopyImageToMemoryEXT>(
        vkGetInstanceProcAddr(VulkanInstance::getVulkanInstance().getInstance(), "vkCopyImageToMemoryEXT"));
    if (vkCopyImageToMemoryEXT)
        vkCopyImageToMemoryEXT(m_device, &copyinfo);
#else
    (void)(ptr);
#endif
}


#define UP_DIV(x, y) (((x) + (y) - 1) / (y))
void VulkanImage::convertNCHWToRGBA(const float *data, std::vector<int> nchw)
{
    int batch = nchw[0];//1
    int depth = nchw[1];//6
    int height = nchw[2];//17
    int width = nchw[3];//2

    int stride_w = 1;
    int stride_h = width;
    int stride_c = width * height;
    int stride_n = width * height * depth;
    int realdepth = UP_DIV(depth, 4);
    int realwidth = width * UP_DIV(depth, 4);

    // since format is VK_FORMAT_R32G32B32A32_SFLOAT
    float *ptr = (float *)malloc(batch * height * realdepth * realwidth * 4 * sizeof(float));

    uint32_t rowPitch = realwidth * 4 * sizeof(float);
    float *dst = reinterpret_cast<float *>(ptr);
    for (int b = 0; b < batch; b++) {
        float* batchstart = reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(ptr) + b * height * rowPitch);
        for (int c = 0; c < realdepth; c++) {
            dst = reinterpret_cast<float *>(batchstart) + c * 4 * width;
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int offset = b * stride_n + 4 * c * stride_c + h * stride_h + w * stride_w;

                    float r = data[offset];
                    float g = (4 * c + 1 < depth) ? data[stride_c + offset] : 0.0f;
                    float b = (4 * c + 2 < depth) ? data[2 * stride_c + offset] : 0.0f;
                    float a = (4 * c + 3 < depth) ? data[3 * stride_c + offset] : 0.0f;

                    // Write RGBA values to the Vulkan image memory
                    dst[w * 4 + 0] = r;
                    dst[w * 4 + 1] = g;
                    dst[w * 4 + 2] = b;
                    dst[w * 4 + 3] = a;
                }
                // Move to the next row in the Vulkan image memory
                dst = reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(dst) + rowPitch);
            }
        }
    }
    hostImageCopyFrom(ptr);
    free(ptr);
}


std::vector<float> VulkanImage::convertRGBAToNCHW(std::vector<int> nchw) {
    int batch = nchw[0];
    int depth = nchw[1];
    int height = nchw[2];
    int width = nchw[3];

    int stride_w = 1;
    int stride_h = width;
    int stride_c = width * height;
    int stride_n = width * height * depth;

    int realdepth = UP_DIV(depth, 4);
    int realwidth = width * UP_DIV(depth, 4);
    
    std::vector<float> retdata(batch * height * depth * width);
    std::vector<float> tmp(batch * height * realdepth * realwidth * 4);
    float *ptr = tmp.data();
    float *data = retdata.data();
    hostImageCopyTo(ptr);

    uint32_t rowPitch = realwidth * 4 * sizeof(float);
    // since format is VK_FORMAT_R32G32B32A32_SFLOAT
    float *dst = reinterpret_cast<float *>(ptr);
    for (int b = 0; b < batch; b++) {
        float* batchstart = reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(ptr) + b * height * rowPitch);
        for (int c = 0; c < realdepth; c++) {
            dst = reinterpret_cast<float *>(batchstart) + c * width * 4;
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int offset = b * stride_n + 4 * c * stride_c + h * stride_h + w * stride_w;
                    data[offset] = dst[w * 4 + 0];
                    data[stride_c + offset] = (4 * c + 1 < depth) ? dst[w * 4 + 1] : 0.0f;
                    data[stride_c * 2 + offset] = (4 * c + 2 < depth) ? dst[w * 4 + 1] : 0.0f;
                    data[stride_c * 3 + offset] = (4 * c + 3 < depth) ? dst[w * 4 + 1] : 0.0f;
                }
                dst = reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(dst) + rowPitch);
            }
        }
    }
    return retdata;
}



}