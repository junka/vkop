#include "VulkanLib.hpp"
#include "VulkanInstance.hpp"
#include "VulkanImage.hpp"

#include <bits/stdint-uintn.h>
#include <iostream>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace vkop {

VulkanImage::VulkanImage(VkPhysicalDevice physicalDevice, const uint32_t queueFamilyIndex, VkDevice device, VkExtent3D dim,
        VkFormat format, VkImageUsageFlags usage, VkMemoryPropertyFlags requireProperties)
: VulkanResource(physicalDevice, queueFamilyIndex, device), m_dim(dim), m_format(format),
  m_usage(usage), m_layout(VK_IMAGE_LAYOUT_UNDEFINED),
  m_access(0), m_rowPitch(0)
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
    calcImageSize();
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
    if (usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) {
        createStagingBuffer(false);
    } else {
        createStagingBuffer(true);
    }
}

void VulkanImage::calcImageSize() {
    switch(m_format) {
        case VK_FORMAT_R8_UNORM:
        case VK_FORMAT_R8_SNORM:
        case VK_FORMAT_R8_USCALED:
        case VK_FORMAT_R8_SSCALED:
        case VK_FORMAT_R8_UINT:
        case VK_FORMAT_R8_SINT:
        case VK_FORMAT_R8_SRGB:
        case VK_FORMAT_S8_UINT:
            m_chansize = 1;
            m_chans = 1;
            break;
        case VK_FORMAT_R8G8_UNORM:
        case VK_FORMAT_R8G8_SNORM:
        case VK_FORMAT_R8G8_USCALED:
        case VK_FORMAT_R8G8_SSCALED:
        case VK_FORMAT_R8G8_UINT:
        case VK_FORMAT_R8G8_SINT:
        case VK_FORMAT_R8G8_SRGB:
            m_chansize = 1;
            m_chans = 2;
            break;
        case VK_FORMAT_R16_UNORM:
        case VK_FORMAT_R16_SNORM:
        case VK_FORMAT_R16_USCALED:
        case VK_FORMAT_R16_UINT:
        case VK_FORMAT_R16_SINT:
        case VK_FORMAT_R16_SFLOAT:
        case VK_FORMAT_D16_UNORM:
            m_chansize = 2;
            m_chans = 1;
            break;
        case VK_FORMAT_R8G8B8_UNORM:
        case VK_FORMAT_R8G8B8_SNORM:
        case VK_FORMAT_R8G8B8_USCALED:
        case VK_FORMAT_R8G8B8_SSCALED:
        case VK_FORMAT_R8G8B8_UINT:
        case VK_FORMAT_R8G8B8_SINT:
        case VK_FORMAT_R8G8B8_SRGB:
        case VK_FORMAT_B8G8R8_UNORM:
        case VK_FORMAT_B8G8R8_SNORM:
        case VK_FORMAT_B8G8R8_USCALED:
        case VK_FORMAT_B8G8R8_SSCALED:
        case VK_FORMAT_B8G8R8_UINT:
        case VK_FORMAT_B8G8R8_SINT:
        case VK_FORMAT_B8G8R8_SRGB:
            m_chansize = 1;
            m_chans = 3;
            break;
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_SNORM:
        case VK_FORMAT_R8G8B8A8_USCALED:
        case VK_FORMAT_R8G8B8A8_SSCALED:
        case VK_FORMAT_R8G8B8A8_UINT:
        case VK_FORMAT_R8G8B8A8_SINT:
        case VK_FORMAT_R8G8B8A8_SRGB:
        case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_B8G8R8A8_SNORM:
        case VK_FORMAT_B8G8R8A8_USCALED:
        case VK_FORMAT_B8G8R8A8_SSCALED:
        case VK_FORMAT_B8G8R8A8_UINT:
        case VK_FORMAT_B8G8R8A8_SINT:
        case VK_FORMAT_B8G8R8A8_SRGB:
        case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
        case VK_FORMAT_A8B8G8R8_SNORM_PACK32:
        case VK_FORMAT_A8B8G8R8_USCALED_PACK32:
        case VK_FORMAT_A8B8G8R8_SSCALED_PACK32:
        case VK_FORMAT_A8B8G8R8_UINT_PACK32:
        case VK_FORMAT_A8B8G8R8_SINT_PACK32:
        case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
            m_chansize = 1;
            m_chans = 4;
            break;
        case VK_FORMAT_R16G16_UNORM:
        case VK_FORMAT_R16G16_SNORM:
        case VK_FORMAT_R16G16_USCALED:
        case VK_FORMAT_R16G16_SSCALED:
        case VK_FORMAT_R16G16_UINT:
        case VK_FORMAT_R16G16_SINT:
        case VK_FORMAT_R16G16_SFLOAT:
            m_chansize = 2;
            m_chans = 2;
            break;
        case VK_FORMAT_R32_UINT:
        case VK_FORMAT_R32_SINT:
        case VK_FORMAT_R32_SFLOAT:
            m_chansize = 4;
            m_chans = 1;
            break;
        case VK_FORMAT_R16G16B16_UNORM:
        case VK_FORMAT_R16G16B16_SNORM:
        case VK_FORMAT_R16G16B16_USCALED:
        case VK_FORMAT_R16G16B16_SSCALED:
        case VK_FORMAT_R16G16B16_UINT:
        case VK_FORMAT_R16G16B16_SINT:
        case VK_FORMAT_R16G16B16_SFLOAT:
            m_chansize = 2;
            m_chans = 3;
            break;
        case VK_FORMAT_R16G16B16A16_UNORM:
        case VK_FORMAT_R16G16B16A16_SNORM:
        case VK_FORMAT_R16G16B16A16_USCALED:
        case VK_FORMAT_R16G16B16A16_SSCALED:
        case VK_FORMAT_R16G16B16A16_UINT:
        case VK_FORMAT_R16G16B16A16_SINT:
        case VK_FORMAT_R16G16B16A16_SFLOAT:
            m_chansize = 2;
            m_chans = 4;
            break;
        case VK_FORMAT_R32G32_UINT:
        case VK_FORMAT_R32G32_SINT:
        case VK_FORMAT_R32G32_SFLOAT:
            m_chansize = 4;
            m_chans = 2;
            break;
        case VK_FORMAT_R64_UINT:
        case VK_FORMAT_R64_SINT:
        case VK_FORMAT_R64_SFLOAT:
            m_chansize = 8;
            m_chans = 1;
            break;
        case VK_FORMAT_R32G32B32_UINT:
        case VK_FORMAT_R32G32B32_SINT:
        case VK_FORMAT_R32G32B32_SFLOAT:
            m_chansize = 4;
            m_chans = 3;
            break;
        case VK_FORMAT_R32G32B32A32_UINT:
        case VK_FORMAT_R32G32B32A32_SINT:
        case VK_FORMAT_R32G32B32A32_SFLOAT:
            m_chansize = 4;
            m_chans = 4;
            break;
        case VK_FORMAT_R64G64_UINT:
        case VK_FORMAT_R64G64_SINT:
        case VK_FORMAT_R64G64_SFLOAT:
            m_chansize = 8;
            m_chans = 2;
            break;
        case VK_FORMAT_R64G64B64_UINT:
        case VK_FORMAT_R64G64B64_SINT:
        case VK_FORMAT_R64G64B64_SFLOAT:
            m_chansize = 8;
            m_chans = 3;
            break;
        case VK_FORMAT_R64G64B64A64_UINT:
        case VK_FORMAT_R64G64B64A64_SINT:
        case VK_FORMAT_R64G64B64A64_SFLOAT:
            m_chansize = 8;
            m_chans = 4;
            break;
        default:
            break;
    }

}

VulkanImage::~VulkanImage() {
    destroyImage();
}

void VulkanImage::createStagingBuffer(bool writeonly)
{
    auto size = getImageSize();
    if (writeonly) {
        m_stagingBuffer = std::make_unique<VulkanBuffer>(m_physicalDevice, m_queueFamilyIndex, m_device, size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    } else {
        m_stagingBuffer = std::make_unique<VulkanBuffer>(m_physicalDevice, m_queueFamilyIndex, m_device, size,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    }
}
void VulkanImage::createImage()
{
    m_imagetype = VK_IMAGE_TYPE_3D;
    if (m_dim.depth == 1) {
        m_imagetype = VK_IMAGE_TYPE_2D;
        if (m_dim.height == 1) {
            m_imagetype = VK_IMAGE_TYPE_1D;
        }
    }

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
                VkAccessFlags dstAccessMask, VkPipelineStageFlags sourceStage, VkPipelineStageFlags destinationStage) {
    std::cout << "transitionImageLayout " << newLayout << std::endl;
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

    vkCmdPipelineBarrier(
        commandBuffer, sourceStage, destinationStage,
        0, 0, nullptr, 0, nullptr,
        1, &barrier
    );

    m_layout = newLayout;
    m_access = dstAccessMask;
}

void VulkanImage::transferBarrier(VkCommandBuffer commandBuffer, VkImageLayout newLayout, VkAccessFlags dstAccessMask)
{    
    VkPipelineStageFlags sourceStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkPipelineStageFlags destinationStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT;
    transitionImageLayout(commandBuffer, newLayout, dstAccessMask, sourceStage, destinationStage);
}

void VulkanImage::readBarrier(VkCommandBuffer commandBuffer)
{
    transitionImageLayout(commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
}

void VulkanImage::writeBarrier(VkCommandBuffer commandBuffer)
{
    transitionImageLayout(commandBuffer, VK_IMAGE_LAYOUT_GENERAL,
        VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
}


void VulkanImage::copyImageToBuffer(VkCommandBuffer commandBuffer, VulkanBuffer& buffer)
{
    VkBuffer bufferHandle = buffer.getBuffer();
    VkImageLayout old_layout = m_layout;
    VkAccessFlags old_access = m_access;
    // Ensure the image is in a readable layout
    if (m_layout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
        std::cout << "image to VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL" << std::endl;
        transferBarrier(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
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

    vkCmdCopyImageToBuffer(commandBuffer, m_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, bufferHandle, 1, &region);

    // Optionally transition the image back to its original layout
    transferBarrier(commandBuffer, old_layout, old_access);
    std::cout << "image to " << old_layout << std::endl;
}


void VulkanImage::copyBufferToImage(VkCommandBuffer commandBuffer, VulkanBuffer& buffer)
{
    VkBuffer srcbuff = buffer.getBuffer();
    VkImageLayout old_layout = m_layout;
    VkAccessFlags old_access = m_access;
    // Ensure the image is in a readable layout
    if (m_layout != VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        std::cout << "image to VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL" << std::endl;
        transferBarrier(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              VK_ACCESS_TRANSFER_WRITE_BIT);
    }

    VkBufferImageCopy region = {};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = m_dim;

    vkCmdCopyBufferToImage(commandBuffer, srcbuff, m_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // Optionally transition the image back to its original layout
    transferBarrier(commandBuffer, old_layout, old_access);
    std::cout << "image to " << old_layout << std::endl;
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

void VulkanImage::hostImageCopyToDevice(void *ptr) {
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

void VulkanImage::hostImageCopyToHost(void *ptr) {
#ifdef VK_EXT_host_image_copy
    VkImageToMemoryCopy region = {};
    region.sType = VK_STRUCTURE_TYPE_IMAGE_TO_MEMORY_COPY_EXT;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = m_dim;
    region.pHostPointer = ptr;
    region.memoryRowLength = 0;
    region.memoryImageHeight = 0;

    VkCopyImageToMemoryInfo copyinfo = {};
    copyinfo.sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_MEMORY_INFO_EXT;
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

void VulkanImage::stagingBufferCopyToImage(VkCommandBuffer commandBuffer, void *ptr)
{
    float *dst = static_cast<float *>(m_stagingBuffer->getMappedMemory());
    auto imagesize = getImageSize();
    float *src = static_cast<float*>(ptr);
    memcpy(dst, src, imagesize);
    m_stagingBuffer->unmapMemory();

    std::cout << "copy size " << imagesize << std::endl;
    for (int i = 0; i < getImageHeight(); i++) {
        for (int j = 0; j < getImageWidth(); j++) {
            std::cout << dst[getImageWidth() * i * 4 + j * 4] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    for (int i = 0; i < getImageHeight(); i++) {
        for (int j = 0; j < getImageWidth(); j++) {
            std::cout << dst[getImageWidth() * i * 4 + j * 4+1] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    for (int i = 0; i < getImageHeight(); i++) {
        for (int j = 0; j < getImageWidth(); j++) {
            std::cout << dst[getImageWidth() * i * 4 + j * 4+2] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    for (int i = 0; i < getImageHeight(); i++) {
        for (int j = 0; j < getImageWidth(); j++) {
            std::cout << dst[getImageWidth() * i * 4 + j * 4+3] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    copyBufferToImage(commandBuffer, *m_stagingBuffer);
}


void VulkanImage::stagingBufferCopyToHost(VkCommandBuffer commandBuffer, void *ptr)
{
    auto imagesize = getImageSize();
    copyImageToBuffer(commandBuffer, *m_stagingBuffer);
    
    void *m = m_stagingBuffer->getMappedMemory();

    float *dst = static_cast<float*>(ptr);
    float *src = static_cast<float *>(m);
    memcpy(dst, src, imagesize);
        
    m_stagingBuffer->unmapMemory();
    std::cout << "copy size " << imagesize << std::endl;
    for (int i = 0; i < getImageHeight(); i++) {
        for (int j = 0; j < getImageWidth(); j++) {
            std::cout << dst[getImageWidth() * i * 4 + j * 4] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    for (int i = 0; i < getImageHeight(); i++) {
        for (int j = 0; j < getImageWidth(); j++) {
            std::cout << dst[getImageWidth() * i * 4 + j * 4+1] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    for (int i = 0; i < getImageHeight(); i++) {
        for (int j = 0; j < getImageWidth(); j++) {
            std::cout << dst[getImageWidth() * i * 4 + j * 4+2] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    for (int i = 0; i < getImageHeight(); i++) {
        for (int j = 0; j < getImageWidth(); j++) {
            std::cout << dst[getImageWidth() * i * 4 + j * 4+3] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

}



}