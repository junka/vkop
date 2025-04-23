#include "VulkanLib.hpp"
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
    std::cout << "VulkanImage::~VulkanImage()" << std::endl;
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
    imageCreateInfo.tiling = VK_IMAGE_TILING_LINEAR;
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

    vkCopyMemoryToImage(m_device, &copyinfo);
#else

#endif
}

}