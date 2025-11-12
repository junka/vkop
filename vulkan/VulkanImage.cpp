// Copyright 2025 @junka
#include "vulkan/VulkanImage.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "vulkan/VulkanLib.hpp"
#include <cstddef>

namespace vkop {

VulkanImage::VulkanImage(std::shared_ptr<VulkanDevice> &vdev, VkExtent3D dim,
                         VkImageUsageFlags usage,
                         VkMemoryPropertyFlags requireProperties,
                         VkFormat format, int ext_fd)
    : VulkanResource(vdev), m_dim_(dim), m_format_(format), m_usage_(usage) {
    if (m_format_ == VK_FORMAT_UNDEFINED) {
        throw std::runtime_error("Invalid Vulkan image format.");
    }
    if (m_usage_ == 0) {
        throw std::runtime_error("Invalid Vulkan image usage.");
    }
    calcImageSize();
    createImage();
#ifndef USE_VMA
#ifdef VK_KHR_get_memory_requirements2
    VkMemoryRequirements2 mem_requirements2 = {};
    mem_requirements2.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
    VkImageMemoryRequirementsInfo2 image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2;
    image_info.image = m_image_;
    vkGetImageMemoryRequirements2(m_vdev_->getLogicalDevice(), &image_info,
                                  &mem_requirements2);
    VkMemoryRequirements memory_requirements =
        mem_requirements2.memoryRequirements;
#else
    VkMemoryRequirements memoryRequirements;
    vkGetImageMemoryRequirements(m_device, m_image, &memoryRequirements);
#endif
    if (!allocMemory(memory_requirements, requireProperties, ext_fd)) {
        destroyImage();
        throw std::runtime_error("failed to allocate image memory!");
    }
    if (vkBindImageMemory(m_vdev_->getLogicalDevice(), m_image_, getMemory(),
                          0)) {
        destroyImage();
        throw std::runtime_error("failed to bind image memory!");
    }
#else
    (void)ext_fd;
    (void)requireProperties;
#endif
    createImageView();
    createSampler();
    if (usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) {
        createStagingBuffer(false);
    } else {
        createStagingBuffer(true);
    }
}

void VulkanImage::calcImageSize() {
    switch (m_format_) {
    case VK_FORMAT_R8_UNORM:
    case VK_FORMAT_R8_SNORM:
    case VK_FORMAT_R8_USCALED:
    case VK_FORMAT_R8_SSCALED:
    case VK_FORMAT_R8_UINT:
    case VK_FORMAT_R8_SINT:
    case VK_FORMAT_R8_SRGB:
    case VK_FORMAT_S8_UINT:
        m_chansize_ = 1;
        m_chans_ = 1;
        break;
    case VK_FORMAT_R8G8_UNORM:
    case VK_FORMAT_R8G8_SNORM:
    case VK_FORMAT_R8G8_USCALED:
    case VK_FORMAT_R8G8_SSCALED:
    case VK_FORMAT_R8G8_UINT:
    case VK_FORMAT_R8G8_SINT:
    case VK_FORMAT_R8G8_SRGB:
        m_chansize_ = 1;
        m_chans_ = 2;
        break;
    case VK_FORMAT_R16_UNORM:
    case VK_FORMAT_R16_SNORM:
    case VK_FORMAT_R16_USCALED:
    case VK_FORMAT_R16_UINT:
    case VK_FORMAT_R16_SINT:
    case VK_FORMAT_R16_SFLOAT:
    case VK_FORMAT_D16_UNORM:
        m_chansize_ = 2;
        m_chans_ = 1;
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
        m_chansize_ = 1;
        m_chans_ = 3;
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
        m_chansize_ = 1;
        m_chans_ = 4;
        break;
    case VK_FORMAT_R16G16_UNORM:
    case VK_FORMAT_R16G16_SNORM:
    case VK_FORMAT_R16G16_USCALED:
    case VK_FORMAT_R16G16_SSCALED:
    case VK_FORMAT_R16G16_UINT:
    case VK_FORMAT_R16G16_SINT:
    case VK_FORMAT_R16G16_SFLOAT:
        m_chansize_ = 2;
        m_chans_ = 2;
        break;
    case VK_FORMAT_R32_UINT:
    case VK_FORMAT_R32_SINT:
    case VK_FORMAT_R32_SFLOAT:
        m_chansize_ = 4;
        m_chans_ = 1;
        break;
    case VK_FORMAT_R16G16B16_UNORM:
    case VK_FORMAT_R16G16B16_SNORM:
    case VK_FORMAT_R16G16B16_USCALED:
    case VK_FORMAT_R16G16B16_SSCALED:
    case VK_FORMAT_R16G16B16_UINT:
    case VK_FORMAT_R16G16B16_SINT:
    case VK_FORMAT_R16G16B16_SFLOAT:
        m_chansize_ = 2;
        m_chans_ = 3;
        break;
    case VK_FORMAT_R16G16B16A16_UNORM:
    case VK_FORMAT_R16G16B16A16_SNORM:
    case VK_FORMAT_R16G16B16A16_USCALED:
    case VK_FORMAT_R16G16B16A16_SSCALED:
    case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_R16G16B16A16_SINT:
    case VK_FORMAT_R16G16B16A16_SFLOAT:
        m_chansize_ = 2;
        m_chans_ = 4;
        break;
    case VK_FORMAT_R32G32_UINT:
    case VK_FORMAT_R32G32_SINT:
    case VK_FORMAT_R32G32_SFLOAT:
        m_chansize_ = 4;
        m_chans_ = 2;
        break;
    case VK_FORMAT_R64_UINT:
    case VK_FORMAT_R64_SINT:
    case VK_FORMAT_R64_SFLOAT:
        m_chansize_ = 8;
        m_chans_ = 1;
        break;
    case VK_FORMAT_R32G32B32_UINT:
    case VK_FORMAT_R32G32B32_SINT:
    case VK_FORMAT_R32G32B32_SFLOAT:
        m_chansize_ = 4;
        m_chans_ = 3;
        break;
    case VK_FORMAT_R32G32B32A32_UINT:
    case VK_FORMAT_R32G32B32A32_SINT:
    case VK_FORMAT_R32G32B32A32_SFLOAT:
        m_chansize_ = 4;
        m_chans_ = 4;
        break;
    case VK_FORMAT_R64G64_UINT:
    case VK_FORMAT_R64G64_SINT:
    case VK_FORMAT_R64G64_SFLOAT:
        m_chansize_ = 8;
        m_chans_ = 2;
        break;
    case VK_FORMAT_R64G64B64_UINT:
    case VK_FORMAT_R64G64B64_SINT:
    case VK_FORMAT_R64G64B64_SFLOAT:
        m_chansize_ = 8;
        m_chans_ = 3;
        break;
    case VK_FORMAT_R64G64B64A64_UINT:
    case VK_FORMAT_R64G64B64A64_SINT:
    case VK_FORMAT_R64G64B64A64_SFLOAT:
        m_chansize_ = 8;
        m_chans_ = 4;
        break;
    default:
        break;
    }
}

VulkanImage::~VulkanImage() { destroyImage(); }

void VulkanImage::createStagingBuffer(bool writeonly) {
    auto size = getImageSize();
    if (writeonly) {
        m_stagingBuffer_ = std::make_unique<VulkanBuffer>(
            m_vdev_, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    } else {
        m_stagingBuffer_ = std::make_unique<VulkanBuffer>(
            m_vdev_, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    }
}
void VulkanImage::createImage() {
    m_imagetype_ = VK_IMAGE_TYPE_3D;
    if (m_dim_.depth == 1) {
        m_imagetype_ = VK_IMAGE_TYPE_2D;
        if (m_dim_.height == 1) {
            m_imagetype_ = VK_IMAGE_TYPE_1D;
        }
    }

    VkImageCreateInfo image_create_info = {};
    image_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_create_info.imageType = m_imagetype_;
    image_create_info.format = m_format_;
    image_create_info.extent = m_dim_;
    image_create_info.mipLevels = 1;
    image_create_info.arrayLayers = 1;
    image_create_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_create_info.usage = m_usage_;
    image_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_create_info.flags = VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
    image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_create_info.queueFamilyIndexCount = 1;
    uint32_t qidx = m_vdev_->getComputeQueueFamilyIndex();
    image_create_info.pQueueFamilyIndices = &qidx;
#ifdef USE_VMA
    auto ret =
        m_vdev_->getVMA()->createImage(&image_create_info, &m_vma_image_);
#else
    auto ret = vkCreateImage(m_vdev_->getLogicalDevice(), &image_create_info,
                             nullptr, &m_image_);
#endif
    if (ret != VK_SUCCESS) {
        throw std::runtime_error("failed to create image!");
    }
}

void VulkanImage::destroyImage() {
    if (m_sampler_)
        vkDestroySampler(m_vdev_->getLogicalDevice(), m_sampler_, nullptr);
    if (m_imageView_ != VK_NULL_HANDLE)
        vkDestroyImageView(m_vdev_->getLogicalDevice(), m_imageView_, nullptr);
#ifndef USE_VMA
    if (m_image_)
        vkDestroyImage(m_vdev_->getLogicalDevice(), m_image_, nullptr);
#else
    m_vdev_->getVMA()->destroyImage(&m_vma_image_);
#endif
}

void VulkanImage::createImageView() {
    VkImageViewCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
#ifdef USE_VMA
    info.image = m_vma_image_.image;
#else
    info.image = m_image_;
#endif
    info.viewType = VK_IMAGE_VIEW_TYPE_1D;
    if (m_imagetype_ == VK_IMAGE_TYPE_3D) {
        info.viewType = VK_IMAGE_VIEW_TYPE_3D;
    } else if (m_imagetype_ == VK_IMAGE_TYPE_2D) {
        info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    }
    info.format = m_format_;
    info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    info.subresourceRange.baseMipLevel = 0;
    info.subresourceRange.levelCount = 1;
    info.subresourceRange.baseArrayLayer = 0;
    info.subresourceRange.layerCount = 1;

    if (vkCreateImageView(m_vdev_->getLogicalDevice(), &info, nullptr,
                          &m_imageView_) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image view!");
    }
}

void VulkanImage::createSampler() {
    VkFilter filter = VK_FILTER_NEAREST;
    VkSamplerAddressMode mode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VkSamplerCreateInfo sampler_info = {};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = filter;
    sampler_info.minFilter = filter;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler_info.addressModeU = mode;
    sampler_info.addressModeV = mode;
    sampler_info.addressModeW = mode;
    sampler_info.mipLodBias = 0.0F;
    sampler_info.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    sampler_info.anisotropyEnable = VK_FALSE;
    sampler_info.maxAnisotropy = 1.0F;
    sampler_info.compareEnable = VK_FALSE;
    sampler_info.minLod = 0.0F;
    sampler_info.maxLod = 0.0F;
    if (vkCreateSampler(m_vdev_->getLogicalDevice(), &sampler_info, nullptr,
                        &m_sampler_) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture sampler!");
    }
}

std::variant<VkDescriptorImageInfo, VkDescriptorBufferInfo>
VulkanImage::getDescriptorInfo() const {
    VkDescriptorImageInfo image_info{};
    image_info.imageLayout = m_layout_;
    image_info.imageView = m_imageView_;
    image_info.sampler = m_sampler_;
    return image_info;
}

void VulkanImage::transitionImageLayout(VkCommandBuffer commandBuffer,
                                        VkImageLayout newLayout,
                                        VkAccessFlags dstAccessMask,
                                        VkPipelineStageFlags sourceStage,
                                        VkPipelineStageFlags destinationStage) {
    if (m_layout_ == newLayout) {
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
    barrier.oldLayout = m_layout_;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
#ifdef USE_VMA
    barrier.image = m_vma_image_.image;
#else
    barrier.image = m_image_;
#endif
    barrier.subresourceRange = subrange;
    barrier.srcAccessMask = m_access_;
    barrier.dstAccessMask = dstAccessMask;

    vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0,
                         nullptr, 0, nullptr, 1, &barrier);

    m_layout_ = newLayout;
    m_access_ = dstAccessMask;
}

void VulkanImage::transferBarrier(VkCommandBuffer commandBuffer,
                                  VkImageLayout newLayout,
                                  VkAccessFlags dstAccessMask) {
    VkPipelineStageFlags source_stage =
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkPipelineStageFlags destination_stage =
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    transitionImageLayout(commandBuffer, newLayout, dstAccessMask, source_stage,
                          destination_stage);
}
void VulkanImage::transferReadBarrier(VkCommandBuffer commandBuffer) {
    transitionImageLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          VK_ACCESS_TRANSFER_READ_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT);
}
void VulkanImage::transferWriteBarrier(VkCommandBuffer commandBuffer) {
    transitionImageLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT);
}
void VulkanImage::readBarrier(VkCommandBuffer commandBuffer) {
    transitionImageLayout(
        commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
}

void VulkanImage::writeBarrier(VkCommandBuffer commandBuffer) {
    transitionImageLayout(commandBuffer, VK_IMAGE_LAYOUT_GENERAL,
                          VK_ACCESS_SHADER_WRITE_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
}

void VulkanImage::copyImageToBuffer(VkCommandBuffer commandBuffer,
                                    VulkanBuffer &buffer) {
    VkImageLayout old_layout = m_layout_;
    VkAccessFlags old_access = m_access_;
    // Ensure the image is in a readable layout
    if (m_layout_ != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
        transferReadBarrier(commandBuffer);
    }
#ifdef USE_VMA
    VkImage img = m_vma_image_.image;
#else
    VkImage img = m_image_;
#endif

    VkBufferImageCopy region = {};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;   // Tightly packed
    region.bufferImageHeight = 0; // Tightly packed
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = m_dim_;

    assert(m_layout_ == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vkCmdCopyImageToBuffer(commandBuffer, img, m_layout_, buffer.getBuffer(), 1,
                           &region);

    // Optionally transition the image back to its original layout
    transferBarrier(commandBuffer, old_layout, old_access);
}

void VulkanImage::copyBufferToImage(VkCommandBuffer commandBuffer,
                                    VulkanBuffer &buffer) {
    VkImageLayout old_layout = m_layout_;
    VkAccessFlags old_access = m_access_;
    // Ensure the image is in a readable layout
    if (m_layout_ != VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        transferWriteBarrier(commandBuffer);
    }
#ifdef USE_VMA
    VkImage img = m_vma_image_.image;
#else
    VkImage img = m_image_;
#endif
    VkBufferImageCopy region = {};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = m_dim_;
    assert(m_layout_ == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vkCmdCopyBufferToImage(commandBuffer, buffer.getBuffer(), img, m_layout_, 1,
                           &region);

    // Optionally transition the image back to its original layout
    transferBarrier(commandBuffer, old_layout, old_access);
}

void VulkanImage::stagingBufferCopyToImage(VkCommandBuffer commandBuffer,
                                           const void *ptr) {
    auto *dst = m_stagingBuffer_->getMappedMemory();
    auto imagesize = getImageSize();
    const auto *src = static_cast<const float *>(ptr);
    assert(dst != nullptr);
    std::memcpy(dst, src, imagesize);
    m_stagingBuffer_->unmapMemory();

    copyBufferToImage(commandBuffer, *m_stagingBuffer_);
}

void VulkanImage::stagingBufferCopyToHost(VkCommandBuffer commandBuffer) {
    copyImageToBuffer(commandBuffer, *m_stagingBuffer_);
}

void VulkanImage::readStaingBuffer(void *ptr) {
    auto imagesize = getImageSize();
    void *src = m_stagingBuffer_->getMappedMemory();
    auto *dst = static_cast<float *>(ptr);
    memcpy(dst, src, imagesize);
    m_stagingBuffer_->unmapMemory();
}

void VulkanImage::hostImaggeTransition(VkImageLayout newLayout) {
#ifdef VK_EXT_host_image_copy
    VkResult ret;
    VkImageSubresourceRange subrange = {};
    subrange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subrange.baseMipLevel = 0;
    subrange.baseArrayLayer = 0;
    subrange.levelCount = 1;
    subrange.layerCount = 1;
#ifdef USE_VMA
    VkImage img = m_vma_image_.image;
#else
    VkImage img = m_image_;
#endif
    VkHostImageLayoutTransitionInfo transinfo = {};
    transinfo.sType = VK_STRUCTURE_TYPE_HOST_IMAGE_LAYOUT_TRANSITION_INFO;
    transinfo.oldLayout = m_layout_;
    transinfo.newLayout = newLayout;
    transinfo.image = img;
    transinfo.subresourceRange = subrange;
    auto vkTransitionImageLayoutEXT =
        reinterpret_cast<PFN_vkTransitionImageLayoutEXT>(vkGetInstanceProcAddr(
            VulkanInstance::getVulkanInstance().getInstance(),
            "vkTransitionImageLayoutEXT"));
    if (vkTransitionImageLayoutEXT) {
        ret = vkTransitionImageLayoutEXT(m_vdev_->getLogicalDevice(), 1,
                                         &transinfo);
        assert(ret == VK_SUCCESS);
        m_layout_ = newLayout;
    }
#else
    (void)newLayout;
#endif
}

void VulkanImage::hostImageCopyToDevice(void *ptr) {
#ifdef VK_EXT_host_image_copy
    VkResult ret;
    VkMemoryToImageCopy region = {};
    region.sType = VK_STRUCTURE_TYPE_MEMORY_TO_IMAGE_COPY_EXT;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = m_dim_;
    region.pHostPointer = ptr;
#ifdef USE_VMA
    VkImage img = m_vma_image_.image;
#else
    VkImage img = m_image_;
#endif
    VkCopyMemoryToImageInfo copyinfo = {};
    copyinfo.sType = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_IMAGE_INFO_EXT;
    copyinfo.dstImage = img;
    copyinfo.dstImageLayout = m_layout_;
    copyinfo.regionCount = 1;
    copyinfo.pRegions = &region;
    auto vkCopyMemoryToImageEXT = reinterpret_cast<PFN_vkCopyMemoryToImageEXT>(
        vkGetInstanceProcAddr(VulkanInstance::getVulkanInstance().getInstance(),
                              "vkCopyMemoryToImageEXT"));
    if (vkCopyMemoryToImageEXT) {
        ret = vkCopyMemoryToImageEXT(m_vdev_->getLogicalDevice(), &copyinfo);
        assert(ret == VK_SUCCESS);
    }
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
    region.imageExtent = m_dim_;
    region.pHostPointer = ptr;
    region.memoryRowLength = 0;
    region.memoryImageHeight = 0;
#ifdef USE_VMA
    VkImage img = m_vma_image_.image;
#else
    VkImage img = m_image_;
#endif
    VkCopyImageToMemoryInfo copyinfo = {};
    copyinfo.sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_MEMORY_INFO_EXT;
    copyinfo.srcImage = img;
    copyinfo.srcImageLayout = m_layout_;
    copyinfo.regionCount = 1;
    copyinfo.pRegions = &region;
    auto vkCopyImageToMemoryEXT = reinterpret_cast<PFN_vkCopyImageToMemoryEXT>(
        vkGetInstanceProcAddr(VulkanInstance::getVulkanInstance().getInstance(),
                              "vkCopyImageToMemoryEXT"));
    if (vkCopyImageToMemoryEXT)
        vkCopyImageToMemoryEXT(m_vdev_->getLogicalDevice(), &copyinfo);
#else
    (void)(ptr);
#endif
}

} // namespace vkop
