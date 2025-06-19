// Copyright 2025 @junka
#ifndef SRC_VULKANIMAGE_HPP_
#define SRC_VULKANIMAGE_HPP_

#include "vulkan/vulkan.hpp"

#include <cstdint>
#include <memory>
#include <variant>

#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanResource.hpp"

#define UP_DIV(x, y) (((x) + (y)-1) / (y))

namespace vkop {

// VulkanImage class inheriting from VulkanResource
class VulkanImage : public VulkanResource {
  public:
    VulkanImage(VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex,
                VkDevice device, VkExtent3D dim, VkFormat format,
                VkImageUsageFlags usage,
                VkMemoryPropertyFlags requireProperties);
    ~VulkanImage() override;

    ResourceType getResourceType() const override {
        return ResourceType::VK_IMAGE;
    }

    std::variant<VkDescriptorImageInfo, VkDescriptorBufferInfo>
    getDescriptorInfo() const override;

    void transitionImageLayout(VkCommandBuffer commandBuffer,
                               VkImageLayout newLayout,
                               VkAccessFlags dstAccessMask,
                               VkPipelineStageFlags sourceStage,
                               VkPipelineStageFlags destinationStage);

    void transferBarrier(VkCommandBuffer commandBuffer, VkImageLayout newLayout,
                         VkAccessFlags dstAccessMask);
    void transferReadBarrier(VkCommandBuffer commandBuffer);
    void transferWriteBarrier(VkCommandBuffer commandBuffer);

    void readBarrier(VkCommandBuffer commandBuffer);

    void writeBarrier(VkCommandBuffer commandBuffer);

    void copyImageToBuffer(VkCommandBuffer commandBuffer, VulkanBuffer &buffer);
    void copyBufferToImage(VkCommandBuffer commandBuffer, VulkanBuffer &buffer);

    /*
     * For host image copy image layout transition
     * This is done on host
     */
    static void hostImaggeTransition(VkImageLayout newLayout) {
#ifdef VK_EXT_host_image_copy
        VkResult ret;
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
        auto vkTransitionImageLayoutEXT =
            reinterpret_cast<PFN_vkTransitionImageLayoutEXT>(
                vkGetInstanceProcAddr(
                    VulkanInstance::getVulkanInstance().getInstance(),
                    "vkTransitionImageLayoutEXT"));
        if (vkTransitionImageLayoutEXT) {
            ret = vkTransitionImageLayoutEXT(m_device, 1, &transinfo);
            assert(ret == VK_SUCCESS);
            m_layout = newLayout;
        }
#else
        (void)newLayout;
#endif
    }

    static void hostImageCopyToDevice(void *ptr) {
#ifdef VK_EXT_host_image_copy
        VkResult ret;
        VkMemoryToImageCopy region = {};
        region.sType = VK_STRUCTURE_TYPE_MEMORY_TO_IMAGE_COPY_EXT;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = m_dim;
        region.pHostPointer = ptr;

        VkCopyMemoryToImage_info copyinfo = {};
        copyinfo.sType = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_IMAGE_INFO_EXT;
        copyinfo.dstImage = m_image;
        copyinfo.dstImageLayout = m_layout;
        copyinfo.regionCount = 1;
        copyinfo.pRegions = &region;
        auto vkCopyMemoryToImageEXT =
            reinterpret_cast<PFN_vkCopyMemoryToImageEXT>(vkGetInstanceProcAddr(
                VulkanInstance::getVulkanInstance().getInstance(),
                "vkCopyMemoryToImageEXT"));
        if (vkCopyMemoryToImageEXT) {
            ret = vkCopyMemoryToImageEXT(m_device, &copyinfo);
            assert(ret == VK_SUCCESS);
        }
#else
        (void)(ptr);
#endif
    }

    static void hostImageCopyToHost(void *ptr) {
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
        auto vkCopyImageToMemoryEXT =
            reinterpret_cast<PFN_vkCopyImageToMemoryEXT>(vkGetInstanceProcAddr(
                VulkanInstance::getVulkanInstance().getInstance(),
                "vkCopyImageToMemoryEXT"));
        if (vkCopyImageToMemoryEXT)
            vkCopyImageToMemoryEXT(m_device, &copyinfo);
#else
        (void)(ptr);
#endif
    }

    void stagingBufferCopyToImage(VkCommandBuffer commandBuffer,
                                  const void *ptr);
    void stagingBufferCopyToHost(VkCommandBuffer commandBuffer);
    void readStaingBuffer(void *ptr);

    int getImageChannelSize() const { return m_chansize_; }
    int getImageChannelNum() const { return m_chans_; }

  private:
    VkExtent3D m_dim_;
    VkFormat m_format_;
    VkImageUsageFlags m_usage_;
    VkImageType m_imagetype_;
    VkImageLayout m_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkAccessFlags m_access_ = 0;

    VkImage m_image_;
    VkImageView m_imageView_;
    VkSampler m_sampler_;

    int m_chansize_;
    int m_chans_;

    std::unique_ptr<VulkanBuffer> m_stagingBuffer_;

    void calcImageSize();
    int getImageSize() const {
        return m_chansize_ * m_chans_ * m_dim_.width * m_dim_.height *
               m_dim_.depth;
    }

    int getImageWidth() const { return m_dim_.width; }
    int getImageHeight() const { return m_dim_.height; }
    int getImageDepth() const { return m_dim_.depth; }

    void createImage();

    void createImageView();

    void createSampler();

    void destroyImage();

    void createStagingBuffer(bool writeonly);
};
} // namespace vkop
#endif // SRC_VULKANIMAGE_HPP_
