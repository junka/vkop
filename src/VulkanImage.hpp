// Copyright 2025 @junka
#ifndef SRC_VULKANIMAGE_HPP_
#define SRC_VULKANIMAGE_HPP_

#include "vulkan/vulkan.hpp"

#include <cstdint>
#include <memory>
#include <variant>

#include "VulkanBuffer.hpp"
#include "VulkanResource.hpp"

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

    void stagingBufferCopyToImage(VkCommandBuffer commandBuffer, void *ptr);
    void stagingBufferCopyToHost(VkCommandBuffer commandBuffer);
    void readStaingBuffer(void *ptr);

    template <typename T>
    std::vector<T> convertNCHWToRGBA(const T *data, std::vector<int> nchw) {
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
        int realheight = height * batch;

        std::vector<T> tmp(realheight * realwidth * getImageChannelNum() *
                           getImageChannelSize());
        T *ptr = tmp.data();
        uint32_t row_pitch =
            realwidth * getImageChannelNum() * getImageChannelSize();
        T *dst = ptr;
        for (int b = 0; b < batch; b++) {
            T *batchstart = reinterpret_cast<T *>(
                reinterpret_cast<uint8_t *>(ptr) + b * height * row_pitch);
            for (int c = 0; c < realdepth; c++) {
                dst = batchstart + c * 4 * width;
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        int offset = b * stride_n + 4 * c * stride_c +
                                     h * stride_h + w * stride_w;

                        dst[w * 4 + 0] = data[offset];
                        dst[w * 4 + 1] = (4 * c + 1 < depth)
                                             ? data[stride_c + offset]
                                             : 0.0F;
                        dst[w * 4 + 2] = (4 * c + 2 < depth)
                                             ? data[2 * stride_c + offset]
                                             : 0.0F;
                        dst[w * 4 + 3] = (4 * c + 3 < depth)
                                             ? data[3 * stride_c + offset]
                                             : 0.0F;
                    }
                    dst = reinterpret_cast<T *>(
                        reinterpret_cast<uint8_t *>(dst) + row_pitch);
                }
            }
        }
        return tmp;
    }
    template <typename T>
    std::vector<T> convertRGBAToNCHW(T *ptr, std::vector<int> nchw) {
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

        std::vector<T> retdata(batch * height * depth * width);
        T *data = retdata.data();

        uint32_t row_pitch = realwidth * 4 * sizeof(T);
        T *dst;
        for (int b = 0; b < batch; b++) {
            T *batchstart = reinterpret_cast<T *>(
                reinterpret_cast<uint8_t *>(ptr) + b * height * row_pitch);
            for (int c = 0; c < realdepth; c++) {
                dst = batchstart + c * width * 4;
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        int offset = b * stride_n + 4 * c * stride_c +
                                     h * stride_h + w * stride_w;
                        data[offset] = dst[w * 4 + 0];
                        if (4 * c + 1 < depth) {
                            data[stride_c + offset] = dst[w * 4 + 1];
                        }
                        if (4 * c + 2 < depth) {
                            data[stride_c * 2 + offset] = dst[w * 4 + 2];
                        }
                        if (4 * c + 3 < depth) {
                            data[stride_c * 3 + offset] = dst[w * 4 + 3];
                        }
                    }
                    dst = reinterpret_cast<T *>(
                        reinterpret_cast<uint8_t *>(dst) + row_pitch);
                }
            }
        }
        return retdata;
    }

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

    int getImageChannelSize() const { return m_chansize_; }
    int getImageChannelNum() const { return m_chans_; }
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
