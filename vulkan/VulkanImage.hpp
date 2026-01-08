// Copyright 2025 @junka
#ifndef SRC_VULKANIMAGE_HPP_
#define SRC_VULKANIMAGE_HPP_

#include <cstdint>
#include <memory>
#include <variant>
#include <vulkan/vulkan_core.h>

#include "vulkan/VulkanResource.hpp"
namespace vkop {

// VulkanImage class inheriting from VulkanResource
class VulkanImage : public VulkanResource {
  public:
    VulkanImage(std::shared_ptr<VulkanDevice> &vdev, VkExtent3D dim,
                uint32_t layers, VkImageUsageFlags usage,
                VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT,
                VkMemoryPropertyFlags req = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                int ext_fd = -1);
    ~VulkanImage() override;

    ResourceType getResourceType() const override {
        return ResourceType::VK_IMAGE;
    }

    std::variant<VkDescriptorImageInfo *, VkDescriptorBufferInfo *,
                 VkBufferView *>
    getDescriptorInfo() override;

    void transferBarrier(VkCommandBuffer commandBuffer, VkImageLayout newLayout,
                         VkAccessFlags dstAccessMask);
    void transferReadBarrier(VkCommandBuffer commandBuffer);
    void transferWriteBarrier(VkCommandBuffer commandBuffer);

    void readBarrier(VkCommandBuffer commandBuffer);

    void writeBarrier(VkCommandBuffer commandBuffer);
    void readwriteBarrier(VkCommandBuffer commandBuffer);

    void copyImageToBuffer(VkCommandBuffer commandBuffer, VkBuffer buffer,
                           VkDeviceSize offset);
    void copyBufferToImage(VkCommandBuffer commandBuffer, VkBuffer buffer,
                           VkDeviceSize offset);
    void copyImageToImage(VkCommandBuffer commandBuffer,
                          const std::shared_ptr<VulkanImage> &srcimage,
                          VkOffset3D dstOffset, uint32_t dstBaseLayer);
    /*
     * For host image copy image layout transition
     * This is done on host
     */
#ifdef VK_EXT_host_image_copy
    void hostImaggeTransition(VkImageLayout newLayout);
    void hostImageCopyToDevice(void *ptr);
    void hostImageCopyToHost(void *ptr);
#endif
    VkImage getImage() const {
#ifndef USE_VMA
        return m_image_;
#else
        return m_vma_image_.image;
#endif
    }

    int getImageSize() const {
        return m_chansize_ * m_chans_ * m_dim_.width * m_dim_.height *
               m_dim_.depth * m_layers_;
    }
    uint32_t getImageWidth() const { return m_dim_.width; }
    uint32_t getImageHeight() const { return m_dim_.height; }
    uint32_t getImageLayers() const { return m_layers_; }

    int getImageChannelSize() const { return m_chansize_; }
    int getImageChannelNum() const { return m_chans_; }

    void splitImageView(std::vector<int64_t> &layers);

  private:
    VkExtent3D m_dim_;
    uint32_t m_layers_;
    VkFormat m_format_;
    VkImageUsageFlags m_usage_;
    VkImageType m_imagetype_;
    VkImageLayout m_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkAccessFlags m_access_ = 0;
#ifndef USE_VMA
    VkImage m_image_;
#else
    VMA::VmaImage m_vma_image_;
#endif
    VkImageView m_imageView_;
    VkSampler m_sampler_;
    VkDescriptorImageInfo image_info_;
    std::vector<VkImageView> splitImageView_;

    int m_chansize_;
    int m_chans_;

    void calcImageSize();

    void createImage();

    void createImageView();

    void createSampler();

    void destroyImage();

    void transitionImageLayout(VkCommandBuffer commandBuffer,
                               VkImageLayout newLayout,
                               VkAccessFlags dstAccessMask,
                               VkPipelineStageFlags sourceStage,
                               VkPipelineStageFlags destinationStage);
};
} // namespace vkop
#endif // SRC_VULKANIMAGE_HPP_
