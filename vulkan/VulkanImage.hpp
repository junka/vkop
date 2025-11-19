// Copyright 2025 @junka
#ifndef SRC_VULKANIMAGE_HPP_
#define SRC_VULKANIMAGE_HPP_

#include "vulkan/vulkan.hpp"

#include <cstdint>
#include <memory>
#include <variant>

#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanResource.hpp"
#include "vulkan/VulkanStagingBufferPool.hpp"

namespace vkop {

// VulkanImage class inheriting from VulkanResource
class VulkanImage : public VulkanResource {
  public:
    VulkanImage(std::shared_ptr<VulkanDevice> &vdev, VkExtent3D dim,
                VkImageUsageFlags usage,
                VkMemoryPropertyFlags requireProperties,
                VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT,
                int ext_fd = -1);
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

    void copyImageToBuffer(VkCommandBuffer commandBuffer, VkBuffer buffer,
                           VkDeviceSize offset);
    void copyBufferToImage(VkCommandBuffer commandBuffer, VkBuffer buffer,
                           VkDeviceSize offset);

    /*
     * For host image copy image layout transition
     * This is done on host
     */
    void hostImaggeTransition(VkImageLayout newLayout);
    void hostImageCopyToDevice(void *ptr);
    void hostImageCopyToHost(void *ptr);

    int getImageSize() const {
        return m_chansize_ * m_chans_ * m_dim_.width * m_dim_.height *
               m_dim_.depth;
    }

    int getImageChannelSize() const { return m_chansize_; }
    int getImageChannelNum() const { return m_chans_; }

#ifdef USE_VMA
    void *getMappedMemory() override {
        return VMA::getMappedMemory(&m_vma_image_);
    };
#endif

  private:
    VkExtent3D m_dim_;
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

    int m_chansize_;
    int m_chans_;

    void calcImageSize();
    int getImageWidth() const { return m_dim_.width; }
    int getImageHeight() const { return m_dim_.height; }
    int getImageDepth() const { return m_dim_.depth; }

    void createImage();

    void createImageView();

    void createSampler();

    void destroyImage();
};
} // namespace vkop
#endif // SRC_VULKANIMAGE_HPP_
