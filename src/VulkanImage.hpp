#ifndef VULKAN_IMAGE_HPP
#define VULKAN_IMAGE_HPP

#include "vulkan/vulkan.hpp"
#include <cstdint>
#include <variant>
#include "VulkanResource.hpp"
#include "VulkanBuffer.hpp"

namespace vkop {

// VulkanImage class inheriting from VulkanResource
class VulkanImage : public VulkanResource {
public:
    VulkanImage(VkPhysicalDevice physicalDevice, const uint32_t queueFamilyIndex, VkDevice device,
        VkExtent3D dim,  VkFormat format, VkImageUsageFlags usage, VkMemoryPropertyFlags requireProperties,
        VkImageType imagetype);
    ~VulkanImage();

    ResourceType getResourceType() const override {
        return ResourceType::Image;
    }

    std::variant<VkDescriptorImageInfo, VkDescriptorBufferInfo> getDescriptorInfo() const override;

    void transitionImageLayout(VkCommandBuffer commandBuffer, VkImageLayout newLayout,
                VkAccessFlags dstAccessMask);

    void copyImageToBuffer(VkCommandBuffer commandBuffer, VulkanBuffer& buffer);

    uint32_t getRowPitch() const;

    void hostImageCopyFrom(void *ptr);

private:
    VkExtent3D m_dim;
    VkFormat m_format;
    VkImageUsageFlags m_usage;
    VkImageType m_imagetype;
    VkImageLayout m_layout;
    VkAccessFlags m_access;

    VkImage m_image;
    VkImageView m_imageView;
    VkSampler m_sampler;


    void createImage();

    void createImageView();

    void createSampler();

    void destroyImage();
};
}
#endif