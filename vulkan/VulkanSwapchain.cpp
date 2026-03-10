
#include "VulkanSwapchain.hpp"
#include "vulkan/vulkan_core.h"

namespace vkop {

VulkanSwapchain::VulkanSwapchain(std::shared_ptr<VulkanDevice> &vdev,
                                 VkSurfaceKHR surface)
    : vdev_(vdev), surface_(surface) {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> present_modes;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vdev_->getPhysicalDevice(),
                                              surface_, &capabilities);
    uint32_t format_count;
    vkGetPhysicalDeviceSurfaceFormatsKHR(vdev_->getPhysicalDevice(), surface_,
                                         &format_count, nullptr);

    assert(format_count != 0);
    formats.resize(format_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(vdev_->getPhysicalDevice(), surface_,
                                         &format_count, formats.data());

    uint32_t present_mode_count;
    vkGetPhysicalDeviceSurfacePresentModesKHR(
        vdev_->getPhysicalDevice(), surface_, &present_mode_count, nullptr);
    assert(present_mode_count != 0);
    present_modes.resize(present_mode_count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(vdev_->getPhysicalDevice(),
                                              surface_, &present_mode_count,
                                              present_modes.data());
    for (auto &mode : present_modes) {
        printf("present mode %d\n", mode);
    }

    surface_format_ = formats[0];
    for (auto &format : formats) {
        if (format.format == VK_FORMAT_R8G8B8A8_UNORM ||
            format.format == VK_FORMAT_B8G8R8A8_UNORM) {
            surface_format_ = format;
            break;
        }
    }

    VkSwapchainCreateInfoKHR create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    create_info.surface = surface_;
    create_info.minImageCount = capabilities.minImageCount;
    create_info.imageFormat = surface_format_.format;
    create_info.imageColorSpace = surface_format_.colorSpace;
    create_info.imageExtent = capabilities.currentExtent;
    create_info.imageArrayLayers = 1;
    create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    create_info.queueFamilyIndexCount = 0;
    create_info.pQueueFamilyIndices = nullptr;
    create_info.preTransform = capabilities.currentTransform;
    create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    create_info.presentMode = present_modes[0];
    create_info.clipped = VK_TRUE;
    auto ret = vkCreateSwapchainKHR(vdev_->getLogicalDevice(), &create_info,
                                    nullptr, &swapchain_);
    assert(ret == VK_SUCCESS);
    createSwapImages();
}

void VulkanSwapchain::createSwapImages() {
    uint32_t image_count;
    vkGetSwapchainImagesKHR(vdev_->getLogicalDevice(), swapchain_, &image_count,
                            nullptr);
    images_.resize(image_count);
    vkGetSwapchainImagesKHR(vdev_->getLogicalDevice(), swapchain_, &image_count,
                            images_.data());

    image_views_.resize(image_count);
    for (size_t i = 0; i < images_.size(); i++) {
        VkImageViewCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        create_info.image = images_[i];
        create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        create_info.format = surface_format_.format;
        create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        create_info.subresourceRange.baseMipLevel = 0;
        create_info.subresourceRange.levelCount = 1;
        create_info.subresourceRange.baseArrayLayer = 0;
        create_info.subresourceRange.layerCount = 1;
        auto ret = vkCreateImageView(vdev_->getLogicalDevice(), &create_info,
                                     nullptr, &image_views_[i]);
        assert(ret == VK_SUCCESS);
    }
}

} // namespace vkop