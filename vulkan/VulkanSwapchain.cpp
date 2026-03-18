
#include "VulkanSwapchain.hpp"
#include "include/logger.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/vulkan_core.h"
#include <limits>

namespace vkop {

VulkanSwapchain::VulkanSwapchain(std::shared_ptr<VulkanDevice> &vdev,
                                 VkSurfaceKHR surface, VkExtent2D extent)
    : vdev_(vdev), surface_(surface), extent_(extent) {
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
        const char *names[] = {
            "VK_PRESENT_MODE_IMMEDIATE_KHR",
            "VK_PRESENT_MODE_FIFO_KHR",
            "VK_PRESENT_MODE_FIFO_RELAXED_KHR",
            "VK_PRESENT_MODE_MAILBOX_KHR",
            "VK_PRESENT_MODE_SHARED_DEMAND_REFRESH_KHR",
        };
        printf("present mode %s\n", names[mode]);
    }

    surface_format_ = formats[0];
    for (auto &format : formats) {
        printf("surface format %d\n", format.format);
        if (format.format == VK_FORMAT_R8G8B8A8_SRGB ||
            format.format == VK_FORMAT_B8G8R8A8_SRGB) {
            surface_format_ = format;
            break;
        }
    }
    VkExtent2D actual_extent = capabilities.currentExtent;
    if (capabilities.currentExtent.width ==
        std::numeric_limits<uint32_t>::max()) {
        actual_extent = extent_;
        actual_extent.width =
            std::clamp(actual_extent.width, capabilities.minImageExtent.width,
                       capabilities.maxImageExtent.width);
        actual_extent.height =
            std::clamp(actual_extent.height, capabilities.minImageExtent.height,
                       capabilities.maxImageExtent.height);
    }

    uint32_t image_count = capabilities.minImageCount +
                           1; // one more for rendering outside of driver
    if (capabilities.maxImageCount > 0 &&
        image_count > capabilities.maxImageCount) {
        image_count = capabilities.maxImageCount;
    }

    // assume present and graphics queue are the same
    VkSwapchainCreateInfoKHR create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    create_info.surface = surface_;
    create_info.minImageCount = image_count;
    create_info.imageFormat = surface_format_.format;
    create_info.imageColorSpace = surface_format_.colorSpace;
    create_info.imageExtent = actual_extent;
    create_info.imageArrayLayers = 1;
    create_info.imageUsage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    create_info.queueFamilyIndexCount = 0;
    create_info.pQueueFamilyIndices = nullptr;
    if (capabilities.supportedTransforms &
        VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) {
        create_info.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    } else {
        create_info.preTransform = capabilities.currentTransform;
    }
    create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    create_info.presentMode = present_modes[0];
    create_info.clipped = VK_TRUE;
    create_info.oldSwapchain = VK_NULL_HANDLE;
    auto ret = vkCreateSwapchainKHR(vdev_->getLogicalDevice(), &create_info,
                                    nullptr, &swapchain_);
    assert(ret == VK_SUCCESS);
    createSwapImages();
}

VulkanSwapchain::~VulkanSwapchain() {
    for (auto &view : image_views_) {
        vkDestroyImageView(vdev_->getLogicalDevice(), view, nullptr);
    }
    if (swapchain_) {
        vkDestroySwapchainKHR(vdev_->getLogicalDevice(), swapchain_, nullptr);
    }
}

void VulkanSwapchain::createSwapImages() {
    uint32_t image_count;
    vkGetSwapchainImagesKHR(vdev_->getLogicalDevice(), swapchain_, &image_count,
                            nullptr);
    LOG_INFO("swapchain image count: %d\n", image_count);
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