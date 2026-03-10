// Copyright 2026 @junka
#ifndef VULKAN_VULKANSWAPCHAIN_HPP_
#define VULKAN_VULKANSWAPCHAIN_HPP_

#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanLib.hpp"

namespace vkop {

class VulkanSwapchain {
  public:
    VulkanSwapchain(std::shared_ptr<VulkanDevice> &vdev, VkSurfaceKHR surface);
    ~VulkanSwapchain() = default;

    VkSwapchainKHR getSwapchain() const { return swapchain_; }
    std::vector<VkImage> getImages() const { return images_; }

    std::vector<VkImageView> getImageViews() const { return image_views_; }

    VkSurfaceFormatKHR getSurfaceFormat() const { return surface_format_; }

  private:
    std::shared_ptr<VulkanDevice> vdev_;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkSurfaceFormatKHR surface_format_;
    std::vector<VkImage> images_;
    std::vector<VkImageView> image_views_;

    void createSwapImages();
};

}; // namespace vkop

#endif // VULKAN_VULKANSWAPCHAIN_HPP_