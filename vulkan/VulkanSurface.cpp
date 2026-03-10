#include "vulkan/VulkanSurface.hpp"

namespace vkop {
VulkanSurface::VulkanSurface(VkInstance instance,
                             VkPhysicalDevice physicalDevice)
    : instance_(instance), physicalDevice_(physicalDevice) {
    VkResult result = VK_SUCCESS;
    auto display_type = detectDisplayType();
    if (display_type == Type::WAYLAND) {
#ifdef VK_KHR_wayland_surface
        auto vkCreateWaylandSurfaceKHR =
            reinterpret_cast<PFN_vkCreateWaylandSurfaceKHR>(
                vkGetInstanceProcAddr(instance_, "vkCreateWaylandSurfaceKHR"));
        if (!vkCreateWaylandSurfaceKHR) {
            throw std::runtime_error(
                "Failed to get vkCreateWaylandSurfaceKHR function address.");
        }
        result =
            vkCreateWaylandSurfaceKHR(instance_, nullptr, nullptr, &surface_);
#endif
    } else if (display_type == Type::X11_XLIB) {
#ifdef VK_KHR_xlib_surface
        auto vkCreateXlibSurfaceKHR =
            reinterpret_cast<PFN_vkCreateXlibSurfaceKHR>(
                vkGetInstanceProcAddr(instance_, "vkCreateXlibSurfaceKHR"));
        if (!vkCreateXlibSurfaceKHR) {
            throw std::runtime_error(
                "Failed to get vkCreateXlibSurfaceKHR function address.");
        }
        result = vkCreateXlibSurfaceKHR(instance_, nullptr, nullptr, &surface_);
#endif
    } else if (display_type == Type::X11_XCB) {
#ifdef VK_KHR_xcb_surface
        auto vkCreateXcbSurfaceKHR =
            reinterpret_cast<PFN_vkCreateXcbSurfaceKHR>(
                vkGetInstanceProcAddr(instance_, "vkCreateXcbSurfaceKHR"));
        if (!vkCreateXcbSurfaceKHR) {
            throw std::runtime_error(
                "Failed to get vkCreateXcbSurfaceKHR function address.");
        }
        result = vkCreateXcbSurfaceKHR(instance_, nullptr, nullptr, &surface_);
#endif
    } else if (display_type == Type::HEADLESS) {
        surface_ = VK_NULL_HANDLE;
    }
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create surface.");
    }
}
} // namespace vkop