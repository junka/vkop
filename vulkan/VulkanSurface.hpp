#include "vulkan/VulkanLib.hpp"

#ifdef VK_KHR_wayland_surface
#include <wayland-client.h>
#endif

#ifdef VK_KHR_xcb_surface
#include <xcb/xcb.h>
#endif

#ifdef VK_KHR_xlib_surface
#include <X11/Xlib.h>
#endif

#include <stdexcept>
namespace vkop {

class VulkanSurface {
    enum class Type { UNKNOWN, WAYLAND, X11_XCB, X11_XLIB, WIN32, HEADLESS };
    static Type detectDisplayType() {
        const char *wayland_display = std::getenv("WAYLAND_DISPLAY");
        const char *display = std::getenv("DISPLAY");

        if (wayland_display && wayland_display[0] != '\0') {
            return Type::WAYLAND;
        }

        if (display && display[0] != '\0') {
#ifdef VK_KHR_xcb_surface
            xcb_connection_t *conn = xcb_connect(nullptr, nullptr);
            if (conn && xcb_connection_has_error(conn) == 0) {
                xcb_disconnect(conn);
                return Type::X11_XCB;
            }
#endif

#ifdef VK_KHR_xlib_surface
            Display *disp = XOpenDisplay(nullptr);
            if (disp) {
                XCloseDisplay(disp);
                return Type::X11_XLIB;
            }
#endif
            return Type::X11_XCB;
        }

        return Type::HEADLESS;
    }

  public:
    VulkanSurface(VkInstance instance, VkPhysicalDevice physicalDevice,
                  VkSurfaceKHR surface)
        : instance_(instance), physicalDevice_(physicalDevice),
          surface_(surface) {}

    VulkanSurface(VkInstance instance, VkPhysicalDevice physicalDevice);

    ~VulkanSurface() {
        if (surface_ != VK_NULL_HANDLE) {
            vkDestroySurfaceKHR(instance_, surface_, nullptr);
            surface_ = VK_NULL_HANDLE;
        }
    }

    bool isQueueFamilySurfaceSupported(uint32_t queueFamilyIndex) const {
        VkBool32 supported = VK_FALSE;
        VkResult result = vkGetPhysicalDeviceSurfaceSupportKHR(
            physicalDevice_, queueFamilyIndex, surface_, &supported);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to query surface support.");
        }
        return supported == VK_TRUE;
    }

    VkSurfaceKHR getSurface() const { return surface_; }

  private:
    VkInstance instance_;
    VkPhysicalDevice physicalDevice_;
    VkSurfaceKHR surface_;
};

} // namespace vkop