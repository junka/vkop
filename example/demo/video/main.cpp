#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanFrameBuffer.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "vulkan/VulkanCommandPool.hpp"
#include "vulkan/VulkanSurface.hpp"
#include "vulkan/VulkanSwapchain.hpp"
#include "vulkan/VulkanCommandBuffer.hpp"
#include "vulkan/VulkanImage.hpp"
#include "vulkan/VulkanRenderPass.hpp"

#include "include/logger.hpp"
#include <thread>

#ifdef USE_GLFW
#include <GLFW/glfw3.h>
#endif

using vkop::VulkanInstance;
using vkop::VulkanDevice;

namespace {
#ifdef USE_GLFW

bool checkDisplayServer() {
    const char* wayland_display = std::getenv("WAYLAND_DISPLAY");
    const char* display = std::getenv("DISPLAY");
    
    if (!wayland_display && !display) {
        std::cerr << "No display server detected!" << std::endl;
        std::cerr << "Make sure you're running in a graphical environment" << std::endl;
        std::cerr << "For X11: DISPLAY should be set (e.g., :0)" << std::endl;
        std::cerr << "For Wayland: WAYLAND_DISPLAY should be set (e.g., wayland-0)" << std::endl;
        return false;
    }
    
    if (wayland_display) {
        std::cout << "Detected Wayland display: " << wayland_display << std::endl;
    }
    if (display) {
        std::cout << "Detected X11 display: " << display << std::endl;
    }
    
    return true;
}

bool init_window(GLFWwindow** out_window) {
    if (!checkDisplayServer()) {
        return false;
    }
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(800, 600, "Vulkan", nullptr, nullptr);
    if (!window) {
        LOG_ERROR("Failed to create GLFW window");

        int error;
        const char* description;
        while ((error = glfwGetError(&description)) != GLFW_NO_ERROR) {
            std::cerr << "GLFW Error " << error << ": " << description << std::endl;
        }
        
        std::cerr << "\nTroubleshooting suggestions:" << std::endl;
        std::cerr << "1. Check if you're running in a graphical environment" << std::endl;
        std::cerr << "2. For SSH/X11 forwarding, use: ssh -X user@host" << std::endl;
        std::cerr << "3. Install required dependencies:" << std::endl;
        std::cerr << "   - Ubuntu/Debian: sudo apt install libglfw3-dev libx11-dev libxrandr-dev libxi-dev libxcursor-dev libxinerama-dev" << std::endl;
        std::cerr << "   - Fedora: sudo dnf install glfw-devel libX11-devel libXrandr-devel libXi-devel libXcursor-devel libXinerama-devel" << std::endl;
        std::cerr << "4. Try running with WLR_NO_HARDWARE_CURSORS=1 if using Wayland" << std::endl;
        
        return false;
    }
    
    *out_window = window;
    std::cout << "GLFW window created successfully" << std::endl;
    return true;
}
#endif


void draw_circle() {
    
}


}


int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", true);
    const auto& phydevs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
    auto dev = std::make_shared<VulkanDevice>(phydevs[0]);
    if (dev->getDeviceName().find("llvmpipe") != std::string::npos) {
        printf("Please set env VK_ICD_FILENAMES for a valid GPU\n");
        return -1;
    }
    auto cmdpool = std::make_shared<vkop::VulkanCommandPool>(dev);

    printf("using %s\n",dev->getDeviceName().c_str());
    
#ifdef USE_GLFW
    GLFWwindow* window = nullptr;
    if (!init_window(&window)) {
        std::cerr << "\nRunning in headless mode (no window)" << std::endl;
        std::cerr << "To enable window support:" << std::endl;
        std::cerr << "1. Ensure you have a display server running (X11 or Wayland)" << std::endl;
        std::cerr << "2. Install GLFW and its dependencies" << std::endl;
        std::cerr << "3. Check DISPLAY or WAYLAND_DISPLAY environment variables" << std::endl;
        
#ifdef __linux__
        std::cerr << "\nLinux-specific tips:" << std::endl;
        std::cerr << "- Run 'echo $DISPLAY' to check X11 (should show :0 or similar)" << std::endl;
        std::cerr << "- Run 'echo $WAYLAND_DISPLAY' to check Wayland" << std::endl;
        std::cerr << "- If using SSH, connect with: ssh -X user@host (for X11 forwarding)" << std::endl;
        std::cerr << "- For headless rendering, consider using Xvfb:" << std::endl;
        std::cerr << "  Xvfb :99 -screen 0 1024x768x24 &" << std::endl;
        std::cerr << "  export DISPLAY=:99" << std::endl;
#endif
        return -1;
    }
    VkSurfaceKHR surface;
    uint32_t glfw_extension_count = 0;
    const char** glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);
    printf("glfw extensions: %d\n", glfw_extension_count);
    for (uint32_t i = 0; i < glfw_extension_count; i++) {
        printf("glfw extension: %s\n", glfw_extensions[i]);
    }
    VkResult err = glfwCreateWindowSurface(
        VulkanInstance::getVulkanInstance().getInstance(),
        window,
        nullptr,
        &surface
    );
    if (err != VK_SUCCESS) {
        throw std::runtime_error("Failed to create window surface");
    }

    uint32_t width = 800;
    uint32_t height = 600;
    VkExtent3D dim = {width, height, 1};
    vkop::VulkanSurface vks(VulkanInstance::getVulkanInstance().getInstance(),
                                dev->getPhysicalDevice(), surface);
    vkop::VulkanSwapchain swapchain(dev, vks.getSurface());

    auto images = swapchain.getImages();
    auto image_views = swapchain.getImageViews();

    VkDevice device = dev->getLogicalDevice();

    auto graphics_queue = dev->getGraphicsQueues();

    glfwGetFramebufferSize(window, reinterpret_cast<int*>(&width), reinterpret_cast<int*>(&height));

    VkFormat format = swapchain.getSurfaceFormat().format;

    vkop::VulkanRenderPass render_pass(device, format);
    std::vector<std::shared_ptr<vkop::VulkanFrameBuffer>> framebuffers;
    framebuffers.reserve(image_views.size());
    for (auto & image_view : image_views) {
        framebuffers.push_back(std::make_shared<vkop::VulkanFrameBuffer>(dev, render_pass, image_view, dim.width, dim.height));
    }

    printf("Starting render loop...\n");

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        uint32_t image_index = 0;
        VkResult acquire_ret = vkop::vkAcquireNextImageKHR(
            device, 
            swapchain.getSwapchain(), 
            UINT64_MAX, 
            VK_NULL_HANDLE, 
            VK_NULL_HANDLE, 
            &image_index
        );

        if (acquire_ret == VK_ERROR_OUT_OF_DATE_KHR) {
            glfwGetFramebufferSize(window, reinterpret_cast<int*>(&width), reinterpret_cast<int*>(&height));
            continue;
        }
        if (acquire_ret != VK_SUCCESS && acquire_ret != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("Failed to acquire swap chain image!");
        }


        vkop::VulkanCommandBuffer cmd(cmdpool);
        cmd.begin();

        render_pass.begin(cmd.get(), framebuffers[image_index]->getFrameBuffer(), width, height);

        draw_circle();

        render_pass.end(cmd.get());
        
        cmd.end();
        cmd.submit(dev->getComputeQueue());
        cmd.wait();

        VkPresentInfoKHR present_info = {};
        auto *swapch = swapchain.getSwapchain();
        present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present_info.waitSemaphoreCount = 0;
        present_info.pWaitSemaphores = nullptr;
        present_info.swapchainCount = 1;
        present_info.pSwapchains = &swapch;
        present_info.pImageIndices = &image_index;
        
        VkResult ret = vkop::vkQueuePresentKHR(graphics_queue->getQueue(), &present_info);
        
        if (ret == VK_ERROR_OUT_OF_DATE_KHR || ret == VK_SUBOPTIMAL_KHR) {
            glfwGetFramebufferSize(window, reinterpret_cast<int*>(&width), reinterpret_cast<int*>(&height));
            continue;
        }
        if (ret != VK_SUCCESS) {
            throw std::runtime_error("Failed to present swap chain image!");
        }
    }

    
    std::this_thread::sleep_for(std::chrono::seconds(5));

#else
    std::cout << "GLFW not enabled, running in headless mode" << std::endl;
#endif

    return 0;
}