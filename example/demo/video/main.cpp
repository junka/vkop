#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanFrameBuffer.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "vulkan/VulkanCommandPool.hpp"
#include "vulkan/VulkanSurface.hpp"
#include "vulkan/VulkanSwapchain.hpp"
#include "vulkan/VulkanCommandBuffer.hpp"
#include "vulkan/VulkanImage.hpp"
#include "vulkan/VulkanRenderPass.hpp"
#include "vulkan/VulkanGraphicsPipeline.hpp"

#include "include/logger.hpp"
#include "vulkan/vulkan_core.h"
#include <cstdio>
#include <csignal>
#include <thread>


#ifdef USE_GLFW
#include <GLFW/glfw3.h>
#endif

extern "C" {
#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "include/stb_image_resize2.h"
}


using vkop::VulkanInstance;
using vkop::VulkanDevice;

extern unsigned char image_spv[];
extern unsigned int image_spv_len;
extern unsigned char quad_spv[];
extern unsigned int quad_spv_len;

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

bool init_window(GLFWwindow** out_window, int width, int height) {
    if (!checkDisplayServer()) {
        return false;
    }
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(width, height, "Vulkan", nullptr, nullptr);
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

void cleanup(GLFWwindow* window) {
    if (window) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
        glfwPollEvents();
        glfwDestroyWindow(window);
    }
    glfwTerminate();
}

#endif


void drawBoundingBox(unsigned char* pixels, int width, int height, int channels,
                     int x1, int y1, int x2, int y2,
                     unsigned char r = 255, unsigned char g = 0, unsigned char b = 0, int thickness = 2) {
    // 确保坐标在有效范围内
    x1 = std::max(0, std::min(x1, width - 1));
    y1 = std::max(0, std::min(y1, height - 1));
    x2 = std::max(0, std::min(x2, width - 1));
    y2 = std::max(0, std::min(y2, height - 1));

    if (x2 <= x1 || y2 <= y1) return;

    // 绘制上边线
    for (int y = y1; y < std::min(y1 + thickness, y2); y++) {
        for (int x = x1; x < std::min(x2, width); x++) {
            int pixel_idx = (y * width + x) * channels;
            pixels[pixel_idx] = r;           // R
            if (channels > 1) pixels[pixel_idx + 1] = g; // G
            if (channels > 2) pixels[pixel_idx + 2] = b; // B
            if (channels > 3) pixels[pixel_idx + 3] = 255; // A
        }
    }

    // 绘制下边线
    for (int y = std::max(y2 - thickness, y1); y < y2; y++) {
        for (int x = x1; x < x2; x++) {
            int pixel_idx = (y * width + x) * channels;
            pixels[pixel_idx] = r;           // R
            if (channels > 1) pixels[pixel_idx + 1] = g; // G
            if (channels > 2) pixels[pixel_idx + 2] = b; // B
            if (channels > 3) pixels[pixel_idx + 3] = 255; // A
        }
    }

    // 绘制左边线
    for (int y = y1 + thickness; y < y2 - thickness; y++) {
        for (int x = x1; x < std::min(x1 + thickness, x2); x++) {
            int pixel_idx = (y * width + x) * channels;
            pixels[pixel_idx] = r;           // R
            if (channels > 1) pixels[pixel_idx + 1] = g; // G
            if (channels > 2) pixels[pixel_idx + 2] = b; // B
            if (channels > 3) pixels[pixel_idx + 3] = 255; // A
        }
    }

    // 绘制右边线
    for (int y = y1 + thickness; y < y2 - thickness; y++) {
        for (int x = std::max(x2 - thickness, x1); x < x2; x++) {
            int pixel_idx = (y * width + x) * channels;
            pixels[pixel_idx] = r;           // R
            if (channels > 1) pixels[pixel_idx + 1] = g; // G
            if (channels > 2) pixels[pixel_idx + 2] = b; // B
            if (channels > 3) pixels[pixel_idx + 3] = 255; // A
        }
    }
}


std::shared_ptr<vkop::VulkanImage> createTexture(std::shared_ptr<VulkanDevice> dev, const std::shared_ptr<vkop::VulkanCommandPool> &cmdpool, std::string &filepath) {
    int texture_width;
    int texture_height;
    int texture_channels;

    auto *pixels = stbi_load(filepath.c_str(), &texture_width, &texture_height, &texture_channels, STBI_rgb_alpha);
    if (!pixels) {
        LOG_ERROR("Failed to load texture");
        return nullptr;
    }
    auto image_size = texture_width * texture_height * 4;

    VkExtent3D dim = {static_cast<uint32_t>(texture_width), static_cast<uint32_t>(texture_height), 1};

    auto texture = std::make_shared<vkop::VulkanImage>(
        dev,
        dim,
        1,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VK_FORMAT_R8G8B8A8_UNORM
    );

    vkop::VulkanCommandBuffer cmd(cmdpool);

    auto stpool = cmdpool->getStagingBufferPool();
    auto b = stpool->allocate(image_size);
    if (!b) {
        return nullptr;
    }

    drawBoundingBox(pixels, texture_width, texture_height, 4, 0, 0, 100, 100);

    memcpy(b->ptr, pixels, image_size);

    cmd.begin();
    texture->copyBufferToImage(cmd.get(), b->buffer, b->offset);
    texture->readBarrier(cmd.get());
    cmd.end();
    cmd.submit(dev->getGraphicsQueues());
    cmd.wait();

    stbi_image_free(pixels);

    return texture;
}

std::atomic<bool> should_exit(false);
GLFWwindow* window = nullptr;
void signalHandler(int sig) {
    (void)sig;
    should_exit = true;
}

} // namespace

int main(int argc, const char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <image_path>\n", argv[0]);
        return -1;
    }
    std::signal(SIGINT, signalHandler);
    std::string image_path = argv[1];

    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", true);
    const auto& phydevs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
    auto dev = std::make_shared<VulkanDevice>(phydevs[0]);
    if (dev->getDeviceName().find("llvmpipe") != std::string::npos) {
        printf("Please set env VK_ICD_FILENAMES for a valid GPU\n");
        return -1;
    }
    auto cmdpool = std::make_shared<vkop::VulkanCommandPool>(dev, false);

    printf("using %s\n",dev->getDeviceName().c_str());
    
    auto texture = createTexture(dev, cmdpool, image_path);

    uint32_t width = texture->getImageWidth();
    uint32_t height = texture->getImageHeight();
#ifdef USE_GLFW
    if (!init_window(&window, width, height)) {
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

    VkExtent3D dim = {width, height, 1};

    glfwGetFramebufferSize(window, reinterpret_cast<int*>(&width), reinterpret_cast<int*>(&height));
    printf("frame buffer size: %d, %d\n", width, height);

    vkop::VulkanSurface vks(VulkanInstance::getVulkanInstance().getInstance(),
                                dev->getPhysicalDevice(), surface);
    vkop::VulkanSwapchain swapchain(dev, vks.getSurface(), VkExtent2D{width, height});

    auto images = swapchain.getImages();
    auto image_views = swapchain.getImageViews();

    VkDevice device = dev->getLogicalDevice();

    auto graphics_queue = dev->getGraphicsQueues();

    VkFormat format = swapchain.getSurfaceFormat().format;

    vkop::VulkanRenderPass render_pass(device, format);
    std::vector<std::shared_ptr<vkop::VulkanFrameBuffer>> framebuffers;
    framebuffers.reserve(image_views.size());
    for (auto & image_view : image_views) {
        framebuffers.push_back(std::make_shared<vkop::VulkanFrameBuffer>(dev, render_pass, image_view, dim.width, dim.height));
    }

    std::vector<VkDescriptorType> types = {
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
    };
    auto pipeline = std::make_shared<vkop::VulkanGraphicsPipeline>(device, render_pass.getRenderPass(), types, swapchain.getExtent(), reinterpret_cast<const uint32_t *>(quad_spv), quad_spv_len, reinterpret_cast<const uint32_t *>(image_spv), image_spv_len);

    auto* ds = pipeline->allocDescriptorSets();

    auto image_info = texture->getDescriptorInfo();
    auto* img_info_ptr = std::get<VkDescriptorImageInfo*>(image_info);

    VkWriteDescriptorSet write = {};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = ds;
    write.dstBinding = 0;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.descriptorCount = 1;
    write.pImageInfo = img_info_ptr;

    pipeline->updateDescriptorSets({write});

    auto image_semaphore = std::make_unique<vkop::VulkanSemaphore>(dev->getLogicalDevice());
    auto complete_semaphore = std::make_unique<vkop::VulkanSemaphore>(dev->getLogicalDevice());

    printf("Starting render loop...\n");

    while (!should_exit && !glfwWindowShouldClose(window)) {
        glfwPollEvents();

        if (should_exit) break;

        uint32_t image_index = 0;
        VkResult acquire_ret = vkop::vkAcquireNextImageKHR(
            device,
            swapchain.getSwapchain(), 
            1000000000,
            image_semaphore->getSemaphore(),
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

        cmd.bindGraphics(*pipeline, ds, nullptr, nullptr);

        vkop::vkCmdDraw(cmd.get(), 3, 1, 0, 0);
        render_pass.end(cmd.get());

        cmd.end();
        auto* needsigalsem = complete_semaphore->getSemaphore();
        cmd.addWait(image_semaphore->getSemaphore(), 0, needsigalsem, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
        auto subinfo = cmd.buildSubmitInfo();
        std::vector<VkSubmitInfo> submit_infos;
        submit_infos.push_back(subinfo);
        vkop::VulkanCommandBuffer::submit(dev->getGraphicsQueues(), submit_infos);

        // last present the display
        VkPresentInfoKHR present_info = {};
        auto *swapch = swapchain.getSwapchain();
        present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present_info.waitSemaphoreCount = 1;
        present_info.pWaitSemaphores = &needsigalsem;
        present_info.swapchainCount = 1;
        present_info.pSwapchains = &swapch;
        present_info.pImageIndices = &image_index;
        
        VkResult ret = vkop::vkQueuePresentKHR(graphics_queue->getQueue(), &present_info);
        cmd.wait();
        if (ret == VK_ERROR_OUT_OF_DATE_KHR || ret == VK_SUBOPTIMAL_KHR) {
            glfwGetFramebufferSize(window, reinterpret_cast<int*>(&width), reinterpret_cast<int*>(&height));
            continue;
        }
        if (ret != VK_SUCCESS) {
            throw std::runtime_error("Failed to present swap chain image!");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    dev->wait_all_done();
    printf("Exiting...\n");

    cleanup(window);
#else
    std::cout << "GLFW not enabled, running in headless mode" << std::endl;
#endif

    return 0;
}