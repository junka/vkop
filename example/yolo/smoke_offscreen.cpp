// smoke_offscreen.cpp — minimal Vulkan offscreen render → PNG
//
// Verifies the rendering path used by vkyolo:
//   uploadTexture → FullscreenBlitter → OffscreenRenderer → stbi_write_png
//
// Build manually (after CMake has generated shader .c.o files via vkyolo):
//
//   cd build
//   g++ -std=c++17 -O1 -g \
//       ../example/yolo/smoke_offscreen.cpp \
//       -I../include -I../vulkan -I../core \
//       example/yolo/CMakeFiles/vkyolo.dir/shaders/quad.c.o \
//       example/yolo/CMakeFiles/vkyolo.dir/shaders/image.c.o \
//       -L. -lvkop -lvload -lvulkan -ldl -lpthread \
//       -Wl,-rpath,. -o smoke_offscreen
//   ./smoke_offscreen out.png

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h"
#pragma GCC diagnostic pop

#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanImage.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "vulkan/VulkanRenderPass.hpp"
#include "vulkan/VulkanFrameBuffer.hpp"
#include "vulkan/VulkanGraphicsPipeline.hpp"
#include "vulkan/VulkanCommandBuffer.hpp"
#include "vulkan/VulkanCommandPool.hpp"
#include "include/logger.hpp"

#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>
#include <chrono>

using vkop::VulkanInstance;
using vkop::VulkanDevice;
using vkop::VulkanImage;

extern unsigned char quad_spv[];
extern unsigned int  quad_spv_len;
extern unsigned char image_spv[];
extern unsigned int  image_spv_len;

namespace {

class FullscreenBlitter {
public:
    FullscreenBlitter(std::shared_ptr<VulkanDevice> dev, VkRenderPass rp)
        : dev_(std::move(dev)), rp_(rp) {}

    void bindTexture(std::shared_ptr<VulkanImage> tex) {
        if (!pipeline_) ensurePipeline(1, 1);  // extent not used here
        auto info = tex->getDescriptorInfo();
        auto* img_info_ptr = std::get<VkDescriptorImageInfo*>(info);
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = ds_; w.dstBinding = 0; w.dstArrayElement = 0;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.descriptorCount = 1; w.pImageInfo = img_info_ptr;
        pipeline_->updateDescriptorSets({w});
    }

    void recordDraw(vkop::VulkanCommandBuffer& cmd, uint32_t w, uint32_t h) {
        if (!pipeline_) ensurePipeline(w, h);
        cmd.bindGraphics(*pipeline_, ds_, VK_NULL_HANDLE, VK_NULL_HANDLE);
        vkCmdDraw(cmd.get(), 3, 1, 0, 0);
    }

private:
    void ensurePipeline(uint32_t w, uint32_t h) {
        if (pipeline_) return;
        std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER};
        pipeline_ = std::make_shared<vkop::VulkanGraphicsPipeline>(
            dev_->getLogicalDevice(), rp_, types,
            VkExtent2D{w, h},
            reinterpret_cast<const uint32_t*>(quad_spv), quad_spv_len,
            reinterpret_cast<const uint32_t*>(image_spv), image_spv_len);
        ds_ = pipeline_->allocDescriptorSets();
    }
    std::shared_ptr<VulkanDevice> dev_;
    VkRenderPass rp_;
    std::shared_ptr<vkop::VulkanGraphicsPipeline> pipeline_;
    VkDescriptorSet ds_ = VK_NULL_HANDLE;
};

void drawBox(uint8_t* rgba, int W, int H, int x1, int y1, int x2, int y2,
             uint8_t r, uint8_t g, uint8_t b, int thickness) {
    x1 = std::clamp(x1, 0, W-1); x2 = std::clamp(x2, 0, W-1);
    y1 = std::clamp(y1, 0, H-1); y2 = std::clamp(y2, 0, H-1);
    if (x2 <= x1 || y2 <= y1) return;
    auto set = [&](int x, int y){ auto* p=rgba+(y*W+x)*4; p[0]=r; p[1]=g; p[2]=b; p[3]=255; };
    for (int t=0; t<thickness; t++) {
        for (int x=x1; x<=x2; x++) { set(x, y1+t); set(x, y2-t); }
        for (int y=y1+t; y<=y2-t; y++) { set(x1+t, y); set(x2-t, y); }
    }
}

} // anon

int main(int argc, char* argv[]) {
    Logger::getInstance().setLevel(LOG_INFO);

    const auto& phydevs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
    auto dev = std::make_shared<VulkanDevice>(phydevs[0]);
    if (dev->getDeviceName().find("llvmpipe") != std::string::npos) {
        fprintf(stderr, "Need a real GPU\n"); return -1;
    }
    auto cmdpool = std::make_shared<vkop::VulkanCommandPool>(dev);
    fprintf(stderr, "device: %s\n", dev->getDeviceName().c_str());

    const int W = 640, H = 480;
    // Build frame: gray bg + 3 colored boxes (to verify boxes survive the whole path)
    std::vector<uint8_t> rgba(W*H*4, 0);
    // gray bg (180)
    for (size_t i=0; i<rgba.size(); i+=4) { rgba[i]=180; rgba[i+1]=180; rgba[i+2]=180; rgba[i+3]=255; }
    // red box top-left
    drawBox(rgba.data(), W, H, 50, 50, 200, 200, 255, 0, 0, 4);
    // green box center
    drawBox(rgba.data(), W, H, W/2-70, H/2-70, W/2+70, H/2+70, 0, 255, 0, 4);
    // blue box bottom-right
    drawBox(rgba.data(), W, H, W-180, H-150, W-50, H-50, 0, 0, 255, 4);

    auto t0 = std::chrono::steady_clock::now();

    // ---- 1. upload texture ----
    VkExtent3D dim{(uint32_t)W, (uint32_t)H, 1};
    auto tex = std::make_shared<VulkanImage>(
        dev, dim, 1,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VK_FORMAT_R8G8B8A8_UNORM);
    size_t bytes = W*H*4;
    auto stpool = cmdpool->getStagingBufferPool();
    auto sl_tex = stpool->allocate(bytes);
    memcpy(sl_tex->ptr, rgba.data(), bytes);
    stpool->flush(*sl_tex);
    {
        vkop::VulkanCommandBuffer cmd(cmdpool);
        cmd.begin();
        tex->copyBufferToImage(cmd.get(), sl_tex->buffer, sl_tex->offset);
        tex->readBarrier(cmd.get());
        cmd.end();
        cmd.submit(dev->getGraphicsQueues());
        cmd.wait();
    }
    fprintf(stderr, "[1/4] texture uploaded\n");

    // ---- 2. color attachment + renderpass + framebuffer ----
    auto color_fmt = VK_FORMAT_R8G8B8A8_UNORM;
    auto color = std::make_shared<VulkanImage>(
        dev, dim, 1,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        color_fmt, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkImageViewCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = color->getImage();
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = color_fmt;
    vi.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                    VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView color_view;
    vkCreateImageView(dev->getLogicalDevice(), &vi, nullptr, &color_view);

    vkop::VulkanRenderPass rp(dev->getLogicalDevice(), color_fmt);
    vkop::VulkanFrameBuffer fb(dev, rp, color_view, W, H);
    fprintf(stderr, "[2/4] attachment + renderpass ready\n");

    // ---- 3. record render → copy to staging ----
    FullscreenBlitter blitter(dev, rp.getRenderPass());
    blitter.bindTexture(tex);

    vkop::VulkanCommandBuffer cmd(cmdpool);
    cmd.begin();

    VkImageMemoryBarrier ib{};
    ib.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    ib.image = color->getImage();
    ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ib.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ib.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    ib.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd.get(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr, 1, &ib);

    rp.begin(cmd.get(), fb.getFrameBuffer(), W, H);
    blitter.recordDraw(cmd, W, H);
    rp.end(cmd.get());

    ib.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ib.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    ib.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    ib.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd.get(), VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &ib);

    auto sl_out = stpool->allocate(bytes);
    VkBufferImageCopy region = {};
    region.bufferOffset = sl_out->offset;
    region.bufferRowLength = W; region.bufferImageHeight = H;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {W, H, 1};
    vkCmdCopyImageToBuffer(cmd.get(), color->getImage(),
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, sl_out->buffer, 1, &region);

    VkBufferMemoryBarrier bmb = {};
    bmb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bmb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bmb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    bmb.buffer = sl_out->buffer; bmb.offset = sl_out->offset; bmb.size = bytes;
    vkCmdPipelineBarrier(cmd.get(), VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &bmb, 0, nullptr);

    cmd.end();
    cmd.submit(dev->getGraphicsQueues());
    cmd.wait();
    fprintf(stderr, "[3/4] render + copy done\n");

    // ---- 4. write PNG + pixel count ----
    std::vector<uint8_t> out((uint8_t*)sl_out->ptr, (uint8_t*)sl_out->ptr + bytes);
    const char* out_path = argc > 1 ? argv[1] : "/tmp/smoke_offscreen.png";
    if (!stbi_write_png(out_path, W, H, 4, out.data(), W*4)) {
        fprintf(stderr, "[4/4] stbi_write_png FAILED\n"); return 1;
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    int red=0, green=0, blue=0, gray=0;
    for (size_t p=0; p+3<out.size(); p+=4) {
        uint8_t r=out[p], g=out[p+1], b=out[p+2];
        if (r > 200 && g < 120 && b < 120) red++;
        else if (r < 120 && g > 200 && b < 120) green++;
        else if (r < 120 && g < 120 && b > 200) blue++;
        else if (r > 160 && g > 160 && b > 160) gray++;
    }
    fprintf(stderr,
        "[4/4] OK: %s (%dx%d, %.1f ms)\n"
        "  red-box pixels:   %d\n"
        "  green-box pixels: %d\n"
        "  blue-box pixels:  %d\n"
        "  gray-bg pixels:   %d\n",
        out_path, W, H, ms, red, green, blue, gray);
    return (red+green+blue > 100) ? 0 : 1;
}
