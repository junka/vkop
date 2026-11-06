// example/yolo/main.cpp — camera or offline → YOLO inference → window or offscreen PNG
//
// Offline single-image path (no --camera/--frames-dir/--offscreen-out) is
// fully unchanged from the original. Realtime path: frame source → YOLO model
// → CPU postProcess + box draw → Vulkan renderer (swapchain or offscreen).

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
// STB_IMAGE_IMPLEMENTATION is provided by libvkop (function.cpp) — don't
// redefine here, or we get linker multiply-defined errors.
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image.h"
#include "include/stb_image_write.h"
#pragma GCC diagnostic pop

#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanFrameBuffer.hpp"
#include "vulkan/VulkanGraphicsPipeline.hpp"
#include "vulkan/VulkanImage.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "vulkan/VulkanRenderPass.hpp"
#include "vulkan/VulkanSurface.hpp"
#include "vulkan/VulkanSwapchain.hpp"
#include "vulkan/VulkanCommandBuffer.hpp"
#include "vulkan/VulkanCommandPool.hpp"
#include "include/logger.hpp"
#include "include/dev_shaders.hpp"
#include "coco_labels.hpp"

#ifndef YOLO_NO_LABELS
#include "font_freetype.hpp"  // FreeType atlas-based label renderer
#endif
#include "core/function.hpp"
#include "core/Tensor.hpp"
#include "core/runtime.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <signal.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <stdarg.h>
#include <inttypes.h>

#ifdef USE_GLFW
#include <GLFW/glfw3.h>
#endif

namespace fs = std::filesystem;
using namespace std::chrono;

using vkop::VulkanInstance;
using vkop::VulkanDevice;
using vkop::VulkanImage;
using vkop::core::Runtime;

// Shader C arrays (produced by CMake).
extern unsigned char quad_spv[];
extern unsigned int  quad_spv_len;
extern unsigned char image_spv[];
extern unsigned int  image_spv_len;

std::atomic<bool> g_running{true};
static void on_signal(int) { g_running = false; }

// FreeType font singleton — initialized once on first drawLabel call.
#ifndef YOLO_NO_LABELS
namespace {
    yolo_font::FontRenderer g_font;
    bool g_font_inited = false;

    const char* kFontCandidates[] = {
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    };

    // Initialize FreeType. Returns true on success. On failure prints a
    // comprehensive error listing the paths tried so the user can fix it
    // (libfreetype-dev missing, or font missing).
    bool init_font(int pixel_height = 24) {
        if (g_font_inited) return g_font.valid();
        for (auto p : kFontCandidates) {
            if (fs::exists(p)) {
                if (g_font.init(p, pixel_height)) {
                    g_font_inited = true;
                    return true;
                }
            }
        }
        fprintf(stderr,
            "[font] ERROR: could not load any usable TTF/OTF font.\n"
            "  Tried in order:\n");
        for (auto p : kFontCandidates)
            fprintf(stderr, "    %s  %s\n", p, fs::exists(p) ? "(load failed)" : "(not present)");
        fprintf(stderr,
            "  Install fonts (apt install fonts-noto-core or fonts-dejavu-core),\n"
            "  or pass -DYOLO_USE_FREETYPE=OFF to build without labels.\n");
        return false;
    }
}
#endif // !YOLO_NO_LABELS

inline uint64_t now_us() {
    using clock = steady_clock;
    return duration_cast<microseconds>(clock::now().time_since_epoch()).count();
}

// =========================================================================
// Part 1 — Frame capture sources
// =========================================================================

struct Frame {
    std::vector<uint8_t> rgba;
    int w = 0, h = 0;
    uint64_t ts_us = 0;
};

class FrameSource {
public:
    virtual ~FrameSource() = default;
    virtual bool open() = 0;
    virtual bool read_frame(Frame& out) = 0;
    virtual void close() = 0;
};

// --- FramesDirSource ---
class FramesDirSource : public FrameSource {
public:
    FramesDirSource(std::string dir, int loop_count)
        : dir_(std::move(dir)), loop_(loop_count) {}

    bool open() override {
        if (!fs::is_directory(dir_)) {
            fprintf(stderr, "[frames-dir] not a directory: %s\n", dir_.c_str());
            return false;
        }
        for (const auto& e : fs::directory_iterator(dir_)) {
            auto ext = e.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(),
                           [](unsigned char c){ return std::tolower(c); });
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png")
                files_.push_back(e.path());
        }
        std::sort(files_.begin(), files_.end());
        fprintf(stderr, "[frames-dir] found %zu images in %s (loop=%d)\n",
                files_.size(), dir_.c_str(), loop_);
        return !files_.empty();
    }
    bool read_frame(Frame& out) override {
        if (files_.empty()) return false;
        auto p = files_[idx_ % files_.size()];
        int w, h, c;
        auto* px = stbi_load(p.string().c_str(), &w, &h, &c, STBI_rgb_alpha);
        if (!px) return false;
        out.rgba.assign(px, px + w*h*4);
        out.w = w; out.h = h; out.ts_us = now_us();
        stbi_image_free(px);
        idx_++;
        if (loop_ == -1) return true;                              // infinite
        if (loop_ > 0)   return idx_ < files_.size() * loop_;      // fixed loops
        return idx_ < files_.size();                               // once
    }
    void close() override {}

private:
    std::string dir_;
    int loop_;
    std::vector<fs::path> files_;
    size_t idx_ = 0;
};

// --- V4L2Source ---
class V4L2Source : public FrameSource {
public:
    V4L2Source(std::string dev, int w, int h, std::string fmt)
        : dev_(std::move(dev)), w_(w), h_(h), fmt_(std::move(fmt)) {}

    bool open() override {
        fd_ = ::open(dev_.c_str(), O_RDWR | O_NONBLOCK);
        if (fd_ < 0) { fprintf(stderr, "[v4l2] open %s: %s\n", dev_.c_str(), strerror(errno)); return false; }
        struct v4l2_capability cap{};
        if (ioctl(fd_, VIDIOC_QUERYCAP, &cap) < 0 || !(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
            fprintf(stderr, "[v4l2] not VIDEO_CAPTURE: %s\n", dev_.c_str()); return false;
        }
        uint32_t pixfmt = V4L2_PIX_FMT_YUYV;
        auto low = fmt_; std::transform(low.begin(), low.end(), low.begin(),
                                         [](unsigned char c){ return std::tolower(c); });
        if (low == "mjpeg") pixfmt = V4L2_PIX_FMT_MJPEG;
        else if (low == "nv12") pixfmt = V4L2_PIX_FMT_NV12;

        struct v4l2_format vf{};
        vf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        vf.fmt.pix.width = w_; vf.fmt.pix.height = h_;
        vf.fmt.pix.pixelformat = pixfmt;
        vf.fmt.pix.field = V4L2_FIELD_NONE;
        if (ioctl(fd_, VIDIOC_S_FMT, &vf) < 0) {
            fprintf(stderr, "[v4l2] S_FMT fail: %s\n", strerror(errno)); return false;
        }
        w_ = vf.fmt.pix.width; h_ = vf.fmt.pix.height;
        pixfmt_ = vf.fmt.pix.pixelformat;
        fprintf(stderr, "[v4l2] negotiated %dx%d fmt=%u\n", w_, h_, pixfmt_);

        struct v4l2_requestbuffers rb{};
        rb.count = 4;
        rb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        rb.memory = V4L2_MEMORY_MMAP;
        if (ioctl(fd_, VIDIOC_REQBUFS, &rb) < 0 || rb.count < 2) {
            fprintf(stderr, "[v4l2] REQBUFS fail: %s\n", strerror(errno)); return false;
        }
        bufs_.resize(rb.count);
        for (uint32_t i = 0; i < rb.count; i++) {
            struct v4l2_buffer b{};
            b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; b.memory = V4L2_MEMORY_MMAP; b.index = i;
            if (ioctl(fd_, VIDIOC_QUERYBUF, &b) < 0) return false;
            bufs_[i].size = b.length;
            bufs_[i].start = mmap(nullptr, b.length, PROT_READ|PROT_WRITE,
                                  MAP_SHARED, fd_, b.m.offset);
            if (bufs_[i].start == MAP_FAILED) return false;
        }
        for (uint32_t i = 0; i < rb.count; i++) {
            struct v4l2_buffer b{};
            b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; b.memory = V4L2_MEMORY_MMAP; b.index = i;
            if (ioctl(fd_, VIDIOC_QBUF, &b) < 0) return false;
        }
        uint32_t t = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (ioctl(fd_, VIDIOC_STREAMON, &t) < 0) {
            fprintf(stderr, "[v4l2] STREAMON fail: %s\n", strerror(errno)); return false;
        }
        return true;
    }

    bool read_frame(Frame& out) override {
        fd_set fds; FD_ZERO(&fds); FD_SET(fd_, &fds);
        struct timeval tv{1,0};
        int r = select(fd_+1, &fds, nullptr, nullptr, &tv);
        if (r <= 0) return false;
        struct v4l2_buffer b{};
        b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; b.memory = V4L2_MEMORY_MMAP;
        if (ioctl(fd_, VIDIOC_DQBUF, &b) < 0) return false;
        out.w = w_; out.h = h_; out.ts_us = now_us();
        convert_to_rgba(bufs_[b.index].start, b.bytesused, out.rgba);
        struct v4l2_buffer qb{};
        qb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; qb.memory = V4L2_MEMORY_MMAP; qb.index = b.index;
        ioctl(fd_, VIDIOC_QBUF, &qb);
        return true;
    }

    void close() override {
        if (fd_ >= 0) {
            uint32_t t = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            ioctl(fd_, VIDIOC_STREAMOFF, &t);
            for (auto& buf : bufs_) munmap(buf.start, buf.size);
            ::close(fd_); fd_ = -1;
        }
    }

private:
    void convert_to_rgba(void* src, uint32_t nbytes, std::vector<uint8_t>& dst) {
        dst.resize(w_ * h_ * 4);
        auto* s = (uint8_t*)src;
        auto* d = dst.data();
        if (pixfmt_ == V4L2_PIX_FMT_YUYV) {
            uint32_t expected = w_ * h_ * 2;
            uint32_t N = std::min(nbytes, expected);
            int j = 0;
            auto write = [&](int idx, uint8_t Y, uint8_t U, uint8_t V){
                int R = Y + 1.402f*(V-128);
                int G = Y - 0.344136f*(U-128) - 0.714136f*(V-128);
                int B = Y + 1.772f*(U-128);
                uint8_t r=std::clamp(R,0,255), g=std::clamp(G,0,255), b=std::clamp(B,0,255);
                auto* p = d + idx*4; p[0]=r; p[1]=g; p[2]=b; p[3]=255;
            };
            int i = 0;
            while (j + 3 < (int)N) {
                uint8_t y0=s[j], u=s[j+1], y1=s[j+2], v=s[j+3];
                write(i*2,   y0, u, v);
                write(i*2+1, y1, u, v);
                j += 4; i++;
            }
        } else if (pixfmt_ == V4L2_PIX_FMT_MJPEG) {
            int w, h, c;
            auto* img = stbi_load_from_memory(s, nbytes, &w, &h, &c, STBI_rgb_alpha);
            if (!img) { std::fill(dst.begin(), dst.end(), 0); return; }
            if (w == w_ && h == h_) {
                memcpy(dst.data(), img, dst.size());
            } else {
                for (int y=0; y<h_; y++)
                    for (int x=0; x<w_; x++) {
                        int sx = x*w/w_; int sy = y*h/h_;
                        memcpy(dst.data() + (y*w_+x)*4, img + (sy*w+sx)*4, 4);
                    }
            }
            stbi_image_free(img);
        } else {
            std::fill(dst.begin(), dst.end(), 0);
        }
    }

    std::string dev_;
    int w_, h_;
    std::string fmt_;
    int fd_ = -1;
    uint32_t pixfmt_ = V4L2_PIX_FMT_YUYV;
    struct Buf { void* start; size_t size; };
    std::vector<Buf> bufs_;
};

// =========================================================================
// Part 2 — CPU box drawing
// =========================================================================

void drawBox(uint8_t* rgba, int W, int H, int x1, int y1, int x2, int y2,
             uint8_t r=255, uint8_t g=0, uint8_t b=0, int thickness=2) {
    x1 = std::clamp(x1, 0, W-1); x2 = std::clamp(x2, 0, W-1);
    y1 = std::clamp(y1, 0, H-1); y2 = std::clamp(y2, 0, H-1);
    if (x2 <= x1 || y2 <= y1) return;
    auto set = [&](int x, int y){ auto* p=rgba+(y*W+x)*4; p[0]=r; p[1]=g; p[2]=b; p[3]=255; };
    for (int t=0; t<thickness; t++) {
        for (int x=x1; x<=x2; x++) { set(x, y1+t); set(x, y2-t); }
        for (int y=y1+t; y<=y2-t; y++) { set(x1+t, y); set(x2-t, y); }
    }
}

// Fill a solid rectangle (used for label background bar).
static inline void fillRect(uint8_t* rgba, int W, int H,
                            int x1, int y1, int x2, int y2,
                            uint8_t r, uint8_t g, uint8_t b) {
    x1 = std::clamp(x1, 0, W-1); x2 = std::clamp(x2, 0, W-1);
    y1 = std::clamp(y1, 0, H-1); y2 = std::clamp(y2, 0, H-1);
    if (x2 <= x1 || y2 <= y1) return;
    for (int y = y1; y <= y2; y++) {
        auto* row = rgba + y * W * 4;
        for (int x = x1; x <= x2; x++) {
            row[x*4] = r; row[x*4+1] = g; row[x*4+2] = b; row[x*4+3] = 255;
        }
    }
}

// Draw "<class> <conf>" label above a box, with a color-matched background bar.
// Uses FreeType glyph atlas. Font init failure is fatal — the process exits
// with a clear error listing all font paths tried, so the user can fix it.
static void drawLabel(uint8_t* rgba, int W, int H,
                      int bx1, int by1, int bx2, int by2,
                      int class_id, float conf,
                      uint8_t bg_r, uint8_t bg_g, uint8_t bg_b) {
#ifdef YOLO_NO_LABELS
    (void)rgba; (void)W; (void)H;
    (void)bx1; (void)by1; (void)bx2; (void)by2;
    (void)class_id; (void)conf;
    (void)bg_r; (void)bg_g; (void)bg_b;
    return;
#else
    (void)bx2;
    if (!init_font()) {
        std::fprintf(stderr, "[yolo] FATAL: font init failed — see stderr above. Aborting.\n");
        std::exit(1);
    }

    // Build label string: "<coco name> <conf.XX>"
    char conf_buf[32];
    snprintf(conf_buf, sizeof(conf_buf), " %.2f", (double)conf);
    std::string label = std::string(yolo_coco::label(class_id)) + conf_buf;

    // Font size rule: box_h → pixel_height 12/20/28/36 px.
    int box_h = std::max(1, by2 - by1);
    int pixel_h = 12 + 8 * std::clamp(box_h / 32, 0, 3);
    g_font.setPixelHeight(pixel_h);

    // Measure with a dummy 1-pixel buffer (we only need the advance sum).
    int pad = 2;  // extra so antialiasing bleeds have room
    int tw = g_font.drawText(nullptr, 0, 0, label, 0, 0, 0, 0, 0);
    int th = g_font.lineHeight();
    int bar_w = tw + pad * 2;
    int bar_h = th + pad * 2;

    // Position: above box left-aligned. If above would go off-screen, draw inside top.
    int bar_x1 = bx1;
    int bar_x2 = bx1 + bar_w - 1;
    int bar_y2 = by1 - 1;
    int bar_y1 = bar_y2 - bar_h + 1;
    if (bar_y1 < 0) {
        bar_y1 = by1;
        bar_y2 = bar_y1 + bar_h - 1;
    }
    bar_x1 = std::clamp(bar_x1, 0, W-1);
    bar_x2 = std::clamp(bar_x2, 0, W-1);

    // Draw background bar (same color as box border).
    fillRect(rgba, W, H, bar_x1, bar_y1, bar_x1 + bar_w - 1, bar_y2,
             bg_r, bg_g, bg_b);

    // Contrast color: bg luminance → black or white text.
    float luma = 0.299f * bg_r + 0.587f * bg_g + 0.114f * bg_b;
    uint8_t tr = (luma > 140.0f) ? 0   : 255;
    uint8_t tg = (luma > 140.0f) ? 0   : 255;
    uint8_t tb = (luma > 140.0f) ? 0   : 255;

    // FreeType draws relative to baseline. baseline_y sits at bar_y1 + pad + ascent.
    // The exact ascent metric is 0.75 * lineHeight which matches typical fonts.
    int ascent = (int)(g_font.lineHeight() * 0.75);
    int baseline_y = bar_y1 + pad + ascent;
    g_font.drawText(rgba, W, H, label, bar_x1 + pad, baseline_y, tr, tg, tb);
#endif
}

// =========================================================================
// Part 3 — YOLO postProcess (decode + NMS, full letterbox inverse already
// baked into detection coords)
// =========================================================================

struct Detection {
    float bbox[4];   // x1 y1 x2 y2 (image-space, post-letterbox inverse)
    float confidence;
    int   class_id;
};

namespace {
template<typename T>
std::vector<Detection> postProcess(
    const std::shared_ptr<vkop::core::Tensor<T>>& out,
    int img_h, int img_w)
{
    auto shape = out->getShape();
    const int num_classes = 80;
    const int num_detections = shape[2];
    const float conf_thr = 0.5f;
    const float nms_thr  = 0.4f;

    float gain = std::min(640.0f / img_h, 640.0f / img_w);
    int new_w = (int)(img_w * gain);
    int new_h = (int)(img_h * gain);
    int pad_left = (640 - new_w) / 2;
    int pad_top  = (640 - new_h) / 2;

    auto load4 = [&](int det_idx) -> std::array<float,4> {
        if constexpr (std::is_same_v<T, float>) {
            return { (*out)[(0*num_detections)+det_idx], (*out)[(1*num_detections)+det_idx],
                     (*out)[(2*num_detections)+det_idx], (*out)[(3*num_detections)+det_idx] };
        } else {
            auto f = [&](int c){ return vkop::core::ITensor::fp16_to_fp32(
                out->data()[(c*num_detections)+det_idx]); };
            return {f(0), f(1), f(2), f(3)};
        }
    };
    auto load_cls = [&](int det_idx) -> std::pair<int,float> {
        int best=0; float best_p=0;
        for (int c=0; c<num_classes; c++) {
            float p;
            if constexpr (std::is_same_v<T,float>)
                p = (*out)[((4+c)*num_detections)+det_idx];
            else
                p = vkop::core::ITensor::fp16_to_fp32(out->data()[((4+c)*num_detections)+det_idx]);
            if (p > best_p) { best_p = p; best = c; }
        }
        return {best, best_p};
    };

    std::vector<Detection> dets;
    for (int d=0; d<num_detections; d++) {
        auto [cx,cy,bw,bh] = load4(d);
        cx -= pad_left; cy -= pad_top;
        float cxn = cx/gain, cyn = cy/gain, bwn = bw/gain, bhn = bh/gain;
        float x1 = cxn - bwn/2, y1 = cyn - bhn/2;
        float x2 = cxn + bwn/2, y2 = cyn + bhn/2;
        x1 = std::clamp(x1, 0.f, (float)img_w); y1 = std::clamp(y1, 0.f, (float)img_h);
        x2 = std::clamp(x2, 0.f, (float)img_w); y2 = std::clamp(y2, 0.f, (float)img_h);

        auto [cls, prob] = load_cls(d);
        if (prob < conf_thr) continue;
        dets.push_back({{x1,y1,x2,y2}, prob, cls});
    }

    std::vector<bool> suppressed(dets.size(), false);
    for (size_t i=0; i<dets.size(); i++) {
        if (suppressed[i]) continue;
        for (size_t j=i+1; j<dets.size(); j++) {
            if (suppressed[j]) continue;
            auto& a = dets[i], b = dets[j];
            float x1_i = std::max(a.bbox[0], b.bbox[0]);
            float y1_i = std::max(a.bbox[1], b.bbox[1]);
            float x2_i = std::min(a.bbox[2], b.bbox[2]);
            float y2_i = std::min(a.bbox[3], b.bbox[3]);
            float inter = std::max(0.f, x2_i-x1_i) * std::max(0.f, y2_i-y1_i);
            float ai = (a.bbox[2]-a.bbox[0])*(a.bbox[3]-a.bbox[1]);
            float bj = (b.bbox[2]-b.bbox[0])*(b.bbox[3]-b.bbox[1]);
            float u = ai + bj - inter;
            if (u > 0 && inter/u > nms_thr) {
                if (a.confidence < b.confidence) { suppressed[i]=true; break; }
                suppressed[j]=true;
            }
        }
    }
    std::vector<Detection> result;
    for (size_t i=0; i<dets.size(); i++) if (!suppressed[i]) result.push_back(dets[i]);
    return result;
}
} // anon

// =========================================================================
// Part 4 — Vulkan texture upload + renderer
// =========================================================================

// Upload host RGBA8 buffer into a VulkanImage (VK_FORMAT_R8G8B8A8_UNORM),
// ready to sample via the image's default descriptor.
std::shared_ptr<VulkanImage> uploadTexture(
    std::shared_ptr<VulkanDevice> dev,
    std::shared_ptr<vkop::VulkanCommandPool> cmdpool,
    const uint8_t* rgba, int w, int h)
{
    VkExtent3D dim{(uint32_t)w, (uint32_t)h, 1};
    auto tex = std::make_shared<VulkanImage>(
        dev, dim, 1,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VK_FORMAT_R8G8B8A8_UNORM);
    size_t bytes = w * h * 4;
    auto stpool = cmdpool->getStagingBufferPool();
    auto b = stpool->allocate(bytes);
    if (!b) return nullptr;
    memcpy(b->ptr, rgba, bytes);
    stpool->flush(*b);
    vkop::VulkanCommandBuffer cmd(cmdpool);
    cmd.begin();
    tex->copyBufferToImage(cmd.get(), b->buffer, b->offset);
    tex->readBarrier(cmd.get());
    cmd.end();
    cmd.submit(dev->getGraphicsQueues());
    cmd.wait();
    return tex;
}

// Upload host RGBA8 into a host-visible buffer via staging pool and return
// the raw VkBuffer handle + offset so callers can issue vkCmdCopyImageToBuffer
// (offscreen render-back path). The buffer stays alive as long as the staging
// pool does (i.e. until CmdPool destruction), which is exactly what we need.
struct StagingSlice {
    VkBuffer buffer;
    VkDeviceSize offset;
    void* ptr;
};

StagingSlice uploadToStaging(
    std::shared_ptr<VulkanDevice> dev,
    std::shared_ptr<vkop::VulkanCommandPool> cmdpool,
    const uint8_t* rgba, int w, int h)
{
    (void)dev;
    (void)rgba; (void)w; (void)h;   // placeholder — not actually used in final path; uploadTexture is the real helper.
    auto stpool = cmdpool->getStagingBufferPool();
    size_t bytes = w * h * 4;
    auto b = stpool->allocate(bytes);
    memcpy(b->ptr, rgba, bytes);
    stpool->flush(*b);
    return {b->buffer, b->offset, b->ptr};
}

// Simple fullscreen blit: records a render pass that binds the given texture
// and draws 3 verts covering the framebuffer.
class FullscreenBlitter {
public:
    FullscreenBlitter(std::shared_ptr<VulkanDevice> dev, VkRenderPass rp)
        : dev_(std::move(dev)), rp_(rp) {}

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

    void bindTexture(std::shared_ptr<VulkanImage> tex) {
        (void)tex;  // extent not needed; recordDraw passes fb dims
        if (!pipeline_) {
            std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER};
            pipeline_ = std::make_shared<vkop::VulkanGraphicsPipeline>(
                dev_->getLogicalDevice(), rp_, types,
                VkExtent2D{1, 1},  // placeholder extent; dynamic state
                reinterpret_cast<const uint32_t*>(quad_spv), quad_spv_len,
                reinterpret_cast<const uint32_t*>(image_spv), image_spv_len);
            ds_ = pipeline_->allocDescriptorSets();
        }
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
        ensurePipeline(w, h);
        cmd.bindGraphics(*pipeline_, ds_, VK_NULL_HANDLE, VK_NULL_HANDLE);
        vkCmdDraw(cmd.get(), 3, 1, 0, 0);
    }

private:
    std::shared_ptr<VulkanDevice> dev_;
    VkRenderPass rp_;
    std::shared_ptr<vkop::VulkanGraphicsPipeline> pipeline_;
    VkDescriptorSet ds_ = VK_NULL_HANDLE;
};

// --- DisplayRenderer: GLFW window + VkSwapchain ---
class DisplayRenderer {
public:
    DisplayRenderer(std::shared_ptr<VulkanDevice> dev,
                    std::shared_ptr<vkop::VulkanCommandPool> cmdpool)
        : dev_(std::move(dev)), cmdpool_(std::move(cmdpool)) {}

    bool open() {
#ifdef USE_GLFW
        if (!glfwInit()) { fprintf(stderr, "glfwInit fail\n"); return false; }
        if (!std::getenv("DISPLAY") && !std::getenv("WAYLAND_DISPLAY")) {
            fprintf(stderr, "no display server\n"); glfwTerminate(); return false;
        }
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window_ = glfwCreateWindow(960, 720, "YOLO vkop", nullptr, nullptr);
        if (!window_) {
            const char* ge = nullptr; glfwGetError(&ge);
            fprintf(stderr, "glfwCreateWindow fail err='%s'\n", ge ? ge : "null");
            glfwTerminate(); return false;
        }

        auto& vi = VulkanInstance::getVulkanInstance();
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        if (glfwCreateWindowSurface(vi.getInstance(), window_, nullptr, &surface) != VK_SUCCESS ||
            surface == VK_NULL_HANDLE) {
            const char* ge = nullptr; glfwGetError(&ge);
            fprintf(stderr, "glfwCreateWindowSurface fail err='%s'\n", ge ? ge : "null");
            glfwDestroyWindow(window_); glfwTerminate(); return false;
        }
        vsurface_ = std::make_unique<vkop::VulkanSurface>(
            vi.getInstance(), dev_->getPhysicalDevice(), surface);
        swapchain_ = std::make_unique<vkop::VulkanSwapchain>(
            dev_, vsurface_->getSurface(), VkExtent2D{960, 720});
        auto imgs = swapchain_->getImages();
        auto views = swapchain_->getImageViews();
        swap_fmt_ = swapchain_->getSurfaceFormat().format;
        swap_w_ = swapchain_->getExtent().width;
        swap_h_ = swapchain_->getExtent().height;
        render_pass_ = std::make_unique<vkop::VulkanRenderPass>(
            dev_->getLogicalDevice(), swap_fmt_);
        framebuffers_.clear();
        for (auto& v : views)
            framebuffers_.push_back(std::make_unique<vkop::VulkanFrameBuffer>(
                dev_, *render_pass_, v, swap_w_, swap_h_));
        blitter_ = std::make_unique<FullscreenBlitter>(dev_, render_pass_->getRenderPass());
        img_sem_ = std::make_unique<vkop::VulkanSemaphore>(dev_->getLogicalDevice());
        done_sem_ = std::make_unique<vkop::VulkanSemaphore>(dev_->getLogicalDevice());
        fprintf(stderr, "[display] %ux%u swapchain ready\n", swap_w_, swap_h_);
        return true;
#else
        fprintf(stderr, "[display] GLFW not compiled in\n");
        return false;
#endif
    }

    void present(std::shared_ptr<VulkanImage> tex) {
#ifdef USE_GLFW
        if (glfwWindowShouldClose(window_)) return;
        glfwPollEvents();
        uint32_t idx;
        if (vkop::vkAcquireNextImageKHR(
                dev_->getLogicalDevice(), swapchain_->getSwapchain(),
                1000000000, img_sem_->getSemaphore(), VK_NULL_HANDLE, &idx) != VK_SUCCESS)
            return;

        vkop::VulkanCommandBuffer cmd(cmdpool_);
        cmd.begin();
        render_pass_->begin(cmd.get(), framebuffers_[idx]->getFrameBuffer(), swap_w_, swap_h_);
        blitter_->bindTexture(tex);
        blitter_->recordDraw(cmd, swap_w_, swap_h_);
        render_pass_->end(cmd.get());
        cmd.end();
        auto done_sem = done_sem_->getSemaphore();
        cmd.addWait(img_sem_->getSemaphore(), 0, done_sem,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
        std::vector<VkSubmitInfo> sis{cmd.buildSubmitInfo()};
        vkop::VulkanCommandBuffer::submit(dev_->getGraphicsQueues(), sis);

        VkSwapchainKHR sc = swapchain_->getSwapchain();
        VkPresentInfoKHR pi{};
        pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        pi.waitSemaphoreCount = 1; pi.pWaitSemaphores = &done_sem;
        pi.swapchainCount = 1; pi.pSwapchains = &sc; pi.pImageIndices = &idx;
        vkQueuePresentKHR(dev_->getGraphicsQueues()->getQueue(), &pi);
        cmd.wait();
#else
        (void)tex;
#endif
    }

    bool shouldClose() {
#ifdef USE_GLFW
        glfwPollEvents();
        return glfwWindowShouldClose(window_) || !g_running;
#else
        return !g_running;
#endif
    }

    ~DisplayRenderer() {
#ifdef USE_GLFW
        if (window_) { glfwDestroyWindow(window_); glfwTerminate(); }
#endif
    }

private:
    std::shared_ptr<VulkanDevice> dev_;
    std::shared_ptr<vkop::VulkanCommandPool> cmdpool_;
#ifdef USE_GLFW
    GLFWwindow* window_ = nullptr;
#endif
    std::unique_ptr<vkop::VulkanSurface> vsurface_;
    std::unique_ptr<vkop::VulkanSwapchain> swapchain_;
    std::unique_ptr<vkop::VulkanRenderPass> render_pass_;
    std::vector<std::unique_ptr<vkop::VulkanFrameBuffer>> framebuffers_;
    std::unique_ptr<FullscreenBlitter> blitter_;
    std::unique_ptr<vkop::VulkanSemaphore> img_sem_, done_sem_;
    VkFormat swap_fmt_ = VK_FORMAT_UNDEFINED;
    uint32_t swap_w_ = 0, swap_h_ = 0;
};

// --- OffscreenRenderer: render into a single device image, copy to CPU RGBA ---
class OffscreenRenderer {
public:
    OffscreenRenderer(std::shared_ptr<VulkanDevice> dev,
                      std::shared_ptr<vkop::VulkanCommandPool> cmdpool,
                      uint32_t w, uint32_t h)
        : dev_(std::move(dev)), cmdpool_(std::move(cmdpool)), w_(w), h_(h) {}

    // First call creates the attachment + renderpass + framebuffer + blitter.
    // Subsequent calls just record a new render → copy each frame.
    bool render(std::shared_ptr<VulkanImage> tex, std::vector<uint8_t>& out_rgba) {
        if (!color_) {
            createAttachment();
            rp_holder_ = std::make_unique<vkop::VulkanRenderPass>(
                dev_->getLogicalDevice(), color_format_);
            blitter_ = std::make_unique<FullscreenBlitter>(dev_, rp_holder_->getRenderPass());
            fb_ = std::make_unique<vkop::VulkanFrameBuffer>(
                dev_, *rp_holder_, color_view_, w_, h_);
        }

        vkop::VulkanCommandBuffer cmd(cmdpool_);
        cmd.begin();

        // attachment UNDEFINED → COLOR_ATTACHMENT
        color_att_barrier_.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        color_att_barrier_.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        vkCmdPipelineBarrier(cmd.get(),
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            0, 0, nullptr, 0, nullptr, 1, &color_att_barrier_);

        rp_holder_->begin(cmd.get(), fb_->getFrameBuffer(), w_, h_);
        blitter_->bindTexture(tex);
        blitter_->recordDraw(cmd, w_, h_);
        rp_holder_->end(cmd.get());

        // COLOR_ATTACHMENT → TRANSFER_SRC
        color_copy_barrier_.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color_copy_barrier_.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        vkCmdPipelineBarrier(cmd.get(),
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &color_copy_barrier_);

        // Allocate staging buffer for this frame (staging pool handles lifetime)
        size_t bytes = w_ * h_ * 4;
        auto stpool = cmdpool_->getStagingBufferPool();
        auto sl = stpool->allocate(bytes);
        if (!sl) { cmd.end(); return false; }

        VkBufferImageCopy region = {};
        region.bufferOffset = sl->offset;
        region.bufferRowLength = w_;
        region.bufferImageHeight = h_;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {w_, h_, 1};
        vkCmdCopyImageToBuffer(cmd.get(), color_->getImage(),
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, sl->buffer, 1, &region);

        // Buffer → host
        VkBufferMemoryBarrier bmb = {};
        bmb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        bmb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        bmb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        bmb.buffer = sl->buffer;
        bmb.offset = sl->offset;
        bmb.size = bytes;
        vkCmdPipelineBarrier(cmd.get(),
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
            0, 0, nullptr, 1, &bmb, 0, nullptr);

        cmd.end();
        cmd.submit(dev_->getGraphicsQueues());
        cmd.wait();

        out_rgba.assign((uint8_t*)sl->ptr, (uint8_t*)sl->ptr + bytes);
        return true;
    }

private:
    void createAttachment() {
        VkExtent3D dim{w_, h_, 1};
        color_format_ = VK_FORMAT_R8G8B8A8_UNORM;
        color_ = std::make_shared<VulkanImage>(
            dev_, dim, 1,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            color_format_,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        // Plain attachment view (not a sampled view).
        VkImageViewCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vi.image = color_->getImage();
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = color_format_;
        vi.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                        VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCreateImageView(dev_->getLogicalDevice(), &vi, nullptr, &color_view_);

        color_att_barrier_.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        color_att_barrier_.image = color_->getImage();
        color_att_barrier_.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        color_att_barrier_.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color_att_barrier_.srcAccessMask = 0;
        color_att_barrier_.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        color_att_barrier_.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        color_att_barrier_.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        color_att_barrier_.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        color_copy_barrier_ = color_att_barrier_;
        color_copy_barrier_.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color_copy_barrier_.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        color_copy_barrier_.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        color_copy_barrier_.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    }

    std::shared_ptr<VulkanDevice> dev_;
    std::shared_ptr<vkop::VulkanCommandPool> cmdpool_;
    uint32_t w_, h_;

    VkFormat color_format_ = VK_FORMAT_UNDEFINED;
    std::shared_ptr<VulkanImage> color_;
    VkImageView color_view_ = VK_NULL_HANDLE;
    std::unique_ptr<vkop::VulkanRenderPass> rp_holder_;
    std::unique_ptr<vkop::VulkanFrameBuffer> fb_;
    std::unique_ptr<FullscreenBlitter> blitter_;

    VkImageMemoryBarrier color_att_barrier_{};
    VkImageMemoryBarrier color_copy_barrier_{};
};

// =========================================================================
// Part 5 — main
// =========================================================================

static void usage(const char* prog) {
    fprintf(stderr,
        "Usage:\n"
        "  Offline (unchanged): %s <model.bin> <image.jpg> [trace_node]\n"
        "  Frames-dir + PNG:     %s <model.bin> --frames-dir <dir> --offscreen-out out.png [--frames N]\n"
        "  V4L2 + PNG:           %s <model.bin> --camera /dev/video0 --offscreen-out out.png\n"
        "  V4L2 + window:        %s <model.bin> --camera /dev/video0\n"
        "  Frames-dir + window:  %s <model.bin> --frames-dir <dir>\n",
        prog, prog, prog, prog, prog);
}

// Minimal color palette for boxes (per-class, wraps around).
static const uint8_t kBoxColors[20][3] = {
    {255,  0,  0}, {  0,255,  0}, {  0,  0,255}, {255,255,  0},
    {255,  0,255}, {  0,255,255}, {128,  0,255}, {255,128,  0},
    {  0,128,255}, {128,255,  0}, {255,  0,128}, {  0,255,128},
    {128,  0,  0}, {  0,128,  0}, {  0,  0,128}, {128,128,  0},
    {128,  0,128}, {  0,128,128}, {200,100, 50}, { 50,200,100}
};

int main(int argc, char* argv[]) {
    signal(SIGINT, on_signal);

    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", true);

    // Load the dev shader archive from this build tree (build/vkop_shaders.bin).
    // vkopbin files embedded SPIR-V subsets, but dev binaries like this one
    // (which run arbitrary models during iteration) need the full 165-entry
    // fallback. libvkop carries no SPIR-V of its own, so skipping this means
    // any operator whose vkopbin did not embed its shader will throw.
    vkop::dev::seed_shader_archive_from_build_tree();

    const auto& phydevs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
    auto dev = std::make_shared<VulkanDevice>(phydevs[0]);
    if (dev->getDeviceName().find("llvmpipe") != std::string::npos) {
        fprintf(stderr, "Please set env VK_ICD_FILENAMES for a real GPU\n");
        return -1;
    }
    fprintf(stderr, "[yolo] device: %s\n", dev->getDeviceName().c_str());
    auto cmdpool = std::make_shared<vkop::VulkanCommandPool>(dev);

    // --- Arg parsing ---
    std::string binary_path;
    std::string frames_dir, camera_dev, offscreen_out;
    std::string trace_node;
    int cam_w = 640, cam_h = 480;
    int frame_limit = -1;  // -1 = infinite / frames-dir exhaustive
    int precision = 0;

    if (argc < 2) { usage(argv[0]); return 1; }
    binary_path = argv[1];
    int i = 2;
    while (i < argc) {
        auto a = std::string(argv[i]);
        auto get = [&](const char* flag) -> std::string {
            if (i+1 >= argc) { fprintf(stderr, "%s needs a value\n", flag); exit(1); }
            return argv[++i];
        };
        if      (a == "--frames-dir")      frames_dir    = get("--frames-dir");
        else if (a == "--camera")         camera_dev    = get("--camera");
        else if (a == "--offscreen-out")  offscreen_out = get("--offscreen-out");
        else if (a == "--cam-w")          cam_w         = std::stoi(get("--cam-w"));
        else if (a == "--cam-h")          cam_h         = std::stoi(get("--cam-h"));
        else if (a == "--frames")         frame_limit   = std::stoi(get("--frames"));
        else if (a == "--precision")      precision     = std::stoi(get("--precision"));
        else if (a == "--trace")          trace_node    = get("--trace");
        else if (a == "--help" || a == "-h") { usage(argv[0]); return 0; }
        else {
            // Legacy positional arg for offline single-image path.
            if (frames_dir.empty() && camera_dev.empty() && offscreen_out.empty()
                && trace_node.empty() && i == 2 && argc == 3) {
                trace_node = a;  // interpreted as image path below
            }
        }
        i++;
    }

    bool realtime = !camera_dev.empty() || !frames_dir.empty();

    // ============= Legacy offline single-image path (UNCHANGED) ============
    if (!realtime) {
        std::string image_path = trace_node;   // was argv[2]
        if (image_path.empty() || !fs::exists(image_path)) {
            // argv[2] may have been consumed as trace_node — re-read
            if (argc >= 3) image_path = argv[2];
        }
        if (image_path.empty() || !fs::exists(image_path)) {
            usage(argv[0]); return 1;
        }
        auto rt = std::make_shared<Runtime>(cmdpool, binary_path, precision);
        rt->LoadModel();
        if (argc >= 4) rt->TraceNode(argv[3]);
        fprintf(stderr, "[yolo] model loaded (offline)\n");

        auto [img_h, img_w] = vkop::core::Function::preprocess_jpg(
            image_path.c_str(), cmdpool, rt->GetInput(), true);
        rt->Run();
        rt->ReadResult();
        if (precision == 0) {
            auto out = vkop::core::as_tensor<float>(rt->GetOutput());
            auto dets = postProcess(out, img_h, img_w);
            fprintf(stderr, "[offline] %zu detections\n", dets.size());
            for (auto& d : dets)
                fprintf(stderr, "  x1=%.0f y1=%.0f x2=%.0f y2=%.0f cls=%d conf=%.2f\n",
                        d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3],
                        d.class_id, d.confidence);
        } else {
            auto out = vkop::core::as_tensor<uint16_t>(rt->GetOutput());
            auto dets = postProcess(out, img_h, img_w);
            fprintf(stderr, "[offline] %zu detections\n", dets.size());
            for (auto& d : dets)
                fprintf(stderr, "  x1=%.0f y1=%.0f x2=%.0f y2=%.0f cls=%d conf=%.2f\n",
                        d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3],
                        d.class_id, d.confidence);
        }
        return 0;
    }

    // ============= Realtime path ============
    auto source = !frames_dir.empty()
        ? (std::unique_ptr<FrameSource>)std::make_unique<FramesDirSource>(frames_dir, frame_limit > 0 ? frame_limit : -1)
        : (std::unique_ptr<FrameSource>)std::make_unique<V4L2Source>(camera_dev, cam_w, cam_h, "yuyv");
    if (!source->open()) { fprintf(stderr, "[yolo] source open failed\n"); return 1; }

    auto rt = std::make_shared<Runtime>(cmdpool, binary_path, precision);
    // Set buffer backend BEFORE LoadModel — Runtime reads backend_buffer_
    // during LoadModel to decide which op implementations to instantiate.
    // YOLOv8's final tensors have 8400 channels (2100 layers as fp16 image),
    // exceeding Vulkan's maxImageArrayLayers=2048. Maxpool2d self-falls back
    // to image internally (it ignores backend_buffer flag).
    rt->set_backend_buffer(true);
    rt->LoadModel();
    fprintf(stderr, "[yolo] model loaded (realtime, precision=%d, backend=buffer)\n", precision);

    // Probe first frame to know output W/H (renderer extent).
    Frame first;
    bool probe_ok = source->read_frame(first);
    source->close();
    if (!probe_ok) { fprintf(stderr, "[yolo] no frames available\n"); return 1; }
    int out_w = first.w, out_h = first.h;
    source.reset();
    if (!frames_dir.empty())
        source = std::make_unique<FramesDirSource>(frames_dir, frame_limit > 0 ? frame_limit : -1);
    else
        source = std::make_unique<V4L2Source>(camera_dev, cam_w, cam_h, "yuyv");
    source->open();

    // Input tensor dimensions. YOLOv5s exported ONNX has NCHW input [1,3,640,640].
    const int MODEL_H = 640, MODEL_W = 640, MODEL_C = 3;
    auto input_ptr = rt->GetInput();
    auto input_float = std::dynamic_pointer_cast<vkop::core::Tensor<float>>(input_ptr);
    auto input_fp16  = std::dynamic_pointer_cast<vkop::core::Tensor<uint16_t>>(input_ptr);
    auto output_ptr  = rt->GetOutput();
    auto output_float = std::dynamic_pointer_cast<vkop::core::Tensor<float>>(output_ptr);
    auto output_fp16  = std::dynamic_pointer_cast<vkop::core::Tensor<uint16_t>>(output_ptr);

    // Auto-detect dtype from the actual vkopbin, not from a --precision flag.
    // Input dtype and output dtype can differ (mixed-dtype model).
    bool input_is_fp16  = (input_fp16  != nullptr);
    bool output_is_fp16 = (output_fp16 != nullptr);
    fprintf(stderr, "[yolo] auto-detected dtype: input=%s output=%s (in_f16=%d in_f32=%d out_f16=%d out_f32=%d)\n",
            input_is_fp16  ? "fp16" : "fp32",
            output_is_fp16 ? "fp16" : "fp32",
            !!input_fp16, !!input_float, !!output_fp16, !!output_float);
    if (!input_float && !input_fp16) {
        fprintf(stderr, "[yolo] input tensor dtype is neither Tensor<float> nor Tensor<uint16_t>\n");
        return 1;
    }

    // Renderer
    std::unique_ptr<DisplayRenderer> disp;
    std::unique_ptr<OffscreenRenderer> offscr;
    bool window_mode = offscreen_out.empty();
    if (window_mode) {
        disp = std::make_unique<DisplayRenderer>(dev, cmdpool);
        if (!disp->open()) {
            fprintf(stderr, "[yolo] display init failed — pass --offscreen-out path.png\n");
            return 1;
        }
    } else {
        fprintf(stderr, "[yolo] offscreen → %s (%dx%d)\n", offscreen_out.c_str(), out_w, out_h);
        offscr = std::make_unique<OffscreenRenderer>(dev, cmdpool, (uint32_t)out_w, (uint32_t)out_h);
    }

    // ============ Main loop ============
    std::vector<double> ms_capture, ms_pre, ms_run, ms_post, ms_present;
    uint64_t all_t0 = now_us();
    std::vector<uint8_t> last_offscreen_frame;
    size_t offscreen_frames = 0;

    fprintf(stderr, "[yolo] realtime loop started  (--frames %d, --precision %d)\n", frame_limit, precision);

    int frame_idx = 0;
    while (g_running && (frame_limit <= 0 || frame_idx < frame_limit)) {
        if (window_mode && disp->shouldClose()) break;

        uint64_t f0 = now_us();

        // --- Capture ---
        Frame f;
        if (!source->read_frame(f)) break;
        uint64_t t_cap = now_us();

        // --- Preprocess (letterbox + NCHW + /255, writes input tensor host data) ---
        float gain = std::min((float)MODEL_H / f.h, (float)MODEL_W / f.w);
        int new_h = std::clamp((int)(f.h * gain), 1, MODEL_H);
        int new_w = std::clamp((int)(f.w * gain), 1, MODEL_W);
        int pad_top  = (MODEL_H - new_h) / 2;
        int pad_left = (MODEL_W - new_w) / 2;
        size_t inp_count = MODEL_H * MODEL_W * MODEL_C;

        // Use a host-side temporary buffer for preprocess and upload via
        // external pointer — this avoids dereferencing tensor.data() when the
        // tensor's internal data_ unique_ptr is null (LoadModel creates graph
        // input tensors without calling reserveOnCPU, so data_ stays empty).
        if (input_float) {
            std::vector<float> host(inp_count, 114.0f / 255.0f);
            for (int y=0; y<new_h; y++) {
                for (int x=0; x<new_w; x++) {
                    int sy = std::clamp((int)(y/gain), 0, f.h-1);
                    int sx = std::clamp((int)(x/gain), 0, f.w-1);
                    auto* p = f.rgba.data() + (sy*f.w+sx)*4;
                    int oy = pad_top + y, ox = pad_left + x;
                    host[oy*MODEL_W + ox]                            = p[0] / 255.0f;
                    host[MODEL_H*MODEL_W + oy*MODEL_W + ox]          = p[1] / 255.0f;
                    host[2*MODEL_H*MODEL_W + oy*MODEL_W + ox]        = p[2] / 255.0f;
                }
            }
            if (input_float->is_on_GPU()) input_float->toCPU();
            input_float->copyToGPU(cmdpool, host.data());
        } else if (input_fp16) {
            auto f32_to_f16 = [](float v) -> uint16_t {
                uint32_t x; memcpy(&x, &v, 4);
                uint32_t sign = (x>>31)&1, exp = (x>>23)&0xff, man = x&0x7fffff;
                uint16_t out;
                if (exp == 0) out = 0;
                else if (exp == 0xff) out = man ? 0x7e00 : (0x7c00 | (sign<<15));
                else {
                    int e = exp - 127 + 15;
                    if (e < 0) out = sign << 15;
                    else if (e >= 31) out = 0x7c00 | (sign << 15);
                    else out = (sign << 15) | (e << 10) | (man >> 13);
                }
                return out;
            };
            auto gray_half = f32_to_f16(114.0f / 255.0f);
            std::vector<uint16_t> host(inp_count, gray_half);
            for (int y=0; y<new_h; y++) {
                for (int x=0; x<new_w; x++) {
                    int sy = std::clamp((int)(y/gain), 0, f.h-1);
                    int sx = std::clamp((int)(x/gain), 0, f.w-1);
                    auto* p = f.rgba.data() + (sy*f.w+sx)*4;
                    int oy = pad_top + y, ox = pad_left + x;
                    host[oy*MODEL_W + ox]                     = f32_to_f16(p[0] / 255.0f);
                    host[MODEL_H*MODEL_W + oy*MODEL_W + ox]   = f32_to_f16(p[1] / 255.0f);
                    host[2*MODEL_H*MODEL_W + oy*MODEL_W + ox] = f32_to_f16(p[2] / 255.0f);
                }
            }
            if (input_fp16->is_on_GPU()) input_fp16->toCPU();
            input_fp16->copyToGPU(cmdpool, host.data());
        } else {
            fprintf(stderr, "[yolo] input dtype neither float nor fp16 tensor\n");
            return 1;
        }
        uint64_t t_pre = now_us();

        // --- Inference ---
        rt->Run();
        rt->ReadResult();
        uint64_t t_run = now_us();

        // --- postProcess — use output tensor's actual dtype, not --precision ---
        std::vector<Detection> dets;
        if (output_float) {
            dets = postProcess(output_float, f.h, f.w);
        } else if (output_fp16) {
            dets = postProcess(output_fp16, f.h, f.w);
        } else {
            fprintf(stderr, "[yolo] output dtype neither float nor fp16 tensor\n");
            return 1;
        }
        // Draw boxes + class/conf labels directly on the RGBA buffer.
        // Coords are already letterbox-inversed to image-space by postProcess.
        for (auto& d : dets) {
            auto& c = kBoxColors[d.class_id % 20];
            int x1=(int)d.bbox[0], y1=(int)d.bbox[1];
            int x2=(int)d.bbox[2], y2=(int)d.bbox[3];
            drawBox(f.rgba.data(), f.w, f.h, x1, y1, x2, y2, c[0], c[1], c[2], 2);
            drawLabel(f.rgba.data(), f.w, f.h, x1, y1, x2, y2,
                      d.class_id, d.confidence, c[0], c[1], c[2]);
        }
        uint64_t t_post = now_us();

        // --- Upload + present ---
        auto tex = uploadTexture(dev, cmdpool, f.rgba.data(), f.w, f.h);
        if (!tex) break;
        if (window_mode) {
            disp->present(tex);
        } else {
            if (offscr->render(tex, last_offscreen_frame))
                offscreen_frames++;
        }
        uint64_t t_pres = now_us();

        ms_capture.push_back((t_cap - f0) * 1e-3);
        ms_pre.push_back((t_pre - t_cap) * 1e-3);
        ms_run.push_back((t_run - t_pre) * 1e-3);
        ms_post.push_back((t_post - t_run) * 1e-3);
        ms_present.push_back((t_pres - t_post) * 1e-3);

        frame_idx++;
        if (frame_idx % 10 == 0) {
            double fps = frame_idx * 1e6 / (now_us() - all_t0);
            fprintf(stderr, "[yolo] frame %d  fps=%.1f  last-frame %.1f ms  dets=%zu\n",
                    frame_idx, fps, (t_pres - f0)*1e-3, dets.size());
        }
    }

    // ============ Summary ============
    auto avg = [](const std::vector<double>& v) {
        if (v.empty()) return 0.0;
        double s = 0; for (auto x : v) s += x;
        return s / v.size();
    };
    uint64_t dur_us = now_us() - all_t0;
    double fps = ms_run.size() > 0 ? ms_run.size() * 1e6 / dur_us : 0;

    fprintf(stderr,
        "\n=== Realtime Summary ===\n"
        "frames:        %zu\n"
        "duration:      %.2f s\n"
        "avg FPS:       %.1f\n"
        "avg per-frame: %.2f ms\n"
        "  capture:     %.2f ms\n"
        "  preprocess:  %.2f ms\n"
        "  inference:   %.2f ms\n"
        "  postprocess: %.2f ms\n"
        "  present:     %.2f ms\n",
        ms_run.size(), dur_us * 1e-6, fps,
        avg(ms_capture)+avg(ms_pre)+avg(ms_run)+avg(ms_post)+avg(ms_present),
        avg(ms_capture), avg(ms_pre), avg(ms_run), avg(ms_post), avg(ms_present));

    // Write offscreen PNG + box-pixel count
    if (!offscreen_out.empty() && !last_offscreen_frame.empty()) {
        fprintf(stderr, "[yolo] writing offscreen frame %zu (%dx%d) → %s\n",
                offscreen_frames, out_w, out_h, offscreen_out.c_str());
        if (!stbi_write_png(offscreen_out.c_str(), out_w, out_h, 4,
                            last_offscreen_frame.data(), out_w * 4)) {
            fprintf(stderr, "[yolo] stbi_write_png failed\n");
        } else {
            int red_box_pixels = 0;
            for (size_t p = 0; p + 3 < last_offscreen_frame.size(); p += 4) {
                uint8_t r = last_offscreen_frame[p];
                uint8_t g = last_offscreen_frame[p+1];
                uint8_t b = last_offscreen_frame[p+2];
                // Count pixels that are strong-red (our first box color).
                if (r > 200 && g < 150 && b < 150) red_box_pixels++;
            }
            fprintf(stderr, "[yolo] %d red-ish box-color pixels counted in PNG\n", red_box_pixels);
        }
    }

    source->close();
    dev->wait_all_done();
    return 0;
}
