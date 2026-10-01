// junka @ 2026
// Minimal VAE-decoder driver: loads vae_decoder_512.vkopbin, decodes a raw
// fp32 latent [1,64,1,32,32] and writes the raw fp32 decoded image plus a PNG
// next to it (same path, .png extension).
//
// Usage: vae_gen <vae.vkopbin> <latent.raw> <decoded_out.raw> [num_elems]
//
// Build: cmake -DENABLE_IMAGE_GEN=ON .. && make vae_gen

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <vector>
#include <string>

#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "vulkan/VulkanCommandPool.hpp"
#include "core/Tensor.hpp"
#include "core/runtime.hpp"

using vkop::VulkanInstance;
using vkop::VulkanDevice;
using vkop::VulkanCommandPool;
using vkop::core::ITensor;
using vkop::core::Runtime;
using vkop::core::as_tensor;

// 8-bit RGB/RGBA PNG writer. Deflate uses stored (uncompressed) blocks, so the
// only compression code needed is the zlib/adler wrapper -- no libpng/stb link.
namespace png {

static uint32_t crc(const uint8_t* p, size_t n) {
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xedb88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
        init = true;
    }
    uint32_t c = 0xffffffffu;
    for (size_t i = 0; i < n; ++i) c = table[(c ^ p[i]) & 0xff] ^ (c >> 8);
    return c ^ 0xffffffffu;
}

static void chunk(FILE* f, const char* type, const std::vector<uint8_t>& data) {
    uint8_t hdr[8];
    uint32_t len = (uint32_t)data.size();
    hdr[0] = len >> 24; hdr[1] = len >> 16; hdr[2] = len >> 8; hdr[3] = len;
    memcpy(hdr + 4, type, 4);
    fwrite(hdr, 1, 8, f);
    if (!data.empty()) fwrite(data.data(), 1, data.size(), f);
    std::vector<uint8_t> body(4 + data.size());
    memcpy(body.data(), type, 4);
    if (!data.empty()) memcpy(body.data() + 4, data.data(), data.size());
    uint32_t c = crc(body.data(), body.size());
    uint8_t tail[4] = { uint8_t(c >> 24), uint8_t(c >> 16), uint8_t(c >> 8), uint8_t(c) };
    fwrite(tail, 1, 4, f);
}

// rows: H scanlines of W*channels bytes, already in image order.
static bool write(FILE* f, int w, int h, int channels, const uint8_t* rgb) {
    if (channels != 3 && channels != 4) return false;
    const size_t stride = (size_t)w * channels;
    std::vector<uint8_t> raw;
    raw.reserve((size_t)h * (stride + 1));
    for (int y = 0; y < h; ++y) {
        raw.push_back(0);  // filter: none
        const uint8_t* row = rgb + (size_t)y * stride;
        raw.insert(raw.end(), row, row + stride);
    }
    uint32_t a = 1, b = 0;
    for (uint8_t v : raw) { a = (a + v) % 65521u; b = (b + a) % 65521u; }
    std::vector<uint8_t> idat;
    idat.push_back(0x78); idat.push_back(0x01);
    for (size_t off = 0;; off += 65535) {
        size_t n = std::min<size_t>(65535, raw.size() - off);
        idat.push_back(off + n >= raw.size() ? 1 : 0);
        idat.push_back(n & 0xff); idat.push_back(n >> 8);
        idat.push_back(~n & 0xff); idat.push_back(~(n >> 8) & 0xff);
        idat.insert(idat.end(), raw.begin() + off, raw.begin() + off + n);
        if (off + n >= raw.size()) break;
    }
    uint32_t adler = (b << 16) | a;
    for (int i = 0; i < 4; ++i)
        idat.push_back(uint8_t(adler >> (24 - 8 * i)));
    uint8_t sig[8] = { 0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a };
    fwrite(sig, 1, 8, f);
    std::vector<uint8_t> ihdr(13);
    ihdr[0] = w >> 24; ihdr[1] = w >> 16; ihdr[2] = w >> 8; ihdr[3] = w;
    ihdr[4] = h >> 24; ihdr[5] = h >> 16; ihdr[6] = h >> 8; ihdr[7] = h;
    ihdr[8] = 8; ihdr[9] = (uint8_t)(channels == 4 ? 6 : 2);
    chunk(f, "IHDR", ihdr);
    chunk(f, "IDAT", idat);
    chunk(f, "IEND", {});
    return true;
}

}  // namespace png

// decoded NCHW fp32 in [-1,1] -> 8-bit HWC. The 4th channel is kept as alpha.
static bool to_png(const std::vector<float>& planar, int c, int h, int w,
                   const char* path) {
    std::vector<uint8_t> rgb((size_t)h * w * c);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int q = 0; q < c; ++q) {
                float v = planar[((size_t)q * h + y) * w + x];
                float u = (v + 1.0f) * 0.5f * 255.0f;
                rgb[((size_t)y * w + x) * c + q] =
                    (uint8_t)std::min<float>(255.0f, std::max<float>(0.0f, u));
            }
        }
    }
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    bool ok = png::write(f, w, h, c, rgb.data());
    fclose(f);
    return ok;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <vae.vkopbin> <latent.raw> <out.raw> [n]\n", argv[0]);
        return 1;
    }
    const char* model_path = argv[1];
    const char* latent_path = argv[2];
    const char* out_path = argv[3];
    int n = argc > 4 ? atoi(argv[4]) : 64 * 32 * 32;

    std::vector<float> latent(n);
    FILE* f = fopen(latent_path, "rb");
    if (!f) { fprintf(stderr, "cannot read %s\n", latent_path); return 1; }
    if (fread(latent.data(), sizeof(float), n, f) != (size_t)n) {
        fprintf(stderr, "short read of %s (expected %d floats)\n", latent_path, n);
        return 1;
    }
    fclose(f);

    const auto& phydevs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
    if (phydevs.empty()) { fprintf(stderr, "no vulkan device\n"); return 1; }
    auto device = std::make_shared<VulkanDevice>(phydevs[0]);
    auto cmdpool = std::make_shared<VulkanCommandPool>(device);

    printf("[load] %s\n", model_path);
    auto rt = std::make_shared<Runtime>(cmdpool, model_path, /*precision=*/1);
    // Buffer (SSBO) backend by default: the VAE mixes Resize with the
    // SSBO-only shape ops (Expand/Squeeze/...), and only the buffer backend
    // covers both. VKOP_BUFFER_BACKEND=0 selects the image backend.
    rt->set_backend_buffer(getenv("VKOP_BUFFER_BACKEND") == nullptr ||
                           getenv("VKOP_BUFFER_BACKEND")[0] != '0');
    auto t0 = std::chrono::steady_clock::now();
    rt->LoadModel();
    printf("[time] load %.2fs\n",
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
               .count());

    auto input = rt->GetInput("latent");
    if (!input) { fprintf(stderr, "no 'latent' input\n"); return 1; }
    auto shp = input->getShape();
    printf("[io] latent shape:");
    for (size_t d = 0; d < shp.size(); ++d) printf(" %u", shp[d]);
    printf(" dtype=%s\n", input->dtype() == typeid(float) ? "fp32" : "fp16/other");

    if (input->dtype() == typeid(float)) {
        auto tt = as_tensor<float>(input);
        tt->fillToCPU(latent.data());
        tt->copyToGPU(cmdpool);
    } else {
        std::vector<uint16_t> h(n);
        for (int i = 0; i < n; ++i) h[i] = ITensor::fp32_to_fp16(latent[i]);
        auto tt = as_tensor<uint16_t>(input);
        tt->fillToCPU(h.data());
        tt->copyToGPU(cmdpool);
    }

    printf("[run] decoding...\n");
    auto trun = std::chrono::steady_clock::now();
    rt->Run();
    rt->ReadResult();
    printf("[time] run %.2fs\n",
           std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         trun)
               .count());

    auto output = rt->GetOutput("decoded");
    if (!output) { fprintf(stderr, "no 'decoded' output\n"); return 1; }
    auto oshp = output->getShape();
    int on = 1;
    for (size_t d = 0; d < oshp.size(); ++d) on *= (int)oshp[d];
    printf("[io] decoded shape:");
    for (size_t d = 0; d < oshp.size(); ++d) printf(" %u", oshp[d]);
    printf(" (%d elems)\n", on);

    std::vector<float> out(on);
    if (output->dtype() == typeid(float)) {
        auto tt = as_tensor<float>(output);
        tt->copyToCPU(cmdpool);
        const auto& d = tt->data();
        for (int i = 0; i < on && i < (int)d.size(); ++i) out[i] = d[i];
    } else {
        auto tt = as_tensor<uint16_t>(output);
        tt->copyToCPU(cmdpool);
        const auto& d = tt->data();
        for (int i = 0; i < on && i < (int)d.size(); ++i)
            out[i] = ITensor::fp16_to_fp32(d[i]);
    }

    FILE* g = fopen(out_path, "wb");
    if (!g) { fprintf(stderr, "cannot write %s\n", out_path); return 1; }
    fwrite(out.data(), sizeof(float), out.size(), g);
    fclose(g);
    printf("[done] wrote %s (%zu B)\n", out_path, out.size() * sizeof(float));

    // [N, C, T, H, W] with T == 1 -> 8-bit PNG (4-channel decodes as RGBA).
    const int ch = (int)oshp[1];
    const int ph = (int)oshp[oshp.size() - 2];
    const int pw = (int)oshp[oshp.size() - 1];
    const int pt = (int)(oshp.size() > 3 ? oshp[2] : 1);
    if (pt != 1) {
        printf("[png] skipped: temporal dim %d != 1\n", pt);
        return 0;
    }
    std::string png_path(out_path);
    size_t dot = png_path.rfind('.');
    if (dot != std::string::npos) png_path = png_path.substr(0, dot);
    png_path += ".png";
    if (!to_png(out, ch, ph, pw, png_path.c_str())) {
        fprintf(stderr, "[png] failed to write %s\n", png_path.c_str());
        return 1;
    }
    printf("[png] wrote %s (%d x %d, %d ch)\n", png_path.c_str(), pw, ph, ch);
    return 0;
}
