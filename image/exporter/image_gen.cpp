// junka @ 2026
// Image generation driver for Qwen-Image-2.1 (text tower + DiT + VAE decoder).
// Text prompt -> tokenized -> text tower -> DiT prefill -> denoising loop ->
// VAE decode -> PNG. Four graphs are loaded one at a time and released between
// stages (each DiT/text-tower graph is ~14 GB; two never fit in 36 GB).
//
// Usage:
//   image_gen <dit_prefill.vkopbin> <dit_decode.vkopbin> <prompt>
//             [steps] [seed] [--size 512|1024] [--ref DIR]
//
// Without --ref everything is computed here; the text tower's three artifacts
// and the VAE graph come from the env vars listed in the usage error below.
// Build: cmake -DENABLE_IMAGE_GEN=ON .. && make image_gen

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <chrono>
#include <random>
#include <fstream>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "vulkan/VulkanCommandPool.hpp"
#include "core/Tensor.hpp"
#include "core/runtime.hpp"
#include "model/load.hpp"
#include "tokenizer.hpp"

// ---- Minimal PNG writer (from vae_gen.cpp) ----
namespace png {
static uint32_t crc(const uint8_t* p, size_t n) {
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k)
                c = (c & 1) ? 0xedb88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
        init = true;
    }
    uint32_t c = 0xffffffffu;
    for (size_t i = 0; i < n; ++i)
        c = table[(c ^ p[i]) & 0xff] ^ (c >> 8);
    return c ^ 0xffffffffu;
}
static void chunk(FILE* f, const char* type,
                  const std::vector<uint8_t>& data) {
    uint8_t hdr[8];
    uint32_t len = (uint32_t)data.size();
    hdr[0] = len >> 24;
    hdr[1] = len >> 16;
    hdr[2] = len >> 8;
    hdr[3] = len;
    memcpy(hdr + 4, type, 4);
    fwrite(hdr, 1, 8, f);
    if (!data.empty())
        fwrite(data.data(), 1, data.size(), f);
    std::vector<uint8_t> body(4 + data.size());
    memcpy(body.data(), type, 4);
    if (!data.empty())
        memcpy(body.data() + 4, data.data(), data.size());
    uint32_t c = crc(body.data(), body.size());
    uint8_t tail[4] = {(uint8_t)(c >> 24), (uint8_t)(c >> 16),
                       (uint8_t)(c >> 8), (uint8_t)c};
    fwrite(tail, 1, 4, f);
}
static bool write_png(FILE* f, int w, int h, int channels,
                      const uint8_t* rgb) {
    if (channels != 3 && channels != 4)
        return false;
    const size_t stride = (size_t)w * channels;
    std::vector<uint8_t> raw;
    raw.reserve((size_t)h * (stride + 1));
    for (int y = 0; y < h; ++y) {
        raw.push_back(0);
        const uint8_t* row = rgb + (size_t)y * stride;
        raw.insert(raw.end(), row, row + stride);
    }
    uint32_t a = 1, b = 0;
    for (uint8_t v : raw) {
        a = (a + v) % 65521u;
        b = (b + a) % 65521u;
    }
    std::vector<uint8_t> idat;
    idat.push_back(0x78);
    idat.push_back(0x01);
    for (size_t off = 0;; off += 65535) {
        size_t n = std::min<size_t>(65535, raw.size() - off);
        idat.push_back(off + n >= raw.size() ? 1 : 0);
        idat.push_back(n & 0xff);
        idat.push_back(n >> 8);
        idat.push_back(~n & 0xff);
        idat.push_back(~(n >> 8) & 0xff);
        idat.insert(idat.end(), raw.begin() + off,
                    raw.begin() + off + n);
        if (off + n >= raw.size())
            break;
    }
    uint32_t adler = (b << 16) | a;
    for (int i = 0; i < 4; ++i)
        idat.push_back(uint8_t(adler >> (24 - 8 * i)));
    uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
    fwrite(sig, 1, 8, f);
    std::vector<uint8_t> ihdr(13);
    ihdr[0] = w >> 24;
    ihdr[1] = w >> 16;
    ihdr[2] = w >> 8;
    ihdr[3] = w;
    ihdr[4] = h >> 24;
    ihdr[5] = h >> 16;
    ihdr[6] = h >> 8;
    ihdr[7] = h;
    ihdr[8] = 8;
    ihdr[9] = (uint8_t)(channels == 4 ? 6 : 2);
    chunk(f, "IHDR", ihdr);
    chunk(f, "IDAT", idat);
    chunk(f, "IEND", {});
    return true;
}
} // namespace png

// Decoded NCHW fp32 in [-1,1] -> 8-bit HWC PNG. Takes first 3 of 4 channels.
static bool save_latent_as_png(const std::vector<float>& planar_4ch, int h,
                               int w, const char* path) {
    // planar_4ch: [4, H, W] in NCHW order, values in [-1, 1]
    std::vector<uint8_t> rgb((size_t)h * w * 3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int q = 0; q < 3; ++q) { // RGB only, skip alpha
                float v = planar_4ch[((size_t)q * h + y) * w + x];
                float u = (v + 1.0f) * 0.5f * 255.0f;
                rgb[((size_t)y * w + x) * 3 + q] =
                    (uint8_t)std::min<float>(
                        255.0f, std::max<float>(0.0f, u));
            }
        }
    }
    FILE* f = fopen(path, "wb");
    if (!f)
        return false;
    bool ok = png::write_png(f, w, h, 3, rgb.data());
    fclose(f);
    return ok;
}

using vkop::VulkanInstance;
using vkop::VulkanDevice;
using vkop::VulkanCommandBuffer;
using vkop::VulkanCommandPool;
using vkop::core::ITensor;
using vkop::core::Runtime;
using vkop::core::as_tensor;

namespace {

// 官方 QwenImage21Pipeline 的固定系统提示词，模板的 system 段就是它。
constexpr const char* kSystemPrompt =
    "Comprehend and analyze the provided prompt.";

// ---- DiT 的 σ 调度与 joint rope ----
// 这两张表原来只在 gen_dit_ref.py 里算（--schedule diffusers / --real-rope），
// 驱动必须靠 --ref 目录才能拿到。搬进 C++ 之后纯 vkop 出图不再需要任何 torch
// 侧产物。常量逐字来自权重目录：transformer/config.json 的
// axes_dims_rope=[16,56,56]、attention_head_dim=128，和
// scheduler/scheduler_config.json 的 base/max_image_seq_len、base/max_shift、
// shift_terminal。
constexpr float kRopeTheta = 10000.0f;
constexpr int kRopeHeadDim = 128;
constexpr int kAxesRope[3] = {16, 56, 56};  // 三轴维度之和 = head_dim
constexpr int kSchedBaseSeq = 256, kSchedMaxSeq = 8192;
constexpr double kSchedBaseShift = 0.5, kSchedMaxShift = 0.9, kSchedTerminal = 0.02;

// 官方调度：sigmas = linspace(1, 1/N, N) 经 exponential 动态位移（mu 由图像段
// 长度线性插值得到），再整体拉伸到 shift_terminal，末尾补一个 0（长度 steps+1）。
// 全程 double。diffusers 那侧 set_timesteps 先 `astype(np.float32)`、后续每一步
// 都在 float32 数组上做（NEP 50 下 python 标量不升精度），所以两边不是逐位相同：
// 40 步表实测 41 个值里 23 个差 1 个 fp32 ulp（max 5.96e-8），末位 σ 差 2e-8
// —— 那个位置 python 侧是 `1 - 0.975/0.99487` 这种抵消运算，float32 下丢掉十几个
// ulp，本函数的 double 结果反而更接近 shift_terminal 本身。要逐位对齐就走 --ref
// 读 sigmas.raw；这里要的是"同一条公式、误差可解释"。
std::vector<float> official_sigmas(int steps, int target_len) {
    const double m =
        (kSchedMaxShift - kSchedBaseShift) / (double)(kSchedMaxSeq - kSchedBaseSeq);
    const double mu = (double)target_len * m + (kSchedBaseShift - m * kSchedBaseSeq);
    const double e = std::exp(mu);
    const double stop = 1.0 / (double)steps;
    std::vector<double> s(steps);
    // numpy linspace(1, 1/N, N)：步长是 (stop-start)/(N-1)，末点精确等于 stop；
    // N==1 时它返回 [start]，没有可除的 N-1。
    // 少除那个 N-1 会让 σ 跑到 1 以上、甚至负数，整条调度就废了。
    for (int k = 0; k < steps; ++k)
        s[k] = steps == 1 ? 1.0 : 1.0 + (stop - 1.0) * k / (steps - 1);
    s.back() = stop;
    // 括号是舍入意义上的，不是可读性：numpy 侧先算 `(1/t - 1)` 再加 exp(mu)，
    // 写成 e + 1/v - 1 会让 σ0 = 1 变成 0.9999999999999999，而下一步的 scale 正是
    // 拿 (1 - σ_last) 当分子 —— 那 1 个 ulp 会被放大成整条曲线塌到 shift_terminal。
    for (double& v : s) v = e / (e + (1.0 / v - 1.0));
    // stretch_shift_to_terminal：把曲线重新拉到 σ_last = shift_terminal。
    // steps==1 时唯一的 σ 位移后仍精确等于 1，分子为 0（diffusers 在这里是
    // 0/0 = NaN），一步也没有可拉伸的区间，跳过。
    if (steps > 1) {
        const double scale = (1.0 - s.back()) / (1.0 - kSchedTerminal);
        for (double& v : s) v = 1.0 - (1.0 - v) / scale;
    }
    std::vector<float> out((size_t)steps + 1, 0.0f);
    for (int k = 0; k < steps; ++k) out[k] = (float)s[k];
    printf("[sched] mu=%.6f σ=[%.6f … %.6f] +终值 0（%d 步）\n", mu, out[0],
           out[steps - 1], steps);
    return out;
}

// joint rope 表，(txt_len + lh*lw, head_dim) 的 fp32 cos/sin。文本段三轴同值
// 0..txt_len-1；图像段 frame 轴整块冻结在 txt_len，h/w 轴取以 0 为中心的网格
// -(n-n/2)..n/2-1（行主序 t = h*lw + w）。每行是 cat(ang, ang)，即 cos[d] ==
// cos[d+hd/2] —— rotate_half 融合成 vkop RotaryEmbedding 后要的就是这个铺满
// 形态，而不是复数表的 (hd/2,) 半行。
void joint_rope(int txt_len, int lh, int lw, std::vector<float>& cos_out,
                std::vector<float>& sin_out) {
    const int rows = txt_len + lh * lw, hd = kRopeHeadDim, half = hd / 2;
    int first[3], cnt[3];
    float inv[3][kRopeHeadDim];
    for (int a = 0, col = 0; a < 3; ++a) {
        first[a] = col;
        cnt[a] = kAxesRope[a] / 2;  // 每轴贡献 dim/2 个频率对
        for (int i = 0; i < cnt[a]; ++i, ++col)
            inv[a][col] =
                1.0f / std::pow(kRopeTheta, (float)(2 * i) / (float)kAxesRope[a]);
        assert(col == first[a] + cnt[a]);
    }
    cos_out.assign((size_t)rows * hd, 0.0f);
    sin_out.assign((size_t)rows * hd, 0.0f);
    for (int r = 0; r < rows; ++r) {
        float pos[3];
        if (r < txt_len) {
            pos[0] = pos[1] = pos[2] = (float)r;
        } else {
            const int t = r - txt_len;
            pos[0] = (float)txt_len;
            pos[1] = (float)(t / lw - (lh - lh / 2));
            pos[2] = (float)(t % lw - (lw - lw / 2));
        }
        for (int a = 0; a < 3; ++a)
            for (int i = 0; i < cnt[a]; ++i) {
                const int c = first[a] + i;
                const float ang = pos[a] * inv[a][c];
                cos_out[(size_t)r * hd + c] = cos_out[(size_t)r * hd + half + c] =
                    std::cos(ang);
                sin_out[(size_t)r * hd + c] = sin_out[(size_t)r * hd + half + c] =
                    std::sin(ang);
            }
    }
}

// 取 DiT 要的两段表，与 gen_dit_ref.py 的 cos_full 拼法逐位同：
//   文本行取自 txt_len=prefix_len 的表（补零行也占位，位置连着排），
//   图像行的 frame 轴按**真实** token 数 valid_len 冻结 —— 官方没有 padding，
//   图像段的位置不能跟着补零一起往后挪。
// 表以 fp16 交付（图的 cos/sin 输入就是 fp16）：先在 fp32 里算 cos/sin 再一次
// 舍入，和 numpy 的 `.astype(np.float16)` 同口径；逐步每行都重算既浪费也容易和
// 参考差出 ulp。
struct RopeTables {
    std::vector<uint16_t> cos_text, sin_text, cos_img, sin_img;  // [rows, head_dim]
};

RopeTables build_rope(int prefix_len, int valid_len, int lh, int lw) {
    auto rows16 = [](const std::vector<float>& f, size_t first, size_t last) {
        std::vector<uint16_t> o((last - first) * kRopeHeadDim);
        for (size_t i = 0; i < o.size(); ++i)
            o[i] = ITensor::fp32_to_fp16(f[(first * kRopeHeadDim) + i]);
        return o;
    };
    const size_t n_img = (size_t)lh * lw;
    RopeTables t;
    std::vector<float> cos_full, sin_full;
    joint_rope(prefix_len, lh, lw, cos_full, sin_full);
    t.cos_text = rows16(cos_full, 0, (size_t)prefix_len);
    t.sin_text = rows16(sin_full, 0, (size_t)prefix_len);
    // 图像行 = 以 valid_len 为 frame 的整张表去掉前 valid_len 行。
    joint_rope(valid_len, lh, lw, cos_full, sin_full);
    t.cos_img = rows16(cos_full, (size_t)valid_len, valid_len + n_img);
    t.sin_img = rows16(sin_full, (size_t)valid_len, valid_len + n_img);
    return t;
}

// ---- VAE latent de-normalization (vae/config.json, z_dim=64) ----
// DiT 工作在归一化潜空间，VAE decoder 工作在原始潜空间。diffusers 的
// QwenImagePipeline 在 vae.decode 前做逐通道 latents = latents * std + mean
// （源码里写成 latents / (1/std) + mean），少了这一步解码出来是彩色噪声块。
static const float kLatentsMean[64] = {
    0.512600f, 0.772100f, -0.063100f, 1.350600f, -0.785500f, -2.102500f, -0.345800f, 1.372200f,
    1.887300f, -1.717700f, -0.651000f, 0.273200f, 0.756200f, -0.616300f, -1.027700f, 3.836300f,
    2.021000f, 0.047200f, 0.932000f, 2.008700f, 2.495400f, -0.139100f, -1.424900f, 1.846400f,
    -0.523600f, 1.282600f, 3.704600f, -1.303500f, 2.728600f, -1.451800f, -1.903600f, -1.995500f,
    -0.034200f, -1.026500f, -0.763600f, 3.055500f, 0.074600f, -3.075100f, -0.107600f, 1.737600f,
    -1.091400f, -1.943500f, -0.278400f, -1.368000f, 0.480900f, -0.443300f, 0.376400f, 0.572900f,
    -2.059500f, 1.096000f, -1.326000f, -2.021100f, -5.017900f, 0.527500f, 4.016200f, 1.850500f,
    0.302600f, 1.937300f, 1.493700f, 0.263200f, 0.554700f, -1.712100f, -0.156200f, 0.030400f,
};
static const float kLatentsStd[64] = {
    3.200100f, 3.293600f, 3.432100f, 3.009100f, 3.106100f, 4.037900f, 4.070500f, 3.791000f,
    3.078500f, 3.650000f, 3.930800f, 3.090400f, 2.877800f, 3.767500f, 3.732000f, 5.075600f,
    3.286400f, 4.039700f, 3.131700f, 4.044300f, 2.924900f, 3.945400f, 3.098800f, 4.248900f,
    3.489600f, 3.851300f, 3.932300f, 3.471900f, 3.749800f, 4.283000f, 3.569400f, 4.246700f,
    3.903700f, 3.294700f, 5.077000f, 3.507500f, 3.270000f, 3.476700f, 2.806300f, 5.112500f,
    3.532700f, 4.783300f, 3.128600f, 4.181900f, 3.852700f, 3.831200f, 3.560500f, 4.387500f,
    3.962400f, 4.016800f, 3.564300f, 4.055000f, 5.561400f, 4.296300f, 4.408000f, 3.495900f,
    3.874700f, 3.760800f, 3.573500f, 3.149000f, 3.766200f, 3.674600f, 3.456300f, 3.816100f,
};

// ---- Random latent generator ----
void generate_random_latent(float* data, int numel, uint32_t seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < numel; ++i) {
        data[i] = dist(gen);
    }
}

// ---- Save raw fp32 tensor to file ----
void save_raw(const char* path, const float* data, int numel) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "Cannot open %s for writing\n", path);
        return;
    }
    fwrite(data, sizeof(float), numel, f);
    fclose(f);
    printf("[save] Saved %d floats (%.1f MB) to %s\n", 
           numel, numel * 4.0f / 1e6f, path);
}

// ---- ORT reference alignment (--ref DIR) ----
// Load a binary file into a byte vector.
bool read_file_bytes(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    out.resize(f.tellg());
    f.seekg(0);
    f.read(reinterpret_cast<char*>(out.data()), out.size());
    return true;
}

struct RefEntry {
    std::vector<uint32_t> dims;
    int elem_size = 0;
    std::vector<uint8_t> bytes;
};

// Parse DIR/shapes.txt ("<name> d0xd1x... <elem_size>") and slurp every file.
bool load_ref_dir(const std::string& dir,
                  std::map<std::string, RefEntry>& ref) {
    std::ifstream shapes(dir + "/shapes.txt");
    if (!shapes) {
        fprintf(stderr, "[ref] missing %s/shapes.txt\n", dir.c_str());
        return false;
    }
    std::string name, dimstr;
    int esz;
    while (shapes >> name >> dimstr >> esz) {
        RefEntry e;
        e.elem_size = esz;
        size_t pos = 0;
        while (pos < dimstr.size()) {
            size_t nxt = dimstr.find('x', pos);
            if (nxt == std::string::npos) nxt = dimstr.size();
            e.dims.push_back(std::stoul(dimstr.substr(pos, nxt - pos)));
            pos = nxt + 1;
        }
        if (!read_file_bytes(dir + "/" + name, e.bytes)) {
            fprintf(stderr, "[ref] missing file %s/%s\n", dir.c_str(), name.c_str());
            return false;
        }
        ref[name] = std::move(e);
    }
    printf("[ref] loaded %zu tensors from %s\n", ref.size(), dir.c_str());
    return true;
}

// ---- 文本塔：prompt -> prompt_embeds（Qwen3-VL 7B，13.9 GB 权重）----
// 复刻 encode_prompt_real.py 的纯文生图前处理链：raw ChatML 模板串（**不走**
// tokenizer 的 chat template，两者分词不同）、丢掉开头 drop_idx 个 system 段
// token、截到 valid_len 后右补**精确零**到 prefix_len。
//
// 补零行在两个地方各有讲究：文本塔的图长 P = drop_idx + prefix_len，第
// drop_idx+valid_len 行往后喂的是零，因果 mask 保证它们污染不到真实行（真实行
// 只 attend ≤ 自己）；而 DiT 的文本 KV 是**双向**消费这 64 行的，所以那
// prefix_len-valid_len 列必须在 decode 的 attention_bias 里屏蔽掉，否则零向量会
// 当成 key 进图像段的注意力。
struct PromptEmbeds {
    std::vector<uint16_t> data;  // [prefix_len, ctx_dim] fp16
    int prefix_len = 0;
    int ctx_dim = 0;
    int valid_len = 0;
};

// token_id -> embedding 的 fp16 表（[vocab, ctx_dim]，1.24 GB）。mmap 而不读进
// 内存：一条 prompt 只碰其中几十页，全量 fread 会把 RSS 白涨一个多 G。
class EmbedTable {
public:
    ~EmbedTable() { reset(); }
    bool open(const std::string& path, int ctx_dim) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            fprintf(stderr, "[te] cannot open %s\n", path.c_str());
            return false;
        }
        struct stat st;
        if (fstat(fd_, &st) != 0 || st.st_size <= 0) {
            fprintf(stderr, "[te] cannot stat %s\n", path.c_str());
            reset();
            return false;
        }
        size_ = (size_t)st.st_size;
        if (size_ % (2 * (size_t)ctx_dim) != 0) {
            fprintf(stderr, "[te] %s 的大小 %zu 不是 ctx_dim=%d 的整数倍\n",
                    path.c_str(), size_, ctx_dim);
            reset();
            return false;
        }
        void* p = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (p == MAP_FAILED) {
            fprintf(stderr, "[te] mmap %s failed\n", path.c_str());
            reset();
            return false;
        }
        base_ = (const uint16_t*)p;
        cols_ = ctx_dim;
        rows_ = (int)(size_ / (2 * (size_t)ctx_dim));
        printf("[te] embed table %s: %d x %d fp16 (%.2f GB, mmap)\n",
               path.c_str(), rows_, cols_, size_ / 1e9);
        return true;
    }
    int rows() const { return rows_; }
    const uint16_t* row(int id) const { return base_ + (size_t)id * cols_; }

private:
    void reset() {
        if (base_) munmap((void*)base_, size_);
        if (fd_ >= 0) ::close(fd_);
        base_ = nullptr;
        fd_ = -1;
    }
    size_t size_ = 0;
    int rows_ = 0, cols_ = 0, fd_ = -1;
    const uint16_t* base_ = nullptr;
};

// 模板里那两个 ChatML 边界符的字面量。bin 里没注册就返回 false —— 少一个特殊
// token 的话分词结果会整段错位，绝不能拿裸字符串继续算。
bool chatml_tags(const qwen::Tokenizer& tok, std::string& im_start,
                 std::string& im_end) {
    // 停止符表里挑出真正是 ChatML 轮末标签的那一个：Phi/GLM 的表里没有它，
    // 这里就返回 false，图像塔的模板绝不能拿裸字符串凑。
    for (const int32_t end : tok.stop_token_ids()) {
        const std::string end_piece = tok.id_to_piece(static_cast<uint32_t>(end), false);
        // im_start 没有具名入口（LLM 只关心终止符）；它的 id 是 im_end 的前一个，
        // 这是 Qwen 词表里唯一一处序号依赖。
        const std::string start_piece =
            tok.id_to_piece(static_cast<uint32_t>(end - 1), false);
        if (start_piece.find("im_start") == std::string::npos ||
            end_piece.find("im_end") == std::string::npos) continue;
        im_end = end_piece;
        im_start = start_piece;
        return true;
    }
    return false;
}

// 官方 pipeline 的纯文生图模板（和 encode_prompt_real.py::build_template 一字
// 同源）。system_only 那份单独可取 —— drop_idx 是它的 token 数，也就是"模板里
// 属于 system 段的那几行"，切 embeds 时整段丢掉。
std::string chatml_system(const std::string& im_start,
                          const std::string& im_end) {
    return im_start + "system\n" + kSystemPrompt + im_end + "\n";
}

std::string chatml_template(const std::string& im_start,
                            const std::string& im_end, const char* prompt) {
    return chatml_system(im_start, im_end) + im_start + "user\n" +
           (prompt ? prompt : "") + im_end + "\n" + im_start + "assistant\n";
}

// 跑一次文本塔，产出 DiT prefill 要的 [prefix_len, ctx_dim]。图长 P 与 ctx_dim
// 都从 vkopbin 的头里读、不写死：换 prefix_len 重导之后这里不用改，而文本塔和 DiT
// 两张图对不对得上由调用方按 prefill 的实际形状把关。
bool encode_prompt(const std::string& te_path, const std::string& table_path,
                   const std::string& tok_path, const char* prompt,
                   const std::shared_ptr<VulkanCommandPool>& cmdpool,
                   const std::shared_ptr<VulkanDevice>& dev, PromptEmbeds& out) {
    vkop::load::VkModel hdr(te_path.c_str());
    int P = 0, ctx_dim = 0;
    for (const auto& s : hdr.inputs)
        if (s.name == "inputs_embeds" && s.dims.size() == 3) {
            P = s.dims[1];
            ctx_dim = s.dims[2];
        }
    if (P <= 0 || ctx_dim <= 0) {
        fprintf(stderr, "[te] %s 里没有 3 维的 inputs_embeds\n", te_path.c_str());
        return false;
    }
    printf("[te] graph %s: P=%d ctx_dim=%d\n", te_path.c_str(), P, ctx_dim);

    EmbedTable table;
    if (!table.open(table_path, ctx_dim)) return false;

    qwen::Tokenizer tok;
    try {
        tok.load(tok_path);
    } catch (const std::exception& e) {
        fprintf(stderr, "[te] tokenizer %s: %s\n", tok_path.c_str(), e.what());
        return false;
    }
    std::string im_start, im_end;
    if (!chatml_tags(tok, im_start, im_end)) {
        fprintf(stderr, "[te] %s 里没有成对的 ChatML im_start/im_end\n",
                tok_path.c_str());
        return false;
    }

    // drop_idx = system 段自己的 token 数，现算而不是写死 14：换 tokenizer 或改
    // 系统提示词都会让它动，而它一动 [drop_idx, P) 的切法就整体错位。
    const auto sys_ids = tok.encode(chatml_system(im_start, im_end));
    const auto all_ids =
        tok.encode(chatml_template(im_start, im_end, prompt));
    const int drop_idx = (int)sys_ids.size();
    const int total = (int)all_ids.size();
    if (drop_idx <= 0 || total <= drop_idx) {
        fprintf(stderr, "[te] 分词异常：system=%d 全序列=%d\n", drop_idx, total);
        return false;
    }
    const int valid_len = total - drop_idx;
    if (valid_len > P - drop_idx) {
        fprintf(stderr,
                "[te] prompt 实际 %d token > 图的文本段 %d（P=%d, drop_idx=%d）；"
                "缩短 prompt，或按更大的 prefix_len 重导文本塔和 DiT\n",
                valid_len, P - drop_idx, P, drop_idx);
        return false;
    }
    const int prefix_len = P - drop_idx;
    for (uint32_t id : all_ids)
        if ((int)id >= table.rows()) {
            fprintf(stderr, "[te] token id %u 越界（表 %d 行）\n", id, table.rows());
            return false;
        }
    printf("[te] drop_idx=%d total=%d valid=%d -> padded %d (ctx_dim=%d)\n",
           drop_idx, total, valid_len, prefix_len, ctx_dim);

    if (getenv("VKOP_TE_DUMP_IDS")) {
        FILE* f = fopen(getenv("VKOP_TE_DUMP_IDS"), "w");
        if (f) {
            for (int k = 0; k < total; ++k)
                fprintf(f, "%s%u", k ? "," : "", all_ids[k]);
            fclose(f);
        }
    }

    // 查表得到图的输入：真实行按 token 顺序，第 total 行往后是零。
    std::vector<uint16_t> embeds((size_t)P * ctx_dim, 0);
    for (int k = 0; k < total; ++k)
        memcpy(embeds.data() + (size_t)k * ctx_dim, table.row(all_ids[k]),
               (size_t)ctx_dim * sizeof(uint16_t));

    auto t0 = std::chrono::high_resolution_clock::now();
    auto te_rt = std::make_shared<Runtime>(cmdpool, te_path.c_str(), /*precision=*/1);
    te_rt->set_backend_buffer(true);
    te_rt->LoadModel();
    printf("[te] loaded in %.2fs\n",
           std::chrono::duration<double>(
               std::chrono::high_resolution_clock::now() - t0).count());

    auto input = te_rt->GetInput("inputs_embeds");
    if (!input) {
        fprintf(stderr, "[te] 没有 inputs_embeds 输入\n");
        return false;
    }
    auto in_tt = as_tensor<uint16_t>(input);
    in_tt->fillToCPU(embeds.data());
    in_tt->copyToGPU(cmdpool);

    auto t1 = std::chrono::high_resolution_clock::now();
    te_rt->Run();
    te_rt->ReadResult();
    printf("[te] forward %.2fs\n",
           std::chrono::duration<double>(
               std::chrono::high_resolution_clock::now() - t1).count());

    auto output = te_rt->GetOutput("hidden_states");
    if (!output) {
        fprintf(stderr, "[te] 没有 hidden_states 输出\n");
        return false;
    }
    auto out_tt = as_tensor<uint16_t>(output);
    out_tt->copyToCPU(cmdpool);
    const auto& hid = out_tt->data();
    if (hid.size() < (size_t)P * ctx_dim) {
        fprintf(stderr, "[te] hidden_states 只有 %zu 个元素，图长要 %d×%d\n",
                hid.size(), P, ctx_dim);
        return false;
    }
    double sq = 0, absmax = 0;
    size_t bad = 0;
    for (size_t i = 0; i < (size_t)P * ctx_dim; ++i) {
        const float v = ITensor::fp16_to_fp32(hid[i]);
        if (!std::isfinite(v)) { ++bad; continue; }
        sq += (double)v * v;
        absmax = std::max(absmax, (double)std::abs(v));
    }
    printf("[te] hidden absmax=%.4f rms=%.4f nonfinite=%zu\n", absmax,
           std::sqrt(sq / (double)std::max<size_t>(1, (size_t)P * ctx_dim)), bad);

    // 丢掉 system 段，取 valid_len 行，右补精确零到 prefix_len。
    out.ctx_dim = ctx_dim;
    out.prefix_len = prefix_len;
    out.valid_len = valid_len;
    out.data.assign((size_t)prefix_len * ctx_dim, 0);
    memcpy(out.data.data(), hid.data() + (size_t)drop_idx * ctx_dim,
           (size_t)valid_len * ctx_dim * sizeof(uint16_t));

    if (const char* d = getenv("VKOP_TE_DUMP_EMBEDS")) {
        FILE* f = fopen(d, "wb");
        if (f) {
            fwrite(out.data.data(), sizeof(uint16_t), out.data.size(), f);
            fclose(f);
            printf("[te] prompt_embeds (%d,%d) -> %s\n", prefix_len, ctx_dim, d);
        }
    }

    // 顺序装载：文本塔的 13.9 GB 必须先还给设备，DiT prefill 才装得下。
    te_rt.reset();
    dev->wait_all_done();
    cmdpool->reset();
    printf("[mem] text tower released\n");
    return true;
}

} // anonymous namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, 
                "Usage: %s <dit_prefill.vkopbin> <dit_decode.vkopbin> <prompt> "
                "[steps] [seed] [--size 512|1024] [--ref DIR]\n"
                "  no --ref: the prompt is tokenized and encoded on the GPU by the\n"
                "  text tower; its files come from the env (defaults shown):\n"
                "    TEXT_ENCODER_VKOPBIN=text_encoder.vkopbin\n"
                "    TEXT_ENCODER_EMBEDS=text_encoder_embeds.bin\n"
                "    TOKENIZER_BIN=llm/tokenizer/qwen3_vl.bin\n", argv[0]);
        return 1;
    }
    
    const char* prefill_path = argv[1];
    const char* decode_path = argv[2];
    const char* prompt = argv[3];
    
    int steps = 20;
    uint32_t seed = 42;
    int img_size = 512;
    std::string ref_dir; // ORT reference dir for numeric alignment (--ref DIR)

    // Parse optional args
    for (int i = 4; i < argc; ++i) {
        if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            img_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ref") == 0 && i + 1 < argc) {
            ref_dir = argv[++i];
        } else if (isdigit(argv[i][0])) {
            if (steps == 20) {
                steps = atoi(argv[i]);
            } else {
                seed = atoi(argv[i]);
            }
        }
    }
    bool ref_mode = !ref_dir.empty();
    std::map<std::string, RefEntry> ref;
    if (ref_mode && !load_ref_dir(ref_dir, ref)) return 1;

    // 文本塔的三份产物。图和查表和 DiT 的 vkopbin 一样放在 image/exporter/，
    // tokenizer 在 llm/tokenizer/ 下 —— 而 BASELINE.md 的命令既可能在
    // image/exporter 里跑也可能在仓库根跑，所以按候选顺序取第一个存在的
    // （显式给 env 时以 env 为准）。
    auto env_or = [](const char* var, const char* def) {
        const char* v = getenv(var);
        return std::string(v && *v ? v : def);
    };
    const std::string te_path =
        env_or("TEXT_ENCODER_VKOPBIN", "text_encoder.vkopbin");
    const std::string te_table_path =
        env_or("TEXT_ENCODER_EMBEDS", "text_encoder_embeds.bin");
    std::string tok_path = getenv("TOKENIZER_BIN") ? getenv("TOKENIZER_BIN") : "";
    if (tok_path.empty())
        for (const char* c : {"llm/tokenizer/qwen3_vl.bin",
                              "../../llm/tokenizer/qwen3_vl.bin"}) {
            struct stat st;
            if (stat(c, &st) == 0) { tok_path = c; break; }
        }
    if (tok_path.empty()) tok_path = "llm/tokenizer/qwen3_vl.bin";
    
    printf("[init] Qwen-Image-2.1 generation driver\n");
    printf("  prompt: \"%s\"\n", prompt);
    printf("  steps: %d, seed: %u, size: %dx%d\n", steps, seed, img_size, img_size);
    
    // Latent dimensions: B×C×H×W where C=64, H=W=img_size/16
    int latent_h = img_size / 16;
    int latent_w = img_size / 16;
    int latent_c = 64; // Will be adjusted based on model's actual input shape
    
    printf("  latent shape: 1 x %d x %d x %d\n", latent_c, latent_h, latent_w);
    
    // Initialize Vulkan runtime (mirrors llm_chat pattern)
    const auto& phydevs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
    if (phydevs.empty()) { fprintf(stderr, "no vulkan device\n"); return 1; }
    auto dev = std::make_shared<VulkanDevice>(phydevs[0]);
    if (dev->getDeviceName().find("llvmpipe") != std::string::npos) {
        fprintf(stderr, "no valid vulkan device\n"); return 1;
    }
    
    auto cmdpool = std::make_shared<VulkanCommandPool>(dev);
    
    // fillToCPU only stages data on the host; the input SSBO must be uploaded
    // explicitly before Run (same fill+copyToGPU pairing as llm_chat).
    auto fill16 = [&](const std::shared_ptr<ITensor>& t,
                      const std::vector<uint16_t>& d) {
        auto tt = as_tensor<uint16_t>(t);
        tt->fillToCPU(d.data());
        tt->copyToGPU(cmdpool);
    };
    auto fill32 = [&](const std::shared_ptr<ITensor>& t,
                      const std::vector<float>& d) {
        auto tt = as_tensor<float>(t);
        tt->fillToCPU(d.data());
        tt->copyToGPU(cmdpool);
    };
    
    // Fill a graph input (fp16 or fp32 by element size) from a reference file.
    auto ref_fill = [&](Runtime* rt, const std::string& input,
                        const std::string& file) -> bool {
        auto it = ref.find(file);
        if (it == ref.end()) {
            fprintf(stderr, "[ref] missing tensor %s\n", file.c_str());
            return false;
        }
        rt->ResizeInput(input, it->second.dims);
        auto t = rt->GetInput(input);
        if (!t) {
            fprintf(stderr, "[ref] no input '%s' in model\n", input.c_str());
            return false;
        }
        const uint8_t* p = it->second.bytes.data();
        size_t n = it->second.bytes.size();
        if (it->second.elem_size == 2) {
            std::vector<uint16_t> d(n / 2);
            memcpy(d.data(), p, n);
            fill16(t, d);
        } else {
            std::vector<float> d(n / 4);
            memcpy(d.data(), p, n);
            fill32(t, d);
        }
        return true;
    };
    
    // Load models (using buffer backend like llm_chat)
    // 文本塔排在 DiT 之前、用完即释放：它和 prefill 各要 13.9/13.8 GB，36 GB
    // 统一内存里两段不能同时在。
    PromptEmbeds text_pe;
    if (!ref_mode) {
        if (!encode_prompt(te_path, te_table_path, tok_path, prompt, cmdpool, dev,
                           text_pe))
            return 1;
    }

    printf("[load] Loading DiT prefill graph...\n");
    auto prefill_rt = std::make_shared<Runtime>(cmdpool, prefill_path, /*precision=*/1);
    prefill_rt->set_backend_buffer(true);
    prefill_rt->LoadModel();
    printf("[load] Prefill loaded\n");

    // Header-only shape probe: the flatbuffer is mmap'd, so reading input dims
    // costs nothing and lets the decode graph stay unloaded until the prefill is
    // done. Each 7.12B graph is ~13 GB of device weights; both at once plus
    // activations do not fit in 36 GB.
    auto header_dims = [](const char* path, bool input,
                          const std::string& name) -> std::vector<int32_t> {
        vkop::load::VkModel m(path);
        const auto& list = input ? m.inputs : m.outputs;
        for (const auto& s : list)
            if (s.name == name) return s.dims;
        return {};
    };

    // [probe] report-only: checksum the initializers embedded in each vkopbin
    // so silently-zeroed weights are caught before any GPU run.
    if (std::getenv("VKOP_DUMP_WEIGHTS")) {
        auto f16 = [](const uint8_t* p) {
            _Float16 h;
            memcpy(&h, p, 2);
            return (float)h;
        };
        for (const char* wp : {prefill_path, decode_path}) {
            vkop::load::VkModel m(wp);
            printf("[weights] %s: %zu entries\n", wp, m.initializers.size());
            for (const auto& kv : m.initializers) {
                const auto& init = kv.second;
                size_t cnt = 1;
                for (auto d : init.dims) cnt *= d;
                size_t esz = init.dtype == "float16" || init.dtype == "bfloat16" ? 2
                           : init.dtype == "float32" || init.dtype == "int32" ? 4
                           : init.dtype == "int64" ? 8 : 0;
                auto oit = m.initializer_offsets.find(kv.first);
                if (!m.initializer_memory || esz == 0 || oit == m.initializer_offsets.end()) {
                    printf("    %-40s %s elems=%zu SKIP\n", kv.first.c_str(),
                           init.dtype.c_str(), cnt);
                    continue;
                }
                const uint8_t* base = m.initializer_memory + oit->second;
                size_t nz = 0;
                double sum = 0;
                float vmin = 1e30f, vmax = -1e30f;
                for (size_t i = 0; i < cnt; ++i) {
                    const uint8_t* p = base + i * esz;
                    float v = 0.0f;
                    if (init.dtype == "float16") {
                        v = f16(p);
                    } else if (init.dtype == "float32") {
                        memcpy(&v, p, 4);
                    } else if (init.dtype == "int64") {
                        int64_t q;
                        memcpy(&q, p, 8);
                        v = (float)(q % 1000000);
                    } else if (init.dtype == "int32") {
                        int32_t q;
                        memcpy(&q, p, 4);
                        v = (float)q;
                    }
                    if (v != 0.0f) ++nz;
                    sum += v;
                    vmin = std::min(vmin, v);
                    vmax = std::max(vmax, v);
                }
                printf("    %-40s %s elems=%zu nz=%zu sum=%.6g min=%.4g max=%.4g\n",
                       kv.first.c_str(), init.dtype.c_str(), cnt, nz, sum, vmin, vmax);
            }
        }
    }
    
    // Detect latent channel dimension from the decode model's header
    int target_len = latent_h * latent_w;
    auto target_latents_dims = header_dims(decode_path, true, "target_latents");
    if (target_latents_dims.size() >= 3 && target_latents_dims[2] > 0) {
        latent_c = target_latents_dims[2]; // Last dim is C in BLC format
        printf("[model] Detected latent_c=%d from decode model\n", latent_c);
    }
    
    int latent_n = 1 * latent_c * target_len;
    std::vector<float> latent(latent_n);
    if (ref_mode) {
        // Reference mode: the latent AND all per-step inputs come from ORT's
        // ref/ dir so vkop and ORT consume bit-identical data.
        const auto& e = ref["latent_init.raw"];
        target_len = (int)e.dims[1];
        latent_c = (int)e.dims[2];
        latent_n = target_len * latent_c;
        latent.assign(reinterpret_cast<const float*>(e.bytes.data()),
                      reinterpret_cast<const float*>(e.bytes.data()) + latent_n);
        printf("[ref] latent from latent_init.raw: target_len=%d latent_c=%d\n",
               target_len, latent_c);
    } else {
        generate_random_latent(latent.data(), latent_n, seed);
    }
    
    // σ 调度。ref 模式读 gen_dit_ref.py 写的 sigmas.raw（长度 steps+1，末尾是
    // 终值 0），其余情况由 official_sigmas 现算 —— 同一条公式，所以两边可以
    // 逐位对拍。模型的 timestep 输入就是位移后的 σ 本身（pipeline 传 t/1000，
    // 而 t = σ·1000），Euler 更新是 x += (σ_next − σ)·v。
    std::vector<float> sigmas;
    if (ref_mode && ref.count("sigmas.raw")) {
        const auto &e = ref["sigmas.raw"];
        sigmas.assign(reinterpret_cast<const float *>(e.bytes.data()),
                      reinterpret_cast<const float *>(e.bytes.data()) +
                          e.bytes.size() / 4);
        if (steps != (int)sigmas.size() - 1) {
            printf("[ref] steps=%d -> %zu (from sigmas.raw)\n", steps,
                   sigmas.size() - 1);
            steps = (int)sigmas.size() - 1;
        }
        printf("[ref] official schedule: %zu sigmas, σ0=%.4f μ-shifted\n",
               sigmas.size(), sigmas[0]);
    } else if (!ref_mode) {
        sigmas = official_sigmas(steps, target_len);
    }
    // Re-anchoring the latent on ORT's per-step state isolates each decode step
    // (no divergence carry-over) — right for numeric alignment, wrong for a
    // real sample, which must follow vkop's own trajectory. VKOP_REF_FREE=1
    // turns re-anchoring off.
    const bool ref_free_run =
        ref_mode && getenv("VKOP_REF_FREE") != nullptr;
    
    const int hd = kRopeHeadDim;  // cos/sin 的列数
    RopeTables rope;  // 非 ref 路径在 prefill 前算好，decode 每一步都要用
    
    // ---- KV Cache Integration ----
    // Prefill outputs present_kv_* tensors, decode needs past_kv_* inputs
    // Dynamically detect the number of KV layers from the model
    
    // prefix_len 是 DiT 侧固定消费的文本段行数（下面从图里读），valid_len 是其中
    // 真实 token 的行数，剩下的行是补零 —— 由文本塔现算。
    int prefix_len = 64;
    int valid_len = 64;
    if (!ref_mode) valid_len = text_pe.valid_len;
    
    printf("[prefill] Running prompt encoding to get initial KV cache...\n");
    
    // Prepare prefill inputs
    int seq_len = prefix_len;
    
    // Detect context_dim from prefill model's prompt_embeds input
    auto prompt_embeds_check = prefill_rt->GetInput("prompt_embeds");
    int context_dim = 4096; // Default for full model
    int graph_text_len = 0; // 图自己要的文本段行数
    if (prompt_embeds_check) {
        auto check_shape = prompt_embeds_check->getShape();
        if (check_shape.size() >= 3 && check_shape[2] > 0) {
            context_dim = check_shape[2];
            graph_text_len = check_shape[1];
            printf("[model] Detected context_dim=%d from prefill model\n", context_dim);
        }
    }
    
    if (ref_mode) {
        // Bit-identical inputs from the ORT reference dir.
        const auto& pe = ref["prompt_embeds.raw"];
        seq_len = prefix_len = (int)pe.dims[1];
        context_dim = (int)pe.dims[2];
        if (!ref_fill(prefill_rt.get(), "prompt_embeds", "prompt_embeds.raw")) return 1;
        if (!ref_fill(prefill_rt.get(), "cos", "cos_prefill.raw")) return 1;
        if (!ref_fill(prefill_rt.get(), "sin", "sin_prefill.raw")) return 1;
        prefill_rt->ResizeInput("timestep_zero", {1u});
        std::vector<float> timestep_zero_data(1, 0.0f);
        fill32(prefill_rt->GetInput("timestep_zero"), timestep_zero_data);
        if (!ref_fill(prefill_rt.get(), "attention_bias", "bias_prefill.raw")) return 1;
        printf("[ref] prefill inputs loaded (seq_len=%d ctx_dim=%d)\n",
               seq_len, context_dim);
    } else {
        // prompt_embeds: [1, prefix_len, context_dim] fp16，来自文本塔（查表 ->
        // 36 层 -> 去 system 段 -> 右补零）。文本塔的图长 P = drop_idx + prefix_len
        // 和 DiT 的 prefix_len 是同一批导出时定死的，行数或宽度对不上就是拿错了
        // 配对 —— 必须在这里停，而不是把 64 行的向量塞进要 56 行的输入。
        prefix_len = seq_len = graph_text_len;
        if (text_pe.ctx_dim != context_dim || text_pe.prefix_len != prefix_len) {
            fprintf(stderr,
                    "[error] 文本塔给出 (%d,%d)，prefill 图要 (%d,%d)；"
                    "text_encoder 和 dit_*_static 不是同一批导出的\n",
                    text_pe.prefix_len, text_pe.ctx_dim, prefix_len, context_dim);
            return 1;
        }
        prefill_rt->ResizeInput("prompt_embeds", {1u, (uint32_t)seq_len, (uint32_t)context_dim});
        auto prompt_embeds_t = prefill_rt->GetInput("prompt_embeds");
        if (!prompt_embeds_t) {
            fprintf(stderr, "[error] Failed to get prompt_embeds input tensor\n");
            return 1;
        }
        fill16(prompt_embeds_t, text_pe.data);
        
        // joint rope 的两段表：文本行 + 图像行（图像段的 frame 位置按 valid_len
        // 冻结，见 build_rope）。fp32 算完再整体落 fp16，与 gen_dit_ref.py 的
        // `torch.cos(...).astype(np.float16)` 同一舍入点。
        rope = build_rope(prefix_len, valid_len, latent_h, latent_w);
        printf("[rope] txt %d 行 + img %d 行, axes={%d,%d,%d}, theta=%g\n",
               prefix_len, target_len, kAxesRope[0], kAxesRope[1], kAxesRope[2],
               (double)kRopeTheta);
        // 把驱动自己算的表和 gen_dit_ref.py --real-rope --schedule diffusers 的
        // 产物逐位对拍用（"纯 vkop 出图"这条路上没有任何东西保证公式搬对了）。
        if (const char* dd = getenv("VKOP_DUMP_TABLES")) {
            mkdir(dd, 0755);
            auto w = [&](const char* name, const std::vector<uint16_t>& v) {
                FILE* f = fopen((std::string(dd) + "/" + name).c_str(), "wb");
                if (f) { fwrite(v.data(), 2, v.size(), f); fclose(f); }
            };
            w("cos_prefill.raw", rope.cos_text);
            w("sin_prefill.raw", rope.sin_text);
            w("cos_decode.raw", rope.cos_img);
            w("sin_decode.raw", rope.sin_img);
            FILE* f = fopen((std::string(dd) + "/sigmas.raw").c_str(), "wb");
            if (f) { fwrite(sigmas.data(), 4, sigmas.size(), f); fclose(f); }
            printf("[tables] -> %s\n", dd);
        }
        prefill_rt->ResizeInput("cos", {(uint32_t)seq_len, (uint32_t)hd});
        fill16(prefill_rt->GetInput("cos"), rope.cos_text);
        prefill_rt->ResizeInput("sin", {(uint32_t)seq_len, (uint32_t)hd});
        fill16(prefill_rt->GetInput("sin"), rope.sin_text);
        
        // timestep_zero: [1]
        prefill_rt->ResizeInput("timestep_zero", {1u});
        auto timestep_zero_t = prefill_rt->GetInput("timestep_zero");
        std::vector<float> timestep_zero_data(1, 0.0f);
        fill32(timestep_zero_t, timestep_zero_data);
        
        // attention_bias: [1, 1, prefix_len, prefix_len] with causal mask (fp16)
        prefill_rt->ResizeInput("attention_bias", {1u, 1u, (uint32_t)seq_len, (uint32_t)seq_len});
        auto attn_bias_t = prefill_rt->GetInput("attention_bias");
        std::vector<float> attn_bias_prefill(seq_len * seq_len, 0.0f);
        for (int i = 0; i < seq_len; ++i)
            for (int j = i + 1; j < seq_len; ++j)
                attn_bias_prefill[i * seq_len + j] = -65504.0f;
        std::vector<uint16_t> attn_bias_fp16(attn_bias_prefill.size());
        for (size_t i = 0; i < attn_bias_prefill.size(); ++i)
            attn_bias_fp16[i] = ITensor::fp32_to_fp16(attn_bias_prefill[i]);
        fill16(attn_bias_t, attn_bias_fp16);
    }
    
    // [probe] VKOP_REF_SKIP_PREFILL=1 (ref mode only): don't execute the vkop
    // prefill at all — decode KV already comes from ORT files. If the
    // VK_ERROR_DEVICE_LOST crash disappears, the prefill's mis-shaped
    // present_kv_1 output (256-elem SSBO written as 4096) is the fault source.
    const bool skip_prefill_run =
        ref_mode && getenv("VKOP_REF_SKIP_PREFILL") != nullptr;

    // Run prefill
    auto t_prefill_start = std::chrono::high_resolution_clock::now();
    if (!skip_prefill_run) {
        prefill_rt->Run();
        prefill_rt->ReadResult(); // drain GPU queues + read back real outputs
    }
    auto t_prefill_end = std::chrono::high_resolution_clock::now();
    double t_prefill = std::chrono::duration<double>(t_prefill_end - t_prefill_start).count();
    
    // Count how many present_kv_* outputs exist
    int num_kv_layers = 0;
    while (true) {
        std::string kv_name = "present_kv_" + std::to_string(num_kv_layers);
        auto kv_out = prefill_rt->GetOutput(kv_name);
        if (!kv_out) break;
        num_kv_layers++;
    }
    
    printf("[prefill] Done in %.3fs, found %d KV layers\n", t_prefill, num_kv_layers);
    
    if (num_kv_layers == 0) {
        fprintf(stderr, "[error] No present_kv outputs found in prefill model\n");
        return 1;
    }
    
    // Extract present_kv outputs and store as past_kv for decode
    std::vector<std::vector<uint16_t>> past_kv_cache(num_kv_layers);
    std::vector<std::vector<int>> past_kv_shapes(num_kv_layers); // Store shapes for decode resize
    
    for (int layer = 0; !skip_prefill_run && layer < num_kv_layers; ++layer) {
        std::string kv_name = "present_kv_" + std::to_string(layer);
        auto kv_out = prefill_rt->GetOutput(kv_name);
        if (!kv_out) {
            fprintf(stderr, "[error] Missing output: %s\n", kv_name.c_str());
            return 1;
        }
        
        auto kv_tensor = as_tensor<uint16_t>(kv_out);
        kv_tensor->copyToCPU(cmdpool);
        const auto& kv_data = kv_tensor->data();
        past_kv_cache[layer].assign(kv_data.begin(), kv_data.end());
        
        // Store the shape for later use in decode
        past_kv_shapes[layer] = kv_tensor->getShape();
        
        if (ref_mode) {
            // Dump vkop's own prefill KV so it can be compared against ORT's.
            mkdir("ref_out", 0755);
            auto shp = kv_tensor->getShape();
            printf("[dbg] present_kv_%d data_n=%zu shape=", layer, kv_data.size());
            for (size_t d = 0; d < shp.size(); ++d) printf("%d,", shp[d]);
            printf("\n");
            std::string out_path = "ref_out/vkop_present_kv_" + std::to_string(layer) + ".raw";
            FILE* fp = fopen(out_path.c_str(), "wb");
            if (fp) {
                fwrite(kv_data.data(), sizeof(uint16_t), kv_data.size(), fp);
                fclose(fp);
            }
        }
    }
    
    if (ref_mode && !getenv("VKOP_REF_VK_KV")) {
        // Use ORT's KV as decode inputs so the decode graph is isolated from
        // any prefill divergence (both compared separately).
        for (int layer = 0; layer < num_kv_layers; ++layer) {
            const auto& e = ref["ort_present_kv_" + std::to_string(layer) + ".raw"];
            const uint16_t* p = reinterpret_cast<const uint16_t*>(e.bytes.data());
            past_kv_cache[layer].assign(p, p + e.bytes.size() / 2);
            past_kv_shapes[layer] =
                std::vector<int>(e.dims.begin(), e.dims.end());
        }
        printf("[ref] decode past_kv taken from ORT prefill outputs\n");
    } else if (ref_mode) {
        printf("[ref] decode past_kv taken from vkop's own prefill\n");
    }
    
    printf("[kv] Cached %d KV layers (prefix_len=%d)\n", num_kv_layers, prefix_len);

    // Sequential loading (same strategy as the ORT baseline): the prefill graph
    // is released before the decode graph takes its weights, so a 7.12B model
    // needs one graph's worth of device memory at a time.
    prefill_rt.reset();

    // Force GPU to finish and release MoltenVK/Metal resources before loading
    // the next large model. On UMA (M5 Max), wait_all_done ensures Metal can
    // reclaim GPU heap back to system RAM.
    cmdpool->getVulkanDevice()->wait_all_done();
    cmdpool->reset();
    printf("[mem] Prefill released, device idle\n");

    // VAE_ONLY mode: skip DiT decode entirely, load latent from file.
    const char* vae_only_env = getenv("VAE_ONLY");
    bool vae_only_mode = vae_only_env && vae_only_env[0] != '0';

    if (!vae_only_mode) {
        printf("[load] Loading DiT decode graph...\n");
        auto decode_rt = std::make_shared<Runtime>(cmdpool, decode_path, /*precision=*/1);
        decode_rt->set_backend_buffer(true);
        try {
            decode_rt->LoadModel();
            printf("[load] Decode loaded\n");
        } catch (const std::exception& e) {
            fprintf(stderr, "[error] Failed to load decode model: %s\n", e.what());
            return 1;
        }

        printf("[gen] Starting denoising loop (%d steps)...\n", steps);
        auto t_start = std::chrono::high_resolution_clock::now();
    
    // Denoising loop with real DiT decode inference
    for (int step = 0; step < steps; ++step) {
        float t, sigma;
        if (!sigmas.empty()) {
            sigma = sigmas[step];
            t = sigma; // pipeline feeds timestep = t/1000, and t = sigma·1000
        } else {
            // 只有 ref 目录里没有 sigmas.raw（gen_dit_ref.py 的 naive 调度）才走到
            // 这里：那条参考循环用的就是 t=linspace(1,0,N,endpoint=False)、σ=1-t。
            t = 1.0f - (float)step / (float)steps;
            sigma = 1.0f - t;
        }

        if (ref_mode && step > 0 && !ref_free_run) {
            // Re-anchor the latent on ORT's post-step state so every step
            // compares decode in isolation (no divergence carry-over).
            std::string lf = "ort_latent_after_step" + std::to_string(step - 1) + ".raw";
            auto it = ref.find(lf);
            if (it != ref.end()) {
                const uint16_t* p = reinterpret_cast<const uint16_t*>(it->second.bytes.data());
                latent.resize(it->second.bytes.size() / 2);
                for (size_t i = 0; i < latent.size(); ++i) {
                    latent[i] = ITensor::fp16_to_fp32(p[i]);
                }
            }
        }
        
        // Resize and fill target_latents: [1, target_len, latent_c]
        decode_rt->ResizeInput("target_latents", {1u, (uint32_t)target_len, (uint32_t)latent_c});
        auto target_lat = decode_rt->GetInput("target_latents");
        std::vector<uint16_t> latent_fp16(latent_n);
        for (int i = 0; i < latent_n; ++i) {
            latent_fp16[i] = ITensor::fp32_to_fp16(latent[i]);
        }
        fill16(target_lat, latent_fp16);
        
        // Resize and fill timestep: [1]
        decode_rt->ResizeInput("timestep", {1u});
        auto timestep_t = decode_rt->GetInput("timestep");
        std::vector<float> timestep_data(1, t);
        fill32(timestep_t, timestep_data);
        
        if (ref_mode) {
            if (!ref_fill(decode_rt.get(), "cos", "cos_decode.raw")) return 1;
            if (!ref_fill(decode_rt.get(), "sin", "sin_decode.raw")) return 1;
        } else {
            // 图像段的 rope 行每步都一样（位置不含 σ），但 SSBO 仍要重新上传：
            // 上一次 Run 之后图里的 buffer 内容不作保证。
            decode_rt->ResizeInput("cos", {(uint32_t)target_len, (uint32_t)hd});
            fill16(decode_rt->GetInput("cos"), rope.cos_img);
            decode_rt->ResizeInput("sin", {(uint32_t)target_len, (uint32_t)hd});
            fill16(decode_rt->GetInput("sin"), rope.sin_img);
        }
        
        // Feed all past_kv inputs (dynamic count and shape based on prefill outputs)
        for (int layer = 0; layer < num_kv_layers; ++layer) {
            std::string kv_name = "past_kv_" + std::to_string(layer);
            const auto& kv_shape = past_kv_shapes[layer];
            decode_rt->ResizeInput(kv_name, {
                (uint32_t)kv_shape[0], 
                (uint32_t)kv_shape[1], 
                (uint32_t)kv_shape[2], 
                (uint32_t)kv_shape[3], 
                (uint32_t)kv_shape[4]
            });
            auto kv_t = decode_rt->GetInput(kv_name);
            fill16(kv_t, past_kv_cache[layer]);
        }
        
        // attention_bias: [1, 1, target_len, kv_len] (fp16)
        if (ref_mode) {
            if (!ref_fill(decode_rt.get(), "attention_bias", "bias_decode.raw")) return 1;
        } else {
            int total_kv_len = prefix_len + target_len;
            decode_rt->ResizeInput("attention_bias", {1u, 1u, (uint32_t)target_len, (uint32_t)total_kv_len});
            auto attn_bias_decode_t = decode_rt->GetInput("attention_bias");
            std::vector<float> attn_bias_decode(target_len * total_kv_len, 0.0f);
            // 文本 KV 段的前 valid_len 列是真 token，剩下 prefix_len-valid_len 列是
            // 补零。图像段对文本是**双向**注意力，零向量当 key 会实打实地改变注意力
            // 权重，所以这些列必须屏蔽（-65504 是 fp16 最小值，softmax 后为 0）。
            // 与 gen_dit_ref.py 的 bias_t[0,0,:,valid_len:prefix_len] 同一口径。
            for (int i = 0; i < target_len; ++i)
                for (int j = valid_len; j < prefix_len; ++j)
                    attn_bias_decode[i * total_kv_len + j] = -65504.0f;
            if (valid_len < prefix_len)
                printf("[bias] decode: 屏蔽 %d 个 prompt 补零列\n", prefix_len - valid_len);
            std::vector<uint16_t> attn_bias_decode_fp16(attn_bias_decode.size());
            for (size_t i = 0; i < attn_bias_decode.size(); ++i) {
                attn_bias_decode_fp16[i] = ITensor::fp32_to_fp16(attn_bias_decode[i]);
            }
            fill16(attn_bias_decode_t, attn_bias_decode_fp16);
        }
        
        if (ref_mode) {
            // [probe] Read the inputs back FROM THE GPU before Run to see
            // whether what the shaders will consume is byte-identical to the
            // ref files the host intended to upload.
            const char* probe_names[] = {"target_latents", "timestep", "cos",
                                         "sin", "attention_bias"};
            for (const char* pn : probe_names) {
                auto t = decode_rt->GetInput(pn);
                std::string f = "ref_out/echo_step" + std::to_string(step) +
                                "_" + pn + ".raw";
                FILE* fp = fopen(f.c_str(), "wb");
                if (!fp) continue;
                if (t->dtype() == typeid(float)) {
                    auto tt = as_tensor<float>(t);
                    tt->copyToCPU(cmdpool);
                    fwrite(tt->data().data(), sizeof(float),
                           tt->data().size(), fp);
                } else {
                    auto tt = as_tensor<uint16_t>(t);
                    tt->copyToCPU(cmdpool);
                    fwrite(tt->data().data(), sizeof(uint16_t),
                           tt->data().size(), fp);
                }
                fclose(fp);
            }
            for (int layer = 0; layer < num_kv_layers; ++layer) {
                auto t = decode_rt->GetInput("past_kv_" + std::to_string(layer));
                auto tt = as_tensor<uint16_t>(t);
                tt->copyToCPU(cmdpool);
                std::string f = "ref_out/echo_step" + std::to_string(step) +
                                "_past_kv_" + std::to_string(layer) + ".raw";
                FILE* fp = fopen(f.c_str(), "wb");
                if (!fp) continue;
                fwrite(tt->data().data(), sizeof(uint16_t),
                       tt->data().size(), fp);
                fclose(fp);
            }
        }

        // Run decode
        decode_rt->Run();
        decode_rt->ReadResult();
        
        // Get output: sample [1, target_len, latent_c]
        auto sample_out = decode_rt->GetOutput("sample");
        auto sample_tensor = as_tensor<uint16_t>(sample_out);
        sample_tensor->copyToCPU(cmdpool);
        const auto& sample_data = sample_tensor->data();
        
        // Convert fp16 back to fp32 and compute velocity
        std::vector<float> velocity(latent_n);
        for (int i = 0; i < latent_n; ++i) {
            velocity[i] = ITensor::fp16_to_fp32(sample_data[i]);
        }
        
        if (ref_mode) {
            // Dump for the compare script.
            mkdir("ref_out", 0755);
            auto shp = sample_tensor->getShape();
            uint64_t sum = 0;
            for (size_t i = 0; i < std::min<size_t>(sample_data.size(), 64); ++i)
                sum += sample_data[i];
            printf("[dbg] step %d data_n=%zu head64sum=%llu shape=", step,
                   sample_data.size(), (unsigned long long)sum);
            for (size_t d = 0; d < shp.size(); ++d) printf("%d,", shp[d]);
            printf("\n");
            std::string out_path = "ref_out/vkop_velocity_step" + std::to_string(step) + ".raw";
            FILE* fp = fopen(out_path.c_str(), "wb");
            if (fp) {
                fwrite(sample_data.data(), sizeof(uint16_t),
                       std::min<size_t>(sample_data.size(), latent_n), fp);
                fclose(fp);
            }
        }
        
        // Euler step：官方调度下 x += (σ_next − σ)·v；naive 那条（老 ref 目录）
        // 沿用 x -= σ·v，与 gen_dit_ref.py 的 naive 循环一致。
        if (!sigmas.empty()) {
            const float d = sigmas[step + 1] - sigmas[step];
            for (int i = 0; i < latent_n; ++i)
                latent[i] += d * velocity[i];
        } else {
            for (int i = 0; i < latent_n; ++i)
                latent[i] -= sigma * velocity[i];
        }
        
        if ((step + 1) % 5 == 0 || step == 0) {
            printf("  Step %d/%d (t=%.3f, σ=%.3f)\n", 
                   step + 1, steps, t, sigma);
        }
    }
    
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    printf("[gen] Denoising complete: %.1fs (%.1f s/step)\n", 
           elapsed, elapsed / steps);
    
    // Save output
    char out_path[256];
    snprintf(out_path, sizeof(out_path), "latent_out_%dx%d.raw", img_size, img_size);
    save_raw(out_path, latent.data(), latent_n);
    
    printf("[done] Raw latent saved (%d elements).\n", (int)latent_n);
} else {
    // VAE_ONLY mode: load latent from file instead of running DiT decode.
    const char* latent_file = getenv("VAE_LATENT_FILE");
    std::string lf = latent_file ? latent_file : "ref_full/ort_latent_after_step0.raw";
    printf("[vae-only] Loading latent from %s\n", lf.c_str());
    FILE* fp = fopen(lf.c_str(), "rb");
    if (!fp) {
        fprintf(stderr, "[vae-only] cannot open %s\n", lf.c_str());
        return 1;
    }
    // Read fp16 latent [1, target_len, latent_c].
    size_t n_fp16 = (size_t)1 * target_len * latent_c;
    std::vector<uint16_t> latent_fp16(n_fp16);
    if (fread(latent_fp16.data(), sizeof(uint16_t), n_fp16, fp) != n_fp16) {
        fprintf(stderr, "[vae-only] short read from %s\n", lf.c_str());
        fclose(fp);
        return 1;
    }
    fclose(fp);
    // Convert to fp32.
    latent.resize(n_fp16);
    for (size_t i = 0; i < n_fp16; ++i) {
        latent[i] = vkop::core::ITensor::fp16_to_fp32(latent_fp16[i]);
    }
    printf("[vae-only] Loaded %zu fp32 values\n", latent.size());
}

// ---- VAE Decode & PNG output ----
    // Latent shape: [1, target_len, latent_c] in BLC layout.
    // VAE expects: [1, latent_c, 1, H_lat, W_lat] in NCDHW where
    //   H_lat * W_lat = target_len. For target_len=1024 -> 32x32 grid.
    int vae_latent_h = 0, vae_latent_w = 0;
    for (int s = 1; s <= target_len; ++s) {
        if (target_len % s == 0) {
            vae_latent_h = s;
            vae_latent_w = target_len / s;
            if (std::abs(vae_latent_h - vae_latent_w) <= 1)
                break;
        }
    }
    if (vae_latent_h == 0) {
        fprintf(stderr, "[vae] cannot factor target_len=%d\n", target_len);
        return 1;
    }
    printf("[vae] Latent grid: %dx%d (target_len=%d, latent_c=%d)\n",
           vae_latent_h, vae_latent_w, target_len, latent_c);

    // Reshape BLC [1, target_len, latent_c] -> NCDHW [1, latent_c, 1, H, W].
    std::vector<float> latent_ncdhw((size_t)1 * latent_c * 1 * vae_latent_h *
                                    vae_latent_w);
    for (int t = 0; t < target_len; ++t) {
        int lh = t / vae_latent_w;
        int lw = t % vae_latent_w;
        for (int c = 0; c < latent_c; ++c) {
            latent_ncdhw[((size_t)c * vae_latent_h + lh) * vae_latent_w + lw] =
                latent[(size_t)t * latent_c + c] * kLatentsStd[c] + kLatentsMean[c];
        }
    }

    // Load VAE decoder.
    const char* vae_path_env = getenv("VAE_VKOPBIN");
    std::string vae_path_str =
        vae_path_env ? vae_path_env : "vae_decoder_512.vkopbin";
    const char* vae_path = vae_path_str.c_str();

    printf("[load] Loading VAE decoder: %s\n", vae_path);
    auto vae_rt =
        std::make_shared<Runtime>(cmdpool, vae_path, /*precision=*/1);
    vae_rt->set_backend_buffer(true);
    try {
        vae_rt->LoadModel();
        printf("[load] VAE loaded\n");
    } catch (const std::exception& e) {
        fprintf(stderr, "[vae] LoadModel failed: %s\n", e.what());
        return 1;
    }

    // Set latent input by name.
    auto vae_input = vae_rt->GetInput("latent");
    if (!vae_input) {
        fprintf(stderr, "[vae] no 'latent' input found\n");
        return 1;
    }
    // Resize and fill the VAE's input tensor.
    auto vae_latent_tensor = vkop::core::as_tensor<float>(vae_input);
    vae_latent_tensor->resize(
        std::vector<int>{1, latent_c, 1, vae_latent_h, vae_latent_w});
    vae_latent_tensor->fillToCPU(latent_ncdhw.data());
    vae_latent_tensor->copyToGPU(cmdpool);

    // Run VAE decode.
    printf("[vae] Running decode...\n");
    auto t_vae_start = std::chrono::high_resolution_clock::now();
    vae_rt->Run();
    auto t_vae_end = std::chrono::high_resolution_clock::now();
    double vae_elapsed =
        std::chrono::duration<double>(t_vae_end - t_vae_start).count();
    printf("[time] vae decode %.2fs\n", vae_elapsed);

    // Read decoded output.
    auto vae_output = vae_rt->GetOutput("decoded");
    if (!vae_output) {
        fprintf(stderr, "[vae] no 'decoded' output\n");
        return 1;
    }
    auto decoded_tensor = vkop::core::as_tensor<float>(vae_output);
    decoded_tensor->copyToCPU(cmdpool);
    const auto& decoded_shape = decoded_tensor->getShape();
    if (decoded_shape.size() < 4) {
        fprintf(stderr, "[vae] unexpected output rank %zu\n",
                decoded_shape.size());
        return 1;
    }
    // Shape is [1, 4, 1, H_img, W_img] in NCDHW.
    int img_h = decoded_shape[decoded_shape.size() - 2];
    int img_w = decoded_shape[decoded_shape.size() - 1];
    int img_c = decoded_shape[1]; // C dimension
    printf("[vae] Decoded: [%d,%d,%d,%d,%d] -> image %dx%d x%d\n",
           (int)decoded_shape[0], (int)decoded_shape[1],
           (int)decoded_shape[2], img_h, img_w, img_h, img_w, img_c);

    std::vector<float> decoded_data(decoded_tensor->num_elements());
    memcpy(decoded_data.data(), decoded_tensor->data().data(),
           decoded_data.size() * sizeof(float));

    char png_path[256];
    snprintf(png_path, sizeof(png_path), "output_%dx%d.png", img_h, img_w);
    if (save_latent_as_png(decoded_data, img_h, img_w, png_path)) {
        printf("[png] Saved %s (%dx%d RGB)\n", png_path, img_w, img_h);
    } else {
        fprintf(stderr, "[png] Failed to write %s\n", png_path);
    }

    return 0;
}
