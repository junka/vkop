// junka @ 2026
// Qwen-Image-2.1 pipeline pieces shared by image_gen (one-shot CLI) and
// image_serve (resident line-protocol server): PNG writer, σ schedule, joint
// rope, latent de-normalisation, and the text tower front end.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <random>
#include <stdexcept>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanCommandPool.hpp"
#include "core/Tensor.hpp"
#include "core/runtime.hpp"
#include "model/load.hpp"
#include "tokenizer.hpp"

namespace qimg {

using vkop::VulkanDevice;
using vkop::VulkanCommandPool;
using vkop::core::ITensor;
using vkop::core::Runtime;
using vkop::core::as_tensor;

// Wall clock since the process (or the current request) started, so the stage
// markers can be costed against the whole run instead of only against each other.
inline std::chrono::high_resolution_clock::time_point g_run_start =
    std::chrono::high_resolution_clock::now();
inline double stage_s() {
    return std::chrono::duration<double>(
               std::chrono::high_resolution_clock::now() - g_run_start)
        .count();
}

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
inline bool save_latent_as_png(const std::vector<float>& planar_4ch, int h,
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
inline std::vector<float> official_sigmas(int steps, int target_len) {
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
inline void joint_rope(int txt_len, int lh, int lw, std::vector<float>& cos_out,
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

inline RopeTables build_rope(int prefix_len, int valid_len, int lh, int lw) {
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
inline void generate_random_latent(float* data, int numel, uint32_t seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < numel; ++i) {
        data[i] = dist(gen);
    }
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
inline bool chatml_tags(const qwen::Tokenizer& tok, std::string& im_start,
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
inline std::string chatml_system(const std::string& im_start,
                          const std::string& im_end) {
    return im_start + "system\n" + kSystemPrompt + im_end + "\n";
}

inline std::string chatml_template(const std::string& im_start,
                            const std::string& im_end, const char* prompt) {
    return chatml_system(im_start, im_end) + im_start + "user\n" +
           (prompt ? prompt : "") + im_end + "\n" + im_start + "assistant\n";
}

// 文本塔里除那张 13.9 GB 的图以外的一切：tokenizer、mmap 的查表、ChatML 标签和
// 从图头读出的 P / ctx_dim。image_serve 整个进程只开一份（分词和查表不必每张图
// 重来），image_gen 每次运行开一份。出错一律抛 std::runtime_error，消息即原因。
class TextFrontEnd {
public:
    void open(const std::string& te_path, const std::string& table_path,
              const std::string& tok_path) {
        te_path_ = te_path;
        vkop::load::VkModel hdr(te_path.c_str());
        for (const auto& s : hdr.inputs)
            if (s.name == "inputs_embeds" && s.dims.size() == 3) {
                P_ = s.dims[1];
                ctx_dim_ = s.dims[2];
            }
        if (P_ <= 0 || ctx_dim_ <= 0)
            throw std::runtime_error(te_path + " 里没有 3 维的 inputs_embeds");
        printf("[te] graph %s: P=%d ctx_dim=%d\n", te_path.c_str(), P_, ctx_dim_);
        if (!table_.open(table_path, ctx_dim_))
            throw std::runtime_error("cannot open embed table " + table_path);
        try {
            tok_.load(tok_path);
        } catch (const std::exception& e) {
            throw std::runtime_error("tokenizer " + tok_path + ": " + e.what());
        }
        if (!chatml_tags(tok_, im_start_, im_end_))
            throw std::runtime_error(tok_path + " 里没有成对的 ChatML im_start/im_end");
        // drop_idx = system 段自己的 token 数，现算而不是写死 14：换 tokenizer 或改
        // 系统提示词都会让它动，而它一动 [drop_idx, P) 的切法就整体错位。
        drop_idx_ = (int)tok_.encode(chatml_system(im_start_, im_end_)).size();
        if (drop_idx_ <= 0 || drop_idx_ >= P_)
            throw std::runtime_error("system 段分词异常：drop_idx=" +
                                     std::to_string(drop_idx_));
    }

    int ctx_dim() const { return ctx_dim_; }
    int drop_idx() const { return drop_idx_; }
    // DiT 侧固定消费的文本段行数，也就是 prompt 最多能占的 token 数。
    int prefix_len() const { return P_ - drop_idx_; }

    // 模板分词；返回整段 ids（含 system 段），valid_len = prompt 段的 token 数。
    // 超过 prefix_len 时抛错 —— 截断会悄悄改掉 prompt 的意思，交给调用方决定。
    std::vector<uint32_t> tokenize(const char* prompt, int& valid_len) const {
        auto ids = tok_.encode(chatml_template(im_start_, im_end_, prompt));
        const int total = (int)ids.size();
        if (total <= drop_idx_)
            throw std::runtime_error("分词异常：system=" + std::to_string(drop_idx_) +
                                     " 全序列=" + std::to_string(total));
        valid_len = total - drop_idx_;
        if (valid_len > prefix_len())
            throw std::runtime_error(
                "prompt 实际 " + std::to_string(valid_len) + " token > 图的文本段 " +
                std::to_string(prefix_len()) +
                "；缩短 prompt，或按更大的 prefix_len 重导文本塔和 DiT");
        for (uint32_t id : ids)
            if ((int)id >= table_.rows())
                throw std::runtime_error("token id " + std::to_string(id) + " 越界");
        return ids;
    }

    // prompt 段 token 数（含模板里 user 轮的标签），不加载任何图。
    int count(const char* prompt) const {
        auto ids = tok_.encode(chatml_template(im_start_, im_end_, prompt));
        return (int)ids.size() - drop_idx_;
    }

    // 跑一次文本塔，产出 DiT prefill 要的 [prefix_len, ctx_dim]：装图、前向、释放。
    // 文本塔和 prefill 各要 ~13.9 GB，36 GB 统一内存里两段不能同时在，所以图
    // 不常驻 —— 常驻的只有上面那些便宜的东西。
    void encode(const char* prompt, const std::shared_ptr<VulkanCommandPool>& cmdpool,
                const std::shared_ptr<VulkanDevice>& dev, PromptEmbeds& out) const {
        int valid_len = 0;
        const auto all_ids = tokenize(prompt, valid_len);
        const int total = (int)all_ids.size(), P = P_, ctx_dim = ctx_dim_;
        printf("[te] drop_idx=%d total=%d valid=%d -> padded %d (ctx_dim=%d)\n",
               drop_idx_, total, valid_len, prefix_len(), ctx_dim);

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
            memcpy(embeds.data() + (size_t)k * ctx_dim, table_.row(all_ids[k]),
                   (size_t)ctx_dim * sizeof(uint16_t));

        auto t0 = std::chrono::high_resolution_clock::now();
        printf("[stage] setup+tokenizer+embed gather: +%.2fs\n", stage_s());
        auto te_rt = std::make_shared<Runtime>(cmdpool, te_path_.c_str(), /*precision=*/1);
        te_rt->set_backend_buffer(true);
        te_rt->LoadModel();
        printf("[te] loaded in %.2fs (+%.2fs)\n",
               std::chrono::duration<double>(
                   std::chrono::high_resolution_clock::now() - t0).count(), stage_s());

        auto input = te_rt->GetInput("inputs_embeds");
        if (!input) throw std::runtime_error("text tower: 没有 inputs_embeds 输入");
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
        if (!output) throw std::runtime_error("text tower: 没有 hidden_states 输出");
        auto out_tt = as_tensor<uint16_t>(output);
        out_tt->copyToCPU(cmdpool);
        const auto& hid = out_tt->data();
        if (hid.size() < (size_t)P * ctx_dim)
            throw std::runtime_error("text tower: hidden_states 元素数不足");
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
        out.prefix_len = prefix_len();
        out.valid_len = valid_len;
        out.data.assign((size_t)out.prefix_len * ctx_dim, 0);
        memcpy(out.data.data(), hid.data() + (size_t)drop_idx_ * ctx_dim,
               (size_t)valid_len * ctx_dim * sizeof(uint16_t));

        if (const char* d = getenv("VKOP_TE_DUMP_EMBEDS")) {
            FILE* f = fopen(d, "wb");
            if (f) {
                fwrite(out.data.data(), sizeof(uint16_t), out.data.size(), f);
                fclose(f);
                printf("[te] prompt_embeds (%d,%d) -> %s\n", out.prefix_len, ctx_dim, d);
            }
        }

        // 顺序装载：文本塔的 13.9 GB 必须先还给设备，DiT prefill 才装得下。
        auto t_te_rel = std::chrono::high_resolution_clock::now();
        te_rt.reset();
        dev->wait_all_done();
        cmdpool->reset();
        printf("[mem] text tower released in %.2fs (+%.2fs)\n",
               std::chrono::duration<double>(
                   std::chrono::high_resolution_clock::now() - t_te_rel)
                   .count(),
               stage_s());
    }

private:
    std::string te_path_;
    EmbedTable table_;
    qwen::Tokenizer tok_;
    std::string im_start_, im_end_;
    int P_ = 0, ctx_dim_ = 0, drop_idx_ = 0;
};

// image_gen 的一次性入口：开一份前端、编码、关掉。
inline bool encode_prompt(const std::string& te_path, const std::string& table_path,
                          const std::string& tok_path, const char* prompt,
                          const std::shared_ptr<VulkanCommandPool>& cmdpool,
                          const std::shared_ptr<VulkanDevice>& dev, PromptEmbeds& out) {
    try {
        TextFrontEnd fe;
        fe.open(te_path, table_path, tok_path);
        fe.encode(prompt, cmdpool, dev, out);
        return true;
    } catch (const std::exception& e) {
        fprintf(stderr, "[te] %s\n", e.what());
        return false;
    }
}

} // namespace qimg
