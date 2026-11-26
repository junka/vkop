// junka @ 2026
// minimax_h3 -- run the MiniMax-H3 denoiser's graph contract on vkop.
//
// MiniMax-H3 is a joint video+audio *diffusion* model, not an LLM. The sampler
// stays on the host; what the vkop artifacts carry is one denoising step
// (`transformer`), the video VAE decoder (`vae_decoder`), and the Qwen3-VL text
// tower (`text_encoder`). This driver owns the host half of that contract so
// the .vkopbin files can be exercised for real:
//
//   * build the packed `[text | video | audio]`-style layout the denoiser reads
//     by index (token_tags / timestep_indices / position_ids / the three row
//     vectors / pack_order),
//   * compute `temb` -- the timestep MLP is deliberately NOT in the graph
//     (see model/exporter/minimax_h3/README.md),
//   * run the graph N times as a sampler would, with an Euler step in between,
//   * build the VAE decoder's rotary position grid,
//
// and then report what came back. It is a *smoke/soak* driver: it proves the
// converted graphs load on this device and produce finite values at the shape
// contract the exporter documented. It does not claim the output looks like a
// video -- with no text encoder wired in here the conditioning is synthetic.
//
// Usage:
//   minimax_h3 --transformer TF.vkopbin [options]
//   minimax_h3 --vae VAE.vkopbin [options]
//   minimax_h3 --transformer TF.vkopbin --vae VAE.vkopbin [options]
//
//   --steps N            denoising steps to run (default 2)
//   --text-tokens N      packed text rows (default 8)
//   --latent-frames N    video latent frames (default 2)
//   --latent-h N         latent height, patch 2 -> rows = h/2 (default 4)
//   --latent-w N         latent width,  patch 2 -> rows = w/2 (default 4)
//   --audio-latents N    audio latents; rows = N * 2 channels (default 4)
//   --no-audio           drive the video-only graph variant
//   --save DIR           write `sample.f32`, `audio_sample.f32`, `video.f16`
//
// Build: cmake -DENABLE_EXAMPLE=ON .. && make minimax_h3
//
// Why these numbers: the denoiser's `hidden_states` feature count is
// `in_channels * pt*ph*pw` (24 * 1*2*2 = 96) and `audio_hidden_states` is
// `audio_in_channels` (32). Both are weight shapes and are static in the graph,
// which is why they are compile-time constants here and the interesting knobs
// are the ROW counts.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "core/Tensor.hpp"
#include "core/runtime.hpp"
#include "h3_host.hpp"
#include "include/logger.hpp"

#if defined(H3_HAVE_TOKENIZER)
#include "tokenizer.hpp"
#endif
#include "vulkan/VulkanCommandPool.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"

using vkop::VulkanCommandPool;
using vkop::VulkanDevice;
using vkop::VulkanInstance;
using vkop::core::as_tensor;
using vkop::core::ITensor;
using vkop::core::Runtime;

namespace {

// ---------------------------------------------------------------------------
// MiniMax-H3 constants the host owns
// ---------------------------------------------------------------------------

// Modality tags, from diffusers' MINIMAX_H3_{VIDEO,TEXT,AUDIO}_TAG. The AdaLN
// table is addressed as timestep_indices * MODALITY_NUM + token_tags.
constexpr int64_t kVideoTag = 0;
constexpr int64_t kTextTag = 1;
constexpr int64_t kAudioTag = 2;
constexpr int64_t kModalityNum = 3;

// Weight shapes: `in_channels * pt*ph*pw`, `audio_in_channels`, and the text
// tower's hidden width (`text_dim` in the transformer config).
constexpr int kVideoFeatureDim = 96;
constexpr int kAudioFeatureDim = 32;
constexpr int kTextFeatureDim = 5120;
// `time_embed_dim` -- the WIDTH of the temb graph input.
constexpr int kTembDim = 2688;
// VAE latent channels, and its (patch_size_t, patch_size) = (4, 16).
constexpr int kVaeLatentChannels = 24;
constexpr int kVaePatchT = 4;
constexpr int kVaePatch = 16;
// Register tokens (4) + the single all-zero CLS row the ViT appends.
constexpr int kVaeSuffixTokens = 5;

struct Options {
    std::string transformer_path;
    std::string vae_path;
    std::string text_encoder_path;
    // The REAL conditioning path. `prompt` + tokenizer + text encoder replace
    // the synthetic ids/embeds; `checkpoint` supplies the timestep MLP the
    // graph deliberately omits.
    std::string prompt;
    std::string tokenizer_path;
    std::string checkpoint_dir;
    // The VAE's `config.json`, for `latents_mean`/`latents_std`.
    std::string vae_config;
    int steps = 2;
    int text_tokens = 8;
    int latent_frames = 2;
    int latent_h = 4;
    int latent_w = 4;
    int audio_latents = 4;
    bool with_audio = true;
    std::string save_dir;
};

void usage(const char *argv0) {
    std::fprintf(
        stderr,
        "Usage: %s [--transformer TF.vkopbin] [--vae VAE.vkopbin]\n"
        "          [--text-encoder TE.vkopbin] [options]\n"
        "  --prompt TEXT      real prompt (needs --tokenizer and --text-encoder)\n"
        "  --tokenizer BIN    tokenizer .bin (qwen3_vl.bin works: H3's is the\n"
        "                     same Qwen2 tokenizer, byte-identical)\n"
        "  --checkpoint DIR   H3 checkpoint dir (supplies the timestep MLP)\n"
        "  --vae-config JSON  the VAE's config.json (latents_mean/std); with it\n"
        "                     the decoder receives de-normalized latents\n"
        "  --steps N          denoising steps (default 2)\n"
        "  --text-tokens N    synthetic text rows when no --prompt (default 8)\n"
        "  --latent-frames N  video latent frames (default 2)\n"
        "  --latent-h N       latent height, multiple of 2 (default 4)\n"
        "  --latent-w N       latent width, multiple of 2 (default 4)\n"
        "  --audio-latents N  audio latents; rows = N*2 (default 4)\n"
        "  --no-audio         run the video-only denoiser variant\n"
        "  --save DIR         write outputs to DIR\n"
        "\nAt least one of --transformer / --vae / --text-encoder is required.\n",
        argv0);
}

bool next_int(int argc, char **argv, int &i, int &out) {
    if (i + 1 >= argc) return false;
    out = std::atoi(argv[++i]);
    return true;
}

bool parse_args(int argc, char **argv, Options &o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            usage(argv[0]);
            return false;
        } else if (a == "--transformer" && i + 1 < argc) {
            o.transformer_path = argv[++i];
        } else if (a == "--vae" && i + 1 < argc) {
            o.vae_path = argv[++i];
        } else if (a == "--text-encoder" && i + 1 < argc) {
            o.text_encoder_path = argv[++i];
        } else if (a == "--prompt" && i + 1 < argc) {
            o.prompt = argv[++i];
        } else if (a == "--tokenizer" && i + 1 < argc) {
            o.tokenizer_path = argv[++i];
        } else if (a == "--checkpoint" && i + 1 < argc) {
            o.checkpoint_dir = argv[++i];
        } else if (a == "--vae-config" && i + 1 < argc) {
            o.vae_config = argv[++i];
        } else if (a == "--steps" && next_int(argc, argv, i, o.steps)) {
        } else if (a == "--text-tokens" && next_int(argc, argv, i, o.text_tokens)) {
        } else if (a == "--latent-frames" &&
                   next_int(argc, argv, i, o.latent_frames)) {
        } else if (a == "--latent-h" && next_int(argc, argv, i, o.latent_h)) {
        } else if (a == "--latent-w" && next_int(argc, argv, i, o.latent_w)) {
        } else if (a == "--audio-latents" &&
                   next_int(argc, argv, i, o.audio_latents)) {
        } else if (a == "--no-audio") {
            o.with_audio = false;
        } else if (a == "--save" && i + 1 < argc) {
            o.save_dir = argv[++i];
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            usage(argv[0]);
            return false;
        }
    }
    if (o.transformer_path.empty() && o.vae_path.empty() &&
        o.text_encoder_path.empty()) {
        std::fprintf(stderr,
                     "error: give --transformer and/or --vae and/or "
                     "--text-encoder\n");
        usage(argv[0]);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// host-side tensor filling
// ---------------------------------------------------------------------------

// Upload one input from host bytes. The dtype dispatch mirrors every other
// driver in this tree: an unknown type falls back to uint16, which is vkop's
// fp16 storage.
template <typename T>
void upload(const std::shared_ptr<Runtime> &rt,
            const std::shared_ptr<VulkanCommandPool> &pool,
            const std::string &name, const std::vector<T> &host,
            const std::vector<uint32_t> &dims) {
    rt->ResizeInput(name, dims);
    auto t = rt->GetInput(name);
    if (!t) {
        throw std::runtime_error("no such graph input: " + name);
    }
    auto typed = as_tensor<T>(t);
    if (!typed) {
        throw std::runtime_error("graph input " + name +
                                 " does not have the expected storage type");
    }
    typed->fillToCPU(host.data());
    typed->copyToGPU(pool);
}

std::vector<uint16_t> f32_to_f16(const std::vector<float> &src) {
    std::vector<uint16_t> out(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        out[i] = vkop::core::ITensor::fp32_to_fp16(src[i]);
    }
    return out;
}

// Upload `host` as fp32 or fp16 depending on what the graph declared. A
// converted fp16 graph records every activation input as float16, but a
// fp32 conversion records float32, and this driver is used with both.
void upload_float(const std::shared_ptr<Runtime> &rt,
                  const std::shared_ptr<VulkanCommandPool> &pool,
                  const std::string &name, const std::vector<float> &host,
                  const std::vector<uint32_t> &dims) {
    auto probe = rt->GetInput(name);
    if (!probe) {
        throw std::runtime_error("no such graph input: " + name);
    }
    if (probe->dtype() == typeid(float)) {
        upload<float>(rt, pool, name, host, dims);
    } else if (probe->elem_kind() == vkop::core::ElemKind::kBFloat16) {
        // bf16 rides the same uint16_t container as fp16, so `dtype()` alone
        // cannot tell them apart -- and f32_to_f16 on a bf16-declared input
        // hands the graph fp16 bytes under a bf16 label. The graph then reads
        // them with bf16's decode and every activation is garbage (measured:
        // Add nodes in a bf16 artifact receiving mixed 14/2 elem kinds).
        std::vector<uint16_t> bits(host.size());
        for (size_t i = 0; i < host.size(); ++i) {
            bits[i] = static_cast<uint16_t>(
                vkop::core::float_to_bf16(host[i]));
        }
        upload<uint16_t>(rt, pool, name, bits, dims);
    } else {
        upload<uint16_t>(rt, pool, name, f32_to_f16(host), dims);
    }
}

void upload_i64(const std::shared_ptr<Runtime> &rt,
                const std::shared_ptr<VulkanCommandPool> &pool,
                const std::string &name, const std::vector<int64_t> &host,
                const std::vector<uint32_t> &dims) {
    upload<int64_t>(rt, pool, name, host, dims);
}

std::vector<float> random_f32(size_t n, float lo, float hi, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> out(n);
    for (auto &v : out) v = dist(rng);
    return out;
}

// `randn_tensor`: the initial noise of both streams is standard normal, drawn
// video-latents-first then audio-rows (the order is part of what the reference
// generator reproduces, though a single request only needs it self-consistent).
std::vector<float> randn_f32(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> out(n);
    for (auto &v : out) v = dist(rng);
    return out;
}

// ---------------------------------------------------------------------------
// a minimal PNG writer (RGB8, no interlace) -- the only output format the
// board can show without a codec
// ---------------------------------------------------------------------------

namespace png {

inline uint32_t crc32(const uint8_t *data, size_t n) {
    static uint32_t table[256];
    static bool ready = false;
    if (!ready) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1u) ? (0xedb88320u ^ (c >> 1)) : (c >> 1);
            }
            table[i] = c;
        }
        ready = true;
    }
    uint32_t c = 0xffffffffu;
    for (size_t i = 0; i < n; ++i) {
        c = table[(c ^ data[i]) & 0xffu] ^ (c >> 8);
    }
    return c ^ 0xffffffffu;
}

inline void put_u32(std::vector<uint8_t> &v, uint32_t x) {
    v.push_back(static_cast<uint8_t>(x >> 24));
    v.push_back(static_cast<uint8_t>(x >> 16));
    v.push_back(static_cast<uint8_t>(x >> 8));
    v.push_back(static_cast<uint8_t>(x));
}

inline void chunk(std::vector<uint8_t> &out, const char tag[5],
                  const std::vector<uint8_t> &body) {
    put_u32(out, static_cast<uint32_t>(body.size()));
    const size_t crc_start = out.size();
    for (int i = 0; i < 4; ++i) {
        out.push_back(static_cast<uint8_t>(tag[i]));
    }
    out.insert(out.end(), body.begin(), body.end());
    const uint32_t c = crc32(out.data() + crc_start, 4 + body.size());
    put_u32(out, c);
}

// `rgb` is width*height*3 bytes. Row-major, top-left origin.
inline bool write_rgb(const std::string &path, int width, int height,
                      const std::vector<uint8_t> &rgb) {
    if (rgb.size() != static_cast<size_t>(width) * height * 3) {
        return false;
    }
    std::vector<uint8_t> out = {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
    std::vector<uint8_t> ihdr;
    put_u32(ihdr, static_cast<uint32_t>(width));
    put_u32(ihdr, static_cast<uint32_t>(height));
    ihdr.push_back(8);  // bit depth
    ihdr.push_back(2);  // colour type: truecolour
    ihdr.push_back(0);  // deflate
    ihdr.push_back(0);  // adaptive filtering
    ihdr.push_back(0);  // no interlace
    chunk(out, "IHDR", ihdr);

    // Raw scanlines with filter byte 0, wrapped in zlib "stored" blocks. No
    // compression: these frames are a few hundred KB and this keeps the writer
    // dependency-free.
    std::vector<uint8_t> raw;
    raw.reserve((static_cast<size_t>(width) * 3 + 1) * height);
    for (int y = 0; y < height; ++y) {
        raw.push_back(0);
        const uint8_t *row = rgb.data() + static_cast<size_t>(y) * width * 3;
        raw.insert(raw.end(), row, row + static_cast<size_t>(width) * 3);
    }
    std::vector<uint8_t> z;
    z.push_back(0x78);
    z.push_back(0x01);
    size_t pos = 0;
    while (pos < raw.size()) {
        const size_t n = std::min<size_t>(65535, raw.size() - pos);
        z.push_back(pos + n == raw.size() ? 1 : 0);
        z.push_back(static_cast<uint8_t>(n));
        z.push_back(static_cast<uint8_t>(n >> 8));
        z.push_back(static_cast<uint8_t>(~n));
        z.push_back(static_cast<uint8_t>((~n) >> 8));
        z.insert(z.end(), raw.begin() + static_cast<std::ptrdiff_t>(pos),
                 raw.begin() + static_cast<std::ptrdiff_t>(pos + n));
        pos += n;
    }
    uint32_t a = 1, b = 0;
    for (uint8_t c : raw) {
        a = (a + c) % 65521u;
        b = (b + a) % 65521u;
    }
    put_u32(z, (b << 16) | a);
    chunk(out, "IDAT", z);
    chunk(out, "IEND", {});

    std::FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) {
        return false;
    }
    std::fwrite(out.data(), 1, out.size(), f);
    std::fclose(f);
    return true;
}

}  // namespace png

// ---------------------------------------------------------------------------
// the packed layout the denoiser reads by index
// ---------------------------------------------------------------------------

// One packed sequence, built exactly as the reference's
// `MiniMaxH3PrepareLayoutStep.build_packed_sequence` does for the t2va task
// with no keyframe anchors: text rows first, then the audio rows, then the
// video rows. `pack_order` is the permutation from that concat order to the
// packed order, because the wrapper concatenates `[text | video | audio]` and
// then gathers:
//
//     packed = cat([text_embeds, video_embeds, audio_embeds], dim=1)
//              .index_select(1, pack_order)
//
// so `pack_order[packed_row] = row in cat([text | video | audio])`.
struct Layout {
    int n_text = 0;
    int n_audio = 0;
    int n_video = 0;
    int seq = 0;
    std::vector<int64_t> token_tags;
    std::vector<int64_t> timestep_indices;
    std::vector<float> position_ids;   // seq * 3
    std::vector<int64_t> video_indices;
    std::vector<int64_t> audio_indices;
    std::vector<int64_t> text_indices;
    std::vector<int64_t> pack_order;
};

Layout build_layout(const Options &o) {
    Layout l;
    l.n_text = o.text_tokens;
    l.n_video = o.latent_frames * (o.latent_h / 2) * (o.latent_w / 2);
    l.n_audio = o.with_audio ? o.audio_latents * 2 : 0;
    l.seq = l.n_text + l.n_audio + l.n_video;

    const int audio_start = l.n_text;
    const int video_start = audio_start + l.n_audio;

    l.token_tags.assign(l.seq, kVideoTag);
    l.timestep_indices.assign(l.seq, 0);

    // position_ids: text rows walk the time axis at their row index; the media
    // rows continue that clock. Only the T axis advances here because a smoke
    // run does not need the reference's exact (t,h,w) spiral -- but the
    // SHAPE (seq, 3) and the per-row-timestep contract are the graph's.
    l.position_ids.assign(static_cast<size_t>(l.seq) * 3, 0.0F);
    for (int r = 0; r < l.seq; ++r) {
        l.position_ids[r * 3 + 0] = static_cast<float>(r);
        l.position_ids[r * 3 + 1] = 0.0F;
        l.position_ids[r * 3 + 2] = 0.0F;
    }

    for (int r = 0; r < l.n_text; ++r) {
        l.token_tags[r] = kTextTag;
        l.text_indices.push_back(r);
    }
    for (int r = 0; r < l.n_audio; ++r) {
        l.token_tags[audio_start + r] = kAudioTag;
        l.audio_indices.push_back(audio_start + r);
    }
    for (int r = 0; r < l.n_video; ++r) {
        l.token_tags[video_start + r] = kVideoTag;
        l.video_indices.push_back(video_start + r);
    }

    // video_indices / audio_indices / text_indices are positions in the PACKED
    // sequence; with the concat order already [text | audio | video] and the
    // packed order [text | audio | video], they are the same numbers. Build
    // pack_order as the identity of the concat order so the contract is
    // explicit rather than accidental:
    //   cat order   = [text(0..) | video(..) | audio(..)]
    //   packed order= [text | audio | video]
    // pack_order[packed_row] = concat_row
    std::vector<int> concat_of(l.seq);
    for (int r = 0; r < l.n_text; ++r) concat_of[r] = r;
    {
        int c = l.n_text;
        for (int r = 0; r < l.n_video; ++r) concat_of[video_start + r] = c++;
        for (int r = 0; r < l.n_audio; ++r) concat_of[audio_start + r] = c++;
    }
    l.pack_order.assign(l.seq, 0);
    for (int packed_row = 0; packed_row < l.seq; ++packed_row) {
        l.pack_order[packed_row] = concat_of[packed_row];
    }
    return l;
}

// ---------------------------------------------------------------------------
// the timestep MLP (host-side by design)
// ---------------------------------------------------------------------------

// `diffusers.get_timestep_embedding(t, freq_dim=256, flip_sin_to_cos, downscale)
// -> time_proj -> time_embedder`. The graph takes the POST-MLP `temb`
// `(num_timesteps, 2688)` because `arange` on a meta build cannot fold; see the
// exporter README, "Host vs device split".
//
// With `--checkpoint` the real `time_embedder.linear_{1,2}` weights drive this
// (see `h3::TimestepMLP`). Without them this emits a deterministic, finite temb
// of the right SHAPE, which is what the graph contract needs and all a
// weights-free smoke run can check.
std::vector<float> make_temb(int num_timesteps, uint32_t seed) {
    return random_f32(static_cast<size_t>(num_timesteps) * kTembDim,
                      -0.05F, 0.05F, seed);
}

// ---------------------------------------------------------------------------
// the video VAE decoder's rotary grid
// ---------------------------------------------------------------------------

// `MiniMaxH3VideoViTDecoder3d.forward`'s grid, hoisted by the exporter into
// `build_position_grid`: each axis normalized to [-1, 1) as `2*(i+0.5)/size-1`,
// stacked as (t,h,w) over the latent voxels, then `num_register_tokens + 1`
// zero rows appended (all at position 0).
std::vector<float> build_vae_position_grid(int frames, int height, int width) {
    const int tokens = frames * height * width;
    std::vector<float> grid(static_cast<size_t>(tokens + kVaeSuffixTokens) * 3,
                            0.0F);
    auto coord = [](int i, int size) {
        return 2.0F * ((static_cast<float>(i) + 0.5F) / static_cast<float>(size)) - 1.0F;
    };
    int row = 0;
    for (int f = 0; f < frames; ++f) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                grid[static_cast<size_t>(row) * 3 + 0] = coord(f, frames);
                grid[static_cast<size_t>(row) * 3 + 1] = coord(h, height);
                grid[static_cast<size_t>(row) * 3 + 2] = coord(w, width);
                ++row;
            }
        }
    }
    return grid;
}

// ---------------------------------------------------------------------------
// reporting
// ---------------------------------------------------------------------------

struct Stats {
    size_t n = 0;
    size_t nonfinite = 0;
    float min = 0.0F;
    float max = 0.0F;
    double mean = 0.0;
};

// Read one output back and summarize it. `fp16` picks the decoder for the
// stored bytes; a converted fp16 graph writes halfs even for an fp32-typed
// accessor, so the caller passes the graph's dtype.
Stats summarize(const std::shared_ptr<Runtime> &rt,
                const std::shared_ptr<VulkanCommandPool> &pool,
                const std::string &name, bool fp16) {
    Stats s;
    auto t = rt->GetOutput(name);
    if (!t) {
        std::printf("  %-16s <no such graph output>\n", name.c_str());
        return s;
    }
    std::vector<float> host;
    if (fp16) {
        auto typed = as_tensor<uint16_t>(t);
        typed->copyToCPU(pool);
        host.resize(typed->num_elements());
        for (size_t i = 0; i < host.size(); ++i) {
            host[i] = vkop::core::ITensor::fp16_to_fp32((*typed)[i]);
        }
    } else {
        auto typed = as_tensor<float>(t);
        typed->copyToCPU(pool);
        host.assign(typed->data().begin(), typed->data().end());
    }
    s.n = host.size();
    if (host.empty()) {
        std::printf("  %-16s empty\n", name.c_str());
        return s;
    }
    s.min = s.max = host[0];
    for (float v : host) {
        if (!std::isfinite(v)) ++s.nonfinite;
        s.min = std::min(s.min, v);
        s.max = std::max(s.max, v);
        s.mean += v;
    }
    s.mean /= static_cast<double>(host.size());
    std::printf("  %-16s n=%-8zu finite=%-5s min=%.4g max=%.4g mean=%.4g\n",
                name.c_str(), s.n, s.nonfinite == 0 ? "yes" : "NO", s.min,
                s.max, s.mean);
    return s;
}

// One Euler step in H3's convention: the transformer predicts a DATA-WARD
// velocity, so the clean estimate is `x0 = x_t + sigma * v` (the `+` is the
// opposite of diffusers' usual flow-match sign). The update is then the
// `x_t`/`x0` blend at the next sigma, `x_next = r*x_t + (1-r)*x0` with
// `r = sigma_next / sigma`. This is `MiniMaxH3Scheduler.step`'s "blend" form
// (euler_update_form="blend", the checkpoint's default) for the video schedule.
void euler_step(std::vector<float> &x, const std::vector<float> &v,
                float sigma, float sigma_next) {
    const float ratio = sigma_next / sigma;
    const size_t n = std::min(x.size(), v.size());
    for (size_t i = 0; i < n; ++i) {
        const float x0 = x[i] + sigma * v[i];
        x[i] = ratio * x[i] + (1.0F - ratio) * x0;
    }
}

} // namespace

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    // Line-buffer stdout even when redirected: a run is tens of GPU-seconds per
    // step, and block buffering hides progress until the process exits.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    Options opt;
    if (!parse_args(argc, argv, opt)) {
        return 2;
    }

    Logger::getInstance().setLevel(LOG_ERROR);

    // A missing `--save` directory would silently drop every output (the
    // writers just fail to open their file), which reads as "the run produced
    // nothing". Create it up front.
    if (!opt.save_dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(opt.save_dir, ec);
        if (ec) {
            std::fprintf(stderr, "cannot create %s: %s\n", opt.save_dir.c_str(),
                         ec.message().c_str());
            return 2;
        }
    }

    std::shared_ptr<VulkanDevice> dev;
    try {
        const auto &phys = VulkanInstance::getVulkanInstance().getPhysicalDevices();
        for (auto *pdev : phys) {
            auto candidate = std::make_shared<VulkanDevice>(pdev);
            // llvmpipe is a software device; picking it would time a CPU
            // rasterizer and prove nothing about the target GPU.
            if (candidate->getDeviceName().find("llvmpipe") != std::string::npos) {
                continue;
            }
            dev = candidate;
        }
    } catch (const std::exception &e) {
        std::fprintf(stderr, "vulkan init failed: %s\n", e.what());
        return 1;
    }
    if (!dev) {
        std::fprintf(stderr, "no usable Vulkan device\n");
        return 1;
    }
    std::printf("gpu: %s\n", dev->getDeviceName().c_str());
    auto pool = std::make_shared<VulkanCommandPool>(dev);

    int failures = 0;

    // -----------------------------------------------------------------
    // REAL conditioning, computed before the sampler runs
    // -----------------------------------------------------------------
    //
    // Three host-side pieces the graphs deliberately omit:
    //   tokenizer      -> input_ids          (Qwen2 BPE, same as the LLM's)
    //   text_encoder   -> prompt_embeds      (Qwen3-VL tower, hidden_states[50])
    //   checkpoint MLP -> temb               (time_proj + time_embedder)
    // When any is missing the denoiser falls back to synthetic inputs, which
    // still exercises the graph but produces noise.
    std::vector<float> prompt_embeds;      // [n_text, 5120]
    std::vector<float> real_temb;          // [num_timesteps, 2688]
    int n_text_tokens = 0;
    // The denoiser's last `sample`, as `[C, F, H, W]` latents: the VAE's input.
    // Empty when the transformer did not run.
    std::vector<float> video_latents;


    // The sigma grids the sampler below walks. They are built here, before the
    // timestep MLP, because `temb` needs rows for every DISTINCT timestep (the
    // AdaLN modulation is a function of t).
    //
    // Video and audio run their own schedules -- `scheduler/` has shift 12 and
    // `audio_scheduler/` shift 3 -- so one step sees two different timesteps and
    // `timestep_indices` is what tells the graph which row each packed row
    // wants. `MiniMaxH3Scheduler.set_timesteps`: `t = 1 - sigma` over
    // `shift*base/(1+(shift-1)*base)` for `base = linspace(1, 0, steps+1)`.
    const int steps = std::max(1, opt.steps);
    auto build_grid = [steps](float shift) {
        std::vector<float> g(static_cast<size_t>(steps) + 1);
        for (int i = 0; i <= steps; ++i) {
            const float base =
                1.0F - static_cast<float>(i) / static_cast<float>(steps);
            g[static_cast<size_t>(i)] = shift * base / (1.0F + (shift - 1.0F) * base);
        }
        return g;
    };
    const std::vector<float> sigmas = build_grid(12.0F);       // video
    const std::vector<float> audio_sigmas = build_grid(3.0F);  // audio

    const bool want_text = !opt.text_encoder_path.empty() && !opt.prompt.empty();
    if (want_text) {
        if (opt.tokenizer_path.empty()) {
            throw std::runtime_error(
                "--prompt needs --tokenizer <bin> (qwen3_vl.bin works: H3's "
                "tokenizer is byte-identical)");
        }
#if defined(H3_HAVE_TOKENIZER)
        qwen::Tokenizer tok;
        tok.load(opt.tokenizer_path);
        const std::vector<uint32_t> ids = tok.encode(opt.prompt);
        n_text_tokens = static_cast<int>(ids.size());
        std::printf("\n[text] prompt -> %d token(s)\n", n_text_tokens);

        auto rt_te = std::make_shared<Runtime>(pool, opt.text_encoder_path,
                                               /*precision=*/1);
        rt_te->set_backend_buffer(true);
        rt_te->LoadModel();

        std::vector<int64_t> id64(ids.begin(), ids.end());
        upload_i64(rt_te, pool, "input_ids", id64,
                          {1, static_cast<uint32_t>(n_text_tokens)});

        // Strictly-causal additive mask, in the graph's own dtype.
        auto bias_t = rt_te->GetInput("attention_bias");
        if (!bias_t) throw std::runtime_error("text encoder has no attention_bias");
        const bool bias_fp32 = bias_t->dtype() == typeid(float);
        const float neg = bias_fp32 ? -1.0e4F : -65504.0F;
        std::vector<float> bias(static_cast<size_t>(n_text_tokens) * n_text_tokens,
                                0.0F);
        for (int r = 0; r < n_text_tokens; ++r) {
            for (int c = r + 1; c < n_text_tokens; ++c) {
                bias[static_cast<size_t>(r) * n_text_tokens + c] = neg;
            }
        }
        upload_float(rt_te, pool, "attention_bias", bias,
                     {1, 1, static_cast<uint32_t>(n_text_tokens),
                      static_cast<uint32_t>(n_text_tokens)});

        auto ms = rt_te->Run();
        rt_te->ReadResult();
        auto pe = rt_te->GetOutput("prompt_embeds");
        if (!pe) throw std::runtime_error("text encoder produced no prompt_embeds");
        const bool pe_fp16 = pe->dtype() == typeid(uint16_t);
        prompt_embeds.resize(static_cast<size_t>(n_text_tokens) * kTextFeatureDim);
        if (pe_fp16) {
            auto t = as_tensor<uint16_t>(pe);
            for (size_t i = 0; i < prompt_embeds.size(); ++i)
                prompt_embeds[i] = vkop::core::ITensor::fp16_to_fp32((*t)[i]);
        } else {
            auto t = as_tensor<float>(pe);
            prompt_embeds.assign(t->data().begin(), t->data().end());
        }
        double sum = 0.0F, amax = 0.0F;
        size_t nonfinite = 0;
        for (float x : prompt_embeds) {
            if (!std::isfinite(x)) {
                ++nonfinite;
                continue;
            }
            sum += x;
            amax = std::max(amax, static_cast<double>(std::fabs(x)));
        }
        std::printf("[text] prompt_embeds [%d, %d] in %.0f ms  nonfinite=%zu "
                    "absmax=%.3f mean=%.4f\n",
                    n_text_tokens, kTextFeatureDim, ms, nonfinite, amax,
                    sum / static_cast<double>(prompt_embeds.size()));
        if (nonfinite != 0) ++failures;
#else
        throw std::runtime_error(
            "built without the tokenizer (ENABLE_LLM_CHAT off and "
            "llm/tokenizer unavailable); rebuild with the tokenizer");
#endif
    }

    // Per step, the DISTINCT timesteps of the packed sequence, `sorted`
    // ascending, plus each modality's index into them -- exactly
    // `MiniMaxH3PrepareLayoutStep.build_row_timesteps`'s
    // `torch.unique(row_timesteps, sorted=True, return_inverse=True)`.
    // Video and audio run different shifts, so a step usually has two rows; at
    // step 0 both sigmas are 1 and they collapse to one. Text rows inherit the
    // video timestep (they never reach an output head), so they share its row.
    std::vector<int> step_row_offset(static_cast<size_t>(steps) + 1, 0);
    std::vector<int> step_row_count(static_cast<size_t>(steps), 1);
    std::vector<int> audio_row_index(static_cast<size_t>(steps), 0);
    std::vector<float> temb_times;

    for (int i = 0; i < steps; ++i) {
        const float vt = 1.0F - sigmas[static_cast<size_t>(i)];
        const float at = 1.0F - audio_sigmas[static_cast<size_t>(i)];
        step_row_offset[static_cast<size_t>(i)] =
            static_cast<int>(temb_times.size());
        if (at > vt) {
            // sigma_video > sigma_audio for every interior base, so the video t
            // is the smaller of the two.
            temb_times.push_back(vt);
            temb_times.push_back(at);
            step_row_count[static_cast<size_t>(i)] = 2;
            audio_row_index[static_cast<size_t>(i)] = 1;
        } else {
            temb_times.push_back(vt);
            step_row_count[static_cast<size_t>(i)] = 1;
            audio_row_index[static_cast<size_t>(i)] = 0;
        }
    }
    step_row_offset[static_cast<size_t>(steps)] =
        static_cast<int>(temb_times.size());

    if (!opt.checkpoint_dir.empty()) {
        h3::TimestepMLP mlp(opt.checkpoint_dir);
        real_temb = mlp.run(temb_times);
        if (mlp.out_dim() != kTembDim) {
            throw std::runtime_error("timestep MLP out_dim " +
                                     std::to_string(mlp.out_dim()) +
                                     " != the graph's 2688");
        }
        std::printf("[temb] checkpoint timestep MLP loaded (%d -> %d), %zu row(s), "
                    "t = %.4f .. %.4f, %d row(s) on the busiest step\n",
                    mlp.hidden(), mlp.out_dim(), temb_times.size(),
                    temb_times.front(), temb_times.back(),
                    *std::max_element(step_row_count.begin(),
                                      step_row_count.end()));
    }

    // -----------------------------------------------------------------
    // denoiser: one graph forward per denoising step
    // -----------------------------------------------------------------
    if (!opt.transformer_path.empty()) {
        try {
            // The layout uses the REAL prompt length when one was encoded.
            Layout l;
            if (n_text_tokens > 0) {
                h3::LayoutSpec spec;
                spec.num_text_tokens = n_text_tokens;
                spec.num_latent_frames = opt.latent_frames;
                spec.latent_height = opt.latent_h;
                spec.latent_width = opt.latent_w;
                spec.num_audio_latents = opt.audio_latents;
                spec.audio_channels = 2;
                spec.patch_t = 1; spec.patch_h = 2; spec.patch_w = 2;
                const h3::PackedLayout pl = h3::build_packed_layout(spec);
                l.n_text = pl.n_text; l.n_audio = pl.n_audio;
                l.n_video = pl.n_video; l.seq = pl.seq;
                l.token_tags = pl.token_tags;
                l.timestep_indices = pl.timestep_indices;
                l.position_ids = pl.position_ids;
                l.video_indices = pl.video_indices;
                l.audio_indices = pl.audio_indices;
                l.text_indices = pl.text_indices;
                l.pack_order = pl.pack_order;
                if (!l.audio_indices.empty() && !opt.with_audio) {
                    // video-only graph has no audio rows
                    l.n_audio = 0; l.seq = l.n_text + l.n_video;
                }
            } else {
                l = build_layout(opt);
            }
            std::printf(
                "\n[transformer] %s\n  layout: text=%d audio=%d video=%d "
                "seq=%d steps=%d%s\n",
                opt.transformer_path.c_str(), l.n_text, l.n_audio, l.n_video,
                l.seq, opt.steps,
                n_text_tokens > 0 ? " (real text conditioning)" : "");

            // precision=1: the artifacts are converted -q fp16, and the fp16
            // shader family is what they embed.
            auto rt = std::make_shared<Runtime>(pool, opt.transformer_path,
                                                /*precision=*/1);
            // Conv3d/MatMul/all the vector ops below live on the buffer
            // backend; the image backend has no 5-D conv.
            rt->set_backend_buffer(true);
            rt->LoadModel();

            const size_t n_vid =
                static_cast<size_t>(l.n_video) * kVideoFeatureDim;
            const size_t n_aud =
                static_cast<size_t>(l.n_audio) * kAudioFeatureDim;
            const size_t n_txt =
                static_cast<size_t>(l.n_text) * kTextFeatureDim;

            // The noisy rows. The host owns the sampler; the graph only sees
            // the current x_t. When a prompt was encoded, `text` is the real
            // conditioner output; otherwise it is noise (a graph smoke test).
            std::vector<float> video = randn_f32(n_vid, 0x1234u);
            std::vector<float> audio = randn_f32(n_aud, 0x5678u);
            const std::vector<float> text =
                prompt_embeds.empty()
                    ? random_f32(n_txt, -1.0F, 1.0F, 0x9abcu)
                    : prompt_embeds;
            // A real checkpoint supplies this step's rows inside the loop; the
            // synthetic path uploads its single row here.
            const std::vector<float> temb =
                real_temb.empty()
                    ? make_temb(1, 0xdef0u)
                    : std::vector<float>(
                          real_temb.begin(),
                          real_temb.begin() +
                              static_cast<std::ptrdiff_t>(
                                  step_row_count[0] * kTembDim));

            upload_float(rt, pool, "hidden_states", video,
                         {1, static_cast<uint32_t>(l.n_video), kVideoFeatureDim});
            upload_float(rt, pool, "encoder_hidden_states", text,
                         {1, static_cast<uint32_t>(l.n_text),
                          static_cast<uint32_t>(kTextFeatureDim)});
            upload_float(rt, pool, "temb", temb,
                         {static_cast<uint32_t>(temb.size() / kTembDim),
                          static_cast<uint32_t>(kTembDim)});
            upload_i64(rt, pool, "timestep_indices", l.timestep_indices,
                       {static_cast<uint32_t>(l.seq)});
            upload_i64(rt, pool, "token_tags", l.token_tags,
                       {static_cast<uint32_t>(l.seq)});
            upload_float(rt, pool, "position_ids", l.position_ids,
                         {static_cast<uint32_t>(l.seq), 3});
            upload_i64(rt, pool, "video_indices", l.video_indices,
                       {static_cast<uint32_t>(l.n_video)});
            upload_i64(rt, pool, "text_indices", l.text_indices,
                       {static_cast<uint32_t>(l.n_text)});

            const bool joint = l.n_audio > 0;
            if (joint) {
                upload_float(rt, pool, "audio_hidden_states", audio,
                             {1, static_cast<uint32_t>(l.n_audio),
                              kAudioFeatureDim});
                upload_i64(rt, pool, "audio_indices", l.audio_indices,
                           {static_cast<uint32_t>(l.n_audio)});
                upload_i64(rt, pool, "pack_order", l.pack_order,
                           {static_cast<uint32_t>(l.seq)});
            }

            // `sigmas` was built before the timestep MLP: it doubles as the
            // `t` grid that produces `temb`, so it must exist by then.
            const size_t temb_stride = static_cast<size_t>(kTembDim);
            const bool real_mlp = !real_temb.empty();

            for (int step = 0; step < steps; ++step) {
                if (real_mlp) {
                    // This step's DISTINCT timesteps, sorted, and the matching
                    // `timestep_indices`. Uploading both per step is what makes
                    // the trajectory track t; a single fixed row is the
                    // smoke-test shortcut.
                    const int off = step_row_offset[static_cast<size_t>(step)];
                    const int cnt = step_row_count[static_cast<size_t>(step)];
                    const std::vector<float> rows(
                        real_temb.begin() +
                            static_cast<std::ptrdiff_t>(off) * temb_stride,
                        real_temb.begin() +
                            static_cast<std::ptrdiff_t>(off + cnt) * temb_stride);
                    upload_float(rt, pool, "temb", rows,
                                 {static_cast<uint32_t>(cnt),
                                  static_cast<uint32_t>(kTembDim)});
                    // `build_row_timesteps`: text inherits the video row (0);
                    // the audio rows take their own when the two differ.
                    std::vector<int64_t> idx = l.timestep_indices;
                    for (int r : l.audio_indices) {
                        idx[static_cast<size_t>(r)] =
                            audio_row_index[static_cast<size_t>(step)];
                    }
                    upload_i64(rt, pool, "timestep_indices", idx,
                               {static_cast<uint32_t>(l.seq)});
                }
                auto ms = rt->Run();
                std::printf("  step %d/%d  t=%.4f sigma=%.4f->%.4f  run %.1f ms\n",
                            step + 1, steps, 1.0F - sigmas[static_cast<size_t>(step)],
                            sigmas[static_cast<size_t>(step)],
                            sigmas[static_cast<size_t>(step) + 1], ms);

                auto s = rt->GetOutput("sample");
                if (!s) {
                    std::fprintf(stderr, "  no `sample` output on this graph\n");
                    ++failures;
                    break;
                }
                const bool fp16 = s->dtype() == typeid(uint16_t);
                rt->ReadResult();
                // Feed the velocity back through one Euler step, then upload
                // the next x_t. Readback is a host->device round trip, which is
                // exactly what a sampler does between steps.
                Stats st = summarize(rt, pool, "sample", fp16);
                if (st.nonfinite != 0) ++failures;
                if (joint) {
                    Stats at = summarize(rt, pool, "audio_sample", fp16);
                    if (at.nonfinite != 0) ++failures;
                }
                if (st.n == 0) break;

                // Reuse the last step's velocity rather than re-reading it: a
                // real sampler reads `sample` and applies the scheduler, which
                // is the same shape the graph wrote.
                std::vector<float> vel;
                if (fp16) {
                    auto typed = as_tensor<uint16_t>(s);
                    vel.resize(typed->num_elements());
                    for (size_t i = 0; i < vel.size(); ++i)
                        vel[i] = vkop::core::ITensor::fp16_to_fp32((*typed)[i]);
                } else {
                    auto typed = as_tensor<float>(s);
                    vel.assign(typed->data().begin(), typed->data().end());
                }
                euler_step(video, vel, sigmas[step], sigmas[step + 1]);
                upload_float(rt, pool, "hidden_states", video,
                             {1, static_cast<uint32_t>(l.n_video),
                              kVideoFeatureDim});
                if (joint && !audio.empty()) {
                    auto a = rt->GetOutput("audio_sample");
                    if (a) {
                        std::vector<float> avel;
                        if (fp16) {
                            auto typed = as_tensor<uint16_t>(a);
                            avel.resize(typed->num_elements());
                            for (size_t i = 0; i < avel.size(); ++i)
                                avel[i] = vkop::core::ITensor::fp16_to_fp32((*typed)[i]);
                        } else {
                            auto typed = as_tensor<float>(a);
                            avel.assign(typed->data().begin(), typed->data().end());
                        }
                        // The audio rows step down their OWN schedule
                        // (`audio_scheduler/`, shift 3), not the video one.
                        euler_step(audio, avel, audio_sigmas[step],
                                   audio_sigmas[step + 1]);
                        upload_float(rt, pool, "audio_hidden_states", audio,
                                     {1, static_cast<uint32_t>(l.n_audio),
                                      kAudioFeatureDim});
                    }
                }
            }

            // The denoiser speaks patch rows; the VAE speaks volumes. Undo
            // `patchify` here, exactly as `MiniMaxH3AfterDenoiseStep` does
            // (frame-major row order, patch axes innermost).
            video_latents = h3::unpatchify(
                video, kVideoFeatureDim / (1 * 2 * 2), opt.latent_frames,
                opt.latent_h, opt.latent_w, /*pt=*/1, /*ph=*/2, /*pw=*/2);
            std::printf("  latents [%d, %d, %d, %d]\n", kVaeLatentChannels,
                        opt.latent_frames, opt.latent_h, opt.latent_w);

            if (!opt.save_dir.empty()) {
                std::string path = opt.save_dir + "/sample.f32";
                std::FILE *f = std::fopen(path.c_str(), "wb");
                if (!f) {
                    std::fprintf(stderr, "  cannot write %s\n", path.c_str());
                    ++failures;
                } else {
                    std::fwrite(video.data(), sizeof(float), video.size(), f);
                    std::fclose(f);
                    std::printf("  saved %s (%zu floats)\n", path.c_str(),
                                video.size());
                }
            }
        } catch (const std::exception &e) {
            std::fprintf(stderr, "transformer: %s\n", e.what());
            ++failures;
        }
    }

    // -----------------------------------------------------------------
    // text encoder: input_ids -> prompt_embeds
    // -----------------------------------------------------------------
    // Skip this synthetic pass when `--prompt` already ran the tower above: the
    // real conditioning is the one the denoiser saw, so re-encoding random ids
    // would only cost a load.
    if (!opt.text_encoder_path.empty() && !want_text) {
        try {
            const int seq = std::max(1, opt.text_tokens);
            std::printf("\n[text_encoder] %s\n  seq=%d\n",
                        opt.text_encoder_path.c_str(), seq);

            auto rt = std::make_shared<Runtime>(pool, opt.text_encoder_path,
                                                /*precision=*/1);
            rt->set_backend_buffer(true);
            rt->LoadModel();

            // A real run tokenizes the prompt. This smoke driver has no
            // tokenizer, so it feeds in-vocabulary ids 1..seq and a strictly
            // causal additive mask in the graph's own dtype.
            std::vector<int64_t> ids(static_cast<size_t>(seq));
            for (int i = 0; i < seq; ++i) {
                ids[static_cast<size_t>(i)] = 1 + (i % 100);
            }
            upload_i64(rt, pool, "input_ids", ids,
                       {1, static_cast<uint32_t>(seq)});

            // The mask's dtype is whatever the graph declares (fp16 now that
            // the exporter builds the tower coherently; fp32 would mean an
            // older artifact). The runtime requires it to MATCH Q's dtype.
            auto bias_t = rt->GetInput("attention_bias");
            if (!bias_t) {
                throw std::runtime_error("no `attention_bias` input");
            }
            const bool bias_fp32 = bias_t->dtype() == typeid(float);
            const float neg = bias_fp32 ? -1.0e4F : -65504.0F;
            std::vector<float> bias(static_cast<size_t>(seq) * seq, 0.0F);
            for (int r = 0; r < seq; ++r) {
                for (int c = r + 1; c < seq; ++c) {
                    bias[static_cast<size_t>(r) * seq + c] = neg;
                }
            }
            upload_float(rt, pool, "attention_bias", bias,
                         {1, 1, static_cast<uint32_t>(seq),
                          static_cast<uint32_t>(seq)});

            auto ms = rt->Run();
            std::printf("  run %.1f ms\n", ms);
            rt->ReadResult();

            auto pe = rt->GetOutput("prompt_embeds");
            const bool fp16 = pe && pe->dtype() == typeid(uint16_t);
            Stats st = summarize(rt, pool, "prompt_embeds", fp16);
            if (st.nonfinite != 0) ++failures;
        } catch (const std::exception &e) {
            std::fprintf(stderr, "text_encoder: %s\n", e.what());
            ++failures;
        }
    }

    // -----------------------------------------------------------------
    // video VAE decoder: latents -> pixels
    // -----------------------------------------------------------------
    if (!opt.vae_path.empty()) {
        try {
            const int frames = opt.latent_frames;
            const int height = opt.latent_h;
            const int width = opt.latent_w;
            const int tokens = frames * height * width;
            std::printf(
                "\n[vae] %s\n  latents: 1x%d(frames)x%d(h)x%d(w) -> %d tokens "
                "+ %d suffix\n",
                opt.vae_path.c_str(), frames, height, width, tokens,
                kVaeSuffixTokens);

            auto rt = std::make_shared<Runtime>(pool, opt.vae_path,
                                                /*precision=*/1);
            rt->set_backend_buffer(true);
            rt->LoadModel();

            // `MiniMaxH3VideoDecodeStep`: the sampler produces NORMALIZED
            // latents, and the graph starts at `decode`, so the de-normalization
            // `latents * std + mean` (per channel) happens here.
            std::vector<float> latents;
            const bool real_latents =
                video_latents.size() ==
                static_cast<size_t>(kVaeLatentChannels) * frames * height * width;
            if (real_latents) {
                latents = video_latents;
                const std::string cfg_path =
                    opt.vae_config.empty() ? std::string() : opt.vae_config;
                std::vector<float> mean, stdv;
                if (!cfg_path.empty()) {
                    std::ifstream cf(cfg_path);
                    const std::string json((std::istreambuf_iterator<char>(cf)),
                                           std::istreambuf_iterator<char>());
                    mean = h3::read_float_array(json, "latents_mean");
                    stdv = h3::read_float_array(json, "latents_std");
                }
                if (mean.size() == static_cast<size_t>(kVaeLatentChannels) &&
                    stdv.size() == static_cast<size_t>(kVaeLatentChannels)) {
                    const size_t plane =
                        static_cast<size_t>(frames) * height * width;
                    for (int c = 0; c < kVaeLatentChannels; ++c) {
                        const float m = mean[static_cast<size_t>(c)];
                        const float sd = stdv[static_cast<size_t>(c)];
                        for (size_t i = 0; i < plane; ++i) {
                            float &v = latents[static_cast<size_t>(c) * plane + i];
                            v = v * sd + m;
                        }
                    }
                    std::printf("  latents de-normalized from %s\n",
                                cfg_path.c_str());
                } else {
                    std::printf(
                        "  WARNING: no latents_mean/std (pass --vae-config); the "
                        "decoder sees un-normalized latents\n");
                }
            } else {
                std::printf(
                    "  WARNING: no denoised latents; decoding seeded noise\n");
                latents = randn_f32(
                    static_cast<size_t>(kVaeLatentChannels) * frames * height *
                        width,
                    0xc0deu);
            }
            const std::vector<float> grid =
                build_vae_position_grid(frames, height, width);

            upload_float(rt, pool, "latents", latents,
                         {1, kVaeLatentChannels, static_cast<uint32_t>(frames),
                          static_cast<uint32_t>(height),
                          static_cast<uint32_t>(width)});
            upload_float(rt, pool, "position_ids", grid,
                         {1, static_cast<uint32_t>(tokens + kVaeSuffixTokens),
                          3});

            auto ms = rt->Run();
            std::printf("  run %.1f ms\n", ms);
            rt->ReadResult();

            auto v = rt->GetOutput("video");
            const bool fp16 = v && v->dtype() == typeid(uint16_t);
            Stats st = summarize(rt, pool, "video", fp16);
            if (st.nonfinite != 0) ++failures;

            if (!opt.save_dir.empty() && v) {
                std::string path = opt.save_dir + "/video.f16";
                std::FILE *f = std::fopen(path.c_str(), "wb");
                if (f) {
                    if (fp16) {
                        auto typed = as_tensor<uint16_t>(v);
                        std::fwrite(&(*typed)[0], sizeof(uint16_t),
                                    typed->num_elements(), f);
                        std::printf("  saved %s (%d halfs)\n", path.c_str(),
                                    typed->num_elements());
                    } else {
                        auto typed = as_tensor<float>(v);
                        std::fwrite(&(*typed)[0], sizeof(float),
                                    typed->num_elements(), f);
                        std::printf("  saved %s (%d floats)\n", path.c_str(),
                                    typed->num_elements());
                    }
                    std::fclose(f);
                }

                // `MiniMaxH3VideoDecodeStep`: the decoder emits ImageNet-
                // normalized RGB, so `v * std + mean`, then clamp to [0,1].
                const int pw_frames = frames * kVaePatchT;
                const int ph_px = height * kVaePatch;
                const int pw_px = width * kVaePatch;
                const size_t need =
                    static_cast<size_t>(3) * pw_frames * ph_px * pw_px;
                std::vector<float> f32;
                if (fp16) {
                    auto typed = as_tensor<uint16_t>(v);
                    f32.resize(typed->num_elements());
                    for (size_t i = 0; i < f32.size(); ++i)
                        f32[i] = ITensor::fp16_to_fp32((*typed)[i]);
                } else {
                    auto typed = as_tensor<float>(v);
                    f32.assign(typed->data().begin(), typed->data().end());
                }
                if (f32.size() == need) {
                    const float im_mean[3] = {0.485F, 0.456F, 0.406F};
                    const float im_std[3] = {0.229F, 0.224F, 0.225F};
                    const size_t plane =
                        static_cast<size_t>(ph_px) * pw_px;
                    int written = 0;
                    for (int t = 0; t < pw_frames; ++t) {
                        std::vector<uint8_t> rgb(plane * 3);
                        for (int c = 0; c < 3; ++c) {
                            const size_t off =
                                (static_cast<size_t>(c) * pw_frames + t) * plane;
                            for (size_t i = 0; i < plane; ++i) {
                                float x = f32[off + i] * im_std[c] + im_mean[c];
                                x = std::min(1.0F, std::max(0.0F, x));
                                rgb[i * 3 + static_cast<size_t>(c)] =
                                    static_cast<uint8_t>(x * 255.0F + 0.5F);
                            }
                        }
                        char name[64];
                        std::snprintf(name, sizeof(name), "/frame_%03d.png", t);
                        if (png::write_rgb(opt.save_dir + name, pw_px, ph_px,
                                           rgb)) {
                            ++written;
                        }
                    }
                    std::printf("  wrote %d PNG frame(s) %dx%d\n", written,
                                pw_px, ph_px);
                } else {
                    std::printf(
                        "  (video has %zu floats, expected %zu for 3x%dx%dx%d; "
                        "no PNG)\n",
                        f32.size(), need, pw_frames, ph_px, pw_px);
                }
            }
        } catch (const std::exception &e) {
            std::fprintf(stderr, "vae: %s\n", e.what());
            ++failures;
        }
    }

    if (failures) {
        std::printf("\nFAILED (%d problem(s))\n", failures);
        return 1;
    }
    std::printf("\nOK\n");
    return 0;
}
