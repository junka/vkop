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
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "core/Tensor.hpp"
#include "core/runtime.hpp"
#include "include/logger.hpp"
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
        "  --steps N          denoising steps (default 2)\n"
        "  --text-tokens N    text rows (default 8)\n"
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
// A real driver loads `time_embedder.linear_{1,2}` from the checkpoint. This
// example has no weights, so it emits a deterministic, finite temb of the right
// SHAPE -- which is what the graph contract needs and what a smoke run checks.
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
    Options opt;
    if (!parse_args(argc, argv, opt)) {
        return 2;
    }

    Logger::getInstance().setLevel(LOG_ERROR);

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
    // denoiser: one graph forward per denoising step
    // -----------------------------------------------------------------
    if (!opt.transformer_path.empty()) {
        try {
            const Layout l = build_layout(opt);
            std::printf(
                "\n[transformer] %s\n  layout: text=%d audio=%d video=%d "
                "seq=%d steps=%d\n",
                opt.transformer_path.c_str(), l.n_text, l.n_audio, l.n_video,
                l.seq, opt.steps);

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
            // the current x_t.
            std::vector<float> video = random_f32(n_vid, -1.0F, 1.0F, 0x1234u);
            std::vector<float> audio = random_f32(n_aud, -1.0F, 1.0F, 0x5678u);
            const std::vector<float> text = random_f32(n_txt, -1.0F, 1.0F, 0x9abcu);
            const std::vector<float> temb = make_temb(1, 0xdef0u);

            upload_float(rt, pool, "hidden_states", video,
                         {1, static_cast<uint32_t>(l.n_video), kVideoFeatureDim});
            upload_float(rt, pool, "encoder_hidden_states", text,
                         {1, static_cast<uint32_t>(l.n_text), static_cast<uint32_t>(kTextFeatureDim)});
            upload_float(rt, pool, "temb", temb,
                         {1, static_cast<uint32_t>(kTembDim)});
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

            // sigma schedule: linspace(1, 0, steps) through the exponential
            // shift, shift = 12 for video. Kept on the host, as the reference
            // does.
            const int steps = std::max(1, opt.steps);
            std::vector<float> sigmas(static_cast<size_t>(steps) + 1);
            for (int i = 0; i <= steps; ++i) {
                const float base =
                    1.0F - static_cast<float>(i) / static_cast<float>(steps);
                sigmas[i] = 12.0F * base / (1.0F + 11.0F * base);
            }

            for (int step = 0; step < steps; ++step) {
                auto ms = rt->Run();
                std::printf("  step %d/%d  run %.1f ms\n", step + 1, steps, ms);

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
                    // The audio stream has its own schedule (shift 3); a smoke
                    // run steps it with the same ratio, which is fine for
                    // finiteness but is NOT the reference's audio schedule.
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
                        euler_step(audio, avel, sigmas[step], sigmas[step + 1]);
                        upload_float(rt, pool, "audio_hidden_states", audio,
                                     {1, static_cast<uint32_t>(l.n_audio),
                                      kAudioFeatureDim});
                    }
                }
            }

            if (!opt.save_dir.empty()) {
                std::string path = opt.save_dir + "/sample.f32";
                std::FILE *f = std::fopen(path.c_str(), "wb");
                if (f) {
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
    if (!opt.text_encoder_path.empty()) {
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

            const std::vector<float> latents = random_f32(
                static_cast<size_t>(kVaeLatentChannels) * frames * height * width,
                -1.0F, 1.0F, 0xc0deu);
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

            if (!opt.save_dir.empty() && fp16) {
                auto typed = as_tensor<uint16_t>(v);
                std::string path = opt.save_dir + "/video.f16";
                std::FILE *f = std::fopen(path.c_str(), "wb");
                if (f) {
                    std::fwrite(&(*typed)[0], sizeof(uint16_t),
                                typed->num_elements(), f);
                    std::fclose(f);
                    std::printf("  saved %s (%d halfs)\n", path.c_str(),
                                typed->num_elements());
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
