// MiniMax-H3 FL2VA / Ref2VA denoiser driver (ComfyUI single-file checkpoint).
//
// Runs the text-to-audio-video sampling loop around the one-step DiT graph
// that model/exporter/minimax_h3/h3_export_fl2va.py exports. The host side
// mirrors ComfyUI (comfy/ldm/minimax/model.py; t2va, and ref2va with
// reference images; no keyframes):
//
//   * packed layout [text | ref images | audio (stereo, channel-major) |
//     video (2x2 patches)] and its (t, h, w) RoPE positions; the graph gets
//     cos/sin of the 48 split-half angles. A reference image is one latent
//     frame at its own size, placed at t = text_len + i ahead of the targets,
//     re-injected every step at timestep max(t_v, 0.999) (never denoised),
//   * per-text-token adaLN tags (--text-tags: 1 text, 0 vision pads),
//   * per-stream timesteps t = 1 - sigma, the audio stream on its own shift
//     (video 12, audio 3); adaLN reads the curve table at each distinct t,
//   * Euler on each stream's own sigma grid: the head predicts x0 - noise,
//     so x <- x + (sigma - sigma_next) * out.
//
// The text tower and the VAEs are not wired yet: without --text-states the
// conditioning is synthetic, so the result exercises the denoiser, not the
// prompt. Outputs (--save DIR): video latent [24, T, H, W] and audio latent
// [32, 2, Ta], raw little-endian fp32, in the sampler's (un-normalized) space.
#include <algorithm>
#include <cctype>
#include <iterator>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "core/Tensor.hpp"
#include "core/runtime.hpp"
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

constexpr int kVideoC = 24;
constexpr int kAudioC = 32;
constexpr int kTextDim = 5120;
constexpr int kRopeAxes = 3;
// Base-model shifts; the 4-step turbo LoRA wants video 6 (--shift-video).
double kShiftVideo = 12.0;
double kShiftAudio = 3.0;
// comfy's spatial VAE tile in px; 0 = auto (Vae::auto_tile), comfy uses 256.
int kTileSize = 0;
constexpr double kFrameRescale = 5.0 / 3.0;
constexpr int kFramePerToken[5] = {1, 4, 4, 4, 4};
constexpr int kFps = 24;
constexpr int kAudioLatentFps = 40;

struct Options {
    std::string dit, host_dir, text_states, save_dir;
    // text tower (h3_export_fl2va_te.py): graph, bf16 embedding table, and
    // the prompt's raw token ids (int64; no template, no special tokens)
    std::string te, embed, prompt_ids;
    // video VAE decoder (h3_export_fl2va_vae.py) and where frames go
    std::string vae, frames_dir;
    // decode this saved video latent [24, T, H, W] instead of sampling
    std::string latent;
    // write the text tower's states here and exit (the DiT then runs in a
    // fresh process with --text-states, after the system reclaimed the tower)
    std::string text_out;
    // audio VAE decoder (h3_export_fl2va_audio_vae.py) -> stereo 32 kHz wav;
    // --audio-latent decodes a saved [32, 2, T] audio latent with only it loaded
    std::string audio_vae, wav, audio_latent;
    // ref2va: normalized reference-image latents [24, 1, HL, WL] ("PATH:HL:WL",
    // in presentation order) and the text tower's per-token adaLN tags (int64,
    // 1 text / 0 vision pad; default all 1)
    std::vector<std::string> ref_latents;
    // ...or reference images as pixels [3, TH, TW] in [0, 1] ("PATH:TH:TW",
    // h3_ref2va_prep.py vae), encoded here by the video VAE encoder
    // (h3_export_fl2va_vae_encoder.py); latents first, then images
    std::vector<std::string> ref_images;
    std::string vae_enc;
    // ref2va text side (h3_ref2va_prep.py qwen --out DIR): ids, tags, MRoPE
    // tables and the vision blocks' patches; the Qwen3-VL vision tower
    // (h3_export_ref2va_vision.py) fills the pads and the DeepStack inputs
    std::string qwen_dir, vision;
    std::string text_tags, dump_layout;
    double cond_aug = 0.999;  // comfy VISUAL_COND_TIMESTEP / noise aug
    // euler (lightx2v / the FL2VA workflow) or res_multistep (comfy's
    // eta-0 second-order multistep; the Ref2VA workflow's sampler)
    std::string sampler = "euler";
    int width = 256, height = 256, length = 22, steps = 8, text_len = 32;
    uint32_t seed = 42;
};

void usage(const char *argv0) {
    std::fprintf(stderr,
                 "usage: %s --dit dit.vkopbin [--host-dir DIR] [--width W] "
                 "[--height H]\n"
                 "          [--length FRAMES] [--steps N] [--seed S] "
                 "[--shift-video 12] [--shift-audio 3] [--vae-tile 0=auto|px] "
                 "[--text-len L | --text-states F.f32]\n"
                 "          [--save DIR]\n"
                 "  --host-dir   rope_inv_freq.f32 + adaln_t_table.f32 "
                 "(default: the DiT's directory)\n"
                 "  --length     frames at 24 fps, snapped up to 17k+5\n"
                 "  --text-states raw fp32 [L, 5120] Qwen3-VL layer-50 states\n"
                 "  --te te.vkopbin --embed embed_tokens.bf16 --prompt-ids ids.i64\n"
                 "               run the text tower on the prompt instead\n"
                 "  --vae vae.vkopbin --frames DIR   decode to DIR/frame_%%04d.ppm\n"
                 "  --audio-vae avae.vkopbin --wav OUT.wav   decode the audio latent\n"
                 "               (float32 stereo, 32 kHz)\n"
                 "  --audio-latent F.f32  decode a saved [32, 2, T] audio latent only\n"
                 "  --ref-latent F.f32:HL:WL  ref2va reference image latent "
                 "[24, 1, HL, WL] (repeatable)\n"
                 "  --ref-image F.f32:TH:TW --vae-enc enc.vkopbin  reference image "
                 "pixels [3, TH, TW]\n"
                 "  --qwen DIR --vision vision.vkopbin  ref2va prompt from "
                 "h3_ref2va_prep.py qwen (with --te/--embed)\n"
                 "  --text-tags F.i64  per-token adaLN tags (1 text, 0 vision pad)\n"
                 "  --cond-aug 0.999   reference timestep / noise augmentation\n"
                 "  --steps 0          stop after the text tower / reference encoding\n"
                 "  --sampler euler|res_multistep  (default euler)\n"
                 "  --dump-layout DIR  write position/pack/mod of step 1 and exit\n",
                 argv0);
}

bool parse(int argc, char **argv, Options &o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char *what) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", what);
                return nullptr;
            }
            return argv[++i];
        };
        const char *v = nullptr;
        if (a == "--dit" && (v = next("--dit"))) o.dit = v;
        else if (a == "--host-dir" && (v = next("--host-dir"))) o.host_dir = v;
        else if (a == "--text-states" && (v = next("--text-states"))) o.text_states = v;
        else if (a == "--save" && (v = next("--save"))) o.save_dir = v;
        else if (a == "--te" && (v = next("--te"))) o.te = v;
        else if (a == "--embed" && (v = next("--embed"))) o.embed = v;
        else if (a == "--prompt-ids" && (v = next("--prompt-ids"))) o.prompt_ids = v;
        else if (a == "--vae" && (v = next("--vae"))) o.vae = v;
        else if (a == "--frames" && (v = next("--frames"))) o.frames_dir = v;
        else if (a == "--latent" && (v = next("--latent"))) o.latent = v;
        else if (a == "--text-out" && (v = next("--text-out"))) o.text_out = v;
        else if (a == "--audio-vae" && (v = next("--audio-vae"))) o.audio_vae = v;
        else if (a == "--wav" && (v = next("--wav"))) o.wav = v;
        else if (a == "--audio-latent" && (v = next("--audio-latent"))) o.audio_latent = v;
        else if (a == "--ref-latent" && (v = next("--ref-latent"))) o.ref_latents.push_back(v);
        else if (a == "--text-tags" && (v = next("--text-tags"))) o.text_tags = v;
        else if (a == "--ref-image" && (v = next("--ref-image"))) o.ref_images.push_back(v);
        else if (a == "--vae-enc" && (v = next("--vae-enc"))) o.vae_enc = v;
        else if (a == "--qwen" && (v = next("--qwen"))) o.qwen_dir = v;
        else if (a == "--vision" && (v = next("--vision"))) o.vision = v;
        else if (a == "--dump-layout" && (v = next("--dump-layout"))) o.dump_layout = v;
        else if (a == "--cond-aug" && (v = next("--cond-aug"))) o.cond_aug = std::atof(v);
        else if (a == "--sampler" && (v = next("--sampler"))) o.sampler = v;
        else if (a == "--width" && (v = next("--width"))) o.width = std::atoi(v);
        else if (a == "--height" && (v = next("--height"))) o.height = std::atoi(v);
        else if (a == "--length" && (v = next("--length"))) o.length = std::atoi(v);
        else if (a == "--steps" && (v = next("--steps"))) o.steps = std::atoi(v);
        else if (a == "--text-len" && (v = next("--text-len"))) o.text_len = std::atoi(v);
        else if (a == "--seed" && (v = next("--seed"))) o.seed = static_cast<uint32_t>(std::atoll(v));
        else if (a == "--shift-video" && (v = next("--shift-video"))) kShiftVideo = std::atof(v);
        else if (a == "--shift-audio" && (v = next("--shift-audio"))) kShiftAudio = std::atof(v);
        else if (a == "--vae-tile" && (v = next("--vae-tile"))) {
            kTileSize = std::max(0, std::atoi(v));
        }
        else return false;
        if (v == nullptr) return false;
    }
    return !o.dit.empty() || !o.latent.empty() || !o.audio_latent.empty();
}

std::vector<float> read_f32(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    const auto n = static_cast<size_t>(f.tellg());
    std::vector<float> out(n / sizeof(float));
    f.seekg(0);
    f.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(n));
    return out;
}

void write_f32(const std::string &path, const std::vector<float> &v) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(v.data()),
            static_cast<std::streamsize>(v.size() * sizeof(float)));
}

std::vector<int64_t> read_i64(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    const auto n = static_cast<size_t>(f.tellg());
    std::vector<int64_t> out(n / sizeof(int64_t));
    f.seekg(0);
    f.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(n));
    return out;
}

template <typename T>
void upload(const std::shared_ptr<Runtime> &rt,
            const std::shared_ptr<VulkanCommandPool> &pool,
            const std::string &name, const std::vector<T> &host,
            const std::vector<uint32_t> &dims) {
    rt->ResizeInput(name, dims);
    auto t = rt->GetInput(name);
    if (!t) throw std::runtime_error("no such graph input: " + name);
    auto typed = as_tensor<T>(t);
    if (!typed)
        throw std::runtime_error("graph input " + name +
                                 " does not have the expected storage type");
    typed->fillToCPU(host.data());
    typed->copyToGPU(pool);
}

// A float input at whatever width the artifact declares: fp32 (--act fp32),
// or bf16 (--act bf16, which retypes the graph inputs too).
void upload_float(const std::shared_ptr<Runtime> &rt,
                  const std::shared_ptr<VulkanCommandPool> &pool,
                  const std::string &name, const std::vector<float> &host,
                  const std::vector<uint32_t> &dims) {
    auto probe = rt->GetInput(name);
    if (!probe) throw std::runtime_error("no such graph input: " + name);
    if (probe->dtype() == typeid(float)) {
        upload<float>(rt, pool, name, host, dims);
        return;
    }
    const auto kind = probe->elem_kind();
    if (kind != vkop::core::ElemKind::kBFloat16 && kind != vkop::core::ElemKind::kFloat16)
        throw std::runtime_error("graph input " + name +
                                 " is neither fp32, fp16 nor bf16");
    std::vector<uint16_t> bits(host.size());
    for (size_t i = 0; i < host.size(); ++i)
        bits[i] = kind == vkop::core::ElemKind::kFloat16
                      ? vkop::core::ITensor::fp32_to_fp16(host[i])
                      : static_cast<uint16_t>(vkop::core::float_to_bf16(host[i]));
    upload<uint16_t>(rt, pool, name, bits, dims);
}

std::vector<float> read_output(const std::shared_ptr<Runtime> &rt,
                               const std::string &name) {
    auto t = rt->GetOutput(name);
    if (!t) throw std::runtime_error("no such output: " + name);
    if (t->dtype() == typeid(float)) {
        auto f = as_tensor<float>(t);
        return {f->data().begin(), f->data().end()};
    }
    const auto kind = t->elem_kind();
    if (kind != vkop::core::ElemKind::kBFloat16 && kind != vkop::core::ElemKind::kFloat16)
        throw std::runtime_error("output " + name + " is neither fp32, fp16 nor bf16");
    auto h = as_tensor<uint16_t>(t);
    std::vector<float> out(h->num_elements());
    for (size_t i = 0; i < out.size(); ++i)
        out[i] = kind == vkop::core::ElemKind::kFloat16
                     ? vkop::core::ITensor::fp16_to_fp32((*h)[i])
                     : vkop::core::bf16_to_float((*h)[i]);
    return out;
}

double time_shift_sigma(double sigma, double from, double to) {
    const double base = sigma / (from + sigma * (1.0 - from));
    return to * base / (1.0 + (to - 1.0) * base);
}

std::vector<double> axis_from_sqrt_area(int dim, int patch, double sqrt_area) {
    const double ratio = dim / sqrt_area;
    const int n = dim / patch;
    std::vector<double> out(n);
    for (int i = 0; i < n; ++i)
        out[i] = (i * (ratio / n) + (1.0 - ratio) / 2.0) * 32.0;
    return out;
}

// The step-independent part of comfy's PackedLayout: t2va, plus ref2va
// reference images (refs = their latent (h, w), one latent frame each).
struct Layout {
    int text_len, latent_t, lat_h, lat_w, audio_t;
    int hh, ww, n_text, n_ref, n_audio, n_video, seq;
    std::vector<double> position;  // [seq, 3] (t, h, w)
    std::vector<int64_t> pack;     // concat([text, refs+video, audio]) -> packed
};

Layout build_layout(int text_len, int latent_t, int lat_h, int lat_w,
                    int audio_t,
                    const std::vector<std::pair<int, int>> &refs = {}) {
    Layout l{};
    l.text_len = text_len;
    l.latent_t = latent_t;
    l.lat_h = lat_h;
    l.lat_w = lat_w;
    l.audio_t = audio_t;
    l.hh = lat_h / 2;
    l.ww = lat_w / 2;
    l.n_text = text_len;
    l.n_audio = audio_t * 2;
    l.n_video = latent_t * l.hh * l.ww;
    l.n_ref = 0;
    for (const auto &r : refs) l.n_ref += (r.first / 2) * (r.second / 2);
    l.seq = l.n_text + l.n_ref + l.n_audio + l.n_video;
    const double area = std::sqrt(static_cast<double>(lat_h) * lat_w);
    const auto ah = axis_from_sqrt_area(lat_h, 2, area);
    const auto aw = axis_from_sqrt_area(lat_w, 2, area);
    l.position.assign(static_cast<size_t>(l.seq) * 3, 0.0);
    auto put = [&](int row, double t, double h, double w) {
        l.position[row * 3 + 0] = t;
        l.position[row * 3 + 1] = h;
        l.position[row * 3 + 2] = w;
    };
    int row = 0;
    for (int i = 0; i < text_len; ++i) put(row++, i, 0.0, 0.0);
    // reference images: one frame each at t = text_len + i, on their own
    // area-normalized grid; the targets' timeline starts after them
    double cursor = text_len;
    for (const auto &r : refs) {
        const double ra = std::sqrt(static_cast<double>(r.first) * r.second);
        const auto rh = axis_from_sqrt_area(r.first, 2, ra);
        const auto rw = axis_from_sqrt_area(r.second, 2, ra);
        for (double y : rh)
            for (double x : rw) put(row++, cursor, y, x);
        cursor += 1.0;
    }
    for (int ch = 0; ch < 2; ++ch)
        for (int t = 0; t < audio_t; ++t)
            put(row++, cursor + t, 0.0, ch == 0 ? aw.front() : aw.back());
    double tpos = cursor;
    for (int t = 0; t < latent_t; ++t) {
        for (int y = 0; y < l.hh; ++y)
            for (int x = 0; x < l.ww; ++x) put(row++, tpos, ah[y], aw[x]);
        tpos += kFrameRescale * kFramePerToken[t % 5];
    }
    // graph concat order is [text, refs + video, audio];
    // packed is [text, refs, audio, video]
    l.pack.resize(l.seq);
    int p = 0;
    const int nv_all = l.n_ref + l.n_video;
    for (int i = 0; i < l.n_text; ++i) l.pack[p++] = i;
    for (int i = 0; i < l.n_ref; ++i) l.pack[p++] = l.n_text + i;
    for (int i = 0; i < l.n_audio; ++i) l.pack[p++] = l.n_text + nv_all + i;
    for (int i = 0; i < l.n_video; ++i) l.pack[p++] = l.n_text + l.n_ref + i;
    return l;
}

// [C, T, H, W] -> [T*(H/2)*(W/2), C*2*2] for any latent (reference images)
std::vector<float> patchify_thw(const std::vector<float> &x, int T, int H, int W) {
    std::vector<float> rows(static_cast<size_t>(T) * (H / 2) * (W / 2) * kVideoC * 4);
    size_t o = 0;
    for (int t = 0; t < T; ++t)
        for (int y = 0; y < H / 2; ++y)
            for (int xw = 0; xw < W / 2; ++xw)
                for (int c = 0; c < kVideoC; ++c)
                    for (int p = 0; p < 2; ++p)
                        for (int q = 0; q < 2; ++q)
                            rows[o++] = x[((static_cast<size_t>(c) * T + t) * H +
                                           (2 * y + p)) * W + (2 * xw + q)];
    return rows;
}

// [C, T, H, W] -> [T*hh*ww, C*2*2] (comfy patchify_video, patch 1x2x2)
std::vector<float> patchify(const std::vector<float> &x, const Layout &l) {
    const int T = l.latent_t, H = l.lat_h, W = l.lat_w;
    std::vector<float> rows(static_cast<size_t>(l.n_video) * kVideoC * 4);
    size_t o = 0;
    for (int t = 0; t < T; ++t)
        for (int y = 0; y < l.hh; ++y)
            for (int xw = 0; xw < l.ww; ++xw)
                for (int c = 0; c < kVideoC; ++c)
                    for (int p = 0; p < 2; ++p)
                        for (int q = 0; q < 2; ++q)
                            rows[o++] = x[((static_cast<size_t>(c) * T + t) * H +
                                           (2 * y + p)) * W + (2 * xw + q)];
    return rows;
}

void add_unpatchified(std::vector<float> &x, const std::vector<float> &rows,
                      const Layout &l, float scale) {
    const int T = l.latent_t, H = l.lat_h, W = l.lat_w;
    size_t o = 0;
    for (int t = 0; t < T; ++t)
        for (int y = 0; y < l.hh; ++y)
            for (int xw = 0; xw < l.ww; ++xw)
                for (int c = 0; c < kVideoC; ++c)
                    for (int p = 0; p < 2; ++p)
                        for (int q = 0; q < 2; ++q)
                            x[((static_cast<size_t>(c) * T + t) * H +
                               (2 * y + p)) * W + (2 * xw + q)] += scale * rows[o++];
}

// [32, 2, Ta] -> [2*Ta, 32] channel-major (comfy pack_audio)
std::vector<float> pack_audio(const std::vector<float> &a, int audio_t) {
    std::vector<float> rows(static_cast<size_t>(2) * audio_t * kAudioC);
    for (int ch = 0; ch < 2; ++ch)
        for (int t = 0; t < audio_t; ++t)
            for (int c = 0; c < kAudioC; ++c)
                rows[(static_cast<size_t>(ch) * audio_t + t) * kAudioC + c] =
                    a[(static_cast<size_t>(c) * 2 + ch) * audio_t + t];
    return rows;
}

void add_unpacked_audio(std::vector<float> &a, const std::vector<float> &rows,
                        int audio_t, float scale) {
    for (int ch = 0; ch < 2; ++ch)
        for (int t = 0; t < audio_t; ++t)
            for (int c = 0; c < kAudioC; ++c)
                a[(static_cast<size_t>(c) * 2 + ch) * audio_t + t] +=
                    scale * rows[(static_cast<size_t>(ch) * audio_t + t) * kAudioC + c];
}

struct Stats {
    double absmax = 0, mean = 0, rms = 0;
    size_t nonfinite = 0;
};

Stats stats(const std::vector<float> &v) {
    Stats s;
    double sum = 0, sq = 0;
    for (float f : v) {
        if (!std::isfinite(f)) {
            ++s.nonfinite;
            continue;
        }
        s.absmax = std::max(s.absmax, static_cast<double>(std::fabs(f)));
        sum += f;
        sq += static_cast<double>(f) * f;
    }
    s.mean = v.empty() ? 0 : sum / v.size();
    s.rms = v.empty() ? 0 : std::sqrt(sq / v.size());
    return s;
}

// ---------------------------------------------------------------- text tower
// Qwen3-VL-32B truncated to 50 layers (comfy/text_encoders/minimax.py): the
// raw prompt tokens, causal attention, plain rotate-half RoPE (theta 5e6; a
// text-only prompt has equal t/h/w MRoPE positions), output = the layer-50
// hidden state, unnormalized.
constexpr int kTeHeadDim = 128;
constexpr double kTeTheta = 5.0e6;

// `te` is one graph or a comma-separated list of consecutive layer slices
// (h3_export_fl2va_te.py --layers): each takes the previous one's hidden as
// its inputs_embeds and is released before the next loads, so peak memory is
// one slice, not the 24 GB tower.
// --------------------------------------------------------- ref2va vision
// Qwen3-VL-32B vision tower, one call per vision block (attention stays in a
// block): pixel_values [S, 1536] + grid_thw [1, 3] -> image_features and
// deepstack_features_0..2, each [S/4, 5120] in token order. Their rows
// replace the block's pads in inputs_embeds, and deepstack_i (zero elsewhere)
// is added after text layer i (graph inputs of a --deepstack text tower).
struct VisionRows {
    struct Block { int start = 0, rows = 0; std::vector<float> feat, ds[3]; };
    std::vector<Block> blocks;
    std::vector<float> rope_cos, rope_sin;  // [L, 128] MRoPE tables
};

// minimal reader for ref2va_qwen.json's "blocks" (written by our own prep)
std::vector<std::map<std::string, std::string>> json_blocks(const std::string &path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    const std::string js((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<std::map<std::string, std::string>> out;
    size_t p = js.find("\"blocks\"");
    if (p == std::string::npos) return out;
    p = js.find('[', p);
    int depth = 0;
    size_t obj = 0;
    for (size_t i = p + 1; i < js.size(); ++i) {
        const char c = js[i];
        if (c == '{' && depth++ == 0) obj = i;
        else if (c == '}' && --depth == 0) {
            const std::string o = js.substr(obj + 1, i - obj - 1);
            std::map<std::string, std::string> kv;
            for (size_t q = o.find('"'); q != std::string::npos; q = o.find('"', q)) {
                const size_t e = o.find('"', q + 1);
                const std::string key = o.substr(q + 1, e - q - 1);
                size_t v = o.find(':', e) + 1;
                while (v < o.size() && std::isspace(static_cast<unsigned char>(o[v]))) ++v;
                size_t ve;
                if (o[v] == '"') { ve = o.find('"', v + 1); kv[key] = o.substr(v + 1, ve - v - 1); ++ve; }
                else if (o[v] == '[') { ve = o.find(']', v) + 1; kv[key] = o.substr(v + 1, ve - v - 2); }
                else { ve = o.find_first_of(",}", v); if (ve == std::string::npos) ve = o.size(); kv[key] = o.substr(v, ve - v); }
                q = o.find('"', ve);
                if (q == std::string::npos) break;
            }
            out.push_back(std::move(kv));
        } else if (c == ']' && depth == 0) break;
    }
    return out;
}

VisionRows run_vision(const std::shared_ptr<VulkanCommandPool> &pool,
                      const std::string &vision, const std::string &dir, int L) {
    VisionRows vr;
    vr.rope_cos = read_f32(dir + "/rope_cos.f32");
    vr.rope_sin = read_f32(dir + "/rope_sin.f32");
    if (vr.rope_cos.size() != static_cast<size_t>(L) * kTeHeadDim ||
        vr.rope_sin.size() != vr.rope_cos.size())
        throw std::runtime_error(dir + ": rope tables are not [L, 128]");
    const auto blocks = json_blocks(dir + "/ref2va_qwen.json");
    if (blocks.empty()) return vr;
    auto t0 = std::chrono::steady_clock::now();
    double ms = 0;
    for (const auto &b : blocks) {
        // A fresh runtime per block: re-running one loaded graph with a new
        // grid_thw of the SAME patch count (e.g. 24x24 then 18x32, both 576)
        // reuses the first grid's grid-derived indices -- the second block
        // came out at cos 0.62. Loading costs ~0.3 s per block.
        auto rt = std::make_shared<Runtime>(pool, vision, /*precision=*/0);
        rt->set_backend_buffer(true);
        rt->LoadModel();
        VisionRows::Block vb;
        vb.start = std::stoi(b.at("start"));
        vb.rows = std::stoi(b.at("rows"));
        const int S = std::stoi(b.at("patch_rows")), D = std::stoi(b.at("patch_dim"));
        std::vector<int64_t> grid;
        for (size_t a = 0, e; a < b.at("grid").size(); a = e + 1) {
            e = b.at("grid").find(',', a);
            if (e == std::string::npos) e = b.at("grid").size();
            grid.push_back(std::stoll(b.at("grid").substr(a, e - a)));
        }
        const auto px = read_f32(dir + "/" + b.at("file"));
        if (px.size() != static_cast<size_t>(S) * D || grid.size() != 3 || S / 4 != vb.rows ||
            vb.start + vb.rows > L)
            throw std::runtime_error(dir + "/" + b.at("file") + ": block does not match the json");
        upload_float(rt, pool, "pixel_values", px, {static_cast<uint32_t>(S), static_cast<uint32_t>(D)});
        upload<int64_t>(rt, pool, "grid_thw", grid, {1, 3});
        ms += rt->Run();
        rt->ReadResult();
        vb.feat = read_output(rt, "image_features");
        for (int i = 0; i < 3; ++i)
            vb.ds[i] = read_output(rt, "deepstack_features_" + std::to_string(i));
        if (vb.feat.size() != static_cast<size_t>(vb.rows) * kTextDim)
            throw std::runtime_error("vision: image_features is not [rows, 5120]");
        const Stats st = stats(vb.feat);
        std::printf("[vision] block at %d: grid %lldx%lld -> %d rows, rms %.3f%s\n", vb.start,
                    static_cast<long long>(grid[1]), static_cast<long long>(grid[2]), vb.rows,
                    st.rms, st.nonfinite ? " NONFINITE" : "");
        if (st.nonfinite) throw std::runtime_error("vision tower produced non-finite features");
        vr.blocks.push_back(std::move(vb));
    }
    std::printf("[vision] %zu block(s), load+run %.1fs (gpu %.0f ms)\n", vr.blocks.size(),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), ms);
    return vr;
}

std::vector<float> run_text_tower(const std::shared_ptr<VulkanCommandPool> &pool,
                                  const std::string &te, const std::string &embed,
                                  const std::vector<int64_t> &ids,
                                  const VisionRows *vr = nullptr) {
    const int L = static_cast<int>(ids.size());
    std::vector<float> emb(static_cast<size_t>(L) * kTextDim);
    {
        std::ifstream f(embed, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open " + embed);
        std::vector<uint16_t> row(kTextDim);
        for (int i = 0; i < L; ++i) {
            f.seekg(static_cast<std::streamoff>(ids[i]) * kTextDim * 2);
            f.read(reinterpret_cast<char *>(row.data()), kTextDim * 2);
            if (!f) throw std::runtime_error("token id out of the embedding table");
            for (int d = 0; d < kTextDim; ++d)
                emb[static_cast<size_t>(i) * kTextDim + d] = vkop::core::bf16_to_float(row[d]);
        }
    }
    // ref2va: vision features over the pads, DeepStack rows beside them
    std::vector<float> ds[3];
    if (vr) {
        for (auto &d : ds) d.assign(emb.size(), 0.f);
        for (const auto &b : vr->blocks) {
            const size_t off = static_cast<size_t>(b.start) * kTextDim;
            std::copy(b.feat.begin(), b.feat.end(), emb.begin() + off);
            for (int i = 0; i < 3; ++i) std::copy(b.ds[i].begin(), b.ds[i].end(), ds[i].begin() + off);
        }
    }
    std::vector<float> bias(static_cast<size_t>(L) * L, 0.f);
    for (int i = 0; i < L; ++i)
        for (int j = i + 1; j < L; ++j) bias[static_cast<size_t>(i) * L + j] = -3.0e38f;
    std::vector<float> rc(static_cast<size_t>(L) * kTeHeadDim), rs(rc.size());
    for (int p = 0; p < L && !vr; ++p)
        for (int i = 0; i < kTeHeadDim / 2; ++i) {
            const float inv = static_cast<float>(
                1.0 / std::pow(kTeTheta, 2.0 * i / kTeHeadDim));
            const float a = static_cast<float>(p) * inv;
            for (int h : {i, i + kTeHeadDim / 2}) {
                rc[static_cast<size_t>(p) * kTeHeadDim + h] = std::cos(a);
                rs[static_cast<size_t>(p) * kTeHeadDim + h] = std::sin(a);
            }
        }
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::string> parts;
    for (size_t a = 0; a <= te.size();) {
        size_t b = te.find(',', a);
        if (b == std::string::npos) b = te.size();
        if (b > a) parts.push_back(te.substr(a, b - a));
        a = b + 1;
    }
    std::vector<float> hidden = std::move(emb);
    double ms = 0;
    const auto u = [](int v) { return static_cast<uint32_t>(v); };
    for (const auto &part : parts) {
        auto rt = std::make_shared<Runtime>(pool, part, /*precision=*/0);
        rt->set_backend_buffer(true);
        rt->LoadModel();
        upload_float(rt, pool, "inputs_embeds", hidden, {u(L), kTextDim});
        upload_float(rt, pool, "attention_bias", bias, {u(L), u(L)});
        upload_float(rt, pool, "rope_cos", vr ? vr->rope_cos : rc, {u(L), kTeHeadDim});
        upload_float(rt, pool, "rope_sin", vr ? vr->rope_sin : rs, {u(L), kTeHeadDim});
        const auto names = rt->ListInputNames();
        for (int i = 0; i < 3; ++i) {
            const std::string n = "deepstack_" + std::to_string(i);
            if (std::find(names.begin(), names.end(), n) == names.end()) continue;
            if (!vr) ds[i].assign(static_cast<size_t>(L) * kTextDim, 0.f);  // plain prompt
            upload_float(rt, pool, n, ds[i], {u(L), kTextDim});
        }
        if (vr && !vr->blocks.empty() &&
            std::find(names.begin(), names.end(), "deepstack_0") == names.end() &&
            &part == &parts.front())
            throw std::runtime_error(part + " has no deepstack inputs: export it with --deepstack");
        ms += rt->Run();
        rt->ReadResult();
        hidden = read_output(rt, "hidden");
        std::printf("[te] %s done\n", part.c_str());
    }  // each slice's runtime is released at the end of its iteration
    const Stats st = stats(hidden);
    std::printf("[te] %d tokens, load+run %.1fs (run %.0f ms), hidden rms %.3f "
                "absmax %.1f%s\n", L,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                ms, st.rms, st.absmax, st.nonfinite ? " NONFINITE" : "");
    if (st.nonfinite) throw std::runtime_error("text tower produced non-finite states");
    return hidden;
}

// ------------------------------------------------------------------ video VAE
// comfy/ldm/minimax/vae.py MiniMaxH3VideoVAE.decode for the ViT3D decoder:
// denormalize, decode overlapping 7-latent clips (decode_temporal), each as
// 256-px spatial tiles blended in place (tiled_decode), finalize to [0, 1].
constexpr int kVaeRatio = 16, kVaeRatioT = 4;
constexpr int kClipLength = 17, kTokenDrop = 3;
constexpr int kTokensChunk = (kClipLength + kVaeRatioT - 1) / kVaeRatioT;      // 5
constexpr int kFramePrePad = ((-kClipLength) % kVaeRatioT + kVaeRatioT) % kVaeRatioT;  // 3
constexpr int kTokenOverlap = ((-kTokenDrop) % kTokensChunk + kTokensChunk) % kTokensChunk;  // 2
constexpr int kFrameOverlap = kTokenOverlap * kVaeRatioT - kFramePrePad > 0
                                  ? kTokenOverlap * kVaeRatioT - kFramePrePad : 0;  // 5
constexpr int kTileOverlapMin = 64;
constexpr float kLatentsMean[24] = {
    0.858090341091156f, -0.9606591463088989f, 1.0661640167236328f, -0.5090325474739075f,
    -0.2727581858634949f, -1.3675414323806763f, -0.2553254961967468f, -0.26907554268836975f,
    -0.5376840829849243f, -0.0464097298681736f, 0.6657370328903198f, 0.19690127670764923f,
    -0.5460608005523682f, -0.4035342037677765f, -0.23683024942874908f, 0.25928452610969543f,
    -0.30133944749832153f, 0.211341992020607f, -1.1206848621368408f, 0.3581933379173279f,
    -0.04225143790245056f, 0.2604829967021942f, 0.22864092886447906f, 0.7056031823158264f};
constexpr float kLatentsStd[24] = {
    1.2223774194717407f, 1.2767263650894165f, 1.6831774711608887f, 1.7549455165863037f,
    1.5636216402053833f, 2.194143533706665f, 0.9653137922286987f, 1.0569885969161987f,
    0.841948926448822f, 0.7729952931404114f, 1.8955937623977661f, 0.946841835975647f,
    0.7996809482574463f, 0.44988900423049927f, 0.7197399735450745f, 0.6936293244361877f,
    2.961095094680786f, 2.7694199085235596f, 3.0496184825897217f, 2.1088054180145264f,
    3.276226282119751f, 3.1627357006073f, 2.2816812992095947f, 2.6127843856811523f};
constexpr float kPixMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kPixStd[3] = {0.229f, 0.224f, 0.225f};

// [C][F][H][W] float video
struct Video {
    int C = 3, F = 0, H = 0, W = 0;
    std::vector<float> v;
    Video() = default;
    Video(int c, int f, int h, int w)
        : C(c), F(f), H(h), W(w), v(static_cast<size_t>(c) * f * h * w, 0.f) {}
    float &at(int c, int f, int y, int x) {
        return v[((static_cast<size_t>(c) * F + f) * H + y) * W + x];
    }
    float at(int c, int f, int y, int x) const {
        return v[((static_cast<size_t>(c) * F + f) * H + y) * W + x];
    }
    // sub-box [f0, f0+nf) x [y0, y0+nh) x [x0, x0+nw)
    Video crop(int f0, int nf, int y0, int nh, int x0, int nw) const {
        Video o(C, nf, nh, nw);
        for (int c = 0; c < C; ++c)
            for (int f = 0; f < nf; ++f)
                for (int y = 0; y < nh; ++y)
                    for (int x = 0; x < nw; ++x)
                        o.at(c, f, y, x) = at(c, f0 + f, y0 + y, x0 + x);
        return o;
    }
};

// comfy blend(a, b, extent, dim): the first `extent` slices of b along `dim`
// (0 frames, 1 height, 2 width) fade in from the last `extent` slices of a.
void blend(const Video &a, Video &b, int extent, int dim) {
    const int alen = dim == 0 ? a.F : dim == 1 ? a.H : a.W;
    const int blen = dim == 0 ? b.F : dim == 1 ? b.H : b.W;
    extent = std::min({alen, blen, extent});
    if (extent <= 0) return;
    for (int c = 0; c < b.C; ++c)
        for (int f = 0; f < b.F; ++f)
            for (int y = 0; y < b.H; ++y)
                for (int x = 0; x < b.W; ++x) {
                    const int p = dim == 0 ? f : dim == 1 ? y : x;
                    if (p >= extent) continue;
                    const int af = dim == 0 ? a.F - extent + p : f;
                    const int ay = dim == 1 ? a.H - extent + p : y;
                    const int ax = dim == 2 ? a.W - extent + p : x;
                    const float wb = static_cast<float>(p) / static_cast<float>(extent);
                    b.at(c, f, y, x) = a.at(c, af, ay, ax) * (1.f - wb) + b.at(c, f, y, x) * wb;
                }
}

void split_tiles(int len, std::vector<int> &idx, std::vector<int> &lens,
                 std::vector<int> &overlaps, int tile = kTileSize) {
    const int kTileSize = tile;
    idx.clear(); lens.clear(); overlaps.clear();
    if (kTileSize >= len) {
        idx = {0};
        lens = {len};
        return;
    }
    int n = (len + kTileSize - 1) / kTileSize;
    int remaining = 0;
    while (true) {
        overlaps.assign(n - 1, kTileOverlapMin);
        remaining = kTileSize * n - kTileOverlapMin * (n - 1) - len;
        if (remaining < 0) ++n; else break;
    }
    for (int i = 0; i < remaining / kVaeRatio; ++i) overlaps[i % (n - 1)] += kVaeRatio;
    idx = {0};
    for (int i = 0; i < n - 1; ++i) idx.push_back(idx.back() + kTileSize - overlaps[i]);
    lens.assign(n, kTileSize);
}

struct Vae {
    std::shared_ptr<Runtime> rt;
    std::shared_ptr<VulkanCommandPool> pool;
    int calls = 0;
    double ms = 0;

    // One decoder call: z [24][T][h][w] (denormalized) -> raw pixels.
    Video decode(const std::vector<float> &z, int T, int h, int w) {
        const int n = T * h * w;
        std::vector<float> tok(static_cast<size_t>(n) * kVideoC);
        for (int c = 0; c < kVideoC; ++c)
            for (int i = 0; i < n; ++i)
                tok[static_cast<size_t>(i) * kVideoC + c] = z[static_cast<size_t>(c) * n + i];
        // comfy RotaryEmbeddingND(48, base 100, 3 axes) over create_token_ids,
        // zero ids for the 4 register tokens and the zero token
        const int rows = n + 5, pairs = 24;
        std::vector<float> rc(static_cast<size_t>(rows) * pairs, 1.f), rs(rc.size(), 0.f);
        const int dims[3] = {T, h, w};
        for (int t = 0; t < T; ++t)
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    const int r = (t * h + y) * w + x;
                    const int pos[3] = {t, y, x};
                    for (int a = 0; a < 3; ++a) {
                        const float id = (static_cast<float>(pos[a]) + 0.5f) / dims[a] * 2.f - 1.f;
                        for (int k = 0; k < 8; ++k) {
                            const float inv = 1.f / std::pow(100.f, 0.125f * k);
                            const float ang = 2.f * static_cast<float>(M_PI) * id * inv;
                            rc[static_cast<size_t>(r) * pairs + a * 8 + k] = std::cos(ang);
                            rs[static_cast<size_t>(r) * pairs + a * 8 + k] = std::sin(ang);
                        }
                    }
                }
        const auto u = [](int v) { return static_cast<uint32_t>(v); };
        upload_float(rt, pool, "z", tok, {u(n), u(kVideoC)});
        upload_float(rt, pool, "rope_cos", rc, {u(rows), u(pairs)});
        upload_float(rt, pool, "rope_sin", rs, {u(rows), u(pairs)});
        ms += rt->Run();
        rt->ReadResult();
        ++calls;
        const auto px = read_output(rt, "pixels");  // [n, 3*4*16*16]
        Video out(3, T * kVaeRatioT, h * kVaeRatio, w * kVaeRatio);
        for (int t = 0; t < T; ++t)
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    const float *p = &px[static_cast<size_t>((t * h + y) * w + x) * 3072];
                    for (int c = 0; c < 3; ++c)
                        for (int a = 0; a < kVaeRatioT; ++a)
                            for (int py = 0; py < kVaeRatio; ++py)
                                for (int pxx = 0; pxx < kVaeRatio; ++pxx)
                                    out.at(c, t * kVaeRatioT + a, y * kVaeRatio + py,
                                           x * kVaeRatio + pxx) =
                                        p[((c * kVaeRatioT + a) * kVaeRatio + py) * kVaeRatio + pxx];
                }
        return out;
    }

    // --vae-tile auto: the decoder only holds up near its 256-px (16-latent)
    // tile -- a whole 512x288 frame (32 latents wide) comes out as blocks --
    // but is fine up to ~320 px, where the tile count drops a lot. Pick the
    // size in [256, 320] with the least decoded area (calls x tile^2).
    static int auto_tile(int H, int W) {
        int best = 256;
        long best_cost = -1;
        for (int t = 256; t <= 320; t += kVaeRatio) {
            std::vector<int> i, l, oy, ox;
            split_tiles(H, i, l, oy, t);
            const int ny = static_cast<int>(i.size());
            split_tiles(W, i, l, ox, t);
            const int nx = static_cast<int>(i.size());
            const long th = std::min(t, H), tw = std::min(t, W);
            const long cost = static_cast<long>(ny) * nx * th * tw;
            if (best_cost < 0 || cost < best_cost) { best = t; best_cost = cost; }
        }
        return best;
    }

    // comfy tiled_decode over one clip z [24][T][Hl][Wl].
    Video tiled(const std::vector<float> &z, int T, int Hl, int Wl) {
        const int H = Hl * kVaeRatio, W = Wl * kVaeRatio;
        const int saved = kTileSize;
        if (kTileSize == 0) kTileSize = auto_tile(H, W);
        struct Restore { int &r; int v; ~Restore() { r = v; } } restore{kTileSize, saved};
        std::vector<int> yi, yl, yo, xi, xl, xo;
        split_tiles(H, yi, yl, yo);
        split_tiles(W, xi, xl, xo);
        Video canvas(3, T * kVaeRatioT, H, W), strip;
        int out_y = 0;
        for (size_t i = 0; i < yi.size(); ++i) {
            const int zi = yi[i] / kVaeRatio, zl = yl[i] / kVaeRatio;
            Video new_strip, left_tail;
            int out_x = 0, tile_h = 0;
            for (size_t j = 0; j < xi.size(); ++j) {
                const int zj = xi[j] / kVaeRatio, zw = xl[j] / kVaeRatio;
                std::vector<float> zt(static_cast<size_t>(kVideoC) * T * zl * zw);
                for (int c = 0; c < kVideoC; ++c)
                    for (int t = 0; t < T; ++t)
                        for (int y = 0; y < zl; ++y)
                            for (int x = 0; x < zw; ++x)
                                zt[((static_cast<size_t>(c) * T + t) * zl + y) * zw + x] =
                                    z[((static_cast<size_t>(c) * T + t) * Hl + zi + y) * Wl + zj + x];
                Video tile = decode(zt, T, zl, zw);
                if (i > 0) blend(strip.crop(0, strip.F, 0, strip.H, xi[j], xl[j]), tile, yo[i - 1], 1);
                if (j > 0) blend(left_tail, tile, xo[j - 1], 2);
                if (j + 1 < xi.size()) {
                    left_tail = tile.crop(0, tile.F, 0, tile.H, tile.W - xo[j], xo[j]);
                    tile = tile.crop(0, tile.F, 0, tile.H, 0, tile.W - xo[j]);
                }
                if (i + 1 < yi.size()) {
                    if (new_strip.v.empty()) new_strip = Video(3, tile.F, yo[i], W);
                    const Video bottom = tile.crop(0, tile.F, tile.H - yo[i], yo[i], 0, tile.W);
                    for (int c = 0; c < 3; ++c)
                        for (int f = 0; f < tile.F; ++f)
                            for (int y = 0; y < yo[i]; ++y)
                                for (int x = 0; x < tile.W; ++x)
                                    new_strip.at(c, f, y, out_x + x) = bottom.at(c, f, y, x);
                    tile = tile.crop(0, tile.F, 0, tile.H - yo[i], 0, tile.W);
                }
                for (int c = 0; c < 3; ++c)
                    for (int f = 0; f < tile.F; ++f)
                        for (int y = 0; y < tile.H; ++y)
                            for (int x = 0; x < tile.W; ++x)
                                canvas.at(c, f, out_y + y, out_x + x) = tile.at(c, f, y, x);
                tile_h = tile.H;
                out_x += tile.W;
            }
            strip = new_strip;
            out_y += tile_h;
        }
        return canvas;
    }
};

void temporal_chunks(int z_len, int &pad, int &num) {
    int pseudo = z_len + kTokenDrop;
    pad = ((-pseudo) % kTokensChunk + kTokensChunk) % kTokensChunk;
    pseudo += pad;
    num = pseudo / kTokensChunk - (kTokenDrop > 0 ? 1 : 0);
    if (num < 1) {
        pad += kTokensChunk;
        num += 1;
    }
}

int temporal_frames(int z_len) {
    int pad = 0, num = 0;
    temporal_chunks(z_len, pad, num);
    const int zl = z_len + pad, chunk_dec = kTokensChunk * kVaeRatioT;
    int total = 0, final_overlap = 0;
    for (int i = 0; i < num; ++i) {
        const int t0 = i * kTokensChunk, t1 = t0 + kTokensChunk + kTokenOverlap;
        const int clip_frames = std::max(0, std::min(t1, zl) - std::min(t0, zl)) * kVaeRatioT;
        for (int j = 0; j < 2; ++j) {
            const int fs = j * chunk_dec, fe = std::min(fs + chunk_dec, clip_frames);
            const int cf = std::max(0, fe - fs - kFramePrePad);
            if (j == 0) total += cf; else final_overlap = cf;
        }
    }
    total += final_overlap;
    int pad_frames = 0;
    const int intra_tail = kClipLength % kVaeRatioT;
    for (int k = 0; k < pad; ++k)
        pad_frames += (intra_tail == 0) ? kVaeRatioT
                      : ((z_len + k) % kTokensChunk == 0 ? intra_tail : kVaeRatioT);
    return total - pad_frames;
}

// comfy decode(): normalized latent [24][T][Hl][Wl] -> pixels in [0, 1].
Video vae_decode(Vae &vae, const std::vector<float> &x, int T, int Hl, int Wl) {
    const size_t plane = static_cast<size_t>(Hl) * Wl;
    int pad = 0, num = 0;
    temporal_chunks(T, pad, num);
    const int Tp = T + pad;
    std::vector<float> z(static_cast<size_t>(kVideoC) * Tp * plane);
    for (int c = 0; c < kVideoC; ++c)
        for (int t = 0; t < Tp; ++t) {
            const int ts = std::min(t, T - 1);  // pad repeats the last latent
            for (size_t p = 0; p < plane; ++p)
                z[(static_cast<size_t>(c) * Tp + t) * plane + p] =
                    x[(static_cast<size_t>(c) * T + ts) * plane + p] * kLatentsStd[c] +
                    kLatentsMean[c];
        }
    const int frames = temporal_frames(T);
    Video dec(3, frames, Hl * kVaeRatio, Wl * kVaeRatio), overlap;
    int write_pos = 0;
    auto write_part = [&](const Video &part) {
        const int n = std::min(part.F, frames - write_pos);
        for (int c = 0; c < 3; ++c)
            for (int f = 0; f < n; ++f)
                for (int y = 0; y < part.H; ++y)
                    for (int xx = 0; xx < part.W; ++xx)
                        dec.at(c, write_pos + f, y, xx) = std::clamp(
                            part.at(c, f, y, xx) * kPixStd[c] + kPixMean[c], 0.f, 1.f);
        write_pos += std::max(0, n);
    };
    const int chunk_dec = kTokensChunk * kVaeRatioT;
    for (int i = 0; i < num; ++i) {
        const int t0 = i * kTokensChunk;
        const int tl = std::min(kTokensChunk + kTokenOverlap, Tp - t0);
        std::vector<float> clip(static_cast<size_t>(kVideoC) * tl * plane);
        for (int c = 0; c < kVideoC; ++c)
            std::copy_n(&z[(static_cast<size_t>(c) * Tp + t0) * plane], tl * plane,
                        &clip[static_cast<size_t>(c) * tl * plane]);
        const Video cd = vae.tiled(clip, tl, Hl, Wl);
        for (int j = 0; j < 2; ++j) {
            const int fs = j * chunk_dec, fe = std::min(fs + chunk_dec, cd.F);
            if (fe - fs - kFramePrePad <= 0) continue;
            Video part = cd.crop(fs + kFramePrePad, fe - fs - kFramePrePad, 0, cd.H, 0, cd.W);
            if (j == 0) {
                if (!overlap.v.empty()) {
                    blend(overlap, part, kFrameOverlap, 0);
                    overlap = Video();
                }
                write_part(part);
            } else {
                overlap = part;
            }
        }
        if (i == num - 1 && !overlap.v.empty()) write_part(overlap);
        std::printf("  [vae] clip %d/%d done (%d decoder calls)\n", i + 1, num, vae.calls);
    }
    return dec;
}

void write_ppm(const std::string &path, const Video &v, int f) {
    std::ofstream o(path, std::ios::binary);
    o << "P6\n" << v.W << " " << v.H << "\n255\n";
    std::vector<uint8_t> row(static_cast<size_t>(v.W) * 3);
    for (int y = 0; y < v.H; ++y) {
        for (int x = 0; x < v.W; ++x)
            for (int c = 0; c < 3; ++c)
                row[static_cast<size_t>(x) * 3 + c] =
                    static_cast<uint8_t>(std::lround(v.at(c, f, y, x) * 255.f));
        o.write(reinterpret_cast<const char *>(row.data()), static_cast<std::streamsize>(row.size()));
    }
}

}  // namespace

int decode_and_write(const Options &o, const std::shared_ptr<VulkanCommandPool> &pool,
                     const std::vector<float> &xv, int latent_t, int lat_h, int lat_w);

// ---------------------------------------------------------------- audio VAE
// MiniMax-H3 audio VAE decode (comfy MiniMaxH3AudioVAE.decode): the
// normalized latent [32, 2, T] goes in as z [2, 32, 1, T] (one row per stereo
// channel; the denormalization is folded into the graph's first conv), and
// one call returns the clamped waveform [2, 1, 1, T*800] at 32 kHz. One call
// covers the whole clip: the activations peak at ~12800*T floats per channel.
constexpr int kAudioHop = 800;
constexpr int kAudioRate = 32000;

void write_wav_f32(const std::string &path, const std::vector<float> &planar,
                   int channels, int rate) {
    const uint32_t n = static_cast<uint32_t>(planar.size() / channels);
    std::vector<float> inter(planar.size());
    for (uint32_t i = 0; i < n; ++i)
        for (int c = 0; c < channels; ++c)
            inter[static_cast<size_t>(i) * channels + c] = planar[static_cast<size_t>(c) * n + i];
    const uint32_t data = static_cast<uint32_t>(inter.size() * 4);
    std::ofstream f(path, std::ios::binary);
    auto u32 = [&](uint32_t v) { f.write(reinterpret_cast<const char *>(&v), 4); };
    auto u16 = [&](uint16_t v) { f.write(reinterpret_cast<const char *>(&v), 2); };
    f.write("RIFF", 4); u32(36 + data); f.write("WAVE", 4);
    f.write("fmt ", 4); u32(16); u16(3 /* IEEE float */); u16(static_cast<uint16_t>(channels));
    u32(static_cast<uint32_t>(rate)); u32(static_cast<uint32_t>(rate * channels * 4));
    u16(static_cast<uint16_t>(channels * 4)); u16(32);
    f.write("data", 4); u32(data);
    f.write(reinterpret_cast<const char *>(inter.data()), data);
}

// ------------------------------------------------------------ VAE encoder
// comfy MiniMaxH3VideoVAE.encode for one reference image (T = 1 always goes
// through tiled_encode): 256-px tiles with >= 64 px overlap, each encoded
// alone (own GroupNorm statistics), blended in latent space against the
// UNBLENDED neighbour above, then the one to the left. The graph folds the
// imagenet pixel norm and the latent normalization: pixels [1, 3, h, w] in
// [0, 1] -> latent [1, 24, h/16, w/16], already in the DiT's space.
constexpr int kEncTile = 256, kEncOverlapMin = 64, kEncRatio = 16;

void enc_tile_plan(int n, std::vector<int> &starts, std::vector<int> &lens,
                   std::vector<int> &ov) {
    starts.clear(); lens.clear(); ov.clear();
    if (kEncTile >= n) { starts = {0}; lens = {n}; return; }
    int N = (n + kEncTile - 1) / kEncTile, rem;
    for (;;) {
        ov.assign(N - 1, kEncOverlapMin);
        rem = kEncTile * N - kEncOverlapMin * (N - 1) - n;
        if (rem >= 0) break;
        ++N;
    }
    for (int i = 0; i < rem / kEncRatio; ++i) ov[i % (N - 1)] += kEncRatio;
    starts = {0};
    for (int i = 0; i < N - 1; ++i) starts.push_back(starts.back() + kEncTile - ov[i]);
    lens.assign(N, kEncTile);
}

struct LatTile {  // [24][h][w]
    int h = 0, w = 0;
    std::vector<float> v;
    float &at(int c, int y, int x) { return v[(static_cast<size_t>(c) * h + y) * w + x]; }
    float at(int c, int y, int x) const { return v[(static_cast<size_t>(c) * h + y) * w + x]; }
};

// comfy blend: b's first ext rows (dim 0) / cols (dim 1) <- a's last ext * (1 - i/ext) + b * i/ext
LatTile enc_blend(const LatTile &a, LatTile b, int ext, int dim) {
    ext = std::min({ext, dim == 0 ? a.h : a.w, dim == 0 ? b.h : b.w});
    for (int c = 0; c < kVideoC; ++c)
        for (int i = 0; i < ext; ++i) {
            const float wb = static_cast<float>(i) / static_cast<float>(ext);
            if (dim == 0)
                for (int x = 0; x < b.w; ++x)
                    b.at(c, i, x) = a.at(c, a.h - ext + i, x) * (1.f - wb) + b.at(c, i, x) * wb;
            else
                for (int y = 0; y < b.h; ++y)
                    b.at(c, y, i) = a.at(c, y, a.w - ext + i) * (1.f - wb) + b.at(c, y, i) * wb;
        }
    return b;
}

// pixels [3, H, W] -> latent [24, H/16, W/16]
std::vector<float> vae_encode_image(const std::shared_ptr<Runtime> &rt,
                                    const std::shared_ptr<VulkanCommandPool> &pool,
                                    const std::vector<float> &px, int H, int W, double &ms) {
    std::vector<int> ys, yl, yo, xs, xl, xo;
    enc_tile_plan(H, ys, yl, yo);
    enc_tile_plan(W, xs, xl, xo);
    std::vector<std::vector<LatTile>> rows(ys.size());
    for (size_t i = 0; i < ys.size(); ++i)
        for (size_t j = 0; j < xs.size(); ++j) {
            const int h = yl[i], w = xl[j];
            std::vector<float> tile(static_cast<size_t>(3) * h * w);
            for (int c = 0; c < 3; ++c)
                for (int y = 0; y < h; ++y)
                    std::memcpy(&tile[(static_cast<size_t>(c) * h + y) * w],
                                &px[(static_cast<size_t>(c) * H + ys[i] + y) * W + xs[j]],
                                sizeof(float) * w);
            upload_float(rt, pool, "pixels", tile, {1, 3, static_cast<uint32_t>(h),
                                                    static_cast<uint32_t>(w)});
            ms += rt->Run();
            rt->ReadResult();
            LatTile t;
            t.h = h / kEncRatio;
            t.w = w / kEncRatio;
            t.v = read_output(rt, "latent");
            rows[i].push_back(std::move(t));
        }
    const int HL = H / kEncRatio, WL = W / kEncRatio;
    std::vector<float> out(static_cast<size_t>(kVideoC) * HL * WL);
    int oy = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
        int ox = 0, th = 0;
        for (size_t j = 0; j < rows[i].size(); ++j) {
            LatTile t = rows[i][j];
            if (i > 0) t = enc_blend(rows[i - 1][j], t, yo[i - 1] / kEncRatio, 0);
            if (j > 0) t = enc_blend(rows[i][j - 1], t, xo[j - 1] / kEncRatio, 1);
            const int keep_h = i + 1 < rows.size() ? t.h - yo[i] / kEncRatio : t.h;
            const int keep_w = j + 1 < rows[i].size() ? t.w - xo[j] / kEncRatio : t.w;
            for (int c = 0; c < kVideoC; ++c)
                for (int y = 0; y < keep_h; ++y)
                    for (int x = 0; x < keep_w; ++x)
                        out[(static_cast<size_t>(c) * HL + oy + y) * WL + ox + x] = t.at(c, y, x);
            ox += keep_w;
            th = keep_h;
        }
        if (ox != WL) throw std::runtime_error("vae encode: tile widths do not add up");
        oy += th;
    }
    if (oy != HL) throw std::runtime_error("vae encode: tile heights do not add up");
    return out;
}

int decode_audio(const Options &o, const std::shared_ptr<VulkanCommandPool> &pool,
                 const std::vector<float> &xa) {
    if (o.audio_vae.empty()) return 0;
    const int T = static_cast<int>(xa.size() / (kAudioC * 2));
    auto t0 = std::chrono::steady_clock::now();
    auto rt = std::make_shared<Runtime>(pool, o.audio_vae, /*precision=*/0);
    rt->set_backend_buffer(true);
    rt->LoadModel();
    std::vector<float> z(xa.size());
    for (int ch = 0; ch < 2; ++ch)
        for (int c = 0; c < kAudioC; ++c)
            for (int t = 0; t < T; ++t)
                z[(static_cast<size_t>(ch) * kAudioC + c) * T + t] =
                    xa[(static_cast<size_t>(c) * 2 + ch) * T + t];
    upload_float(rt, pool, "z", z, {2, kAudioC, 1, static_cast<uint32_t>(T)});
    const double ms = rt->Run();
    rt->ReadResult();
    const auto wave = read_output(rt, "wave");
    const Stats s = stats(wave);
    std::printf("[audio] %d latent frames -> %zu samples x 2 ch (%.2f s at %d Hz), "
                "%.1fs (gpu %.0f ms); rms %.4f absmax %.3f%s\n", T, wave.size() / 2,
                static_cast<double>(wave.size() / 2) / kAudioRate, kAudioRate,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                ms, s.rms, s.absmax, s.nonfinite ? " NONFINITE" : "");
    if (wave.size() != static_cast<size_t>(2) * T * kAudioHop)
        throw std::runtime_error("audio VAE: unexpected output size");
    if (!o.wav.empty()) {
        write_wav_f32(o.wav, wave, 2, kAudioRate);
        std::printf("[audio] wrote %s\n", o.wav.c_str());
    }
    return s.nonfinite ? 3 : 0;
}

std::shared_ptr<VulkanDevice> pick_device() {
    std::shared_ptr<VulkanDevice> dev;
    for (auto *pdev : VulkanInstance::getVulkanInstance().getPhysicalDevices()) {
        auto cand = std::make_shared<VulkanDevice>(pdev);
        if (cand->getDeviceName().find("llvmpipe") != std::string::npos) continue;
        dev = cand;
    }
    return dev;
}

int main(int argc, char **argv) {
    // Line-buffered even into a log file, so per-step progress shows as it
    // happens (a full buffer held 30 steps back on a long run).
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    Options o;
    if (!parse(argc, argv, o)) {
        usage(argv[0]);
        return 2;
    }
    if (o.host_dir.empty())
        o.host_dir = std::filesystem::path(o.dit).parent_path().string();
    if (o.width % 32 || o.height % 32) {
        std::fprintf(stderr, "width/height must be multiples of 32\n");
        return 2;
    }

    // temporal shape (comfy nodes_minimax_h3.temporal_shape)
    int frames = std::max(5, o.length);
    while (frames % 17 != 5) ++frames;
    const int latent_t = frames <= 5 ? 2 : ((frames - 5) / 17) * 5 + 2;
    const int audio_t = static_cast<int>(
        std::lround(static_cast<double>(frames) / kFps * kAudioLatentFps));
    const int lat_h = o.height / 16, lat_w = o.width / 16;

    if (!o.audio_latent.empty()) {
        const auto xa = read_f32(o.audio_latent);
        if (xa.empty() || xa.size() % (kAudioC * 2)) {
            std::fprintf(stderr, "%s is not [32, 2, T]\n", o.audio_latent.c_str());
            return 2;
        }
        auto dev = pick_device();
        if (!dev) return 1;
        auto pool = std::make_shared<VulkanCommandPool>(dev);
        const int rc = decode_audio(o, pool, xa);
        if (rc || o.latent.empty()) return rc;
    }

    if (!o.latent.empty()) {
        // decode only: --width/--height/--length give the latent's shape
        const auto xv = read_f32(o.latent);
        if (xv.size() != static_cast<size_t>(kVideoC) * latent_t * lat_h * lat_w) {
            std::fprintf(stderr, "%s is not [24, %d, %d, %d]\n", o.latent.c_str(),
                         latent_t, lat_h, lat_w);
            return 2;
        }
        auto dev = pick_device();
        if (!dev) return 1;
        auto pool = std::make_shared<VulkanCommandPool>(dev);
        return decode_and_write(o, pool, xv, latent_t, lat_h, lat_w);
    }

    const auto inv_freq = read_f32(o.host_dir + "/rope_inv_freq.f32");
    const auto t_table = read_f32(o.host_dir + "/adaln_t_table.f32");
    const int t_dim = 8;
    const int t_rows = static_cast<int>(t_table.size() / t_dim);
    const int n_freq = static_cast<int>(inv_freq.size());
    const int n_ang = n_freq * kRopeAxes;

    std::vector<float> text;
    int text_len = o.text_len;
    std::vector<int64_t> prompt_ids;
    if (!o.te.empty()) {
        const std::string ids_path = o.qwen_dir.empty() ? o.prompt_ids : o.qwen_dir + "/ids.i64";
        if (o.embed.empty() || ids_path.empty() || (!o.qwen_dir.empty() && o.vision.empty())) {
            std::fprintf(stderr, "--te needs --embed and --prompt-ids (or --qwen DIR --vision V)\n");
            return 2;
        }
        prompt_ids = read_i64(ids_path);
        text_len = static_cast<int>(prompt_ids.size());
    } else if (!o.text_states.empty()) {
        text = read_f32(o.text_states);
        text_len = static_cast<int>(text.size() / kTextDim);
    } else {
        std::mt19937 rng(o.seed ^ 0x5eedU);
        std::normal_distribution<float> nd(0.f, 1.f);
        text.resize(static_cast<size_t>(text_len) * kTextDim);
        for (auto &v : text) v = nd(rng);
    }

    // ref2va reference images: normalized latents [24, 1, HL, WL] -> rows,
    // condition-noise-augmented once (comfy _cond_video_rows: aug * r +
    // (1 - aug) * noise with one fixed seed for the whole run; our RNG, not
    // torch's -- the noise is 0.1% of the row)
    std::vector<std::pair<int, int>> ref_hw;
    std::vector<float> ref_rows;
    for (const auto &spec : o.ref_latents) {
        const size_t c2 = spec.rfind(':'), c1 = spec.rfind(':', c2 - 1);
        if (c1 == std::string::npos || c2 == std::string::npos || c1 == 0) {
            std::fprintf(stderr, "--ref-latent wants PATH:HL:WL, got %s\n", spec.c_str());
            return 2;
        }
        const int rh = std::atoi(spec.c_str() + c1 + 1), rw = std::atoi(spec.c_str() + c2 + 1);
        const auto z = read_f32(spec.substr(0, c1));
        if (rh % 2 || rw % 2 || z.size() != static_cast<size_t>(kVideoC) * rh * rw) {
            std::fprintf(stderr, "%s: %zu floats, want [24, 1, %d, %d] with even HL/WL\n",
                         spec.c_str(), z.size(), rh, rw);
            return 2;
        }
        auto r = patchify_thw(z, 1, rh, rw);
        ref_rows.insert(ref_rows.end(), r.begin(), r.end());
        ref_hw.emplace_back(rh, rw);
    }
    // reference images: sizes now (the layout needs them), encoded once the
    // GPU is up
    std::vector<std::pair<std::string, std::pair<int, int>>> ref_px;
    for (const auto &spec : o.ref_images) {
        const size_t c2 = spec.rfind(':'), c1 = spec.rfind(':', c2 - 1);
        if (o.vae_enc.empty() || c1 == std::string::npos || c2 == std::string::npos || c1 == 0) {
            std::fprintf(stderr, "--ref-image wants PATH:TH:TW and --vae-enc, got %s\n",
                         spec.c_str());
            return 2;
        }
        const int th = std::atoi(spec.c_str() + c1 + 1), tw = std::atoi(spec.c_str() + c2 + 1);
        if (th % 32 || tw % 32 || th <= 0 || tw <= 0) {
            std::fprintf(stderr, "%s: TH/TW must be positive multiples of 32\n", spec.c_str());
            return 2;
        }
        ref_px.push_back({spec.substr(0, c1), {th, tw}});
        ref_hw.emplace_back(th / kEncRatio, tw / kEncRatio);
    }
    // comfy _cond_video_rows: aug * r + (1 - aug) * noise, one fixed seed
    const auto augment = [&](std::vector<float> &rows) {
        if (o.cond_aug >= 1.0) return;
        std::mt19937 arng(o.seed ^ 0xa11ceU);
        std::normal_distribution<float> an(0.f, 1.f);
        const float a = static_cast<float>(o.cond_aug);
        for (auto &v : rows) v = a * v + (1.f - a) * an(arng);
    };
    if (ref_px.empty()) augment(ref_rows);
    std::vector<int64_t> text_tags(text_len, 1);
    const std::string tags_path = !o.text_tags.empty() ? o.text_tags
                                  : !o.qwen_dir.empty() ? o.qwen_dir + "/tags.i64" : "";
    if (!tags_path.empty()) {
        text_tags = read_i64(tags_path);
        if (static_cast<int>(text_tags.size()) != text_len) {
            std::fprintf(stderr, "--text-tags has %zu entries, text has %d tokens\n",
                         text_tags.size(), text_len);
            return 2;
        }
    }

    const Layout l = build_layout(text_len, latent_t, lat_h, lat_w, audio_t, ref_hw);
    std::printf("[fl2va] %dx%d, %d frames -> latent %dx%dx%d, audio_t %d, "
                "text %d%s; seq %d (text %d, refs %d in %zu, audio %d, video %d)\n",
                o.width, o.height, frames, latent_t, lat_h, lat_w, audio_t,
                text_len,
                !o.te.empty() ? " (prompt)"
                : o.text_states.empty() ? " (synthetic)" : "", l.seq,
                l.n_text, l.n_ref, ref_hw.size(), l.n_audio, l.n_video);

    // rope cos/sin are step-independent
    std::vector<float> rope_cos(static_cast<size_t>(l.seq) * n_ang);
    std::vector<float> rope_sin(rope_cos.size());
    for (int s = 0; s < l.seq; ++s)
        for (int a = 0; a < kRopeAxes; ++a)
            for (int f = 0; f < n_freq; ++f) {
                const float pos = static_cast<float>(l.position[s * 3 + a]);
                const float ang = pos * inv_freq[f];
                rope_cos[static_cast<size_t>(s) * n_ang + a * n_freq + f] = std::cos(ang);
                rope_sin[static_cast<size_t>(s) * n_ang + a * n_freq + f] = std::sin(ang);
            }

    // per-step adaLN tables: distinct timesteps -> curve rows, per-row mod index
    // (row * 3 + modality tag: video 0, text 1, audio 2)
    const auto step_tables = [&](double t_v, double t_a, std::vector<float> &t_emb,
                                 std::vector<int64_t> &mod, int64_t &row_v,
                                 int64_t &row_a) {
        const double t_ref = std::max(t_v, o.cond_aug);
        std::set<double> uniq{t_v, t_a};
        if (l.n_ref) uniq.insert(t_ref);
        std::vector<double> unique_t(uniq.begin(), uniq.end());
        auto t_row = [&](double t) {
            return static_cast<int64_t>(
                std::find(unique_t.begin(), unique_t.end(), t) - unique_t.begin());
        };
        t_emb.assign(unique_t.size() * t_dim, 0.f);
        for (size_t u = 0; u < unique_t.size(); ++u) {
            const float pos = std::clamp(static_cast<float>(unique_t[u]), 0.f, 1.f) *
                              static_cast<float>(t_rows - 1);
            const int i0 = std::min(static_cast<int>(std::floor(pos)), t_rows - 2);
            const float fr = pos - static_cast<float>(i0);
            for (int d = 0; d < t_dim; ++d) {
                const float a = t_table[i0 * t_dim + d], b = t_table[(i0 + 1) * t_dim + d];
                t_emb[u * t_dim + d] = a + (b - a) * fr;
            }
        }
        mod.assign(l.seq, 0);
        int r = 0;
        for (int i = 0; i < l.n_text; ++i) mod[r++] = t_row(t_v) * 3 + text_tags[i];
        for (int i = 0; i < l.n_ref; ++i) mod[r++] = t_row(t_ref) * 3 + 0;
        for (int i = 0; i < l.n_audio; ++i) mod[r++] = t_row(t_a) * 3 + 2;
        for (int i = 0; i < l.n_video; ++i) mod[r++] = t_row(t_v) * 3 + 0;
        row_v = t_row(t_v);
        row_a = t_row(t_a);
    };

    // initial noise: video [24, T, H, W] then audio [32, 2, Ta]
    std::mt19937 rng(o.seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> xv(static_cast<size_t>(kVideoC) * latent_t * lat_h * lat_w);
    std::vector<float> xa(static_cast<size_t>(kAudioC) * 2 * audio_t);
    for (auto &v : xv) v = nd(rng);
    for (auto &v : xa) v = nd(rng);

    // sigma grids: video sigma on the shift-12 grid, audio mapped to shift 3
    const int steps = std::max(1, o.steps);
    std::vector<double> sv(steps + 1), sa(steps + 1);
    for (int k = 0; k <= steps; ++k) {
        sv[k] = time_shift_sigma(1.0 - static_cast<double>(k) / steps, 1.0, kShiftVideo);
        sa[k] = time_shift_sigma(sv[k], kShiftVideo, kShiftAudio);
    }

    if (!o.dump_layout.empty()) {
        // step-1 tables, for checking the layout against comfy's PackedLayout
        std::filesystem::create_directories(o.dump_layout);
        const double t_v = 1.0 - std::max(sv[0], 1e-6);
        const double t_a = 1.0 - time_shift_sigma(std::max(sv[0], 1e-6), kShiftVideo, kShiftAudio);
        std::vector<float> t_emb;
        std::vector<int64_t> mod;
        int64_t rv, ra;
        step_tables(t_v, t_a, t_emb, mod, rv, ra);
        auto dump = [&](const std::string &n, const void *p, size_t bytes) {
            std::ofstream f(o.dump_layout + "/" + n, std::ios::binary);
            f.write(static_cast<const char *>(p), static_cast<std::streamsize>(bytes));
        };
        dump("position.f64", l.position.data(), l.position.size() * 8);
        dump("pack.i64", l.pack.data(), l.pack.size() * 8);
        dump("mod.i64", mod.data(), mod.size() * 8);
        dump("t_emb.f32", t_emb.data(), t_emb.size() * 4);
        dump("ref_rows.f32", ref_rows.data(), ref_rows.size() * 4);
        std::printf("[dump] %s: seq %d, %zu distinct t\n", o.dump_layout.c_str(), l.seq,
                    t_emb.size() / t_dim);
        return 0;
    }

    std::shared_ptr<VulkanDevice> dev;
    try {
        for (auto *pdev : VulkanInstance::getVulkanInstance().getPhysicalDevices()) {
            auto cand = std::make_shared<VulkanDevice>(pdev);
            if (cand->getDeviceName().find("llvmpipe") != std::string::npos) continue;
            dev = cand;
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

    if (!o.te.empty()) {
        if (!o.qwen_dir.empty()) {
            const VisionRows vr = run_vision(pool, o.vision, o.qwen_dir, text_len);
            text = run_text_tower(pool, o.te, o.embed, prompt_ids, &vr);
        } else {
            text = run_text_tower(pool, o.te, o.embed, prompt_ids);
        }
    }
    if (!ref_px.empty() && o.text_out.empty()) {
        auto te0 = std::chrono::steady_clock::now();
        auto enc = std::make_shared<Runtime>(pool, o.vae_enc, /*precision=*/0);
        enc->set_backend_buffer(true);
        enc->LoadModel();
        double ems = 0;
        int ri = 0;
        for (const auto &[path, hw] : ref_px) {
            const auto px = read_f32(path);
            if (px.size() != static_cast<size_t>(3) * hw.first * hw.second) {
                std::fprintf(stderr, "%s: %zu floats, want [3, %d, %d]\n", path.c_str(),
                             px.size(), hw.first, hw.second);
                return 2;
            }
            const auto z = vae_encode_image(enc, pool, px, hw.first, hw.second, ems);
            const Stats zs = stats(z);
            std::printf("[vae-enc] %s %dx%d -> latent %dx%d, rms %.3f%s\n", path.c_str(),
                        hw.second, hw.first, hw.first / kEncRatio, hw.second / kEncRatio,
                        zs.rms, zs.nonfinite ? " NONFINITE" : "");
            if (zs.nonfinite) return 3;
            if (!o.save_dir.empty()) {
                std::filesystem::create_directories(o.save_dir);
                write_f32(o.save_dir + "/ref_latent_" + std::to_string(ri) + ".f32", z);
            }
            const auto r = patchify_thw(z, 1, hw.first / kEncRatio, hw.second / kEncRatio);
            ref_rows.insert(ref_rows.end(), r.begin(), r.end());
            ++ri;
        }
        augment(ref_rows);
        std::printf("[vae-enc] %zu image(s) in %.1fs (gpu %.0f ms)\n", ref_px.size(),
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - te0).count(),
                    ems);
    }  // the encoder is released before the DiT loads
    if (o.steps <= 0) return 0;  // --steps 0: prepare (text / refs) only
    if (!o.text_out.empty()) {
        write_f32(o.text_out, text);
        std::printf("[te] wrote %s [%d, %d]\n", o.text_out.c_str(), text_len, kTextDim);
        return 0;
    }

    auto t0 = std::chrono::steady_clock::now();
    auto rt = std::make_shared<Runtime>(pool, o.dit, /*precision=*/0);
    rt->set_backend_buffer(true);
    rt->LoadModel();
    const double load_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    std::printf("[load] %s in %.1fs\n", o.dit.c_str(), load_s);

    const auto u32 = [](size_t v) { return static_cast<uint32_t>(v); };
    // step-independent inputs, uploaded once per round anyway: Run() may have
    // retired them if activation reuse is on, and they are small.
    if (o.sampler != "euler" && o.sampler != "res_multistep") {
        std::fprintf(stderr, "--sampler: euler or res_multistep\n");
        return 2;
    }
    // res_multistep (comfy k_diffusion, eta 0): Euler on the first and the
    // last step, otherwise x <- e^-h x + h (b1 x0 + b2 x0_prev) on t = -log sigma.
    // x0 = x + sigma * out (the head predicts x0 - noise). Applied per stream on
    // its own sigma grid: exact for video (the sampler's sigma); for audio comfy
    // steps the carried (sigma_v / sigma_a)-scaled latent instead, so audio is
    // an approximation of the workflow's trajectory.
    std::vector<float> x0v_prev, x0a_prev;
    const auto res_step = [](std::vector<float> &x, const std::vector<float> &x0,
                             std::vector<float> &x0_prev, const std::vector<double> &sg,
                             int k) {
        const double s = sg[k], sn = sg[k + 1];
        if (k == 0 || sn <= 0.0 || x0_prev.empty()) {
            for (size_t i = 0; i < x.size(); ++i)  // Euler: x + (sn - s) (x - x0) / s
                x[i] = static_cast<float>(x[i] + (sn - s) * (x[i] - x0[i]) / s);
        } else {
            const double t = -std::log(s), tn = -std::log(sn), tp = -std::log(sg[k - 1]);
            const double h = tn - t, c2 = (tp - t) / h;
            const double phi1 = std::expm1(-h) / -h, phi2 = (phi1 - 1.0) / -h;
            const double b1 = phi1 - phi2 / c2, b2 = phi2 / c2, eh = std::exp(-h);
            for (size_t i = 0; i < x.size(); ++i)
                x[i] = static_cast<float>(eh * x[i] + h * (b1 * x0[i] + b2 * x0_prev[i]));
        }
        x0_prev = x0;
    };
    double total_ms = 0;
    for (int k = 0; k < steps; ++k) {
        const double sig_v = std::max(sv[k], 1e-6), sig_a = sa[k];
        const double t_v = 1.0 - sig_v, t_a = 1.0 - time_shift_sigma(sig_v, kShiftVideo, kShiftAudio);
        std::vector<float> t_emb;
        std::vector<int64_t> mod;
        int64_t row_v, row_a;
        step_tables(t_v, t_a, t_emb, mod, row_v, row_a);
        const size_t n_t = t_emb.size() / t_dim;
        std::vector<int64_t> vidx(l.n_video), vt(l.n_video, row_v);
        std::vector<int64_t> aidx(l.n_audio), at(l.n_audio, row_a);
        for (int i = 0; i < l.n_video; ++i) vidx[i] = l.n_text + l.n_ref + l.n_audio + i;
        for (int i = 0; i < l.n_audio; ++i) aidx[i] = l.n_text + l.n_ref + i;

        // refs ride ahead of the target video rows in the same projection
        std::vector<float> vrows = ref_rows;
        {
            const auto tr = patchify(xv, l);
            vrows.insert(vrows.end(), tr.begin(), tr.end());
        }
        upload_float(rt, pool, "video_rows", vrows, {u32(l.n_ref + l.n_video), 96});
        upload_float(rt, pool, "audio_rows", pack_audio(xa, audio_t), {u32(l.n_audio), 32});
        upload_float(rt, pool, "text_states", text, {u32(text_len), kTextDim});
        upload<int64_t>(rt, pool, "pack_index", l.pack, {u32(l.seq)});
        upload_float(rt, pool, "t_emb", t_emb, {u32(n_t), u32(t_dim)});
        upload<int64_t>(rt, pool, "mod_index", mod, {u32(l.seq)});
        upload_float(rt, pool, "rope_cos", rope_cos, {u32(l.seq), u32(n_ang)});
        upload_float(rt, pool, "rope_sin", rope_sin, {u32(l.seq), u32(n_ang)});
        upload<int64_t>(rt, pool, "video_index", vidx, {u32(l.n_video)});
        upload<int64_t>(rt, pool, "video_t", vt, {u32(l.n_video)});
        upload<int64_t>(rt, pool, "audio_index", aidx, {u32(l.n_audio)});
        upload<int64_t>(rt, pool, "audio_t", at, {u32(l.n_audio)});

        const double ms = rt->Run();
        rt->ReadResult();
        total_ms += ms;
        const auto ov = read_output(rt, "video_v");
        const auto oa = read_output(rt, "audio_v");
        const Stats s1 = stats(ov), s2 = stats(oa);
        std::printf("  step %d/%d  sigma_v %.4f sigma_a %.4f  %.0f ms  "
                    "video_out rms %.3f absmax %.2f%s  audio_out rms %.3f absmax %.2f%s\n",
                    k + 1, steps, sig_v, sig_a, ms, s1.rms, s1.absmax,
                    s1.nonfinite ? " NONFINITE" : "", s2.rms, s2.absmax,
                    s2.nonfinite ? " NONFINITE" : "");
        if (s1.nonfinite || s2.nonfinite) {
            std::fprintf(stderr, "non-finite denoiser output; stopping\n");
            return 3;
        }
        if (o.sampler == "euler") {
            add_unpatchified(xv, ov, l, static_cast<float>(sv[k] - sv[k + 1]));
            add_unpacked_audio(xa, oa, audio_t, static_cast<float>(sa[k] - sa[k + 1]));
        } else {
            std::vector<float> x0v = xv, x0a = xa;  // x0 = x + sigma * out
            add_unpatchified(x0v, ov, l, static_cast<float>(sv[k]));
            add_unpacked_audio(x0a, oa, audio_t, static_cast<float>(sa[k]));
            res_step(xv, x0v, x0v_prev, sv, k);
            res_step(xa, x0a, x0a_prev, sa, k);
        }
    }
    const Stats fv = stats(xv), fa = stats(xa);
    std::printf("[done] %d steps, %.2f s/step; video latent rms %.3f absmax %.2f, "
                "audio latent rms %.3f absmax %.2f\n",
                steps, total_ms / 1000.0 / steps, fv.rms, fv.absmax, fa.rms, fa.absmax);
    if (!o.save_dir.empty()) {
        std::filesystem::create_directories(o.save_dir);
        write_f32(o.save_dir + "/video_latent.f32", xv);
        write_f32(o.save_dir + "/audio_latent.f32", xa);
        std::printf("[save] %s/video_latent.f32 [24,%d,%d,%d], audio_latent.f32 "
                    "[32,2,%d]\n", o.save_dir.c_str(), latent_t, lat_h, lat_w, audio_t);
    }
    rt.reset();  // the 20 GB DiT goes before the decoders load
    if (const int rc = decode_audio(o, pool, xa)) return rc;
    return decode_and_write(o, pool, xv, latent_t, lat_h, lat_w);
}

int decode_and_write(const Options &o, const std::shared_ptr<VulkanCommandPool> &pool,
                     const std::vector<float> &xv, int latent_t, int lat_h, int lat_w) {
    if (!o.vae.empty()) {
        auto tv = std::chrono::steady_clock::now();
        Vae vae;
        vae.pool = pool;
        vae.rt = std::make_shared<Runtime>(pool, o.vae, /*precision=*/0);
        vae.rt->set_backend_buffer(true);
        vae.rt->LoadModel();
        const Video video = vae_decode(vae, xv, latent_t, lat_h, lat_w);
        const Stats sv2 = stats(video.v);
        std::printf("[vae] %d frames %dx%d, %d decoder calls, %.1fs (gpu %.1fs); "
                    "pixel mean %.3f%s\n", video.F, video.W, video.H, vae.calls,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - tv).count(),
                    vae.ms / 1000.0, sv2.mean, sv2.nonfinite ? " NONFINITE" : "");
        if (!o.frames_dir.empty()) {
            std::filesystem::create_directories(o.frames_dir);
            char name[64];
            for (int f = 0; f < video.F; ++f) {
                std::snprintf(name, sizeof(name), "/frame_%04d.ppm", f);
                write_ppm(o.frames_dir + name, video, f);
            }
            std::printf("[frames] %s/frame_%%04d.ppm (%d frames, %d fps)\n",
                        o.frames_dir.c_str(), video.F, kFps);
        }
    }
    return 0;
}
