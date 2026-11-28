// MiniMax-H3 FL2VA / Ref2VA denoiser driver (ComfyUI single-file checkpoint).
//
// Runs the text-to-audio-video sampling loop around the one-step DiT graph
// that model/exporter/minimax_h3/h3_export_fl2va.py exports. The host side
// mirrors ComfyUI (comfy/ldm/minimax/model.py, plain t2va: no keyframes, no
// references):
//
//   * packed layout [text | audio (stereo, channel-major) | video (2x2 patches)]
//     and its (t, h, w) RoPE positions; the graph gets cos/sin of the 48
//     split-half angles,
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
constexpr double kShiftVideo = 12.0;
constexpr double kShiftAudio = 3.0;
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
    int width = 256, height = 256, length = 22, steps = 8, text_len = 32;
    uint32_t seed = 42;
};

void usage(const char *argv0) {
    std::fprintf(stderr,
                 "usage: %s --dit dit.vkopbin [--host-dir DIR] [--width W] "
                 "[--height H]\n"
                 "          [--length FRAMES] [--steps N] [--seed S] "
                 "[--text-len L | --text-states F.f32]\n"
                 "          [--save DIR]\n"
                 "  --host-dir   rope_inv_freq.f32 + adaln_t_table.f32 "
                 "(default: the DiT's directory)\n"
                 "  --length     frames at 24 fps, snapped up to 17k+5\n"
                 "  --text-states raw fp32 [L, 5120] Qwen3-VL layer-50 states\n"
                 "  --te te.vkopbin --embed embed_tokens.bf16 --prompt-ids ids.i64\n"
                 "               run the text tower on the prompt instead\n"
                 "  --vae vae.vkopbin --frames DIR   decode to DIR/frame_%%04d.ppm\n",
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
        else if (a == "--width" && (v = next("--width"))) o.width = std::atoi(v);
        else if (a == "--height" && (v = next("--height"))) o.height = std::atoi(v);
        else if (a == "--length" && (v = next("--length"))) o.length = std::atoi(v);
        else if (a == "--steps" && (v = next("--steps"))) o.steps = std::atoi(v);
        else if (a == "--text-len" && (v = next("--text-len"))) o.text_len = std::atoi(v);
        else if (a == "--seed" && (v = next("--seed"))) o.seed = static_cast<uint32_t>(std::atoll(v));
        else return false;
        if (v == nullptr) return false;
    }
    return !o.dit.empty() || !o.latent.empty();
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
    if (probe->elem_kind() != vkop::core::ElemKind::kBFloat16)
        throw std::runtime_error("graph input " + name +
                                 " is neither fp32 nor bf16");
    std::vector<uint16_t> bits(host.size());
    for (size_t i = 0; i < host.size(); ++i)
        bits[i] = static_cast<uint16_t>(vkop::core::float_to_bf16(host[i]));
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
    if (t->elem_kind() != vkop::core::ElemKind::kBFloat16)
        throw std::runtime_error("output " + name + " is neither fp32 nor bf16");
    auto h = as_tensor<uint16_t>(t);
    std::vector<float> out(h->num_elements());
    for (size_t i = 0; i < out.size(); ++i)
        out[i] = vkop::core::bf16_to_float((*h)[i]);
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

// The step-independent part of comfy's PackedLayout for plain t2va.
struct Layout {
    int text_len, latent_t, lat_h, lat_w, audio_t;
    int hh, ww, n_text, n_audio, n_video, seq;
    std::vector<double> position;  // [seq, 3] (t, h, w)
    std::vector<int64_t> pack;     // concat([text, video, audio]) -> packed
};

Layout build_layout(int text_len, int latent_t, int lat_h, int lat_w,
                    int audio_t) {
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
    l.seq = l.n_text + l.n_audio + l.n_video;
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
    const double cursor = text_len;
    for (int ch = 0; ch < 2; ++ch)
        for (int t = 0; t < audio_t; ++t)
            put(row++, cursor + t, 0.0, ch == 0 ? aw.front() : aw.back());
    double tpos = cursor;
    for (int t = 0; t < latent_t; ++t) {
        for (int y = 0; y < l.hh; ++y)
            for (int x = 0; x < l.ww; ++x) put(row++, tpos, ah[y], aw[x]);
        tpos += kFrameRescale * kFramePerToken[t % 5];
    }
    // graph concat order is [text, video, audio]; packed is [text, audio, video]
    l.pack.resize(l.seq);
    int p = 0;
    for (int i = 0; i < l.n_text; ++i) l.pack[p++] = i;
    for (int i = 0; i < l.n_audio; ++i) l.pack[p++] = l.n_text + l.n_video + i;
    for (int i = 0; i < l.n_video; ++i) l.pack[p++] = l.n_text + i;
    return l;
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
std::vector<float> run_text_tower(const std::shared_ptr<VulkanCommandPool> &pool,
                                  const std::string &te, const std::string &embed,
                                  const std::vector<int64_t> &ids) {
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
    std::vector<float> bias(static_cast<size_t>(L) * L, 0.f);
    for (int i = 0; i < L; ++i)
        for (int j = i + 1; j < L; ++j) bias[static_cast<size_t>(i) * L + j] = -3.0e38f;
    std::vector<float> rc(static_cast<size_t>(L) * kTeHeadDim), rs(rc.size());
    for (int p = 0; p < L; ++p)
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
        upload_float(rt, pool, "rope_cos", rc, {u(L), kTeHeadDim});
        upload_float(rt, pool, "rope_sin", rs, {u(L), kTeHeadDim});
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
constexpr int kTileSize = 256, kTileOverlapMin = 64;
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
                 std::vector<int> &overlaps) {
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

    // comfy tiled_decode over one clip z [24][T][Hl][Wl].
    Video tiled(const std::vector<float> &z, int T, int Hl, int Wl) {
        const int H = Hl * kVaeRatio, W = Wl * kVaeRatio;
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
        if (o.embed.empty() || o.prompt_ids.empty()) {
            std::fprintf(stderr, "--te needs --embed and --prompt-ids\n");
            return 2;
        }
        prompt_ids = read_i64(o.prompt_ids);
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

    const Layout l = build_layout(text_len, latent_t, lat_h, lat_w, audio_t);
    std::printf("[fl2va] %dx%d, %d frames -> latent %dx%dx%d, audio_t %d, "
                "text %d%s; seq %d (text %d, audio %d, video %d)\n",
                o.width, o.height, frames, latent_t, lat_h, lat_w, audio_t,
                text_len,
                !o.te.empty() ? " (prompt)"
                : o.text_states.empty() ? " (synthetic)" : "", l.seq,
                l.n_text, l.n_audio, l.n_video);

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

    if (!o.te.empty()) text = run_text_tower(pool, o.te, o.embed, prompt_ids);
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
    double total_ms = 0;
    for (int k = 0; k < steps; ++k) {
        const double sig_v = std::max(sv[k], 1e-6), sig_a = sa[k];
        const double t_v = 1.0 - sig_v, t_a = 1.0 - time_shift_sigma(sig_v, kShiftVideo, kShiftAudio);
        std::set<double> uniq{t_v, t_a};
        std::vector<double> unique_t(uniq.begin(), uniq.end());
        auto t_row = [&](double t) {
            return static_cast<int64_t>(
                std::find(unique_t.begin(), unique_t.end(), t) - unique_t.begin());
        };
        // adaLN curve table, lerped at each distinct t
        std::vector<float> t_emb(unique_t.size() * t_dim);
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
        std::vector<int64_t> mod(l.seq);
        int r = 0;
        for (int i = 0; i < l.n_text; ++i) mod[r++] = t_row(t_v) * 3 + 1;
        for (int i = 0; i < l.n_audio; ++i) mod[r++] = t_row(t_a) * 3 + 2;
        for (int i = 0; i < l.n_video; ++i) mod[r++] = t_row(t_v) * 3 + 0;
        std::vector<int64_t> vidx(l.n_video), vt(l.n_video, t_row(t_v));
        std::vector<int64_t> aidx(l.n_audio), at(l.n_audio, t_row(t_a));
        for (int i = 0; i < l.n_video; ++i) vidx[i] = l.n_text + l.n_audio + i;
        for (int i = 0; i < l.n_audio; ++i) aidx[i] = l.n_text + i;

        upload_float(rt, pool, "video_rows", patchify(xv, l), {u32(l.n_video), 96});
        upload_float(rt, pool, "audio_rows", pack_audio(xa, audio_t), {u32(l.n_audio), 32});
        upload_float(rt, pool, "text_states", text, {u32(text_len), kTextDim});
        upload<int64_t>(rt, pool, "pack_index", l.pack, {u32(l.seq)});
        upload_float(rt, pool, "t_emb", t_emb, {u32(unique_t.size()), u32(t_dim)});
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
        add_unpatchified(xv, ov, l, static_cast<float>(sv[k] - sv[k + 1]));
        add_unpacked_audio(xa, oa, audio_t, static_cast<float>(sa[k] - sa[k + 1]));
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
    rt.reset();  // the 20 GB DiT goes before the decoder loads
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
