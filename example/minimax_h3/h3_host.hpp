// junka @ 2026
// Host-side pieces of the MiniMax-H3 contract that the graphs deliberately do
// NOT contain:
//
//   * the timestep MLP (`time_proj` + `time_embedder`), which the reference
//     runs in-graph but whose `torch.arange` cannot fold on a meta build;
//   * the packed-sequence rotary grid (`build_packed_sequence`), which depends
//     only on the canvas and the prompt length;
//   * patchify, which turns a latent [1,C,F,H,W] into the denoiser's rows.
//
// Everything here mirrors `diffusers/modular_pipelines/minimax_h3/` exactly;
// where a formula is non-obvious the comment names the reference function.
// Nothing is approximated: a wrong rope grid is a silent quality loss, not a
// crash.

#pragma once

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace h3 {

// ---------------------------------------------------------------------------
// a minimal safetensors reader (header only; the payload is read on demand)
// ---------------------------------------------------------------------------

// The denoiser checkpoint's four timestep-MLP tensors are all we need, so this
// parses just enough JSON to find a named tensor's dtype/shape/byte range:
// a safetensors file is `<u64 header_len><json header><payload>`, and each
// entry in the header is
//   "name": {"dtype":"F32","shape":[r,c],"data_offsets":[begin,end]}
// where the offsets are relative to the end of the header.
class SafetensorsFile {
  public:
    explicit SafetensorsFile(const std::string &path) : path_(path) {
        std::ifstream in(path_, std::ios::binary);
        if (!in) {
            throw std::runtime_error("cannot open " + path_);
        }
        uint64_t hlen = 0;
        in.read(reinterpret_cast<char *>(&hlen), 8);
        if (!in || hlen == 0 || hlen > (64u << 20)) {
            throw std::runtime_error("bad safetensors header length in " + path_);
        }
        header_.resize(static_cast<size_t>(hlen));
        in.read(header_.data(), static_cast<std::streamsize>(hlen));
        if (!in) {
            throw std::runtime_error("truncated safetensors header in " + path_);
        }
        base_ = 8 + hlen;
        parse();
    }

    bool has(const std::string &name) const { return entries_.count(name) > 0; }

    struct Entry {
        std::string dtype;
        std::vector<int64_t> shape;
        uint64_t begin = 0;  // payload-relative
        uint64_t end = 0;
    };

    Entry entry(const std::string &name) const {
        auto it = entries_.find(name);
        if (it == entries_.end()) {
            throw std::runtime_error("'" + name + "' is not in " + path_);
        }
        return it->second;
    }

    // Read one F32 tensor as floats.
    std::vector<float> read_f32(const std::string &name) const {
        const Entry e = entry(name);
        if (e.dtype != "F32") {
            throw std::runtime_error("'" + name + "' is " + e.dtype +
                                     ", not F32 (this loader handles only the "
                                     "timestep MLP, which is float32)");
        }
        const uint64_t nbytes = e.end - e.begin;
        std::vector<float> out(nbytes / sizeof(float));
        std::ifstream in(path_, std::ios::binary);
        if (!in) {
            throw std::runtime_error("cannot reopen " + path_);
        }
        in.seekg(static_cast<std::streamoff>(base_ + e.begin));
        in.read(reinterpret_cast<char *>(out.data()),
                static_cast<std::streamsize>(nbytes));
        if (!in) {
            throw std::runtime_error("short read of '" + name + "'");
        }
        return out;
    }

  private:
    static bool is_space(char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r';
    }

    static void skip_space(const std::string &s, size_t *i) {
        while (*i < s.size() && is_space(s[*i])) {
            ++*i;
        }
    }

    // Position of `key` (as a quoted JSON member name) at or after `from`,
    // searched only inside the [from, limit) half-open window.
    static size_t find_key(const std::string &s, const std::string &key,
                           size_t from, size_t limit) {
        const std::string pat = "\"" + key + "\"";
        const size_t p = s.find(pat, from);
        return (p == std::string::npos || p >= limit) ? std::string::npos : p;
    }

    // Extract the bracketed integer list that follows `key` in `s`.
    static std::vector<int64_t> int_list(const std::string &s,
                                         const std::string &key, size_t from,
                                         size_t limit) {
        const size_t p = find_key(s, key, from, limit);
        if (p == std::string::npos) {
            return {};
        }
        const size_t lb = s.find('[', p);
        if (lb == std::string::npos || lb >= limit) {
            return {};
        }
        const size_t rb = s.find(']', lb);
        if (rb == std::string::npos || rb >= limit) {
            return {};
        }
        std::vector<int64_t> out;
        std::string num;
        for (size_t i = lb + 1; i < rb; ++i) {
            const char c = s[i];
            if (c == ',' || is_space(c)) {
                if (!num.empty()) {
                    out.push_back(std::stoll(num));
                    num.clear();
                }
                continue;
            }
            num.push_back(c);
        }
        if (!num.empty()) {
            out.push_back(std::stoll(num));
        }
        return out;
    }

    // Index just past the value that starts at `i` (which must be the first
    // non-space character of that value). Handles objects, arrays and strings
    // by bracket/quote pairing; strings in safetensors headers never contain
    // an unescaped quote, and escapes are skipped explicitly.
    static size_t skip_value(const std::string &s, size_t i) {
        skip_space(s, &i);
        if (i >= s.size()) {
            return i;
        }
        const char c = s[i];
        if (c == '{' || c == '[') {
            const char close = (c == '{') ? '}' : ']';
            int depth = 0;
            bool in_str = false;
            for (; i < s.size(); ++i) {
                const char d = s[i];
                if (in_str) {
                    if (d == '\\') {
                        ++i;
                    } else if (d == '"') {
                        in_str = false;
                    }
                    continue;
                }
                if (d == '"') {
                    in_str = true;
                } else if (d == c) {
                    ++depth;
                } else if (d == close && --depth == 0) {
                    return i + 1;
                }
            }
            return s.size();
        }
        if (c == '"') {
            for (++i; i < s.size(); ++i) {
                if (s[i] == '\\') {
                    ++i;
                } else if (s[i] == '"') {
                    return i + 1;
                }
            }
            return s.size();
        }
        while (i < s.size() && s[i] != ',' && s[i] != '}') {
            ++i;
        }
        return i;
    }

    // Walk the top-level object, `"<tensor name>": { ... }`, and record each
    // entry's dtype/shape/byte range. The value body is delimited by balanced
    // braces FIRST so that the field searches below cannot run off into the
    // neighbouring tensor (which is what made an earlier version of this
    // parser fail to find `time_embedder.linear_1.weight`).
    void parse() {
        size_t i = 0;
        skip_space(header_, &i);
        if (i >= header_.size() || header_[i] != '{') {
            throw std::runtime_error("safetensors header is not a JSON object: " +
                                     path_);
        }
        ++i;
        while (true) {
            skip_space(header_, &i);
            if (i >= header_.size() || header_[i] == '}') {
                break;
            }
            if (header_[i] == ',') {
                ++i;
                continue;
            }
            if (header_[i] != '"') {
                throw std::runtime_error("malformed safetensors header near byte " +
                                         std::to_string(i) + " of " + path_);
            }
            const size_t name_end = header_.find('"', i + 1);
            if (name_end == std::string::npos) {
                break;
            }
            const std::string name = header_.substr(i + 1, name_end - i - 1);
            i = name_end + 1;
            skip_space(header_, &i);
            if (i >= header_.size() || header_[i] != ':') {
                throw std::runtime_error("malformed safetensors header after '" +
                                         name + "' in " + path_);
            }
            ++i;
            skip_space(header_, &i);
            const size_t body = i;
            i = skip_value(header_, i);
            if (name == "__metadata__" || body >= header_.size() ||
                header_[body] != '{') {
                continue;
            }
            Entry e;
            const size_t dp = find_key(header_, "dtype", body, i);
            if (dp != std::string::npos) {
                const size_t v = header_.find('"', header_.find(':', dp) + 1);
                const size_t w =
                    (v == std::string::npos) ? std::string::npos
                                             : header_.find('"', v + 1);
                if (w != std::string::npos && w < i) {
                    e.dtype = header_.substr(v + 1, w - v - 1);
                }
            }
            e.shape = int_list(header_, "shape", body, i);
            const std::vector<int64_t> offs =
                int_list(header_, "data_offsets", body, i);
            if (offs.size() == 2) {
                e.begin = static_cast<uint64_t>(offs[0]);
                e.end = static_cast<uint64_t>(offs[1]);
            }
            entries_[name] = e;
        }
    }

    std::string path_;
    std::string header_;
    uint64_t base_ = 0;
    std::map<std::string, Entry> entries_;
};

// ---------------------------------------------------------------------------
// the timestep MLP  (`time_proj` -> `time_embedder`, both host-side)
// ---------------------------------------------------------------------------

// `diffusers.get_timestep_embedding(t, freq_dim=256, flip_sin_to_cos=True,
// downscale_freq_shift=0)`:  half = 128 frequencies
//   exponent = -ln(10000) * arange(half) / (half - 0)
//   emb      = t * exp(exponent)
//   out      = cat([cos(emb), sin(emb)], -1)     (flip_sin_to_cos puts cos first)
// Then `TimestepEmbedding`: linear_1 (256 -> 5376), SiLU, linear_2 (5376 -> 2688).
//
// The checkpoint stores both layers in F32, and the reference runs them in F32,
// so this stays float throughout and only the RESULT is cast by the caller.
class TimestepMLP {
  public:
    TimestepMLP(const std::string &checkpoint_dir, int freq_dim = 256,
                int hidden = 5376, int out_dim = 2688)
        : freq_dim_(freq_dim) {
        // The 4 tensors live in one shard each; find them by header scan.
        namespace fs = std::filesystem;
        for (const auto &ent : fs::directory_iterator(checkpoint_dir)) {
            if (ent.path().extension() != ".safetensors") {
                continue;
            }
            SafetensorsFile f(ent.path().string());
            if (!f.has("time_embedder.linear_1.weight")) {
                continue;
            }
            w1_ = f.read_f32("time_embedder.linear_1.weight");
            b1_ = f.read_f32("time_embedder.linear_1.bias");
            w2_ = f.read_f32("time_embedder.linear_2.weight");
            b2_ = f.read_f32("time_embedder.linear_2.bias");
            const auto s1 = f.entry("time_embedder.linear_1.weight").shape;
            const auto s2 = f.entry("time_embedder.linear_2.weight").shape;
            if (s1.size() != 2 || s2.size() != 2 || s1[1] != freq_dim ||
                s2[1] != s1[0]) {
                throw std::runtime_error(
                    "timestep MLP shapes disagree with the expected "
                    "[hidden,freq] -> [out,hidden]");
            }
            out_dim_ = static_cast<int>(s2[0]);
            hidden_ok_ = static_cast<int>(s1[0]);
            out_dim = out_dim_;
            (void)hidden;
            (void)out_dim;
            loaded_ = true;
            return;
        }
        throw std::runtime_error(
            "no safetensors with time_embedder.linear_1.weight under " +
            checkpoint_dir);
    }

    bool loaded() const { return loaded_; }
    int hidden() const { return hidden_ok_; }
    int out_dim() const { return out_dim_; }

    // `timesteps` are the DISTINCT values of the sampler's t = 1 - sigma.
    // Returns one row per timestep, `hidden()` floats each.
    std::vector<float> run(const std::vector<float> &timesteps) const {
        const int half = freq_dim_ / 2;
        std::vector<float> freq(static_cast<size_t>(half));
        const double denom = static_cast<double>(half - 0);
        for (int i = 0; i < half; ++i) {
            const double e = -std::log(10000.0) * static_cast<double>(i) / denom;
            freq[static_cast<size_t>(i)] = static_cast<float>(std::exp(e));
        }
        const int n = static_cast<int>(timesteps.size());
        std::vector<float> out(static_cast<size_t>(n) *
                                   static_cast<size_t>(out_dim_),
                               0.0F);
        std::vector<float> h(static_cast<size_t>(hidden_ok_));
        for (int r = 0; r < n; ++r) {
            // sinusoidal embedding, cos first (flip_sin_to_cos=True)
            std::vector<float> emb(static_cast<size_t>(freq_dim_));
            for (int i = 0; i < half; ++i) {
                const float a = timesteps[static_cast<size_t>(r)] * freq[static_cast<size_t>(i)];
                emb[static_cast<size_t>(i)] = std::cos(a);
                emb[static_cast<size_t>(half + i)] = std::sin(a);
            }
            // linear_1: [hidden, freq] x [freq] + bias
            for (int o = 0; o < hidden_ok_; ++o) {
                double acc = b1_[static_cast<size_t>(o)];
                const size_t row = static_cast<size_t>(o) * freq_dim_;
                for (int i = 0; i < freq_dim_; ++i) {
                    acc += static_cast<double>(w1_[row + i]) * emb[static_cast<size_t>(i)];
                }
                // SiLU
                const double x = acc;
                h[static_cast<size_t>(o)] =
                    static_cast<float>(x / (1.0 + std::exp(-x)));
            }
            // linear_2: [out, hidden] x [hidden] + bias
            for (int o = 0; o < out_dim_; ++o) {
                double acc = b2_[static_cast<size_t>(o)];
                const size_t row = static_cast<size_t>(o) * hidden_ok_;
                for (int i = 0; i < hidden_ok_; ++i) {
                    acc += static_cast<double>(w2_[row + i]) * h[static_cast<size_t>(i)];
                }
                out[static_cast<size_t>(r) * static_cast<size_t>(out_dim_) +
                    static_cast<size_t>(o)] = static_cast<float>(acc);
            }
        }
        return out;
    }

  private:
    int freq_dim_ = 256;
    int hidden_ok_ = 0;
    int out_dim_ = 0;
    bool loaded_ = false;
    std::vector<float> w1_, b1_, w2_, b2_;
};

// ---------------------------------------------------------------------------
// the packed sequence's rotary grid
// ---------------------------------------------------------------------------

// `_ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[idx % 5]`: one latent frame
// spans 5/3 * (1,4,4,4,4) rotary units, mirroring the VAE's 17-pixel-frames ->
// 5-latent-frames grouping.
inline constexpr double kRopeFrameRescale = 5.0 / 3.0;
inline constexpr int kRopeFramesPerLatent[5] = {1, 4, 4, 4, 4};
inline constexpr double kRopeSpatialScale = 32.0;

// `_spatial_position_grid`: `dim // patch` coordinates, aspect-normalized to
// the unit interval and scaled by 32. Built as numpy's
// `linspace(left, left+ratio, n, endpoint=False)` — NOT torch.linspace, whose
// arithmetic differs in the last ulp and which the reference deliberately
// reproduces.
inline std::vector<double> spatial_grid(int dim, int patch, double sqrt_area) {
    const double ratio = static_cast<double>(dim) / sqrt_area;
    const double left = (1.0 - ratio) / 2.0;
    const int n = dim / patch;
    const double step = ratio / static_cast<double>(n);
    std::vector<double> g(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        g[static_cast<size_t>(i)] = (left + static_cast<double>(i) * step) *
                                    kRopeSpatialScale;
    }
    return g;
}

// `_temporal_position_grid`: non-uniform spacing, starting at `origin`.
inline std::vector<double> temporal_grid(int num_latent_frames, double origin) {
    std::vector<double> spans(static_cast<size_t>(num_latent_frames));
    for (int i = 0; i < num_latent_frames; ++i) {
        spans[static_cast<size_t>(i)] =
            kRopeFrameRescale * kRopeFramesPerLatent[i % 5];
    }
    std::vector<double> out(static_cast<size_t>(num_latent_frames));
    out[0] = origin;
    for (int i = 1; i < num_latent_frames; ++i) {
        out[static_cast<size_t>(i)] =
            out[static_cast<size_t>(i - 1)] + spans[static_cast<size_t>(i - 1)];
    }
    return out;
}

struct LayoutSpec {
    int num_text_tokens = 0;
    int num_latent_frames = 0;
    int latent_height = 0;
    int latent_width = 0;
    int num_audio_latents = 0;
    int audio_channels = 2;
    int patch_t = 1, patch_h = 2, patch_w = 2;
    int video_tag = 0, text_tag = 1, audio_tag = 2;
};

// The full packed-sequence description, mirroring
// `MiniMaxH3PrepareLayoutStep.build_packed_sequence` for the t2va task with no
// keyframe anchors. Row order is [text | audio | video].
struct PackedLayout {
    int n_text = 0, n_audio = 0, n_video = 0, seq = 0;
    std::vector<int64_t> token_tags;
    std::vector<int64_t> timestep_indices;
    std::vector<float> position_ids;  // seq * 3
    std::vector<int64_t> video_indices, audio_indices, text_indices;
    std::vector<int64_t> pack_order;
};

inline PackedLayout build_packed_layout(const LayoutSpec &s) {
    PackedLayout l;
    l.n_text = s.num_text_tokens;
    const int rows_per_frame =
        (s.latent_height / s.patch_h) * (s.latent_width / s.patch_w);
    l.n_video = s.num_latent_frames * rows_per_frame;
    l.n_audio = s.num_audio_latents * s.audio_channels;
    l.seq = l.n_text + l.n_audio + l.n_video;

    const int audio_start = l.n_text;
    const int video_start = audio_start + l.n_audio;

    l.position_ids.assign(static_cast<size_t>(l.seq) * 3, 0.0F);

    // Text rows sit on the time axis at their row index.
    for (int r = 0; r < l.n_text; ++r) {
        l.position_ids[static_cast<size_t>(r) * 3 + 0] = static_cast<float>(r);
    }

    const double sqrt_area =
        std::sqrt(static_cast<double>(s.latent_height) * s.latent_width);
    const std::vector<double> hgrid =
        spatial_grid(s.latent_height, s.patch_h, sqrt_area);
    const std::vector<double> wgrid =
        spatial_grid(s.latent_width, s.patch_w, sqrt_area);

    // Audio: no height coordinate; pinned to the two extremes of the width
    // grid, channel-major (the first `num_audio_latents` rows are channel 0).
    for (int c = 0; c < s.audio_channels; ++c) {
        for (int i = 0; i < s.num_audio_latents; ++i) {
            const size_t row =
                static_cast<size_t>(audio_start + c * s.num_audio_latents + i);
            l.position_ids[row * 3 + 0] =
                static_cast<float>(static_cast<double>(s.num_text_tokens) + i);
            l.position_ids[row * 3 + 1] = 0.0F;
            l.position_ids[row * 3 + 2] = static_cast<float>(
                c == 0 ? wgrid.front() : wgrid.back());
        }
    }

    // Video: (t,h,w) spiral; t from `_temporal_position_grid`.
    const std::vector<double> tgrid = temporal_grid(
        s.num_latent_frames, static_cast<double>(s.num_text_tokens));
    int row = video_start;
    for (int f = 0; f < s.num_latent_frames; ++f) {
        for (int h = 0; h < static_cast<int>(hgrid.size()); ++h) {
            for (int w = 0; w < static_cast<int>(wgrid.size()); ++w) {
                l.position_ids[static_cast<size_t>(row) * 3 + 0] =
                    static_cast<float>(tgrid[static_cast<size_t>(f)]);
                l.position_ids[static_cast<size_t>(row) * 3 + 1] =
                    static_cast<float>(hgrid[static_cast<size_t>(h)]);
                l.position_ids[static_cast<size_t>(row) * 3 + 2] =
                    static_cast<float>(wgrid[static_cast<size_t>(w)]);
                ++row;
            }
        }
    }

    l.token_tags.assign(static_cast<size_t>(l.seq), s.video_tag);
    l.timestep_indices.assign(static_cast<size_t>(l.seq), 0);
    for (int r = 0; r < l.n_text; ++r) {
        l.token_tags[static_cast<size_t>(r)] = s.text_tag;
        l.text_indices.push_back(r);
    }
    for (int r = 0; r < l.n_audio; ++r) {
        l.token_tags[static_cast<size_t>(audio_start + r)] = s.audio_tag;
        l.audio_indices.push_back(audio_start + r);
    }
    for (int r = 0; r < l.n_video; ++r) {
        l.video_indices.push_back(video_start + r);
    }

    // pack_order[packed_row] = row in cat([text | video | audio]), because the
    // wrapper concatenates in that order and then gathers.
    std::vector<int> concat_of(static_cast<size_t>(l.seq), 0);
    for (int r = 0; r < l.n_text; ++r) {
        concat_of[static_cast<size_t>(r)] = r;
    }
    {
        int c = l.n_text;
        for (int r = 0; r < l.n_video; ++r) {
            concat_of[static_cast<size_t>(video_start + r)] = c++;
        }
        for (int r = 0; r < l.n_audio; ++r) {
            concat_of[static_cast<size_t>(audio_start + r)] = c++;
        }
    }
    l.pack_order.assign(static_cast<size_t>(l.seq), 0);
    for (int r = 0; r < l.seq; ++r) {
        l.pack_order[static_cast<size_t>(r)] = concat_of[static_cast<size_t>(r)];
    }
    return l;
}

// ---------------------------------------------------------------------------
// patchify  (latents -> the denoiser's video rows)
// ---------------------------------------------------------------------------

// `patchify_video_latents`: [B,C,F,H,W] -> [B * (F/pt)*(H/ph)*(W/pw),
// C*pt*ph*pw], frame-major then row-major, with the patches laid out as
// (c, pt, ph, pw) inside each row.
inline std::vector<float> patchify(const std::vector<float> &latents, int C,
                                   int F, int H, int W, int pt, int ph,
                                   int pw) {
    const int nf = F / pt, nh = H / ph, nw = W / pw;
    if (F % pt || H % ph || W % pw) {
        throw std::runtime_error("latent dims are not divisible by the patch");
    }
    const int feat = C * pt * ph * pw;
    std::vector<float> out(static_cast<size_t>(nf) * nh * nw * feat, 0.0F);
    size_t row = 0;
    for (int f = 0; f < nf; ++f) {
        for (int h = 0; h < nh; ++h) {
            for (int w = 0; w < nw; ++w) {
                int k = 0;
                for (int c = 0; c < C; ++c) {
                    for (int a = 0; a < pt; ++a) {
                        for (int b = 0; b < ph; ++b) {
                            for (int d = 0; d < pw; ++d) {
                                const size_t src =
                                    ((static_cast<size_t>(c) * F +
                                      f * pt + a) * H + h * ph + b) * W +
                                    w * pw + d;
                                out[row * feat + static_cast<size_t>(k++)] =
                                    latents[src];
                            }
                        }
                    }
                }
                ++row;
            }
        }
    }
    return out;
}

// The inverse, `MiniMaxH3AfterDenoiseStep`: rows are frame-major then row-major
// in `(nf, nh, nw)` with the patch's axes innermost, so
//   rows.reshape(nf, nh, nw, C, pt, ph, pw).permute(0,3,1,4,2,5,3,6)
// becomes the `[C,F,H,W]` volume the decoder takes. `nf` here already accounted
// for `pt`, exactly as the reference's `num_latent_frames // patch_t`.
inline std::vector<float> unpatchify(const std::vector<float> &rows, int C,
                                     int F, int H, int W, int pt, int ph,
                                     int pw) {
    const int nf = F / pt, nh = H / ph, nw = W / pw;
    const int feat = C * pt * ph * pw;
    if (rows.size() != static_cast<size_t>(nf) * nh * nw * feat) {
        throw std::runtime_error(
            "unpatchify: row count does not match the latent volume");
    }
    std::vector<float> out(static_cast<size_t>(C) * F * H * W, 0.0F);
    size_t row = 0;
    for (int f = 0; f < nf; ++f) {
        for (int h = 0; h < nh; ++h) {
            for (int w = 0; w < nw; ++w) {
                int k = 0;
                for (int c = 0; c < C; ++c) {
                    for (int a = 0; a < pt; ++a) {
                        for (int b = 0; b < ph; ++b) {
                            for (int d = 0; d < pw; ++d) {
                                const size_t dst =
                                    ((static_cast<size_t>(c) * F +
                                      f * pt + a) * H + h * ph + b) * W +
                                    w * pw + d;
                                out[dst] =
                                    rows[row * feat + static_cast<size_t>(k++)];
                            }
                        }
                    }
                }
                ++row;
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// a 24-value float array out of a JSON config
// ---------------------------------------------------------------------------

// `latents_mean` / `latents_std` live in the VAE's `config.json` as plain float
// arrays. Nothing else in that file is needed here, so this reads the one key
// rather than pulling in a JSON dependency.
inline std::vector<float> read_float_array(const std::string &json,
                                           const std::string &key) {
    const std::string pat = "\"" + key + "\"";
    size_t p = json.find(pat);
    if (p == std::string::npos) {
        throw std::runtime_error("'" + key + "' is not in the config");
    }
    p = json.find('[', p);
    if (p == std::string::npos) {
        throw std::runtime_error("'" + key + "' has no array");
    }
    const size_t q = json.find(']', p);
    if (q == std::string::npos) {
        throw std::runtime_error("'" + key + "' has an unterminated array");
    }
    std::vector<float> out;
    std::string num;
    auto flush = [&] {
        if (!num.empty()) {
            out.push_back(std::stof(num));
            num.clear();
        }
    };
    for (size_t i = p + 1; i < q; ++i) {
        const char c = json[i];
        if (c == ',' || c == ' ' || c == '\n' || c == '\r' || c == '\t') {
            flush();
            continue;
        }
        num.push_back(c);
    }
    flush();
    return out;
}

}  // namespace h3
