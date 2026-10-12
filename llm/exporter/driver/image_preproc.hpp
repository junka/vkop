// Copyright 2025 @junka
#ifndef LLM_EXPORTER_IMAGE_PREPROC_HPP_
#define LLM_EXPORTER_IMAGE_PREPROC_HPP_

// C++ reimplementation of the HF Qwen2-VL / Qwen3-VL image processor's
// `pixel_values` construction (image_processing_qwen2_vl.py::_preprocess),
// do_resize=False path. Takes HWC uint8 RGB and produces the fp16
// (seq_len, row) buffer the visual.vkopbin graph expects.
//
// Pipeline (mirrors _preprocess with do_resize=False):
//   1. rescale x/255 -> [0,1]
//   2. normalize (x-mean)/std per channel (CLIP mean/std)
//   3. to channel-first CHW
//   4. temporal pad: single image -> repeat to temporal_patch_size frames
//   5. reshape (grid_t, tps, C, gh//m, m, patch, gw//m, m, patch)
//      -> transpose(0,3,6,4,7,2,1,5,8)
//      -> flatten (grid_t*gh*gw, C*tps*patch*patch)
//
// do_resize=True / smart_resize is NOT handled here — the exported visual.onnx
// bakes grid_thw at export time (224x224 -> grid_thw=[1,16,16]), so the image
// must already be the exported size. Resizing arbitrary images is a later stage.

#include <cstdint>
#include <vector>
#include <string>

#include "core/Tensor.hpp" // for ITensor::fp32_to_fp16

namespace vkop {
namespace export_ {

struct PreprocResult {
    std::vector<uint16_t> pixel_values_fp16; // [seq_len * row] row-major
    int grid_t = 0, grid_h = 0, grid_w = 0;  // grid_thw
    int seq_len = 0, row = 0;
};

// Visual encoder constants (Qwen3-VL-2B: patch_size=16, NOT the Qwen2VL
// default of 14 — Qwen3-VL config overrides it). Verified against
// Qwen2VLImageProcessorFast with the Qwen3-VL processor config.
inline constexpr int kPatch = 16;
inline constexpr int kTemporalPatch = 2;
inline constexpr int kMerge = 2;
inline constexpr int kInChans = 3;
inline constexpr int kRow = kInChans * kTemporalPatch * kPatch * kPatch; // 1536

// Qwen3-VL uses mean=std=0.5 (i.e. (x/255 - 0.5)/0.5 = x/127.5 - 1), NOT the
// CLIP mean/std. Verified via Qwen2VLImageProcessorFast.image_mean/std.
inline constexpr float kClipMean[3] = {0.5f, 0.5f, 0.5f};
inline constexpr float kClipStd[3] = {0.5f, 0.5f, 0.5f};

// Build pixel_values from HWC uint8 RGB. H,W must each be divisible by
// patch*merge (e.g. 224 -> 224/(14*2)=8). Returns empty on bad size.
inline PreprocResult preprocess_image_noresize(const uint8_t *hwc, int H, int W,
                                               int C) {
    PreprocResult r;
    if (C != kInChans || H % (kPatch * kMerge) != 0 ||
        W % (kPatch * kMerge) != 0) {
        return r; // bad size
    }
    const int grid_t = 1;
    const int grid_h = H / kPatch;
    const int grid_w = W / kPatch;
    const int tps = kTemporalPatch;
    const int m = kMerge;
    const int patch = kPatch;
    const int channel = kInChans;

    // 1+2+3: rescale + normalize + to CHW float. Single image = 1 frame; we
    // build a [tps][C][H][W] float buffer directly (frame 1 = frame 0 copy,
    // i.e. the temporal-repeat pad for a single image).
    std::vector<float> chw(tps * channel * H * W, 0.f);
    auto put = [&](int t, int c, int y, int x, float v) {
        chw[((t * channel + c) * H + y) * W + x] = v;
    };
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            for (int c = 0; c < channel; ++c) {
                uint8_t p = hwc[(y * W + x) * C + c];
                float v = (static_cast<float>(p) / 255.f - kClipMean[c]) /
                          kClipStd[c];
                put(0, c, y, x, v); // frame 0
            }
        }
    }
    // temporal repeat: frame 1 = frame 0
    for (int i = 0; i < channel * H * W; ++i) {
        chw[channel * H * W + i] = chw[i];
    }

    // 4+5: HF reshape (grid_t, tps, C, gh//m, m, patch, gw//m, m, patch) then
    // transpose(0,3,6,4,7,2,1,5,8) then flatten (grid_t*gh*gw, row).
    //
    // Instead of materializing the 9-D array, we compute, for each output
    // (seq, row_idx), the source (t, h, w, m_h, m_w, c, tp, ph, pw) and read
    // the corresponding CHW element. The transpose remaps:
    //   out_dim[0]=t, [1]=h//m, [2]=w//m, [3]=m_h, [4]=m_w,
    //   [5]=c, [6]=tp, [7]=ph, [8]=pw
    // and the pre-transpose dims are:
    //   [grid_t, tps, C, gh//m, m, patch, gw//m, m, patch]
    // so out_index -> pre_index via the inverse permutation of (0,3,6,4,7,2,1,5,8).
    //
    // Pre-permute shape: D = [grid_t, tps, C, gh//m, m, patch, gw//m, m, patch]
    const int D[9] = {grid_t, tps, channel, grid_h / m, m, patch,
                      grid_w / m, m, patch};
    // perm[i] = which pre-dim goes to output position i.
    // transpose(0,3,6,4,7,2,1,5,8) means output[i] = pre[perm[i]].
    const int perm[9] = {0, 3, 6, 4, 7, 2, 1, 5, 8};
    // pre-dim sizes for each output position
    int pre_size[9];
    for (int i = 0; i < 9; ++i)
        pre_size[i] = D[perm[i]];

    const int seq_len = grid_t * grid_h * grid_w;
    const int row = kRow;
    r.pixel_values_fp16.resize(seq_len * row);
    r.grid_t = grid_t;
    r.grid_h = grid_h;
    r.grid_w = grid_w;
    r.seq_len = seq_len;
    r.row = row;

    // Iterate over the 9-D output coordinate (c0..c8) in row-major order over
    // pre_size[]. That linear sweep IS the flattened transposed tensor.
    // For each, recover the pre-coordinate and index into CHW.
    int coord[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int out_lin = 0; out_lin < seq_len * row; ++out_lin) {
        // pre_coord[perm[i]] = coord[i]
        int pre_coord[9];
        for (int i = 0; i < 9; ++i)
            pre_coord[perm[i]] = coord[i];
        int t = pre_coord[0];
        int tp = pre_coord[1];
        int c = pre_coord[2];
        int gh_m = pre_coord[3]; // h//m
        int m_h = pre_coord[4];
        int ph = pre_coord[5];
        int gw_m = pre_coord[6]; // w//m
        int m_w = pre_coord[7];
        int pw = pre_coord[8];
        int y = gh_m * m * patch + m_h * patch + ph;
        int x = gw_m * m * patch + m_w * patch + pw;
        // CHW buffer is [tps][C][H][W]; t is the grid_t dim (always 0 for a
        // single image), tp is the within-patch temporal frame. For a single
        // image both frames are identical (temporal repeat), so reading frame
        // tp is correct and equals reading frame t.
        (void)t;
        float v = chw[((tp * channel + c) * H + y) * W + x];
        r.pixel_values_fp16[out_lin] = core::ITensor::fp32_to_fp16(v);

        // increment coord (row-major over pre_size[])
        for (int k = 8; k >= 0; --k) {
            if (++coord[k] < pre_size[k])
                break;
            coord[k] = 0;
        }
    }
    return r;
}

// ---------------------------------------------------------------------------
// get_rope_index: C++ port of HF Qwen3-VL RoPE position-id construction.
//
// Builds the 3D MRoPE position_ids (temporal, height, width) for a prefill
// sequence that mixes text and image runs. Mirrors
// transformers.models.qwen3_vl.modeling_qwen3_vl.get_rope_index +
// get_vision_position_ids exactly (spatial_merge_size=2, temp_merge_size=1,
// time_interval=1 — the only config Qwen3-VL-2B uses).
//
// Inputs (batch B=1 supported; multi-batch is a loop):
//   input_ids : (B, L) int64
//   mtt       : (B, L) int32  mm_token_type_ids — 0=text, 1=image, 2=video
//   am        : (B, L) bool   attention_mask (1=keep, 0=pad). If null, all 1.
//   grid_thw  : flat (n_img*3,) int — [T,H,W] per image, in encounter order.
// Returns:
//   pos_ids   : (3, B, L) int64, row-major [axis][batch][seq]
//   rope_delta: (B,) int64 — llm_positions.max()+1 - L (per batch). Used for
//               decode rounds: position_ids = past_len + rope_delta.
//
// Algorithm (per batch):
//   - Group mtt (after masking) into contiguous runs (modality, start, end).
//   - Text run  : pos = arange(text_len) + current_pos, replicated on all 3
//                 axes; current_pos += text_len.
//   - Image run : consume next grid_thw. llm_grid_t=T, llm_grid_h=H//2,
//                 llm_grid_w=W//2, seq=thw. width  = arange(cp, cp+w) repeated
//                 h*t times; height = arange(cp, cp+h) repeat_interleave w*t;
//                 temporal = cp (constant). current_pos += max(H,W)//2.
//   - rope_delta = max(pos)+1 - len(masked_seq).
// Padding positions (am==0) are written as 0 in pos_ids (HF leaves the
// pre-zeroed tensor unassigned there).
struct RopeIndexResult {
    std::vector<int64_t> pos_ids;       // (3, B, L)
    std::vector<int64_t> rope_delta;    // (B,)
};

inline RopeIndexResult get_rope_index(
    const int64_t *input_ids, const int32_t *mtt, const int8_t *am,
    const int *grid_thw_flat, int n_img, int B, int L) {
    RopeIndexResult res;
    res.pos_ids.assign(3 * B * L, 0);
    res.rope_delta.assign(B, 0);
    const int spatial_merge_size = kMerge; // 2
    const int temp_merge_size = 1;
    const int time_interval = 1;
    (void)input_ids; // not needed for pos computation (only mtt + grid_thw)

    int grid_consumed = 0;
    for (int b = 0; b < B; ++b) {
        const int32_t *mtt_b = mtt + b * L;
        const int8_t *am_b = am ? am + b * L : nullptr;

        // Build the masked run list over mtt. First gather the kept indices,
        // then group consecutive equal modality.
        std::vector<int> kept;       // indices into the original L
        std::vector<int32_t> kept_type;
        kept.reserve(L);
        kept_type.reserve(L);
        for (int i = 0; i < L; ++i) {
            if (am_b == nullptr || am_b[i] != 0) {
                kept.push_back(i);
                kept_type.push_back(mtt_b[i]);
            }
        }
        int masked_len = static_cast<int>(kept.size());

        // Run-length encode kept_type into (modality, start, end) over the
        // kept sequence.
        struct Run { int32_t mod; int start; int end; };
        std::vector<Run> runs;
        if (!kept_type.empty()) {
            int32_t cur = kept_type[0];
            int start = 0;
            for (int i = 1; i <= static_cast<int>(kept_type.size()); ++i) {
                if (i == static_cast<int>(kept_type.size()) ||
                    kept_type[i] != cur) {
                    runs.push_back({cur, start, i});
                    if (i < static_cast<int>(kept_type.size())) {
                        cur = kept_type[i];
                        start = i;
                    }
                }
            }
        }

        int current_pos = 0;
        int64_t llm_max = -1;
        // For each run, append its 3-axis pos values into a per-batch list,
        // then scatter into res.pos_ids at the kept positions.
        for (const auto &run : runs) {
            int run_len = run.end - run.start;
            if (run.mod == 0) {
                // text: arange(run_len) + current_pos on all 3 axes
                for (int t = 0; t < run_len; ++t) {
                    int64_t p = current_pos + t;
                    for (int axis = 0; axis < 3; ++axis) {
                        int seq_idx = run.start + t;
                        int orig = kept[seq_idx];
                        res.pos_ids[(axis * B + b) * L + orig] = p;
                    }
                    llm_max = std::max(llm_max, (int64_t)(current_pos + t));
                }
                current_pos += run_len;
            } else {
                // image (1) or video (2): consume next grid_thw
                if (grid_consumed >= n_img) break; // malformed; stop
                int T = grid_thw_flat[grid_consumed * 3 + 0];
                int H = grid_thw_flat[grid_consumed * 3 + 1];
                int W = grid_thw_flat[grid_consumed * 3 + 2];
                grid_consumed++;
                int lt = T / temp_merge_size;
                int lh = H / spatial_merge_size;
                int lw = W / spatial_merge_size;
                int seq = lt * lh * lw;
                // Build width/height/temporal exactly as get_vision_position_ids:
                //   width  = arange(cp, cp+lw) repeated lh*lt times
                //   height = arange(cp, cp+lh) repeat_interleave lw*lt
                //   temporal = cp (constant), * time_interval
                for (int s = 0; s < seq; ++s) {
                    int t = s / (lh * lw);        // 0..lt-1
                    int rem = s % (lh * lw);
                    int h = rem / lw;             // 0..lh-1
                    int w = rem % lw;             // 0..lw-1
                    (void)t;
                    int64_t p_t = (int64_t)current_pos * time_interval;
                    int64_t p_h = current_pos + h;
                    int64_t p_w = current_pos + w;
                    int seq_idx = run.start + s;
                    if (seq_idx >= run.end) break; // grid larger than run (safe)
                    int orig = kept[seq_idx];
                    res.pos_ids[(0 * B + b) * L + orig] = p_t;
                    res.pos_ids[(1 * B + b) * L + orig] = p_h;
                    res.pos_ids[(2 * B + b) * L + orig] = p_w;
                    llm_max = std::max(llm_max, std::max(p_t, std::max(p_h, p_w)));
                }
                current_pos += std::max(H, W) / spatial_merge_size;
            }
        }
        res.rope_delta[b] = (llm_max + 1) - masked_len;
    }
    return res;
}

} // namespace export_
} // namespace vkop

#endif // LLM_EXPORTER_IMAGE_PREPROC_HPP_
