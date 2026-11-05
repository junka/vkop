// Copyright 2025 @junka

#include "core/KernelSelect.hpp"

#include <algorithm>
#include <cstdlib>

#include "vulkan/VulkanDevice.hpp"

namespace vkop {
namespace core {
namespace {

// ---------------------------------------------------------------------------
// Device classes
// ---------------------------------------------------------------------------

bool is_moltenvk(const DeviceTraits &t) {
    return t.driver_id == static_cast<uint32_t>(VK_DRIVER_ID_MOLTENVK);
}

bool match_mvk_amd(const DeviceTraits &t) {
    return is_moltenvk(t) && t.vendor_id == 0x1002; // AMD
}

bool match_mvk_intel(const DeviceTraits &t) {
    return is_moltenvk(t) && t.vendor_id == 0x8086; // Intel
}

// A VRAM-backed dGPU on a native driver. Kept behavioural (not vendor-keyed)
// because the GTX 1070 profile below is a memory-system profile: any native
// discrete part without coopmat follows the same GEMV-first decode curve.
bool match_native_discrete(const DeviceTraits &t) {
    return !is_moltenvk(t) && !t.unified_memory;
}

struct ClassRule {
    const char *name;
    bool (*match)(const DeviceTraits &);
    DeviceClass cls;
};

// Ordered: first match wins. The generic class is the fallthrough.
const ClassRule kClassRules[] = {
    {"moltenvk-amd", match_mvk_amd, DeviceClass::kMoltenVkAmd},
    {"moltenvk-intel", match_mvk_intel, DeviceClass::kMoltenVkIntel},
    {"native-discrete", match_native_discrete, DeviceClass::kNativeDiscrete},
};

DeviceProfile make_profile(DeviceClass cls, const char *name, int byte_min_m,
                           int fp16_min_m, int wide_n, int byte_wide_min_m,
                           int fp16_wide_min_m, int ks_target_blocks) {
    DeviceProfile p;
    p.device_class = cls;
    p.name = name;
    p.byte_min_m = byte_min_m;
    p.fp16_min_m = fp16_min_m;
    p.wide_n = wide_n;
    p.byte_wide_min_m = byte_wide_min_m;
    p.fp16_wide_min_m = fp16_wide_min_m;
    p.ks_target_blocks = ks_target_blocks;
    // ksplit_floor / wide thresholds / cache_resident_bytes keep the shipped
    // defaults from the header. A device that measures differently overrides
    // them here (one more field, one more measurement).
    return p;
}

// One row per DeviceClass, indexed by the enum. Values come from the
// calibration measurements that used to be written three times as if-blocks in
// VulkanDevice::getProperties() and again as comments in ops/Matmul.hpp.
const DeviceProfile kProfiles[static_cast<int>(DeviceClass::kCount)] = {
    // kGeneric: native iGPU / unknown. ARL 8x8x16 and RTX 4000 Ada defaults.
    make_profile(DeviceClass::kGeneric, "generic", 2, 3, 0, 2, 3, 128),
    // kMoltenVkAmd (RX 560X behind MVK): the byte GEMV wins the whole
    // batch*m <= 8 window (m=8 ties the tile, m=8/N=151936 wins 1.8x), so the
    // tile starts at 9. fp16 pays 2x the row bytes and crosses at m=4 (N=6144)
    // / m=6 (N=151936); 16 CUs saturate the tile at ~32 blocks.
    make_profile(DeviceClass::kMoltenVkAmd, "moltenvk-amd", 9, 4, 65536, 9, 6,
                 32),
    // kMoltenVkIntel (UHD 630 behind MVK): both crossovers land past the
    // split-K window, so the GEMV keeps every row its gate admits. 24 EUs
    // saturate near 32 blocks.
    make_profile(DeviceClass::kMoltenVkIntel, "moltenvk-intel", 9, 9, 0, 2, 3,
                 32),
    // kNativeDiscrete (GTX 1070: 256-bit GDDR5, ~224 GB/s measured read
    // ceiling, no coopmat). The byte GEMV wins the whole split-K window and
    // the tile takes over at 9. fp16 crosses at m=9 for narrow N and at m=7
    // for the wide band (N >= 6144), where the per-row weight re-read is 2x
    // the bytes. ks_target stays at the ARL/Ada 128: this part is bandwidth
    // bound, not block bound, and no measurement has shown a lower saturation
    // point.
    make_profile(DeviceClass::kNativeDiscrete, "native-discrete", 9, 9, 6144, 9,
                 7, 128),
};

// ---------------------------------------------------------------------------
// MatMul rules
// ---------------------------------------------------------------------------

inline int up_div(int a, int b) { return (a + b - 1) / b; }

struct Ctx {
    const DeviceTraits *dev;
    const DeviceProfile *prof;
    const MatmulShape *s;
    const MatmulFeatures *f;
    const MatmulAvailability *av;
    const MatmulKnobs *kn;

    bool weight_byte = false;
    bool weight_4bit = false;
    bool fused = false;
    bool tiled = false;
    bool ksplit_ok = false;
    bool w8_quad = false;
    bool w4_quad = false;
    bool l2_resident = false;
    bool wide_disabled = false;
    bool wide_forced = false;
    bool ks512 = false;
    int tiled_min_m = 0;
    int ks_n = 0;
    // Profile value unless a knob overrides it; resolved once in make_ctx so
    // the predicates never reach back into the knobs.
    int wide_thr_fp16 = 0;
    int wide_thr_byte = 0;
    // Configured by the winning row:
    int q = 16;
    bool wide = false;
};

int ks_target_blocks(const Ctx &c) {
    if (c.kn->ks_target_pin > 0) {
        return c.kn->ks_target_pin;
    }
    return c.prof->ks_target_blocks;
}

// MatMul::ks_target_n(): the tile-split-K depth that lifts `blocks` workgroups
// to the device's saturation target, rounded down to a power of two (the shader
// keeps slice widths a whole number of TILE_K where it can).
int ks_depth_for(const Ctx &c, long blocks) {
    if (c.kn->ks_n_pin > 0) {
        return c.kn->ks_n_pin;
    }
    const int target = ks_target_blocks(c);
    if (blocks >= target) {
        return 1;
    }
    long want = (target + blocks - 1) / blocks;
    long p = 1;
    while (p * 2 <= want) {
        p *= 2;
    }
    return p < 2 ? 2 : static_cast<int>(p);
}

Ctx make_ctx(const DeviceTraits &dev, const DeviceProfile &prof,
             const MatmulShape &s, const MatmulFeatures &f,
             const MatmulAvailability &av, const MatmulKnobs &kn) {
    Ctx c{};
    c.dev = &dev;
    c.prof = &prof;
    c.s = &s;
    c.f = &f;
    c.av = &av;
    c.kn = &kn;

    c.weight_byte = f.weight == WeightKind::kInt8 || f.weight == WeightKind::kFp8;
    c.weight_4bit = f.weight == WeightKind::kInt4 || f.weight == WeightKind::kNvfp4;
    c.fused = f.fp16 && ((s.n & 1) == 0);

    const long long b_row_bytes =
        static_cast<long long>(s.n) * (c.weight_byte ? 1 : 2);
    const bool b_stride_pow2_big =
        b_row_bytes >= 65536 && (b_row_bytes & (b_row_bytes - 1)) == 0;
    const bool wide_band = prof.wide_n > 0 && s.n >= prof.wide_n;
    const int dtype_min_m =
        c.weight_byte
            ? (wide_band ? prof.byte_wide_min_m : prof.byte_min_m)
            : (wide_band ? prof.fp16_wide_min_m : prof.fp16_min_m);
    c.l2_resident =
        static_cast<long long>(s.k) * b_row_bytes <=
        static_cast<long long>(prof.cache_resident_bytes);

    const bool coop16 = av.coop_bm >= 32;
    c.tiled_min_m = b_stride_pow2_big
                        ? 12
                        : (c.weight_byte
                               ? dtype_min_m
                               : ((coop16 && c.l2_resident) ? 12 : dtype_min_m));

    const bool tiled_ok_n = (s.n % 64 == 0) || (s.n % 8 == 0);
    c.tiled = c.fused && !kn.tile_disabled && !c.weight_4bit &&
              (s.k % 16 == 0) && (s.m >= c.tiled_min_m) && !f.epi_bias &&
              tiled_ok_n && (!c.weight_byte || (!f.transB && (s.n % 4 == 0)));

    if (c.tiled && !c.weight_byte && !f.transB && s.k >= 256) {
        const long blocks = static_cast<long>(s.batch) * up_div(s.n, 64) *
                            up_div(s.m, 64);
        const int want = ks_depth_for(c, blocks);
        if (want > 1) {
            c.ks_n = std::min<int>(want, s.k / 16);
        }
    }

    const bool ksplit = c.fused && !f.transB && (s.n % 4 == 0) &&
                        ((s.k % 2) == 0) && (s.batch * s.m <= 8) && (s.k >= 256);
    c.ksplit_ok = ksplit && !c.tiled;

    c.w8_quad = c.fused && c.weight_byte && !f.transB && (s.n % 4 == 0);
    c.w4_quad = c.fused && c.weight_4bit;

    c.wide_disabled = kn.wide_force < 0;
    c.wide_forced = kn.wide_force > 0;
    c.wide_thr_fp16 = kn.wide_threshold_fp16 > 0 ? kn.wide_threshold_fp16
                                                 : prof.wide_threshold_fp16;
    c.wide_thr_byte = kn.wide_threshold_byte > 0 ? kn.wide_threshold_byte
                                                 : prof.wide_threshold_byte;
    c.ks512 = kn.ksplit512_enabled && !c.weight_4bit;
    return c;
}

using Pred = bool (*)(const Ctx &);
using Configure = void (*)(Ctx &);

struct MatmulRule {
    MatmulRoute route;
    const char *tag;
    Pred applies;
    Configure configure;
};

bool attn_softmax_applies(const Ctx &c) {
    const MatmulShape &s = *c.s;
    const MatmulFeatures &f = *c.f;
    return f.attn_softmax && f.fp16 && ((s.n & 1) == 0) &&
           f.weight == WeightKind::kFp16 && s.n > 0 && s.n <= 16384 &&
           s.k <= 1024;
}

bool coopmat_i8_applies(const Ctx &c) {
    const MatmulFeatures &f = *c.f;
    return c.kn->coop_i8_enabled && f.fp16 && f.weight == WeightKind::kInt8 &&
           !f.transB && (c.s->n % 8 == 0) && c.av->coopmat_i8;
}

bool coopmat_gate(const Ctx &c) {
    // Opt-in bypass for PoC — forces coopmat regardless of shape thresholds.
    if (c.kn->force_coop) return true;
    const MatmulShape &s = *c.s;
    if (c.av->coop_bm < 32) {
        // 8x8x16: a 16x32 workgroup tile. Below one full TILE_M of rows the
        // shared-memory tile pads most of its block, so coop keeps that band;
        // past it the tile's 64x64 traffic ratio wins.
        return s.batch * s.m > 8 && (s.m < 64 || !c.tiled);
    }
    // 16x16x16: a 32-row workgroup M, so m=2..7 pads for free; it wins from m=2
    // on DRAM-bound weights and only needs the block count to clear one wave
    // on cache-resident ones.
    if (s.batch * s.m > 8 || (s.m >= 2 && !c.l2_resident)) {
        return true;
    }
    if (c.av->coop_bm <= 0 || c.av->coop_bn <= 0) {
        return false;
    }
    return s.m >= 8 &&
           up_div(s.n, c.av->coop_bn) * up_div(s.m, c.av->coop_bm) * s.batch >= 96;
}

bool coopmat_applies(const Ctx &c) {
    const MatmulFeatures &f = *c.f;
    return f.fp16 && f.weight == WeightKind::kFp16 && !c.kn->coop_disabled &&
           c.av->coopmat && !f.epi_bias && coopmat_gate(c);
}

// coopmat + split-K hybrid: only when explicitly opted in via VKOP_COOP_SPLITK
// or VKOP_COOP_SPLITK_FORCE. Reuses the same coopmat eligibility as kCoopmat
// (fp16, no epi_bias, gate passes) but additionally requires K deep enough to
// split across (K >= 32, so each of 2+ slices still gets >= 1 MMA tile).
// Returns true only if EITHER forced depth >=2 AND valid for K, OR auto-split
// computes depth >=2.
bool coopmat_splitk_applies(const Ctx &c) {
    const MatmulFeatures &f = *c.f;
    const MatmulShape &s = *c.s;
    // coop_split 语义: 0 = off, -1 = auto (env opt-in), >= 2 = forced depth.
    if (c.kn->coop_split == 0) return false;
    if (c.kn->coop_split == 1) return false;
    if (!f.fp16 || f.weight != WeightKind::kFp16) return false;
    if (c.kn->coop_disabled || !c.av->coopmat) return false;
    if (f.epi_bias) return false;
    if (s.k < 32) return false;
    if (!coopmat_gate(c)) return false;

    // Forced depth: must be valid for K.
    if (c.kn->coop_split >= 2) {
        int n = c.kn->coop_split;
        // Each slice needs >= BK (=16) to do at least one MMA.
        return n >= 2 && n <= s.k / 16;
    }

    // Auto: compute depth the same way coop_splitk_configure would, and reject
    // if it came out <2 (grid already saturated on plain coopmat).
    const int bm = c.av->coop_bm > 0 ? c.av->coop_bm : 16;
    const int bn = c.av->coop_bn > 0 ? c.av->coop_bn : 32;
    long coop_blocks = static_cast<long>(s.batch) * up_div(s.n, bn) *
                       up_div(s.m, bm);
    int want = ks_depth_for(c, coop_blocks);
    int max_by_k = s.k / 16;
    int depth = std::min<int>(want, max_by_k);
    return depth >= 2;
}

// Configure coopmat_splitk: resolve the split-K depth. Same occupancy logic
// as ks_depth_for (tiled_ks) but gated by coop_split knob.
void coop_splitk_configure(Ctx &c) {
    const MatmulShape &s = *c.s;
    // Forced from env.
    if (c.kn->coop_split >= 2) {
        c.ks_n = c.kn->coop_split;
        return;
    }
    // Auto: same heuristic as ks_depth_for, but the coopmat grid counts the
    // coop workgroups (one tile per BMxBN output block) rather than tiled's
    // 64x64 tiles.
    const int bm = c.av->coop_bm > 0 ? c.av->coop_bm : 16;
    const int bn = c.av->coop_bn > 0 ? c.av->coop_bn : 32;
    long coop_blocks = static_cast<long>(s.batch) * up_div(s.n, bn) *
                       up_div(s.m, bm);
    const int want = ks_depth_for(c, coop_blocks);
    // But also constrained by K / BK (each slice needs >= BK). BK=16 for all
    // listed coop tiles (8x8x16 CK=16, 16x16x16 CK=16).
    const int max_by_k = s.k / 16;
    c.ks_n = std::min<int>(want, max_by_k);
    if (c.ks_n < 2) c.ks_n = 0; // 退化为不分
}

// coopmat + split-K in-workgroup (opt-in via VKOP_COOP_INWG / VKOP_COOP_INWG_K).
// Single dispatch, no global scratch — coop_split serial passes of the K loop
// inside one workgroup, cross-pass fp32 accumulate in shared memory. Shared
// coop_split knob with splitk (same env-driven cs value), but applies is
// stricter: ACC_M==ACC_N==1 only (Ada 2x2/2x1 variants have no INWG build,
// and the cross-pass accumulator assumes a single fp32 tile per subgroup
// matching CM=CN), N even (scatter_tile_half2 guard), and K/cs >= 2*K_BLOCK
// (each slice must have >= 2 MMA tiles or serial-pass overhead swallows gain).
bool coopmat_inwg_applies(const Ctx &c) {
    // DISABLED while shader + host dispatch branch land together. Re-enable
    // after (a) Ada & ARL both ship -DSPLITK_INWG SPIR-V, (b) Matmul.hpp
    // has the kCoopmatInwg dispatch branch: single dispatch,
    // grid = (⌈N/BN⌉, ⌈M/BM⌉, batch), uPC.ksplit = cs, half2 epilogue
    // directly to uOutput, no scratch / barrier / pack.
    (void)c;
    return false;
}
#if 0
{
    // Full applies body — re-enable below by removing #if 0 once dispatch lands
    ...(existing full logic)...
}
#endif

void coop_inwg_configure(Ctx &c) {
    const MatmulShape &s = *c.s;
    if (c.kn->coop_split >= 2) {
        // Forced cs, clamp to [2, 8] and verify K/cs >= 2*K_BLOCK (= 32).
        int n = std::min<int>(c.kn->coop_split, 8);
        n = std::max<int>(n, 2);
        if (n > s.k / 32) n = std::max<int>(s.k / 32, 0);
        c.ks_n = n;
    } else {
        // Auto: reuse splitk occupancy heuristic, clamp [2, 8], verify K.
        const int bm = c.av->coop_bm > 0 ? c.av->coop_bm : 16;
        const int bn = c.av->coop_bn > 0 ? c.av->coop_bn : 32;
        long coop_blocks = static_cast<long>(s.batch) * up_div(s.n, bn) *
                           up_div(s.m, bm);
        int want = std::min<int>(ks_depth_for(c, coop_blocks), 8);
        want = std::max<int>(want, 2);
        int max_by_k = s.k / 32; // K / (2 * K_BLOCK) — each slice needs >= 2 tiles
        c.ks_n = std::min<int>(want, max_by_k);
    }
}

// coopmat + split-K "last block reduce" — true parallel K split via
// grid.z = batch * SK, each slice writes fp32 partials to a global buffer,
// one designated tail block in the tile reduces and writes half2 directly.
//
// Host-side guards mirror INWG but for grid parallelism:
//   * SK ∈ [2, 8] capped (8 subgroups is arbitrary; shader clamp matches)
//   * K/SK >= 2*K_BLOCK (= 32) so every slice does >= 2 MMA tiles
//   * N even (scatter_tile_half2 requires it)
//   * ACC_M = ACC_N = 1 only (larger geometries have no SK_LAST build)
//   * fp16, no epi_bias, coopmat_gate passes — same as INWG / splitK
//   * coop_split knob must be valid — user opted in via VKOP_COOP_SK or
//     VKOP_COOP_SK_LAST (sets coop_split = forced value or -1 for auto).
// SPIR-V key availability is checked in dispatch (Matmul.hpp), not here:
// the selector must produce a plan even on devices without the key, so the
// caller can fall through to splitK / plain coopmat without re-running.
bool sk_last_applies(const Ctx &c) {
    const MatmulFeatures &f = *c.f;
    const MatmulShape &s = *c.s;
    if (c.kn->coop_split == 0) return false;
    if (c.kn->coop_split == 1) return false;
    if (!f.fp16 || f.weight != WeightKind::kFp16) return false;
    if (c.kn->coop_disabled || !c.av->coopmat) return false;
    if (f.epi_bias) return false;
    if (!coopmat_gate(c)) return false;
    // N even — scatter_tile_half2 requires it for the HALF2_EPI variant;
    // the fp32 fallback also writes flat fp32 to binding 3 regardless.
    // Keep N-even as guard because half2 direct is the primary performance
    // path for this route (otherwise splitK DEPOSIT/FOLD is always faster).
    if ((s.n & 1) != 0) return false;
    // K deep enough to split into SK pieces each with >= 2 MMA tiles.
    if (c.kn->coop_split >= 2) {
        int n = std::min<int>(c.kn->coop_split, 8);
        if (n > s.k / 32) return false;
        return true;
    }
    // Auto: compute SK the same way splitk does, clamp [2, 8], verify K.
    const int bm = c.av->coop_bm > 0 ? c.av->coop_bm : 16;
    const int bn = c.av->coop_bn > 0 ? c.av->coop_bn : 32;
    long coop_blocks = static_cast<long>(s.batch) * up_div(s.n, bn) *
                       up_div(s.m, bm);
    int depth = std::min<int>(ks_depth_for(c, coop_blocks), 8);
    depth = std::max<int>(depth, 2);
    if (depth > s.k / 32) return false;
    return true;
}

void sk_last_configure(Ctx &c) {
    const MatmulShape &s = *c.s;
    if (c.kn->coop_split == 1) {
        c.ks_n = 1; // user explicitly opted for no split — sklast dispatched
                    // with cs=1, useful for comparing MMA against DEFAULT
        return;
    }
    if (c.kn->coop_split >= 2) {
        int n = std::min<int>(c.kn->coop_split, 8);
        n = std::max<int>(n, 2);
        if (n > s.k / 32) n = std::max<int>(s.k / 32, 0);
        c.ks_n = n;
    } else {
        const int bm = c.av->coop_bm > 0 ? c.av->coop_bm : 16;
        const int bn = c.av->coop_bn > 0 ? c.av->coop_bn : 32;
        long coop_blocks = static_cast<long>(s.batch) * up_div(s.n, bn) *
                           up_div(s.m, bm);
        int want = std::min<int>(ks_depth_for(c, coop_blocks), 8);
        want = std::max<int>(want, 2);
        int max_by_k = s.k / 32;
        c.ks_n = std::min<int>(want, max_by_k);
    }
}

bool down_act_applies(const Ctx &c) {
    const MatmulShape &s = *c.s;
    const MatmulFeatures &f = *c.f;
    return f.down_act && c.av->down_act && f.fp16 && !f.transB && s.n_up >= 0 &&
           (s.n % 4 == 0) && s.n <= 2048 && (s.n_up % 4 == 0) && s.k >= 1;
}

bool tiled_ks_applies(const Ctx &c) {
    return c.tiled && c.ks_n > 1 && c.av->tile_ks_split &&
           c.av->tile_ks_fold;
}

bool tiled_applies(const Ctx &c) { return c.tiled; }

bool w8_wide_ok(const Ctx &c) {
    return c.ksplit_ok && !c.wide_disabled && c.weight_byte &&
           (c.s->n % 16 == 0) &&
           (c.wide_forced || c.s->n >= c.wide_thr_byte);
}

bool ksplit_w8_wide512_applies(const Ctx &c) {
    return w8_wide_ok(c) && c.ks512 && c.av->w8_wide && c.av->w8_wide512;
}

bool ksplit_w8_wide_applies(const Ctx &c) {
    return w8_wide_ok(c) && c.av->w8_wide;
}

bool fp16_wide_ok(const Ctx &c) {
    return c.ksplit_ok && !c.wide_disabled && !c.weight_byte &&
           (c.s->n % 8 == 0) &&
           (c.wide_forced || c.s->n >= c.wide_thr_fp16);
}

bool ksplit_wide512_applies(const Ctx &c) {
    return fp16_wide_ok(c) && c.ks512 && c.av->ksplit512;
}

bool ksplit_wide_applies(const Ctx &c) { return fp16_wide_ok(c); }

bool ksplit_512_applies(const Ctx &c) {
    return c.ksplit_ok && c.ks512 && c.av->ksplit512;
}

bool ksplit_applies(const Ctx &c) { return c.ksplit_ok; }

bool quad4_applies(const Ctx &c) {
    return c.fused && (c.w8_quad || c.w4_quad);
}

bool fused_applies(const Ctx &c) { return c.fused; }

bool reduce_pack_applies(const Ctx &c) { return c.f->fp16; }

bool reduce_applies(const Ctx &) { return true; }

// q search: the block floor is the only reason to split K, so take the
// shallowest split that clears it, starting from the route's natural depth.
void q_plain(Ctx &c) {
    int q = c.kn->ksplit_q > 0 ? c.kn->ksplit_q : 16;
    while (q > 4 && up_div(c.s->n / 4, q) < 32) {
        q /= 2;
    }
    c.q = q;
    c.wide = false;
}

void q_fp16_wide(Ctx &c) {
    int q = c.kn->ksplit_q;
    if (q <= 0) {
        // Shallow K is the one exception: K < 1024 regressed 23% at q=256.
        q = (c.s->k >= 1024) ? 256 : 16;
    }
    while (q > 4 && up_div(c.s->n / 8, q) < 32) {
        q /= 2;
    }
    c.q = q;
    c.wide = true;
}

void q_byte_wide(Ctx &c) {
    int q = c.kn->ksplit_q;
    if (q <= 0) {
        // 16 columns per lane, so the byte WIDE block count is UP_DIV(n/16, q)
        // and the profile's floor (8) is the only constraint.
        q = 256;
        while (q > 4 && up_div(c.s->n / 16, q) < c.kn->ksplit_floor) {
            q /= 2;
        }
    }
    c.q = q;
    c.wide = true;
}

// Order matters and is the entire policy. The comments name the constraint that
// keeps an earlier row out of a later row's territory.
const MatmulRule kMatmulRules[] = {
    // 1. The fused QK^T+Softmax epilogue owns the dispatch; it pre-empts
    //    coopmat (which cannot reduce a row) and the split-K GEMV.
    {MatmulRoute::kAttnSoftmax, "attn_softmax", attn_softmax_applies, nullptr},
    // 2. W8A8 SINT8 MMA, opt-in: the MMA saves no weight bandwidth and loses
    //    to the weight-only GEMV at decode.
    {MatmulRoute::kCoopmatI8, "coopmat_i8", coopmat_i8_applies, nullptr},
    // 3. SwiGLU down fold: a different module and a different shape mapping,
    //    so it must pre-empt coopmat. A folded node's MatMul carries Wg/Wu as
    //    bindings 7/8 and its result is silu(x.Wg)*(x.Wu) . Wd -- the coopmat
    //    descriptor layout has neither the bindings nor the epilogue, so
    //    letting it win here computes a plain down GEMM at a folded node and
    //    returns silently wrong logits. That only bites once the shape is
    //    large enough for coopmat to apply (M >= the tile minimum) -- at
    //    M <= 8 the table fell through to here by accident and was right.
    {MatmulRoute::kDownActFold, "down_act_gemv", down_act_applies, nullptr},
    // 4. fp16 cooperative-matrix routes. Priority order (first match wins):
    //    SK_LASTBLOCK (parallel grid.z) > INWG (workgroup serial) > splitK
    //    (DEPOSIT/FOLD two dispatch) > plain coopmat (catch-all).
    //    SK_LAST currently disabled (applies returns false) while the shader
    //    lands; when enabled, it takes priority over INWG.
    // 4a. coopmat + split-K grid-parallel "last block reduce" (opt-in via
    //     VKOP_COOP_SK_LAST / VKOP_COOP_SK).
    {MatmulRoute::kCoopmatSkLast, "coopmat_sklast", sk_last_applies,
     sk_last_configure},
    // 4b. coopmat + split-K in-workgroup (opt-in via VKOP_COOP_INWG / _K).
    {MatmulRoute::kCoopmatInwg, "coopmat_inwg", coopmat_inwg_applies,
     coop_inwg_configure},
    // 4c. coopmat + split-K DEPOSIT/FOLD hybrid (opt-in via VKOP_COOP_SPLITK
    //     / _FORCE). Split across grid.z + global scratch round-trip.
    {MatmulRoute::kCoopmatSplitK, "coopmat_splitk", coopmat_splitk_applies,
     coop_splitk_configure},
    // 4d. Plain coopmat, no K split.
    {MatmulRoute::kCoopmat, "coopmat", coopmat_applies, nullptr},
    // 5. The shared-memory tile, split across K when the grid is too shallow.
    {MatmulRoute::kTiledKs, "tiled_ks", tiled_ks_applies, nullptr},
    {MatmulRoute::kTiled, "tiled", tiled_applies, nullptr},
    // 6. Split-K GEMV. The WIDE variants are per-N auto-enabled; a driver that
    //    refused their module falls through to the 4-column kernel below.
    {MatmulRoute::kKsplitW8Wide512, "ksplit_w8wide512",
     ksplit_w8_wide512_applies, q_byte_wide},
    {MatmulRoute::kKsplitW8Wide, "ksplit_w8wide", ksplit_w8_wide_applies,
     q_byte_wide},
    {MatmulRoute::kKsplitWide512, "ksplit_wide512", ksplit_wide512_applies,
     q_fp16_wide},
    {MatmulRoute::kKsplitWide, "ksplit_wide", ksplit_wide_applies, q_fp16_wide},
    {MatmulRoute::kKsplit512, "ksplit512", ksplit_512_applies, q_plain},
    {MatmulRoute::kKsplit, "ksplit", ksplit_applies, q_plain},
    // 7. Column-parallel fallbacks, one kernel per (dtype, N parity) pair.
    {MatmulRoute::kQuad4, "quad4", quad4_applies, nullptr},
    {MatmulRoute::kFusedFp16, "fused_fp16", fused_applies, nullptr},
    {MatmulRoute::kReducePack, "reduce_pack", reduce_pack_applies, nullptr},
    {MatmulRoute::kReduce, "reduce", reduce_applies, nullptr},
};

MatmulPlan apply_rule(Ctx &c, const MatmulRule &rule) {
    if (rule.configure != nullptr) {
        rule.configure(c);
    }
    MatmulPlan p;
    p.route = rule.route;
    p.tag = rule.tag;
    p.ksplit_q = c.q;
    p.wide = c.wide;
    p.tiled = c.tiled;
    p.tiled_min_m = c.tiled_min_m;
    p.device = c.prof;
    p.ks_n = (rule.route == MatmulRoute::kTiledKs) ? c.ks_n : 0;
    // coopmat-split-K and coopmat-inwg both carry their depth in coop_split,
    // not ks_n (configure wrote it to c.ks_n, we move it here).
    p.coop_split =
        (rule.route == MatmulRoute::kCoopmatSplitK ||
         rule.route == MatmulRoute::kCoopmatInwg ||
         rule.route == MatmulRoute::kCoopmatSkLast)
            ? c.ks_n
            : 0;
    return p;
}

} // namespace

const char *device_class_name(DeviceClass cls) {
    switch (cls) {
    case DeviceClass::kMoltenVkAmd:
        return "moltenvk-amd";
    case DeviceClass::kMoltenVkIntel:
        return "moltenvk-intel";
    case DeviceClass::kNativeDiscrete:
        return "native-discrete";
    case DeviceClass::kGeneric:
    default:
        return "generic";
    }
}

const DeviceProfile &device_profile_for(const DeviceTraits &traits) {
    for (const ClassRule &rule : kClassRules) {
        if (rule.match(traits)) {
            return kProfiles[static_cast<int>(rule.cls)];
        }
    }
    return kProfiles[static_cast<int>(DeviceClass::kGeneric)];
}

DeviceTraits device_traits_from(VulkanDevice &dev) {
    DeviceTraits t;
    // Query the driver/subgroup chain directly instead of going through
    // VulkanDevice::getProperties(): that is private and re-runs the legacy
    // tile-profile if-chain on every call. Only the physical-device handle is
    // needed, and it is public. Call once per device, cache the result.
    VkPhysicalDeviceSubgroupProperties subgroup{};
    subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceDriverProperties driver{};
    driver.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
    driver.pNext = &subgroup;
    VkPhysicalDeviceProperties2 props{};
    props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props.pNext = &driver;
    vkGetPhysicalDeviceProperties2(dev.getPhysicalDevice(), &props);

    t.vendor_id = props.properties.vendorID;
    t.device_id = props.properties.deviceID;
    t.driver_id = static_cast<uint32_t>(driver.driverID);
    t.unified_memory = dev.is_unified_memory();
    t.coopmat = dev.is_support_cooperate_matrix() &&
                dev.coopmat_supported_stages() != 0;
    t.coopmat_fp16_8x8x16 = dev.supports_coopmat_fp16_tile(8, 8, 16);
    t.coopmat_fp16_16x16x16 = dev.supports_coopmat_fp16_tile(16, 16, 16);
    t.coopmat_sint8_8x8x32 = dev.supports_coopmat_sint8_tile(8, 8, 32);
    t.subgroup_arithmetic = dev.supportsSubgroupArithmetic();
    t.subgroup_size = subgroup.subgroupSize;
    t.max_storage_buffer_range = props.properties.limits.maxStorageBufferRange;
    // Enabled-on-the-device view of the shader-level probes; see DeviceTraits.
    const ShaderCapabilities &sc = dev.shader_caps();
    t.f32_global_atomic_add = sc.f32_global_atomic_add.enabled;
    t.f32_shared_atomic_add = sc.f32_shared_atomic_add.enabled;
    t.f16x2_global_atomic_add = sc.f16x2_global_atomic_add.enabled;
    t.shader_float16 = sc.shader_float16.enabled;
    t.storage_buffer_8bit = sc.storage_buffer_8bit.enabled;
    t.subgroup_rotate = sc.subgroup_rotate.enabled;
    t.wmem_explicit_layout = sc.wmem_explicit_layout.enabled;
    t.zero_init_wmem = sc.zero_init_wmem.enabled;
    return t;
}

const char *matmul_route_tag(MatmulRoute route) {
    switch (route) {
    case MatmulRoute::kReduce:
        return "reduce";
    case MatmulRoute::kReducePack:
        return "reduce_pack";
    case MatmulRoute::kFusedFp16:
        return "fused_fp16";
    case MatmulRoute::kQuad4:
        return "quad4";
    case MatmulRoute::kKsplit:
        return "ksplit";
    case MatmulRoute::kKsplit512:
        return "ksplit512";
    case MatmulRoute::kKsplitWide:
        return "ksplit_wide";
    case MatmulRoute::kKsplitWide512:
        return "ksplit_wide512";
    case MatmulRoute::kKsplitW8Wide:
        return "ksplit_w8wide";
    case MatmulRoute::kKsplitW8Wide512:
        return "ksplit_w8wide512";
    case MatmulRoute::kTiled:
        return "tiled";
    case MatmulRoute::kTiledKs:
        return "tiled_ks";
    case MatmulRoute::kCoopmat:
        return "coopmat";
    case MatmulRoute::kCoopmatSplitK:
        return "coopmat_splitk";
    case MatmulRoute::kCoopmatInwg:
        return "coopmat_inwg";
    case MatmulRoute::kCoopmatSkLast:
        return "coopmat_sklast";
    case MatmulRoute::kCoopmatI8:
        return "coopmat_i8";
    case MatmulRoute::kDownActFold:
        return "down_act_gemv";
    case MatmulRoute::kAttnSoftmax:
        return "attn_softmax";
    case MatmulRoute::kCoopmatSkLast:
        return "coopmat_sklast";
    case MatmulRoute::kCoopmatInwg:
        return "coopmat_inwg";
    }
    return "reduce";
}

MatmulKnobs matmul_knobs_from_env() {
    MatmulKnobs k;
    if (const char *e = std::getenv("VKOP_KSPLIT_FLOOR")) {
        const int v = std::atoi(e);
        if (v >= 1) {
            k.ksplit_floor = v;
        }
    }
    if (const char *e = std::getenv("VKOP_KSPLIT_Q")) {
        const int v = std::atoi(e);
        if (v > 0) {
            k.ksplit_q = v;
        }
    }
    if (const char *e = std::getenv("VKOP_KSPLIT_WIDE")) {
        if (e[0] == '\0') {
            k.wide_force = 0;
        } else if (e[0] == '0') {
            k.wide_force = -1;
        } else {
            k.wide_force = 1;
        }
    }
    if (const char *e = std::getenv("VKOP_KSPLIT_WIDE_N")) {
        if (e[0] != '\0') {
            const int v = std::atoi(e);
            if (v >= 0) {
                k.wide_threshold_fp16 = v;
                k.wide_threshold_byte = v;
            }
        }
    }
    if (const char *e = std::getenv("VKOP_KSPLIT512")) {
        k.ksplit512_enabled = (e[0] == '1');
    }
    if (const char *e = std::getenv("VKOP_KS_N")) {
        k.ks_n_pin = std::atoi(e);
    }
    if (const char *e = std::getenv("VKOP_KS_TARGET")) {
        k.ks_target_pin = std::atoi(e);
    }
    auto flag = [](const char *name) {
        const char *e = std::getenv(name);
        return e != nullptr && e[0] != '\0' && e[0] != '0';
    };
    k.tile_disabled = flag("VKOP_NO_TILE");
    k.coop_disabled = flag("VKOP_NO_COOP");
    k.coop_i8_enabled = flag("VKOP_COOP_I8");
    k.force_coop = flag("VKOP_FORCE_COOP");
    if (const char *e = std::getenv("VKOP_COOP_SPLITK_FORCE")) {
        const int v = std::atoi(e);
        if (v >= 0) {
            // >= 2 means "force K-split to this depth"; 0 or 1 means "off but
            // the generic opt-in flag can still enable auto-split".
            k.coop_split = v;
        }
    }
    if (k.coop_split == 0 && flag("VKOP_COOP_SPLITK")) {
        // Bare opt-in: selector picks the K depth from occupancy.
        k.coop_split = -1; // sentinel: "auto-split allowed, actual depth decided by selector"
    }
    // VKOP_COOP_INWG takes priority over VKOP_COOP_SPLITK because INWG has
    // stricter guards and is a single-dispatch variant. When both are set,
    // the route table entry for kCoopmatInwg wins (listed first), and we
    // write coop_split here so the selector uses that depth for both routes'
    // configure phases — splitK's configure ignores INWG-only guards, so
    // shapes that fail INWG's N-even guard will still fall through to splitK.
    if (const char *e = std::getenv("VKOP_COOP_INWG_K")) {
        const int v = std::atoi(e);
        if (v >= 2 && v <= 8) {
            k.coop_split = v; // force cs ∈ [2, 8]
        }
    }
    if (flag("VKOP_COOP_INWG") && k.coop_split == 0) {
        k.coop_split = -1; // auto depth
    }
    // VKOP_COOP_SK_LAST takes highest priority. Parsed last so it overwrites
    // coop_split set by SPLITK or INWG; the route table orders SkLast first,
    // so applies/configure resolve the correct depth here. When both SK_LAST
    // and INWG are set, SkLast wins the route and coop_split carries the SK
    // value; INWG's applies will see coop_split > 1 but fall through because
    // SkLast wins first — but if SK_LAST is disabled (applies false), INWG's
    // own applies will pick up that coop_split and route correctly.
    if (const char *e = std::getenv("VKOP_COOP_SK")) {
        const int v = std::atoi(e);
        if (v >= 1 && v <= 8) {
            k.coop_split = v;
        }
    }
    if (flag("VKOP_COOP_SK_LAST") && k.coop_split == 0) {
        k.coop_split = -1; // auto depth
    }
    return k;
}

const MatmulKnobs &matmul_knobs() {
    static const MatmulKnobs knobs = matmul_knobs_from_env();
    return knobs;
}

MatmulPlan select_matmul(const DeviceTraits &traits, const MatmulShape &shape,
                         const MatmulFeatures &features,
                         const MatmulAvailability &availability,
                         const MatmulKnobs &knobs) {
    const DeviceProfile &profile = device_profile_for(traits);
    Ctx c = make_ctx(traits, profile, shape, features, availability, knobs);
    std::fprintf(stderr, "[ksel] m=%d n=%d k=%d fp16=%d weight_fp16=%d coop_split=%d force_coop=%d coopmat_av=%d\n",
                 shape.m, shape.n, shape.k, features.fp16,
                 features.weight == WeightKind::kFp16, knobs.coop_split,
                 knobs.force_coop, availability.coopmat);
    for (const MatmulRule &rule : kMatmulRules) {
        bool a = rule.applies(c);
        std::fprintf(stderr, "[ksel] rule %s -> %d\n", rule.tag, a);
        if (a) {
            MatmulPlan p = apply_rule(c, rule);
            std::fprintf(stderr, "[ksel] CHOSE %s coop_split=%d ks_n=%d\n",
                         rule.tag, p.coop_split, p.ks_n);
            return p;
        }
    }
    // The last row is unconditional; reaching this is a table bug.
    return MatmulPlan();
}

} // namespace core
} // namespace vkop
