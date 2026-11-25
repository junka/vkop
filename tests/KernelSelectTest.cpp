// Table-driven kernel selection: profile table + MatMul rule table.
//
// Standalone (no gtest, no torch, no GPU): every input is a DeviceTraits /
// MatmulShape value, so the test runs on any host and pins the policy before
// ops/Matmul.hpp starts consuming it. The expected routes are the ones the
// current inline cascade produces; the parity check against the live op is
// VKOP_MATMUL_ROUTE=1 on bench_matmul / matmul_m_sweep.
//
// Run: ./KernelSelectTest

#include <cstdio>
#include <string>

#include "core/KernelSelect.hpp"

using namespace vkop::core;

namespace {

int g_fail = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);        \
            ++g_fail;                                                          \
        }                                                                      \
    } while (0)

void check_route(const MatmulPlan &plan, MatmulRoute want, int line) {
    if (plan.route != want) {
        std::printf("FAIL %s:%d: route %s, want %s\n", __FILE__, line,
                    matmul_route_tag(plan.route), matmul_route_tag(want));
        ++g_fail;
    }
}

#define CHECK_ROUTE(plan, want) check_route((plan), MatmulRoute::want, __LINE__)

DeviceTraits traits_native_discrete() {
    DeviceTraits t;
    t.vendor_id = 0x10de;
    t.driver_id = 4; // VK_DRIVER_ID_NVIDIA_PROPRIETARY
    t.unified_memory = false;
    return t;
}

DeviceTraits traits_mvk_amd() {
    DeviceTraits t;
    t.vendor_id = 0x1002;
    t.driver_id = 14; // VK_DRIVER_ID_MOLTENVK
    t.unified_memory = false;
    return t;
}

DeviceTraits traits_mvk_intel() {
    DeviceTraits t;
    t.vendor_id = 0x8086;
    t.driver_id = 14;
    t.unified_memory = true;
    return t;
}

DeviceTraits traits_mvk_apple() {
    DeviceTraits t;
    t.vendor_id = 0x106B;
    t.driver_id = 14;
    t.unified_memory = true;
    return t;
}

// Ada-class: native discrete + the 16x16x16 fp16 MMA.
DeviceTraits traits_ada() {
    DeviceTraits t = traits_native_discrete();
    t.coopmat = true;
    t.coopmat_fp16_16x16x16 = true;
    return t;
}

MatmulAvailability fp16_only() {
    MatmulAvailability a; // no optional modules
    return a;
}

MatmulAvailability fp16_splitk() {
    MatmulAvailability a;
    a.ksplit512 = true;
    a.w8_wide512 = false;
    a.tile_ks_split = true;
    a.tile_ks_fold = true;
    return a;
}

MatmulAvailability int8_modules() {
    MatmulAvailability a;
    a.w8_wide = true;
    a.tile_ks_split = true;
    a.tile_ks_fold = true;
    return a;
}

// DP4A: the int8 W8A8 modules plus the dot-product one, on a device whose
// packed 4x8 signed form is accelerated.
MatmulAvailability dp4a_modules() {
    MatmulAvailability a = int8_modules();
    a.w8a8_dp4a = true;
    return a;
}

DeviceTraits traits_dp4a() {
    DeviceTraits t = traits_native_discrete();
    t.integer_dot_product_packed = true;
    return t;
}

MatmulAvailability coop_modules() {
    MatmulAvailability a;
    a.coopmat = true;
    a.coop_bm = 32; // 16x16x16 workgroup tile
    a.coop_bn = 64;
    a.tile_ks_split = true;
    a.tile_ks_fold = true;
    return a;
}

MatmulFeatures fp16_w(bool transB = false) {
    MatmulFeatures f;
    f.fp16 = true;
    f.transB = transB;
    return f;
}

MatmulFeatures weight8(bool transB = false) {
    MatmulFeatures f;
    f.fp16 = true;
    f.transB = transB;
    f.weight = WeightKind::kInt8;
    return f;
}

// int8 activation + int8 weight: the only combination the DP4A route accepts.
MatmulFeatures w8a8(bool transB = false) {
    MatmulFeatures f = weight8(transB);
    f.a_int8 = true;
    return f;
}

MatmulShape sh(int batch, int m, int k, int n) {
    MatmulShape s;
    s.batch = batch;
    s.m = m;
    s.k = k;
    s.n = n;
    return s;
}

void test_apple_coop8_gate() {
    const DeviceTraits apple = traits_mvk_apple();
    CHECK(device_supports_apple_coop8(apple));
    const DeviceTraits amd = traits_mvk_amd();
    const DeviceTraits intel = traits_mvk_intel();
    CHECK(!device_supports_apple_coop8(amd));
    CHECK(!device_supports_apple_coop8(intel));
    // Apple vendor on a non-MoltenVK driver is not the off-spec path either.
    DeviceTraits native_apple = traits_mvk_apple();
    native_apple.driver_id = 0;
    CHECK(!device_supports_apple_coop8(native_apple));
}

void test_profiles() {
    // The traits are bound to named locals: device_profile_for returns a
    // reference into the static table, and GCC's -Wdangling-reference cannot
    // tell that from "returning a reference derived from the temporary
    // argument", so passing a temporary here is a -Werror build break.
    const DeviceTraits native_traits = traits_native_discrete();
    const DeviceTraits amd_traits = traits_mvk_amd();
    const DeviceTraits intel_traits = traits_mvk_intel();
    const DeviceProfile &native = device_profile_for(native_traits);
    CHECK(native.device_class == DeviceClass::kNativeDiscrete);
    CHECK(native.byte_min_m == 9);
    CHECK(native.fp16_min_m == 9);
    CHECK(native.wide_n == 6144);
    CHECK(native.fp16_wide_min_m == 7);
    CHECK(native.ks_target_blocks == 128);
    CHECK(native.ksplit_floor == 8);
    CHECK(native.wide_threshold_fp16 == 2048);
    CHECK(native.wide_threshold_byte == 896);

    const DeviceProfile &amd = device_profile_for(amd_traits);
    CHECK(amd.device_class == DeviceClass::kMoltenVkAmd);
    CHECK(amd.byte_min_m == 9);
    CHECK(amd.fp16_min_m == 4);
    CHECK(amd.wide_n == 65536);
    CHECK(amd.fp16_wide_min_m == 6);
    CHECK(amd.ks_target_blocks == 32);

    const DeviceProfile &intel = device_profile_for(intel_traits);
    CHECK(intel.device_class == DeviceClass::kMoltenVkIntel);
    CHECK(intel.byte_min_m == 9);
    CHECK(intel.fp16_min_m == 9);
    CHECK(intel.wide_n == 0);
    CHECK(intel.ks_target_blocks == 32);

    const DeviceTraits apple_traits = traits_mvk_apple();
    const DeviceProfile &apple = device_profile_for(apple_traits);
    CHECK(apple.device_class == DeviceClass::kMoltenVkApple);
    CHECK(apple.byte_min_m == 9);
    CHECK(apple.fp16_min_m == 9);
    CHECK(apple.wide_n == 0);
    CHECK(apple.ksplit_floor == 32);
    CHECK(std::string(device_class_name(apple.device_class)) ==
          "moltenvk-apple");

    // An unknown native iGPU keeps the ARL / Ada calibration.
    DeviceTraits igpu = traits_mvk_intel();
    igpu.driver_id = 0; // native
    CHECK(device_profile_for(igpu).device_class == DeviceClass::kGeneric);
    CHECK(device_profile_for(igpu).fp16_min_m == 3);
}

// The four Qwen3-2B decode GEMMs on the GTX 1070 profile, transB == 0.
void test_decode_shapes_native() {
    const DeviceTraits dev = traits_native_discrete();
    const MatmulAvailability av = fp16_only();

    // 8 MB: n = 2048 reaches the fp16 WIDE threshold exactly.
    MatmulPlan p = select_matmul(dev, sh(1, 1, 2048, 2048), fp16_w(), av);
    CHECK_ROUTE(p, kKsplitWide);
    CHECK(p.ksplit_q == 8); // 2048/8 = 256 cols; q=8 -> 32 blocks
    CHECK(p.wide);

    p = select_matmul(dev, sh(1, 1, 2048, 6144), fp16_w(), av);
    CHECK_ROUTE(p, kKsplitWide);

    p = select_matmul(dev, sh(1, 1, 6144, 2048), fp16_w(), av);
    CHECK_ROUTE(p, kKsplitWide);

    // 622 MB lm_head.
    p = select_matmul(dev, sh(1, 1, 2048, 151936), fp16_w(), av);
    CHECK_ROUTE(p, kKsplitWide);
    CHECK(p.ksplit_q == 256); // plenty of blocks, no K split beyond the load

    // WIDE off: the 4-column split-K kernel takes the same shape.
    MatmulKnobs kn;
    kn.wide_force = -1;
    p = select_matmul(dev, sh(1, 1, 2048, 2048), fp16_w(), av, kn);
    CHECK_ROUTE(p, kKsplit);
    CHECK(!p.wide);

    // WIDE forced on for a narrow row: q walks down to the 32-block floor.
    kn = MatmulKnobs();
    kn.wide_force = 1;
    p = select_matmul(dev, sh(1, 1, 2048, 1024), fp16_w(), av, kn);
    CHECK_ROUTE(p, kKsplitWide);
    CHECK(p.ksplit_q == 4); // 1024/8 = 128 cols; 128/4 = 32 blocks

    // 512-lane build, when the archive has it.
    kn = MatmulKnobs();
    kn.ksplit512_enabled = true;
    MatmulAvailability av512 = fp16_splitk();
    av512.ksplit512 = true;
    p = select_matmul(dev, sh(1, 1, 2048, 2048), fp16_w(), av512, kn);
    CHECK_ROUTE(p, kKsplitWide512);

    // The b_row_bytes power-of-two >= 64 KiB override: fp16 N = 32768 is a
    // 64 KiB row stride, whose tile staging reads fall to 6-13 GB/s. The tile
    // must not claim it before m = 12.
    CHECK_ROUTE(select_matmul(dev, sh(1, 11, 2048, 32768), fp16_w(true),
                              fp16_only()),
                kFusedFp16);
    // 512 column tiles already clear the 128-block target, so no K split.
    CHECK_ROUTE(select_matmul(dev, sh(1, 12, 2048, 32768), fp16_w(true),
                              fp16_splitk()),
                kTiled);
}

void test_tiled_band_native() {
    const DeviceTraits dev = traits_native_discrete();
    const MatmulAvailability av = fp16_splitk();

    // m = 7 crosses the wide band (N >= 6144) on this device. Tile-split-K
    // needs transB == 0 (it walks whole K rows), so keep B in [K, N] layout.
    MatmulPlan p =
        select_matmul(dev, sh(1, 7, 2048, 6144), fp16_w(), av);
    CHECK_ROUTE(p, kTiledKs);
    CHECK(p.tiled);
    CHECK(p.ks_n > 1);
    // 6144/64 = 96 blocks already below the 128 target: depth 2.
    CHECK(p.ks_n == 2);

    // transB == 1 forbids the K split but not the tile itself.
    p = select_matmul(dev, sh(1, 7, 2048, 6144), fp16_w(true), av);
    CHECK_ROUTE(p, kTiled);
    CHECK(p.ks_n == 0);

    // m = 6 is still GEMV territory in the wide band.
    p = select_matmul(dev, sh(1, 6, 2048, 6144), fp16_w(), av);
    CHECK_ROUTE(p, kKsplitWide);

    // Narrow N keeps the tile out until m = 9.
    CHECK_ROUTE(select_matmul(dev, sh(1, 8, 2048, 2048), fp16_w(), av),
                kKsplitWide);
    p = select_matmul(dev, sh(1, 9, 2048, 2048), fp16_w(), fp16_splitk());
    CHECK_ROUTE(p, kTiledKs);

    // Tile-split-K needs both modules; without them the same shape runs the
    // unsplit tile (the shader ignores the K-range header).
    p = select_matmul(dev, sh(1, 9, 2048, 2048), fp16_w(true), fp16_only());
    CHECK_ROUTE(p, kTiled);
    CHECK(p.ks_n == 0);
}

void test_byte_weight_native() {
    const DeviceTraits dev = traits_native_discrete();
    const MatmulAvailability av = int8_modules();

    // lm_head: 16 columns/lane byte WIDE, q = 256 is allowed by the floor.
    MatmulPlan p =
        select_matmul(dev, sh(1, 1, 2048, 151936), weight8(), av);
    CHECK_ROUTE(p, kKsplitW8Wide);
    CHECK(p.wide);
    CHECK(p.ksplit_q == 256);

    // n = 2048 is above the byte threshold (896).
    p = select_matmul(dev, sh(1, 1, 2048, 2048), weight8(), av);
    CHECK_ROUTE(p, kKsplitW8Wide);

    // n = 512 is below it: the 4-column kernel with q walked to the floor.
    p = select_matmul(dev, sh(1, 1, 2048, 512), weight8(), av);
    CHECK_ROUTE(p, kKsplit);
    CHECK(p.ksplit_q == 4); // 512/4 = 128 cols; 128/4 = 32 blocks

    // The WIDE module missing (driver refusal) falls through, it does not fail.
    p = select_matmul(dev, sh(1, 1, 2048, 2048), weight8(), fp16_only());
    CHECK_ROUTE(p, kKsplit);

    // m = 9 leaves the split-K window and the tile takes over. The byte tile's
    // staging path reads whole K rows, so the tile never splits K for int8.
    p = select_matmul(dev, sh(1, 9, 2048, 2048), weight8(), av);
    CHECK_ROUTE(p, kTiled);
    CHECK(p.ks_n == 0);

    // transB = 1 is not the layout the byte tile loader reads: stay fused.
    p = select_matmul(dev, sh(1, 9, 2048, 2048), weight8(true), av);
    CHECK_ROUTE(p, kFusedFp16);

    // Decode + transB byte weight: the split-K GEMV's [N, K] form. Its lanes
    // are mapped K-contiguous (slice = lane % n_slice), so q is capped at 8 to
    // keep n_slice >= 32 and one warp on one 128-byte line. Uncapped, q_plain
    // would leave this shape at its default 16 (N/4 = 1280 quads is already
    // past the 32-block floor); the cap pulls it to 8.
    p = select_matmul(dev, sh(1, 1, 5120, 5120), weight8(true), av);
    CHECK_ROUTE(p, kKsplit);
    CHECK(p.ksplit_q == 8);
    // A narrow N still walks q down for block count, below the cap.
    p = select_matmul(dev, sh(1, 1, 5120, 512), weight8(true), av);
    CHECK_ROUTE(p, kKsplit);
    CHECK(p.ksplit_q == 4); // 512/4 = 128 quads; 128/4 = 32 blocks
    // The cap is an upper bound only: a pin may ask for fewer, never more.
    MatmulKnobs kn2;
    kn2.ksplit_q = 64;
    p = select_matmul(dev, sh(1, 1, 5120, 5120), weight8(true), av, kn2);
    CHECK(p.ksplit_q == 8);
    kn2.ksplit_q = 4;
    p = select_matmul(dev, sh(1, 1, 5120, 5120), weight8(true), av, kn2);
    CHECK(p.ksplit_q == 4);
}

void test_coopmat_gate() {
    const DeviceTraits ada = traits_ada();
    const MatmulAvailability av = coop_modules();

    // One full 32-row coop workgroup of rows: coop wins outright.
    CHECK_ROUTE(select_matmul(ada, sh(1, 16, 2048, 6144), fp16_w(true), av),
                kCoopmat);

    // Decode, cache-resident weight: ksplit keeps the per-row re-read in L2
    // and beats the coop tile across the whole batch*m <= 8 window.
    MatmulPlan p =
        select_matmul(ada, sh(1, 1, 2048, 6144), fp16_w(), av);
    CHECK_ROUTE(p, kKsplitWide);
    CHECK(!p.tiled);

    // m = 3, still cache-resident: the tile floor is raised to 12 on a
    // 16x16x16 device so neither tile nor coop intercepts the GEMV window.
    p = select_matmul(ada, sh(1, 3, 2048, 6144), fp16_w(), av);
    CHECK_ROUTE(p, kKsplitWide);
    CHECK(p.tiled_min_m == 12);

    // Same shape but the weight no longer fits the 32 MB cache budget
    // (k=6144, n=2048 fp16 -> 24 MB... use n=6144 -> 75 MB): coop takes m=3.
    p = select_matmul(ada, sh(1, 3, 6144, 6144), fp16_w(), av);
    CHECK_ROUTE(p, kCoopmat);

    // epi_bias has no binding in the coop descriptor layout, and the tile
    // refuses an epilogue too: the column-parallel fused kernel takes it.
    MatmulFeatures f = fp16_w();
    f.epi_bias = true;
    p = select_matmul(ada, sh(1, 16, 2048, 6144), f, av);
    CHECK_ROUTE(p, kFusedFp16);

    // VKOP_NO_COOP.
    MatmulKnobs kn;
    kn.coop_disabled = true;
    p = select_matmul(ada, sh(1, 16, 2048, 6144), fp16_w(), av, kn);
    CHECK_ROUTE(p, kTiledKs);
}

// Two ways a shape WOULD pass the coop shape gate but must not take the route,
// because the module cannot express the node: an odd N on a self-packing
// kernel, and a SwiGLU down fold (which has its own layout and epilogue). Both
// used to be enforced only by rule-table ORDER, which the smart selector
// bypasses by running first.
void test_coopmat_shape_holes() {
    const DeviceTraits ada = traits_ada();
    MatmulAvailability av = coop_modules();
    av.coop_self_pack = true;

    // Even N, self-packing: coop takes it (the control).
    CHECK_ROUTE(select_matmul(ada, sh(1, 16, 2048, 6144), fp16_w(), av),
                kCoopmat);

    // Odd N: the self-packing epilogue pairs (col, col+1) within a row, so a
    // row-major flat word index lands one half-word off on every other row.
    // Fall through to the odd-N reduce->pack pair, whose scratch pass re-pairs
    // the words.
    CHECK_ROUTE(select_matmul(ada, sh(1, 16, 2048, 6143), fp16_w(), av),
                kReducePack);

    // A module that buffers fp32 in scratch and lets the pack pass form the
    // words has no such hole, so odd N stays eligible.
    av.coop_self_pack = false;
    CHECK_ROUTE(select_matmul(ada, sh(1, 16, 2048, 6143), fp16_w(), av),
                kCoopmat);
    av.coop_self_pack = true;

    // SwiGLU down fold at a coop-eligible shape (m=15: the real prefill row
    // count). The folded node's inputs are [x, Wd, Wg, Wu] and its result is
    // silu(x.Wg)*(x.Wu).Wd -- no coop module has those bindings, so coop must
    // lose to the fold row even though M is past the tile minimum.
    MatmulFeatures f = fp16_w();
    f.down_act = true;
    MatmulShape s = sh(1, 15, 2048, 2048);
    s.n_up = 6144;
    MatmulAvailability dav = av;
    dav.down_act = true;
    CHECK_ROUTE(select_matmul(ada, s, f, dav), kDownActFold);

    // Same node with the fold module missing: the plain routes take it, and
    // coop is still not one of them (it would compute a bare down GEMM).
    dav.down_act = false;
    MatmulPlan p = select_matmul(ada, s, f, dav);
    CHECK(p.route != MatmulRoute::kCoopmat);
}

void test_other_routes() {
    const DeviceTraits dev = traits_native_discrete();
    const MatmulAvailability av = fp16_only();

    // fp32: no fused kernel, straight to reduce.
    MatmulFeatures f;
    f.fp16 = false;
    CHECK_ROUTE(select_matmul(dev, sh(1, 1, 512, 512), f, av), kReduce);

    // fp16 odd N: the half2 fused kernel cannot express it.
    CHECK_ROUTE(select_matmul(dev, sh(1, 1, 512, 513), fp16_w(), av),
                kReducePack);

    // Attention QK^T + fused Softmax, even N, K <= 1024.
    f = fp16_w();
    f.attn_softmax = true;
    CHECK_ROUTE(select_matmul(dev, sh(1, 1, 128, 1024), f, av), kAttnSoftmax);

    // SwiGLU down fold: x's true width rides in K, the contraction in n_up.
    f = fp16_w();
    f.down_act = true;
    MatmulShape s = sh(1, 1, 2048, 2048);
    s.n_up = 8192;
    MatmulAvailability dav;
    dav.down_act = true;
    CHECK_ROUTE(select_matmul(dev, s, f, dav), kDownActFold);
    // Without the module the shape falls back to plain routing, i.e. the
    // decode GEMV for this n.
    CHECK_ROUTE(select_matmul(dev, s, f, av), kKsplitWide);

    // 4-bit is excluded from coop; the split-K window claims decode
    // (matmul.comp dequantizes in the ksplit entry too).
    f = fp16_w();
    f.weight = WeightKind::kInt4;
    CHECK_ROUTE(select_matmul(dev, sh(1, 1, 512, 512), f, av), kKsplit);
    // Without a group size (the fixture's default) the tile cannot walk the
    // scale rows, so past the split-K window the quad kernel keeps it.
    CHECK_ROUTE(select_matmul(dev, sh(1, 9, 512, 512), f, av), kQuad4);
    // A wide N must NOT promote a 4-bit weight to kKsplitWide. That route
    // builds from a separate -DW8_WIDE module whose main() has no uPC.w4
    // branch, so it reads packed nibbles with unpackHalf2x16: right byte
    // count, garbage values, no error. This is the decode path that made the
    // first int4 27B run emit ",,,,," -- the narrow case above passed while
    // every real shape (n >= 10240) went wide.
    CHECK_ROUTE(select_matmul(dev, sh(1, 1, 5120, 10240), f, av), kKsplit);
    CHECK_ROUTE(select_matmul(dev, sh(1, 1, 5120, 248320), f, av), kKsplit);

    // With a group size recorded (every real int4 weight has one) a wide 4-bit
    // shape takes kKsplitW4Wide instead: the 8-column nibble kernel in the
    // -DW8_WIDE module, launched UP_DIV(n/8, q). The cases above stay on
    // kKsplit only because the fixture leaves group_size at 0, which that
    // kernel's per-group scale walk cannot use.
    f.group_size = 64;
    CHECK_ROUTE(select_matmul(dev, sh(1, 1, 5120, 10240), f, av),
                kKsplitW4Wide);
    // Below the wide threshold the narrow kernel still wins.
    CHECK_ROUTE(select_matmul(dev, sh(1, 1, 512, 512), f, av), kKsplit);
    // Past the split-K window a grouped 4-bit weight takes the 64x64 tile
    // (tile_load_b_w4 stages the dequantized, group-scaled half), from the
    // first row the GEMV gives up: batch*m = 9. Never the 128 tile, coopmat or
    // the tile-split-K pair -- none of them has a nibble loader.
    CHECK_ROUTE(select_matmul(dev, sh(1, 9, 512, 512), f, av), kTiled);
    CHECK_ROUTE(select_matmul(dev, sh(1, 1024, 2048, 6144), f, av), kTiled);
    CHECK_ROUTE(select_matmul(dev, sh(3, 3, 512, 512), f, av), kTiled);
    CHECK_ROUTE(select_matmul(dev, sh(1, 8, 512, 512), f, av), kKsplit);
    // A K that is not whole groups keeps the quad kernel past the window.
    CHECK_ROUTE(select_matmul(dev, sh(1, 9, 5152, 512), f, av), kQuad4);
    // A K that is not a whole number of groups would leave the scale walk
    // scaling a partial group, so the predicate refuses it. 5152 = 80*64 + 32.
    CHECK_ROUTE(select_matmul(dev, sh(1, 1, 5152, 10240), f, av), kKsplit);
}

void test_fallthrough_bits() {
    const DeviceTraits dev = traits_native_discrete();

    // A tile-split-K route needs BOTH modules. Half a pair is not a route: the
    // deposit's partials would never be folded, so the unsplit tile must take
    // the shape instead of the selector ignoring the missing bit.
    MatmulAvailability av = fp16_splitk();
    av.tile_ks_fold = false;
    MatmulPlan p = select_matmul(dev, sh(1, 9, 2048, 2048), fp16_w(), av);
    CHECK_ROUTE(p, kTiled);
    CHECK(p.ks_n == 0);

    av = fp16_splitk();
    av.tile_ks_split = false;
    CHECK_ROUTE(select_matmul(dev, sh(1, 9, 2048, 2048), fp16_w(), av), kTiled);

    // 512-lane byte WIDE needs the W8 WIDE module AND the 512 build; losing
    // the 512 one drops to the 256-lane WIDE row, not to the 4-column kernel.
    MatmulAvailability b512;
    b512.w8_wide = true;
    b512.w8_wide512 = true;
    MatmulKnobs kn;
    kn.ksplit512_enabled = true;
    p = select_matmul(dev, sh(1, 1, 2048, 2048), weight8(), b512, kn);
    CHECK_ROUTE(p, kKsplitW8Wide512);
    b512.w8_wide512 = false;
    CHECK_ROUTE(select_matmul(dev, sh(1, 1, 2048, 2048), weight8(), b512, kn),
                kKsplitW8Wide);
}

// int8 x int8 packed dot product (VKOP_DP4A). Every check here is a boundary
// that, if it slipped, would either run a float kernel over byte payloads or
// let a signed 32-bit accumulator wrap.
void test_dp4a_routes() {
    const DeviceTraits dev = traits_dp4a();
    const MatmulAvailability av = dp4a_modules();
    const MatmulShape decode = sh(1, 1, 2048, 2048);

    MatmulKnobs on;
    on.dp4a_enabled = true;

    // Default OFF: the same device, availability and shape keep the byte WIDE
    // GEMV. This is the shipped behaviour and the reason the route is opt-in.
    MatmulPlan p = select_matmul(dev, decode, w8a8(), av, MatmulKnobs());
    CHECK_ROUTE(p, kKsplitW8Wide);

    // Opted in on a device with the packed form accelerated: DP4A wins, even
    // though the byte WIDE row would also apply (n >= 896). That ordering is
    // deliberate -- see the rule comment.
    p = select_matmul(dev, decode, w8a8(), av, on);
    CHECK_ROUTE(p, kKsplitW8A8Dp4a);
    CHECK(!p.wide);
    // Same block-floor walk as the byte WIDE kernel, in 4-column groups: n/4 =
    // 512 groups, and q walks down from 256 while UP_DIV(groups, q) stays under
    // the floor of 8 (2 -> 4 -> 8), landing on q = 64.
    CHECK(p.ksplit_q == 64);

    // An fp16 activation is the ordinary case: the DP4A gate must not claim it
    // even with the knob on.
    p = select_matmul(dev, decode, weight8(), av, on);
    CHECK_ROUTE(p, kKsplitW8Wide);

    // The int32 ceiling. The safe bound is K * 128 * 128 < 2^31, i.e.
    // K <= 131071; every K below is a multiple of 4 so the dot-word gate does
    // not mask the overflow gate. One -128 element makes a term 16384, and
    // 131072 of them reach 2^31 exactly.
    p = select_matmul(dev, sh(1, 1, 131068, 2048), w8a8(), av, on);
    CHECK_ROUTE(p, kKsplitW8A8Dp4a);
    p = select_matmul(dev, sh(1, 1, 131072, 2048), w8a8(), av, on);
    CHECK(p.route != MatmulRoute::kKsplitW8A8Dp4a);
    // 133148 sits INSIDE the published "K < 133152" window but already
    // overflows under the symmetric 127*127 bound (133145 is the last K that
    // fits with 127s), so a gate written to the published figure accepts it.
    p = select_matmul(dev, sh(1, 1, 133148, 2048), w8a8(), av, on);
    CHECK(p.route != MatmulRoute::kKsplitW8A8Dp4a);

    // Non-int8 weight (fp8 shares the byte kernels but is not a signed dot
    // product operand -- its bytes are a float encoding).
    MatmulFeatures f8 = w8a8();
    f8.weight = WeightKind::kFp8;
    CHECK(select_matmul(dev, decode, f8, av, on).route !=
          MatmulRoute::kKsplitW8A8Dp4a);

    // int8 weight with a float activation (weight-only quantization).
    CHECK(select_matmul(dev, decode, weight8(), av, on).route !=
          MatmulRoute::kKsplitW8A8Dp4a);

    // 4-bit weight is not a byte operand at all.
    MatmulFeatures w4 = w8a8();
    w4.weight = WeightKind::kInt4;
    CHECK(select_matmul(dev, decode, w4, av, on).route !=
          MatmulRoute::kKsplitW8A8Dp4a);

    // Device without the feature (or without the accelerated packed form):
    // the flag alone must not route. This is the exact failure the old
    // api_version < 1.3 probe produced -- feature "on" nowhere, kernel
    // rejected at pipeline creation -- and the gate has to fail closed.
    DeviceTraits no_dot = traits_native_discrete();
    CHECK(select_matmul(no_dot, decode, w8a8(), av, on).route !=
          MatmulRoute::kKsplitW8A8Dp4a);

    // Module missing (archive without the key, or the driver refused it): the
    // availability bit is what the op clears, and the table must fall through
    // rather than emit a route with nothing to dispatch.
    CHECK_ROUTE(select_matmul(dev, decode, w8a8(), int8_modules(), on),
                kKsplitW8Wide);

    // K and N must be whole dot-words (4 bytes each) -- the kernel reads one
    // 32-bit word per side.
    CHECK(select_matmul(dev, sh(1, 1, 2046, 2048), w8a8(), av, on).route !=
          MatmulRoute::kKsplitW8A8Dp4a);
    CHECK(select_matmul(dev, sh(1, 1, 2048, 2046), w8a8(), av, on).route !=
          MatmulRoute::kKsplitW8A8Dp4a);

    // Decode-shaped only (batch * m <= 8), like every split-K GEMV row: past
    // it the tile owns the shape and a row-per-block kernel is the wrong
    // launch.
    CHECK(select_matmul(dev, sh(1, 9, 2048, 2048), w8a8(), av, on).route !=
          MatmulRoute::kKsplitW8A8Dp4a);
    // batch * m, not m alone: 2 x 3 = 6 is inside the window, 3 x 4 = 12 is
    // not.
    CHECK_ROUTE(select_matmul(dev, sh(2, 3, 2048, 2048), w8a8(), av, on),
                kKsplitW8A8Dp4a);
    CHECK(select_matmul(dev, sh(3, 4, 2048, 2048), w8a8(), av, on).route !=
          MatmulRoute::kKsplitW8A8Dp4a);

    // K < 256 is not worth a dot loop (one block would cover the whole row).
    CHECK(select_matmul(dev, sh(1, 1, 128, 2048), w8a8(), av, on).route !=
          MatmulRoute::kKsplitW8A8Dp4a);

    // transB: the DP4A module walks [K, N] rows like the WIDE kernels, so a
    // folded-transpose weight must stay on the 4-column split-K form.
    CHECK(select_matmul(dev, sh(1, 1, 2048, 2048), w8a8(true), av, on).route !=
          MatmulRoute::kKsplitW8A8Dp4a);

    // Odd N has no fused epilogue for the byte kernels either.
    MatmulKnobs on_odd = on;
    CHECK(select_matmul(dev, sh(1, 1, 2048, 2047), w8a8(), av, on_odd).route !=
          MatmulRoute::kKsplitW8A8Dp4a);
}

void test_mvk_routes() {
    const MatmulAvailability av = int8_modules();

    // AMD 560X: the byte GEMV keeps the whole split-K window, so m = 8 is
    // still a GEMV and m = 9 is the tile.
    const DeviceTraits amd = traits_mvk_amd();
    CHECK_ROUTE(select_matmul(amd, sh(1, 8, 2048, 2048), weight8(), av),
                kKsplitW8Wide);
    CHECK_ROUTE(select_matmul(amd, sh(1, 9, 2048, 2048), weight8(), av),
                kTiled);

    // fp16 crosses at m = 4 on that part, and 96 blocks already clear its
    // 32-block saturation target, so the tile takes m = 4 without a K split.
    MatmulPlan p = select_matmul(amd, sh(1, 4, 2048, 6144), fp16_w(), av);
    CHECK_ROUTE(p, kTiled);
    CHECK(p.ks_n == 0);
    p = select_matmul(amd, sh(1, 3, 2048, 6144), fp16_w(), av);
    CHECK_ROUTE(p, kKsplitWide);

    // Intel UHD 630: both crossovers sit past the window, so every row the
    // GEMV gate admits stays GEMV and the tile starts at 9.
    const DeviceTraits intel = traits_mvk_intel();
    CHECK_ROUTE(select_matmul(intel, sh(1, 8, 2048, 2048), fp16_w(), av),
                kKsplitWide);
    CHECK_ROUTE(select_matmul(intel, sh(1, 9, 2048, 2048), fp16_w(), av),
                kTiled);
}

void print_native_decode_table() {
    const DeviceTraits dev = traits_native_discrete();
    const MatmulAvailability av = int8_modules();
    struct Row {
        const char *name;
        int k, n;
        WeightKind w;
        bool transB;
    };
    const Row rows[] = {
        {"q/k/v/o", 2048, 2048, WeightKind::kFp16, false},
        {"gate/up", 2048, 6144, WeightKind::kFp16, false},
        {"down", 6144, 2048, WeightKind::kFp16, false},
        {"lm_head", 2048, 151936, WeightKind::kFp16, false},
    };
    std::printf("\nnative-discrete (GTX 1070) m=1 decode:\n");
    std::printf("%-9s %6s %6s %8s %6s %6s\n", "shape", "k", "n", "route", "q",
                "wide");
    for (const Row &r : rows) {
        MatmulFeatures f;
        f.fp16 = true;
        f.weight = r.w;
        f.transB = r.transB;
        const MatmulPlan p = select_matmul(dev, sh(1, 1, r.k, r.n), f, av);
        std::printf("%-9s %6d %6d %8s %6d %6d\n", r.name, r.k, r.n, p.tag,
                    p.ksplit_q, p.wide ? 1 : 0);
    }

    std::printf("byte weight:\n");
    for (const Row &r : rows) {
        MatmulFeatures f;
        f.fp16 = true;
        f.weight = WeightKind::kInt8;
        const MatmulPlan p = select_matmul(dev, sh(1, 1, r.k, r.n), f, av);
        std::printf("%-9s %6d %6d %8s %6d %6d\n", r.name, r.k, r.n, p.tag,
                    p.ksplit_q, p.wide ? 1 : 0);
    }
}

} // namespace

// Apple behind MoltenVK: the split-K GEMV keeps the whole batch*m <= 8 window
// for both dtypes, the tile takes m >= 9, and the byte WIDE floor is 32 blocks.
void test_mvk_apple_routes() {
    const DeviceTraits dev = traits_mvk_apple();

    // fp16 m = 3..8 used to fall to the tile on the generic profile.
    for (int m : {1, 2, 3, 4, 8}) {
        MatmulPlan p = select_matmul(dev, sh(1, m, 2048, 6144), fp16_w(),
                                     fp16_splitk());
        CHECK_ROUTE(p, kKsplitWide);
        p = select_matmul(dev, sh(1, m, 4096, 16384), fp16_w(), fp16_splitk());
        CHECK_ROUTE(p, kKsplitWide);
    }
    MatmulPlan p =
        select_matmul(dev, sh(1, 9, 2048, 6144), fp16_w(), fp16_splitk());
    CHECK(p.tiled);

    // int8: GEMV for m <= 8, tile from 9.
    const MatmulAvailability av8 = int8_modules();
    for (int m : {2, 3, 8}) {
        p = select_matmul(dev, sh(1, m, 2048, 151936), weight8(), av8);
        CHECK_ROUTE(p, kKsplitW8Wide);
    }
    p = select_matmul(dev, sh(1, 9, 2048, 6144), weight8(), av8);
    CHECK(p.tiled);

    // Byte WIDE floor 32: n = 8192 -> 512 16-col groups, q = 16 -> 32 blocks
    // (the generic floor 8 took q = 64 -> 8 blocks).
    p = select_matmul(dev, sh(1, 1, 2048, 8192), weight8(), av8);
    CHECK_ROUTE(p, kKsplitW8Wide);
    CHECK(p.ksplit_q == 16);
    p = select_matmul(dev, sh(1, 1, 2048, 32768), weight8(), av8);
    CHECK(p.ksplit_q == 64);
    // lm_head already clears 32 blocks at q = 256.
    p = select_matmul(dev, sh(1, 1, 2048, 151936), weight8(), av8);
    CHECK(p.ksplit_q == 256);

    // The env knob still overrides the profile.
    MatmulKnobs kn;
    kn.ksplit_floor = 8;
    p = select_matmul(dev, sh(1, 1, 2048, 8192), weight8(), av8, kn);
    CHECK(p.ksplit_q == 64);

    // Other classes keep floor 8.
    p = select_matmul(traits_native_discrete(), sh(1, 1, 2048, 8192),
                      weight8(), av8);
    CHECK(p.ksplit_q == 64);
}

// A bf16 weight holding |w| > 65504 (bf16_wide): every tile route stages B as
// half and would clamp it, so none of them may be picked on any device, shape
// or availability; the in-range bf16 weight still takes the tile.
void test_bf16_wide_routes() {
    const DeviceTraits devs[] = {traits_native_discrete(), traits_ada(),
                                 traits_mvk_apple(), traits_mvk_amd(),
                                 traits_mvk_intel()};
    const MatmulAvailability avs[] = {fp16_only(), fp16_splitk(),
                                      coop_modules()};
    int checked = 0;
    for (const auto &dev : devs) {
        for (const auto &av : avs) {
            for (int batch : {1, 3}) {
                for (int m : {1, 4, 8, 9, 16, 33, 64, 130, 256, 1024}) {
                    for (int k : {64, 256, 1024, 4096}) {
                        for (int n : {64, 96, 192, 2048, 6144, 151936}) {
                            for (bool tb : {false, true}) {
                                MatmulFeatures f = fp16_w(tb);
                                f.weight = WeightKind::kBf16;
                                f.bf16_wide = true;
                                MatmulPlan p = select_matmul(
                                    dev, sh(batch, m, k, n), f, av);
                                CHECK(!p.tiled);
                                CHECK(p.route != MatmulRoute::kTiled);
                                CHECK(p.route != MatmulRoute::kTiledKs);
                                CHECK(p.route != MatmulRoute::kCoopmat);
                                CHECK(p.ks_n == 0);
                                ++checked;
                            }
                        }
                    }
                }
            }
        }
    }
    CHECK(checked > 0);

    // Prefill on Apple: in range the tile (coop8) takes it; wide falls to the
    // column-parallel even-N kernel, which decodes bf16 to fp32.
    const DeviceTraits apple = traits_mvk_apple();
    MatmulFeatures in_range = fp16_w();
    in_range.weight = WeightKind::kBf16;
    MatmulPlan p = select_matmul(apple, sh(1, 64, 128, 64), in_range,
                                 fp16_splitk());
    CHECK(p.tiled);
    MatmulFeatures wide = in_range;
    wide.bf16_wide = true;
    p = select_matmul(apple, sh(1, 64, 128, 64), wide, fp16_splitk());
    CHECK_ROUTE(p, kFusedFp16);
    // Decode is untouched: the split-K GEMV already reads bf16 as fp32.
    p = select_matmul(apple, sh(1, 1, 2048, 6144), wide, fp16_splitk());
    CHECK_ROUTE(p, kKsplitWide);
    // The flag means nothing on a non-bf16 weight.
    MatmulFeatures fp16_flagged = fp16_w();
    fp16_flagged.bf16_wide = true;
    p = select_matmul(apple, sh(1, 64, 128, 64), fp16_flagged, fp16_splitk());
    CHECK(p.tiled);
}

int main() {
    test_profiles();
    test_bf16_wide_routes();
    test_apple_coop8_gate();
    test_mvk_apple_routes();
    test_decode_shapes_native();
    test_tiled_band_native();
    test_byte_weight_native();
    test_coopmat_gate();
    test_coopmat_shape_holes();
    test_other_routes();
    test_fallthrough_bits();
    test_dp4a_routes();
    test_mvk_routes();
    print_native_decode_table();

    if (g_fail != 0) {
        std::printf("\n%d check(s) failed\n", g_fail);
        return 1;
    }
    std::printf("\nall KernelSelect checks passed\n");
    return 0;
}
