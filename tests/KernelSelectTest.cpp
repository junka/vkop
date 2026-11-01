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

MatmulShape sh(int batch, int m, int k, int n) {
    MatmulShape s;
    s.batch = batch;
    s.m = m;
    s.k = k;
    s.n = n;
    return s;
}

void test_profiles() {
    const DeviceProfile &native =
        device_profile_for(traits_native_discrete());
    CHECK(native.device_class == DeviceClass::kNativeDiscrete);
    CHECK(native.byte_min_m == 9);
    CHECK(native.fp16_min_m == 9);
    CHECK(native.wide_n == 6144);
    CHECK(native.fp16_wide_min_m == 7);
    CHECK(native.ks_target_blocks == 128);
    CHECK(native.ksplit_floor == 8);
    CHECK(native.wide_threshold_fp16 == 2048);
    CHECK(native.wide_threshold_byte == 896);

    const DeviceProfile &amd = device_profile_for(traits_mvk_amd());
    CHECK(amd.device_class == DeviceClass::kMoltenVkAmd);
    CHECK(amd.byte_min_m == 9);
    CHECK(amd.fp16_min_m == 4);
    CHECK(amd.wide_n == 65536);
    CHECK(amd.fp16_wide_min_m == 6);
    CHECK(amd.ks_target_blocks == 32);

    const DeviceProfile &intel = device_profile_for(traits_mvk_intel());
    CHECK(intel.device_class == DeviceClass::kMoltenVkIntel);
    CHECK(intel.byte_min_m == 9);
    CHECK(intel.fp16_min_m == 9);
    CHECK(intel.wide_n == 0);
    CHECK(intel.ks_target_blocks == 32);

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

    // 4-bit is excluded from tile and coop; the split-K window still claims
    // decode (matmul.comp dequantizes in the ksplit entry too).
    f = fp16_w();
    f.weight = WeightKind::kInt4;
    CHECK_ROUTE(select_matmul(dev, sh(1, 1, 512, 512), f, av), kKsplit);
    // Past the split-K window only the four-column quad kernel can read a
    // packed 4-bit word.
    CHECK_ROUTE(select_matmul(dev, sh(1, 9, 512, 512), f, av), kQuad4);
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

int main() {
    test_profiles();
    test_decode_shapes_native();
    test_tiled_band_native();
    test_byte_weight_native();
    test_coopmat_gate();
    test_other_routes();
    test_fallthrough_bits();
    test_mvk_routes();
    print_native_decode_table();

    if (g_fail != 0) {
        std::printf("\n%d check(s) failed\n", g_fail);
        return 1;
    }
    std::printf("\nall KernelSelect checks passed\n");
    return 0;
}
