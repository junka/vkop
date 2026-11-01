// Copyright 2025 @junka
#ifndef CORE_KERNEL_SELECT_HPP_
#define CORE_KERNEL_SELECT_HPP_

#include <cstdint>

namespace vkop {
class VulkanDevice;

namespace core {

// ---------------------------------------------------------------------------
// Table-driven kernel routing.
//
// MatMul's device-and-shape decision used to be an if-chain in its own header
// (~300 lines of predicates plus ~900 lines of calibration comments), while the
// device half of the same decision lived in VulkanDevice.cpp as three
// overlapping if-blocks writing a 6-field struct. The two halves could not be
// reviewed together, and a new device meant editing at least two files.
//
// This file splits the decision into three layers:
//
//   1. DeviceTraits   - raw facts, no policy (vendor/driver id, UMA, the
//                       cooperative-matrix combos the device lists, subgroup
//                       arithmetic, buffer limits).
//   2. DeviceProfile  - measured policy for a *class* of device: the tile-vs-
//                       GEMV crossovers, the cache-resident weight budget, the
//                       split-K occupancy targets. One ordered table, first
//                       match wins. VulkanDevice only publishes DeviceTraits.
//   3. <Op>Rule[]     - one ordered array per op. Each row is
//                       {kernel, tag, predicate, configure}; the first row whose
//                       predicate holds is the route. The predicate sees the
//                       device profile, the shape and the op's feature flags;
//                       the configure hook fills route parameters (split-K q,
//                       tile-split-K depth, ...). Adding a kernel is adding a
//                       row; adding a device is adding a profile row.
//
// MatMul is the first (and currently only) op with a real device table, so it
// is the first rule array. select_matmul() is deliberately pure: no getenv, no
// pipeline construction, no Vulkan calls. matmul_knobs_from_env() is the only
// place that parses the VKOP_* overrides (the op reads the cached
// matmul_knobs()), so a test can pin them by value.
//
// ops/Matmul.hpp consumes this table: it builds the shape/feature structs,
// publishes the modules it has built, and maps the plan's route to its existing
// dispatch. A route whose module the driver refuses clears that availability
// bit and re-runs the table, which is the fall-through the old inline cascade
// wrote by hand. The unit test (tests/KernelSelectTest.cpp) pins the routes;
// VKOP_MATMUL_ROUTE=1 on bench_matmul / matmul_m_sweep prints them live.
// ---------------------------------------------------------------------------

// Raw device facts. driver_id is a VkDriverId value; it stays an integer here so
// this header does not pull in vulkan_core.h.
struct DeviceTraits {
    uint32_t vendor_id = 0;
    uint32_t device_id = 0;
    uint32_t driver_id = 0;
    bool unified_memory = false;
    // Usable, not merely advertised: MoltenVK sets the feature bool but
    // reports 0 supported stages, and a pipeline built from that combo fails.
    bool coopmat = false;
    bool coopmat_fp16_8x8x16 = false;
    bool coopmat_fp16_16x16x16 = false;
    bool coopmat_sint8_8x8x32 = false;
    bool subgroup_arithmetic = false;
    uint32_t subgroup_size = 32;
    uint64_t max_storage_buffer_range = 0;
};

// Coarse device classes the tables key on. Keep this list about *behaviour*,
// not vendors: two vendors behind the same driver stack can share a class when
// their measured crossovers agree.
enum class DeviceClass {
    kGeneric = 0,     // native iGPU / unknown: the ARL / Ada calibration
    kMoltenVkAmd,     // AMD dGPU behind MoltenVK
    kMoltenVkIntel,   // Intel iGPU behind MoltenVK
    kNativeDiscrete,  // native driver + VRAM-backed dGPU (GTX 1070 class)
    kCount
};

const char *device_class_name(DeviceClass cls);

// Measured policy for one device class. Every field is a number that a benchmark
// produced, not a hardware query.
struct DeviceProfile {
    const char *name = "generic";
    DeviceClass device_class = DeviceClass::kGeneric;

    // MatMul tile-vs-split-K-GEMV crossover, in output rows m. The shared-memory
    // tile runs a fixed 64-row block, so its cost is flat in m while the split-K
    // GEMV pays one weight pass per row; the crossing point is a property of the
    // memory system, not of the shape.
    int byte_min_m = 2;
    int fp16_min_m = 3;
    // A second band for weights whose row pass is already DRAM-bound (large N).
    // 0 disables the band.
    int wide_n = 0;
    int byte_wide_min_m = 2;
    int fp16_wide_min_m = 3;
    // Workgroups at which the tiled GEMM saturates; tile-split-K targets this.
    int ks_target_blocks = 128;

    // Split-K GEMV occupancy floor: the minimum block count that makes splitting
    // K worth it (fewer, longer K chunks beat more, shorter ones once the device
    // is fed). Was hardcoded in Matmul.hpp before the table existed.
    int ksplit_floor = 8;
    // Per-N auto-enable threshold for the WIDE (8 columns/lane fp16, 16
    // columns/lane byte) GEMV variants.
    int wide_threshold_fp16 = 2048;
    int wide_threshold_byte = 896;
    // Weight bytes (K * row bytes) that still fit the cache the GEMV re-reads
    // through. Above this the tile's single weight pass wins earlier.
    uint64_t cache_resident_bytes = 32ull << 20;
};

// First matching row wins; the last row is the generic default.
const DeviceProfile &device_profile_for(const DeviceTraits &traits);

// Narrows a VulkanDevice to the facts above. This is the only bridge between
// the Vulkan objects and the table, and it belongs at device-init time: cache
// the result, do not call it per dispatch.
DeviceTraits device_traits_from(VulkanDevice &dev);

// ---------------------------------------------------------------------------
// MatMul routing
// ---------------------------------------------------------------------------

// One route per kernel module (or module pair) the op can dispatch. The tags
// match the strings MatMul::set_route() publishes, so a parity test can compare
// a plan against VKOP_MATMUL_ROUTE=1 output unchanged.
enum class MatmulRoute {
    kReduce = 0,
    kReducePack,
    kFusedFp16,
    kQuad4,
    kKsplit,
    kKsplit512,
    kKsplitWide,
    kKsplitWide512,
    kKsplitW8Wide,
    kKsplitW8Wide512,
    kTiled,
    kTiledKs,
    kCoopmat,
    kCoopmatI8,
    kDownActFold,
    kAttnSoftmax,
};

const char *matmul_route_tag(MatmulRoute route);

enum class WeightKind {
    kFp16 = 0,
    kInt8,
    kFp8,
    kInt4,
    kNvfp4,
};

struct MatmulShape {
    int batch = 1;
    int m = 1;
    int k = 0;
    int n = 0;
    // Up-projection width, i.e. the down fold's contraction. Only read when the
    // down_act feature is set; -1 means "not a fold".
    int n_up = -1;
};

struct MatmulFeatures {
    bool fp16 = true;
    bool transB = false;
    WeightKind weight = WeightKind::kFp16;
    int group_size = 0;
    // Attention QK^T epilogue: the converter folded scale/bias into this MatMul.
    bool epi_bias = false;
    // The trailing Softmax can be fused into the QK^T kernel.
    bool attn_softmax = false;
    // SwiGLU down-projection fold (x, Wd, Wg, Wu).
    bool down_act = false;
};

// What the op actually managed to build. A route whose module is missing (driver
// refusal, archive without the key) must fall through, so availability is an
// input to selection rather than something the selector discovers.
struct MatmulAvailability {
    bool coopmat = false;
    int coop_bm = 0;  // built fp16 coop tile's workgroup M (0 = none)
    int coop_bn = 0;
    bool coopmat_i8 = false;
    bool ksplit512 = false;
    bool w8_wide = false;
    bool w8_wide512 = false;
    bool tile_ks_split = false;
    bool tile_ks_fold = false;
    bool down_act = false;
};

// Env-backed tuning knobs, resolved once. Defaults are the shipped values; see
// matmul_knobs_from_env() for the VKOP_* spelling of each.
struct MatmulKnobs {
    int ksplit_floor = 8;
    // VKOP_KSPLIT_Q: pin the column quads / 8-column groups per block. 0 = auto.
    int ksplit_q = 0;
    // VKOP_KSPLIT_WIDE: -1 force off, 0 auto (per-N threshold), +1 force on.
    int wide_force = 0;
    // 0 = take the device profile's threshold.
    int wide_threshold_fp16 = 0;
    int wide_threshold_byte = 0;
    // VKOP_KSPLIT512: opt into the 512-lane split-K builds.
    bool ksplit512_enabled = false;
    // VKOP_KS_N / VKOP_KS_TARGET: pin tile-split-K depth / saturation target.
    int ks_n_pin = 0;
    int ks_target_pin = 0;
    // VKOP_NO_TILE / VKOP_NO_COOP / VKOP_COOP_I8.
    bool tile_disabled = false;
    bool coop_disabled = false;
    bool coop_i8_enabled = false;
};

MatmulKnobs matmul_knobs_from_env();

// Process-wide knobs, resolved once from the environment on first use. Ops read
// this instead of re-parsing VKOP_* themselves: a flag cannot then mean one
// thing to the selector and another to the code that builds the modules behind
// it.
const MatmulKnobs &matmul_knobs();

struct MatmulPlan {
    MatmulRoute route = MatmulRoute::kReduce;
    const char *tag = "reduce";
    // Tile-split-K depth. 0/1 means "no K split".
    int ks_n = 0;
    // Split-K GEMV columns per block (quads, 8-column groups or 16-column
    // groups depending on the route).
    int ksplit_q = 16;
    bool wide = false;
    // Diagnostics: whether the tile claimed the shape, and the crossover that
    // decided it. Kept on the plan so a route log can explain a flip without
    // re-running the selector.
    bool tiled = false;
    int tiled_min_m = 0;
    const DeviceProfile *device = nullptr;
};

// Pure: the same inputs always produce the same plan. Pipeline availability and
// env overrides are explicit inputs.
MatmulPlan select_matmul(const DeviceTraits &traits, const MatmulShape &shape,
                         const MatmulFeatures &features,
                         const MatmulAvailability &availability,
                         const MatmulKnobs &knobs = {});

} // namespace core
} // namespace vkop

#endif // CORE_KERNEL_SELECT_HPP_
