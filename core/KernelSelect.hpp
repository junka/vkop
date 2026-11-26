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
    // Shader-level features that are live on the logical device (enabled, not
    // merely advertised): fp32 atomic add on a storage buffer / in shared
    // memory (VK_EXT_shader_atomic_float), half2 atomic add
    // (VK_NV_shader_atomic_float16_vector, only meaningful with
    // shader_float16), 8-bit storage loads (VK_KHR_8bit_storage),
    // subgroupRotate, explicit workgroup-memory layout, zero-initialized
    // workgroup memory. Nothing in kMatmulRules keys on these yet — the
    // VKOP_DEVCAPS=1 dump in vulkan/VulkanDevice.cpp exists to verify the probe
    // on hardware this box cannot represent before a kernel depends on it.
    bool f32_global_atomic_add = false;
    bool f32_shared_atomic_add = false;
    bool f16x2_global_atomic_add = false;
    bool shader_float16 = false;
    bool storage_buffer_8bit = false;
    bool subgroup_rotate = false;
    bool wmem_explicit_layout = false;
    bool zero_init_wmem = false;
    // VK_EXT_shader_float8: `shader_float8` turns the fp8 weight decode in
    // matmul.comp from a bits reassembly into one hardware conversion, which
    // needs its own module (OpCapability Float8EXT is module-wide, so a device
    // without the extension rejects the whole module at pipeline creation).
    // `shader_float8_coopmat` is the fp8 MMA half and is tracked separately --
    // a device may decode fp8 without an fp8 cooperative matrix.
    bool shader_float8 = false;
    bool shader_float8_coopmat = false;
    // VK_KHR_shader_integer_dot_product is ENABLED and the packed 4x8 signed
    // form is ACCELERATED. Both halves matter: the feature bit is what lets a
    // module carry `OpCapability DotProduct` (without it vkCreatePipeline
    // rejects the module), and the property is what says `dotPacked4x8EXT` is
    // one instruction rather than an emulation. This device has the feature
    // and only the packed form accelerated (8BitSignedAccelerated is false),
    // so the i8vec4 spelling must never be assumed from this flag alone.
    bool integer_dot_product_packed = false;
};

// Coarse device classes the tables key on. Keep this list about *behaviour*,
// not vendors: two vendors behind the same driver stack can share a class when
// their measured crossovers agree.
enum class DeviceClass {
    kGeneric = 0,     // native iGPU / unknown: the ARL / Ada calibration
    kMoltenVkAmd,     // AMD dGPU behind MoltenVK
    kMoltenVkIntel,   // Intel iGPU behind MoltenVK
    kNativeDiscrete,  // native driver + VRAM-backed dGPU (GTX 1070 class)
    kMoltenVkApple,   // Apple GPU behind MoltenVK (M-series, unified memory)
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
    // Same, for the 4-bit WIDE GEMV (matmul16_w4_ksplit_wide). 0 = use
    // wide_threshold_fp16. Its own knob because below the fp16 threshold the
    // 4-bit alternative is matmul16_w4_ksplit, which splits K by whole scale
    // groups over only 16 lanes -- a different trade than fp16's.
    int wide_threshold_w4 = 0;
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

// The 8x8x8 coopmat kernels (matmul_coop8 / attention_flash_coop8 /
// conv_coop8) are an off-spec Apple path: MoltenVK does not list
// VK_KHR_cooperative_matrix, but SPIRV-Cross lowers an 8x8 coopmat to Metal
// simdgroup_matrix. That lowering is an Apple-silicon capability - the
// MoltenVK builds running on AMD/Intel Macs reject the pipeline with
// VK_ERROR_INITIALIZATION_FAILED. Gate on driver and vendor together so those
// machines go straight to the SIMT kernels instead of paying a failed
// vkCreateComputePipelines (and printing a rejection) on first use.
// Calibrated on an M5 Max; the capability is Apple7+, so it is deliberately
// not pinned to one M-series SKU.
bool device_supports_apple_coop8(const DeviceTraits &traits);

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
    // 4-bit weight, eight columns per lane: one uint32 per k holds the lane's
    // whole octet, so the weight load is a quarter of the fp16 WIDE form's
    // bytes for the same column count. A separate route rather than a flag on
    // kKsplitWide because the launch geometry differs (8 columns per lane vs
    // the int8 WIDE kernels' 16) and the pipeline must be the -DW8_WIDE
    // module: the general module has no such kernel, and routing 4-bit there
    // silently ran the 4-column matmul16_w4_ksplit with a quarter of the
    // needed grid.
    kKsplitW4Wide,
    kKsplitW8Wide,
    kKsplitW8Wide512,
    // int8 x int8 split-K GEMV whose inner loop is the packed 4x8 integer dot
    // product (`dotPacked4x8EXT`): four weight bytes and four activation bytes
    // per instruction, accumulated in int32. Opt-in via VKOP_DP4A because the
    // kernel only pays off when the weight stream is the bottleneck; see the
    // rule row in core/KernelSelect.cpp. The int32 accumulator is exact only
    // while K * 128 * 128 < 2^31 (K <= 131071), and the gate refuses past that
    // -- note the commonly quoted K < 133152 bound is itself wrong, see
    // kDp4aMaxK.
    kKsplitW8A8Dp4a,
    kTiled,
    kTiledKs,
    kCoopmat,
    kCoopmatSplitK,
    kCoopmatInwg,
    // coopmat + split-K with grid.z = batch*SK (true parallel K split, not
    // workgroup-internal serial passes). One dispatch, each slice writes fp32
    // partials to a global buffer; the tail block reduces and writes half2
    // directly. Boundary conditions on the partial buffer / atomic counter
    // are resolved by the shader (matmul_coop*.comp -DSK_LASTBLOCK).
    // Host-side guard (not yet active while shader lands): SK ∈ [2, 8],
    // K/SK >= 2*K_BLOCK (= 32), N even, ACC_M=ACC_N=1 only.
    kCoopmatSkLast,
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
    // A bfloat16 weight. Same storage as fp16 (two 16-bit values per word) and
    // no scale, so it shares every fp16 GEMV / tile / reduce route and its
    // gates (the shader swaps the B unpack, uPC.bf16). The rows that test
    // `weight == kFp16` exclude it on purpose: the KHR cooperative-matrix
    // modules (matmul_coop*.comp other than the Apple coop8) and the fused
    // softmax have no bf16 read, and the down_act fold row refuses it
    // explicitly. The Apple coop8 GEMM is reached through the tiled route and
    // does convert a bf16 B in its shared-memory stage.
    kBf16,
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
    // True when A is an int8 tensor (the activation is quantized, not the
    // value the kernel dequantizes). Only the W8A8 DP4A route consumes it;
    // every other route requires a float A, and core::require_float_elem
    // still enforces that for them (see ops/Matmul.hpp).
    bool a_int8 = false;
    int group_size = 0;
    // Attention QK^T epilogue: the converter folded scale/bias into this MatMul.
    bool epi_bias = false;
    // The trailing Softmax can be fused into the QK^T kernel.
    bool attn_softmax = false;
    // SwiGLU down-projection fold (x, Wd, Wg, Wu).
    bool down_act = false;
    // SwiGLU gate/up fold (fuse_swiglu_gate_up): B is Wg/Wu interleaved four
    // columns at a time, [K, 2*N_up], and the result is silu(x.Wg) * (x.Wu).
    // The fp16 WIDE split-K GEMV and the Apple 8x8x8 coopmat GEMM apply the
    // activation in their own epilogue (uPC.wide == 2). Every route that
    // writes fp32 to the flat scratch buffer -- including the generic
    // cooperative-matrix GEMM -- gets it from the pack pass instead
    // (matmul_pack.comp's swiglu mode), so selection does not exclude it.
    bool swiglu = false;
    // A bf16 weight (weight == kBf16) that holds at least one element past
    // fp16's range (|w| > 65504, inf or NaN): the converter's bit scan sets the
    // node attribute bf16_wide=1 (onnx2vkop/bf16_weights.py), and LoadModel
    // sets it too for a file written without the attribute. The tile routes
    // (kTiled, kTiledKs and the Apple coop8 GEMM reached through kTiled) stage
    // B as half, which would clamp those elements, so they reject it; the GEMV
    // and column-parallel routes decode bf16 straight to fp32 and take it.
    // Ignored for every other weight kind.
    bool bf16_wide = false;
};

// What the op actually managed to build. A route whose module is missing (driver
// refusal, archive without the key) must fall through, so availability is an
// input to selection rather than something the selector discovers.
struct MatmulAvailability {
    bool coopmat = false;
    int coop_bm = 0;  // built fp16 coop tile's workgroup M (0 = none)
    int coop_bn = 0;
    // At least one reachable fp16 coop module packs half2 straight into the
    // output instead of writing flat fp32 to scratch for a following pack
    // pass (matmul_coop.comp's DEFAULT build vs coop16.comp's). That epilogue
    // forms each word from (col, col+1) of one row, so it is only correct for
    // EVEN N: with odd N every other word straddles two rows. The
    // scratch+pack family has no such constraint (the pack pass sees a flat
    // element stream). Set by the op from what it actually probed.
    bool coop_self_pack = false;
    bool coopmat_i8 = false;
    bool ksplit512 = false;
    bool w8_wide = false;
    bool w8_wide512 = false;
    // The -DW8_WIDE module built and was accepted by the driver. The 4-bit
    // WIDE GEMV lives in that same module (its main() dispatches on uPC.w4),
    // so w8_wide covers it -- kept separate anyway so a future split of the
    // module does not silently disable one of the two.
    bool w4_wide = false;
    // The DP4A module probed and built (op side). Its route ALSO needs
    // DeviceTraits::integer_dot_product_packed, because the pipeline cannot
    // even be created without the feature.
    bool w8a8_dp4a = false;
    bool tile_ks_split = false;
    bool tile_ks_fold = false;
    bool down_act = false;
};

// Env-backed tuning knobs, resolved once. Defaults are the shipped values; see
// matmul_knobs_from_env() for the VKOP_* spelling of each.
struct MatmulKnobs {
    // VKOP_KSPLIT_FLOOR: the byte WIDE / DP4A GEMV block floor. 0 = take the
    // device profile's ksplit_floor.
    int ksplit_floor = 0;
    // VKOP_KSPLIT_Q: pin the column quads / 8-column groups per block. 0 = auto.
    int ksplit_q = 0;
    // VKOP_KSPLIT_WIDE: -1 force off, 0 auto (per-N threshold), +1 force on.
    int wide_force = 0;
    // 0 = take the device profile's threshold.
    int wide_threshold_fp16 = 0;
    int wide_threshold_byte = 0;
    int wide_threshold_w4 = 0;
    // VKOP_KSPLIT512: opt into the 512-lane split-K builds.
    bool ksplit512_enabled = false;
    // VKOP_DP4A: opt into the int8 x int8 packed-dot-product GEMV. Default
    // OFF: the kernel is a bandwidth play, and on a device whose weights fit
    // VRAM it loses to the weight-only byte GEMV (which reads the same bytes
    // through a wider per-lane load and needs no activation quantization).
    // It exists for the regime where a few GB of weights spill out of VRAM:
    // there the launch is PCIe-limited rather than issue-limited, so halving
    // the fetched weight stream is what matters.
    bool dp4a_enabled = false;
    // VKOP_KS_N / VKOP_KS_TARGET: pin tile-split-K depth / saturation target.
    int ks_n_pin = 0;
    int ks_target_pin = 0;
    // VKOP_NO_TILE / VKOP_NO_COOP / VKOP_COOP_I8.
    bool tile_disabled = false;
    bool coop_disabled = false;
    bool coop_i8_enabled = false;
    // VKOP_FORCE_COOP: opt-in bypass of the coopmat_gate M threshold. Forces
    // coopmat even for M=1 shapes where it would otherwise be excluded as a
    // "waste of MMA lanes." Intended for PoC — DO NOT set in production.
    bool force_coop = false;
    // VKOP_COOP_SPLITK=1: opt-in coopmat + split-K hybrid kernel. When set,
    // the selector may route fp16 coopmat shapes through kCoopmatSplitK,
    // which dispatches multiple K slices of the coopmat main pass + one
    // reduce pass — no standalone pack pass.
    // VKOP_COOP_SPLITK_FORCE=<ks_n>: pin the split-K depth (>=2). 0 = auto
    // (selector picks based on occupancy, like ks_depth_for for tiled_ks).
    int coop_split = 0;
    // VKOP_KERNEL_DEBUG=1: trace every rule walk to stderr. Off by default
    // because selection runs once per MatMul per round -- at 142 MatMuls and
    // ~20 candidate rules that was >2000 lines per decode step, which drowned
    // the server's own request/prefill lines out of the log. It lives in the
    // knobs (not a getenv inside select_matmul) so the selector stays pure and
    // a test can pin it by value, like every other switch here.
    bool trace = false;
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
    // coopmat-split-K depth (opt-in via VKOP_COOP_SPLITK*). 0/1 means
    // "coopmat main pass only, no K split". Also used for coopmat_sklast
    // (grid.z parallel K split) — same field, same dispatch path.
    int coop_split = 0;
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
    // Why this route/BM/BN/cs was chosen. Points to a static string literal
    // produced by coopmat_select_smart() — pure function, no allocation.
    const char *route_reason = nullptr;
    // Workgroup tile chosen by the smart selector. 0 means "use the probed
    // default from coopmat availability". ops/MatMul dispatches coopmat
    // routes with these dimensions when set.
    int bm = 0;
    int bn = 0;
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
