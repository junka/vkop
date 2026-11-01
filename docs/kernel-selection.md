# Kernel selection: table-driven policy for MatMul

Status: **wired**. `core/KernelSelect.{hpp,cpp}` owns MatMul's routing policy;
`ops/Matmul.hpp` only builds the inputs, publishes module availability, and
dispatches the plan it gets back. `VulkanDevice` caches one `DeviceTraits` and
publishes nothing else: the per-driver calibration block it used to run is gone
(§4.4).

## 1. How the decision was made before the refactor

Kept as the reference for the rule table's numbers. Two layers, in two files,
with no shared data structure:

### 1.1 Device layer

`vulkan/VulkanDevice.cpp:135-205` runs three overlapping `if` blocks during
`getProperties()` and writes `matmul_tile_profile_`:

| match | byte_min_m | fp16_min_m | wide_n | byte_wide_min_m | fp16_wide_min_m | ks_target_blocks |
|---|---|---|---|---|---|---|
| MoltenVK + AMD (0x1002) | 9 | 4 | 65536 | 9 | 6 | 32 |
| MoltenVK + Intel (0x8086) | 9 | 9 | 0 | 2 | 3 | 32 |
| native driver + !UMA | 9 | 9 | 6144 | 9 | 7 | 128 |
| default (ARL / Ada) | 2 | 3 | 0 | 2 | 3 | 128 |

The struct is `VulkanDevice::MatmulTileProfile` (`vulkan/VulkanDevice.hpp:189`).
It only carries MatMul's tile-vs-GEMV crossovers (plus the split-K saturation
target); every other tuning number is a constant inside `Matmul.hpp`.

### 1.2 Op layer (removed)

`ops/Matmul.hpp` used to compute a chain of booleans from `(m, k, n, batch,
dtype, transB, epilogue flags, device profile, env)` and dispatch against them:

| old line | decision |
|---|---|
| 812 | `fused_fp16` (fp16 + even N) |
| 844 | `tiled_ok_n` (N % 64 or N % 8) |
| 900-931 | row-stride power-of-two override, wide band, `tiled_min_m` (also the 16x16x16 cache-resident exception) |
| 950 | `tiled` |
| 961-981 | tile-split-K depth `ks_n` |
| 996-1051 | `coopmat` gate (depends on `tiled`, `l2_resident`, built tile geometry) |
| 1070 | `coopmat_i8` (opt-in) |
| 1087-1105 | `ksplit_ok`, `w8_quad`, `w4_quad` |
| 1563-1839 | dispatch: tiled / tiled_ks / ksplit{,_wide,_512,_w8wide} / quad4 / fused / reduce |

Those predicates and the ~900 lines of calibration comments were interleaved
with scratch-buffer sizing, descriptor binding and two-dispatch sequences, so a
new device or kernel had to be validated by reading the whole region. The
predicates now live as rows in `core/KernelSelect.cpp`; the dispatch tail reads
`plan.route` and stays where it was.

### 1.3 What is *not* device-selected

Other ops use capability booleans directly (`is_support_cooperate_matrix()`,
`supportsSubgroupArithmetic()`, `is_unified_memory()`); they have no measured
crossover table today. That is correct as long as their route is a hardware
feature, and becomes a table row only when a *performance* threshold appears.

## 2. Target architecture

Three layers, one direction of dependency:

```
DeviceTraits  (facts)  ->  DeviceProfile (measured policy, ordered table)
                                     |
                        <Op>Rule[]   (ordered rows: kernel + predicate)
                                     |
                                  <Op>Plan (route + parameters)
```

- **DeviceTraits** — vendor/device/driver id, UMA, cooperative-matrix combos,
  subgroup arithmetic/size, buffer limits. No thresholds.
- **DeviceProfile** — one row per `DeviceClass`, keyed by an ordered matcher
  table (MoltenVK AMD, MoltenVK Intel, native discrete, generic). All numbers
  are benchmark results. `ksplit_floor`, the WIDE thresholds and the
  cache-resident byte budget are per-device fields now, not constants in the op.
- **Rule table** — one ordered array per op. Each row is
  `{route, tag, predicate, configure}`. The first predicate that holds wins;
  `configure` fills the route parameters (split-K q, tile-split depth, wide
  flag). The predicate sees `(traits, profile, shape, feature flags,
  availability, knobs)`, so device and shape conditions live in the same row.

The selector is pure: no `getenv`, no pipeline creation, no Vulkan calls.
Environment overrides are resolved once by `matmul_knobs_from_env()` and passed
in by value, which is what lets the unit test pin them.

Adding a thing:

- **device**: add a `ClassRule` matcher (or reuse one) and one `make_profile`
  row with the measured crossovers.
- **kernel**: add an enum value, a tag, an `applies` function and one row in
  priority order.
- **op**: add `select_<op>()` plus a rule array in the same file, reusing the
  device tables.

## 3. Contents

`core/KernelSelect.hpp`

- `DeviceTraits`, `DeviceClass`, `DeviceProfile`, `device_profile_for()`,
  `device_traits_from(VulkanDevice&)`
- `MatmulShape`, `MatmulFeatures`, `MatmulAvailability`, `MatmulKnobs`,
  `MatmulPlan`, `MatmulRoute`, `select_matmul()`

`core/KernelSelect.cpp`

- device class matchers and the profile table
- the MatMul rule array in dispatch priority order:
  `attn_softmax → coopmat_i8 → coopmat → down_act → tiled_ks → tiled →
  ksplit_w8wide512 → ksplit_w8wide → ksplit_wide512 → ksplit_wide →
  ksplit512 → ksplit → quad4 → fused_fp16 → reduce_pack → reduce`
- the same q-search / ks-depth arithmetic as the op
- `matmul_knobs_from_env()` for every `VKOP_*` override the op reads today

`tests/KernelSelectTest.cpp` — standalone, no GPU, no torch. Covers the four
profile rows, the four Qwen3-2B decode shapes for fp16 and int8, the tile
crossovers, the power-of-two stride override, the coopmat gate incl. the
cache-resident exception, WIDE fallback, 512-lane opt-in, down fold, attention
softmax, 4-bit and both reduce fallbacks.

## 4. Wiring (phase 2, done)

What the op does per `execute()` now:

1. `VulkanDevice` caches one `core::DeviceTraits` at construction, right after
   `createLogicalDevice()` settled the cooperative-matrix feature flag. MatMul
   never queries the physical device.
2. MatMul builds `MatmulShape` / `MatmulFeatures` / `MatmulAvailability` from
   values it already has (including the fold's `k_out` / `n_up` from the input
   shapes) and calls `select_matmul()`.
3. Modules that are built on first use (WIDE GEMV, 512-lane split-K,
   tile-split-K, the down fold) start optimistic. If the selected route needs
   one and the driver refuses it, that availability bit is cleared and the
   table is re-run (max 8 attempts) — the fall-through the old cascade wrote by
   hand. Modules built in `set_runtime_device()` (coopmat tiles) are reported
   as facts.
4. The dispatch tail keeps its old structure and reads the plan:
   `tiled` = `kTiled|kTiledKs`, `ksplit` = any `kKsplit*`, plus `w8_quad`,
   `w4_quad`, `para_.tile/wide/ksplit_q/attn_softmax`, `ks_n` and the down-fold
   gate. Nothing else in `Matmul.hpp` decides a route.

The env knobs are read once (`matmul_knobs_from_env()`, process-wide static in
the op) exactly like the old `VKOP_*` static lambdas.

### 4.1 Route readout

`VKOP_MATMUL_ROUTE=1` selects the tag in `Matmul::route_tag()`; `Runtime`
prints it for graph nodes, and both `benchmark/bench_matmul.cpp` and
`llm/exporter/driver/matmul_m_sweep.cpp` print `[route] <tag> batch=…`. The tag
set is unchanged from the pre-refactor names, so old logs compare directly.

### 4.2 Parity evidence (GTX 1070, validation layers off)

- `KernelSelectTest` passes (policy rows, crossovers, fold, attention, quant).
- `matmul_m_sweep` (transB=1, MoE projections) is unchanged:
  `m = 1, 8 → fused_fp16`, `m >= 32 → tiled`, for both `K=2048/N=1408` and
  `K=1408/N=2048`.
- `bench_matmul` decode shapes route `ksplit_wide` with the predicted q:
  `q/k/v/o (K=2048,N=2048) q=8`, `gate/up (N=6144) q=16`,
  `down (K=6144,N=2048) q=8`, `lm_head (N=151936) q=256`.
- Timings match the pre-refactor numbers within run-to-run noise:
  q/k/v/o 0.059-0.063 ms, gate/up 0.140-0.143 ms, down 0.136-0.138 ms,
  lm_head 2.80-2.82 ms (~220 GB/s), and the bench's exact-value probe still
  reports the expected `K * wpat(j)` columns.

### 4.3 Regression tests

`tests/KernelSelectTest.cpp` is the GPU-free policy test (`KernelSelectTest`
target). The op-level suites are in `all_tests`.

### 4.4 Legacy device table removed

`VulkanDevice::MatmulTileProfile`, its three `if` blocks in `getProperties()` and
`matmul_tile_profile()` are deleted. `VulkanDevice` publishes `device_traits()`
only, and logs the resolved profile once at construction; the policy numbers
live in `core/KernelSelect.cpp` alone, so tuning one no longer touches the
Vulkan layer. The measurements behind those numbers stay in
`benchmark/README.md`.

## 5. Small-shape optimization (next)

Measurements on the GTX 1070 (256-bit GDDR5, 4004 MHz → 256 GB/s theoretical),
**validation layers off** — the previous build had `USE_VALIDATION_LAYERS=ON`,
which alone accounted for 30-50% of the small-shape time:

| probe | with validation | without |
|---|---|---|
| empty `vkQueueSubmit` | 0.069 ms | 0.022 ms |
| q/k/v/o 2048x2048 (8 MB) | 0.086-0.096 ms (87-97 GB/s) | 0.062-0.066 ms (128-136 GB/s) |
| gate/up 2048x6144 (24 MB) | 0.165-0.185 ms | 0.137-0.158 ms (159-183 GB/s) |
| down 6144x2048 (24 MB) | 0.160 ms | 0.136-0.155 ms (162-186 GB/s) |
| lm_head 2048x151936 (622 MB) | 2.79 ms | 2.79 ms (223 GB/s) |
| 1 GB GEMV | — | 211-224 GB/s |

Interpretation:

- The read ceiling is ~224 GB/s (87% of theoretical). `lm_head` is already on
  it. There is **no bandwidth headroom left for the large shapes**.
- The 8 MB shape measures 62 µs wall, of which ~22 µs is the harness's own
  per-submit cost; the kernel itself is ~40 µs ≈ 200 GB/s. So the small-shape
  gap is mostly *host submit*, not the kernel. The production graph already
  batches levels into one `vkQueueSubmit` (`core/runtime.cpp:2735-2810`), so the
  real decode path does not pay 22 µs per GEMV.
- Remaining kernel-side headroom is in the 24 MB shapes (160-183 GB/s vs 224)
  and in the block count of the 8 MB shape (n=2048 picks q=8 → 32 blocks; a
  q=4 / 64-block arm is the obvious A/B).

Concrete follow-ups, each a new profile/knob value rather than new `if`s:

1. Add a GPU-timestamp mode to `bench_matmul` (`VK_QUERY_TYPE_TIMESTAMP` around
   the replayed dispatch) so submit cost and kernel time are reported
   separately. All further tuning needs that split.
2. Sweep `ksplit_q` on the 8/24 MB shapes for the `native-discrete` profile;
   pin the winning q per shape band (n < 2048, 2048-8192, > 8192) as profile
   fields.
3. Re-check `wide_threshold_fp16 = 2048` on this device: n=2048 is exactly at
   the boundary, and the standard 4-column kernel reads 128-bit per lane pair.
4. Consider one `vkQueueSubmit` per decode round in the benchmark harness (the
   runtime already does it) before attributing small-shape time to the kernel.
5. Only after 1-4, revisit the kernel: the 24 MB cases are the only ones with
   >10% bandwidth headroom.

## 6. Build note

`build/` was configured with `USE_VALIDATION_LAYERS=ON`; performance work
reconfigured it to `OFF`. Restore it with
`cmake -S . -B build -DUSE_VALIDATION_LAYERS=ON` when validation coverage is
wanted again.
