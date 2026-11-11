# Running a second Vulkan SDK side by side (for float8)

This note covers preparing a Vulkan toolchain newer than the 1.3.275 stack
that both the dev box and `b3` ship, without disturbing it, and using that
toolchain to exercise `VK_EXT_shader_float8`.

## What is actually missing

The driver is not the problem. NVIDIA 595.99.02 on the RTX 4000 Ada reports:

```
VK_EXT_shader_float8   extension revision 1
VkPhysicalDeviceShaderFloat8FeaturesEXT:
    shaderFloat8                  = true
    shaderFloat8CooperativeMatrix = true
VkCooperativeMatrixPropertiesKHR:
    fp8 config 16x16x32  A=E4M3 B=E4M3  C=fp16 / fp32
```

`vulkaninfo --text` against the SDK loader shows the same thing. What the
1.3.275 install lacks is three pieces of the *build and load* stack:

| piece | 1.3.275 (system) | 1.4.363.1 (SDK) |
|---|---|---|
| headers (`vulkan_core.h`) | no `VK_VERSION_1_4`, no `VK_EXT_shader_float8` | both present |
| loader (`libvulkan.so.1`) | 1.3.275, caps the instance at 1.3 | 1.4.363, instance can be 1.4 |
| shader compiler (`glslc`) | shaderc 2023.8 — rejects `GL_EXT_float_e4m3` | shaderc 2026.4 — accepts it |

**Version naming trap**: `1.4.317` is a *Vulkan-Headers* git tag, not a LunarG
SDK release. `https://vulkan.lunarg.com/sdk/versions.json` lists SDKs; the
lowest 1.4 release is `1.4.321.0` and the current latest is `1.4.363.1`.
Anything `>= 1.4.321.0` satisfies "1.4.317 or newer" for this purpose.

## Install (one command)

```bash
tools/float8_probe/setup_sdk.sh 1.4.363.1 ~/vulkan-sdk
```

It downloads the SDK tarball, extracts it under `~/vulkan-sdk/<version>/`,
compiles the two probe shaders, asserts the resulting SPIR-V really carries
`OpCapability Float8EXT` / `Float8CooperativeMatrixEXT`, and links the probe
against the SDK headers and loader. Nothing is installed system-wide.

On `b3` the SDK is already at `~/vulkan-sdk/1.4.363.1`; re-running with
`SKIP_DOWNLOAD=1` only rebuilds the probe.

### Why not just set `VULKAN_SDK` and rebuild vkop

The vkop CMake already honours `VULKAN_SDK` and `Vulkan_INCLUDE_DIRS`, so a
second build tree works:

```bash
cmake -S . -B /tmp/build_vk14 \
      -DVulkan_INCLUDE_DIRS=~/vulkan-sdk/1.4.363.1/x86_64/include \
      -DGLSLC_EXECUTABLE=~/vulkan-sdk/1.4.363.1/x86_64/bin/glslc \
      -DYOLO_USE_FREETYPE=OFF -DYOLO_USE_GLFW=OFF
cmake --build /tmp/build_vk14 -j
```

That build compiles cleanly (verified: all of `vulkan/`, `core/`, `ops/` and
the 103 standalone shaders). Do it as a *second* build directory; the default
`build/` stays on 1.3.275.

Two gotchas when pointing at the SDK:

- Use `Vulkan_INCLUDE_DIRS`, not `VULKAN_SDK` alone. `find_package(Vulkan)`
  finds the system `libvulkan.so` either way, so the *loader* at link time is
  still 1.3.275 unless you also put the SDK loader on `LD_LIBRARY_PATH`.
- The SDK ships `vk_video/` as a sibling of `include/vulkan/`, which
  `vulkan_core.h` includes as `"vk_video/..."`. Copying only `include/vulkan`
  into a scratch prefix fails; keep the SDK's `include/` root.

### Running a 1.4-built binary on a 1.3 loader

`VulkanInstance::createInstance` clamps the requested apiVersion to the
headers this build was compiled against *and* to what
`vkEnumerateInstanceVersion` reports. The consequence is intentional and
visible in the probe output:

| binary built with | loader on `LD_LIBRARY_PATH` | instance apiVersion |
|---|---|---|
| 1.3.275 headers | system 1.3.275 | 1.3.275 |
| 1.4.363 headers | system 1.3.275 | **1.3.275** (clamped by loader) |
| 1.4.363 headers | SDK 1.4.363 loader | **1.4.0** |

So a 1.4-built binary on the 1.3 loader still runs, but the feature gating in
`VulkanDevice::createFeatureDescriptors` takes the EXT path (subgroup rotate,
host image copy) instead of the 1.4-core structs. Put the SDK loader first
when you want the 1.4 paths exercised:

```bash
LD_LIBRARY_PATH=~/vulkan-sdk/1.4.363.1/x86_64/lib/VulkanLoader/lib ./your_binary
```

`VulkanInstance` creates the instance at `VK_MAKE_API_VERSION(0,1,4,0)` when
the loader supports it, and the existing `min(instance, device)` gating in
`VulkanDevice.cpp` then selects the `VkPhysicalDeviceVulkan14Features`
descriptor. On `b3` that moves `subgroup_rotate` from "not queried" (1.3
build) to `driver=yes device=yes`.

**This table does not apply to float8.** `VK_EXT_shader_float8` is an
extension, not one of the features promoted into the 1.4 core struct, so it is
enabled whenever the driver advertises it — the instance apiVersion is
irrelevant. Verified on `b3`: a 1.4-header build reports
`shader_float8 driver=yes device=yes` under both the system 1.3.275 loader and
the SDK's 1.4.363 one, and the fp8 hardware decode runs at the same speed in
both. What gates float8 is the *header* version at build time (the
`#ifdef VK_EXT_shader_float8` block compiles out against 1.3.275 headers) and
the glslc version, not the loader.


## float8 probe

```bash
LD_LIBRARY_PATH=~/vulkan-sdk/1.4.363.1/x86_64/lib/VulkanLoader/lib \
    /tmp/f8build/f8_probe /tmp/f8build/f8_storage.spv /tmp/f8build/f8_coopmat.spv
```

Two independent checks, because a device can support one without the other:

- **storage** (`f8_storage.comp`): `float -> floate4m3_t -> float` round trip
  through a storage buffer. Needs only `shaderFloat8`; covers
  `OpTypeFloat 8 Float8E4M3EXT` and the conversion constructors.
- **coopmat** (`f8_coopmat.comp`): one 16x16x32 E4M3 x E4M3 -> FP32 MMA.
  Needs `shaderFloat8CooperativeMatrix` too, and the exact tile must appear in
  `VkCooperativeMatrixPropertiesKHR`.

Both compare against a host-side integer reference and require *exact*
equality (all probe inputs are exactly representable in E4M3, so any error is
a real miscompute, not rounding). Expected output on Ada:

```
[f8probe] features: shaderFloat8=1 shaderFloat8CooperativeMatrix=1 cooperativeMatrix=1
[f8probe] coopmat fp8 config: 16x16x32 A=1000491002 B=1000491002 C=1 R=1 scope=3
[f8probe] storage RESULT PASS: 0/256 bad, maxerr=0
[f8probe] coopmat RESULT PASS: 0/256 bad, maxerr=0
[f8probe] OVERALL PASS
```

On the Intel ARL dev box it prints `SKIP: VK_EXT_shader_float8 not advertised`
and exits 0 — the extension is NVIDIA-only there.

## Shader syntax notes (glslang 11.1.0 / shaderc 2026.4)

The extension names and type names differ from the Vulkan-side naming, and
the cooperative-matrix declaration dropped the K parameter:

```glsl
#extension GL_EXT_float_e4m3 : require   // not GL_EXT_float8
#extension GL_EXT_float_e5m2 : require
floate4m3_t v;                          // not float8_e4m3_t

#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require   // required by gl_ScopeSubgroup
coopmat<floate4m3_t, gl_ScopeSubgroup, 16, 32, gl_MatrixUseA> matA;  // 5 type params
coopmat<float,       gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> matC;
```

`coopmat` takes `type, scope, rows, cols, use`; K is inferred by
`coopMatMulAdd` from the operand types. Writing the 6-parameter
`type, scope, M, N, K, use` form gives `coopmat incorrect number of type
parameters`. A cooperative matrix also cannot be indexed (`matC[0][0]` is
rejected); read it back with `coopMatStore`.

## fp8 in vkop: what is wired up now

Two things were true before this work and are worth keeping straight:

- The **encoder** side was always software. `quantize_linear.comp` writes E4M3 /
  E5M2 bytes by hand, and `matmul.comp`'s `e4m3_val` / `e5m2_val` decode them by
  reassembling the exponent and mantissa. That is why fp8 works on 1.3.275 and
  on MoltenVK, neither of which has the extension.
- The **decode** is the expensive half. Measured on Ada at 896x4864, one byte
  per weight: int8 87 GB/s, fp8 with the software decode 38 GB/s. The field
  reassembly is most of the deficit.

### The hardware decode (implemented)

`shaders/buffer/matmul.comp` gained `-DFP8_HW`, which replaces
`e4m3_val`/`e5m2_val` with `uintBitsToFloate4m3EXT` / `...e5m2EXT` and converts
a whole 32-bit word at once:

```glsl
fe4m3vec4 v = fe4m3vec4(uintBitsToFloate4m3EXT(int8_t(word)), ...);
return vec4(v);          // one 32-bit load, one OpFConvert %v4float
```

It is a **separate module, not a runtime branch**: `OpCapability Float8EXT` is
module-wide, so a device without the extension rejects the module at pipeline
creation rather than falling back. Five keys are emitted, one per path a byte
weight can take:

| key | route |
|---|---|
| `buffer_matmul_fp8_hw_fp16` | general fp16 module |
| `buffer_matmul_w8_wide_fp8_hw_fp16` | WIDE GEMV (the decode-time path) |
| `buffer_matmul_w8_wide_bda_fp8_hw_fp16` | WIDE GEMV + BDA |
| `buffer_matmul_ksplit512_w8_wide_fp8_hw_fp16` | 512-lane WIDE (opt-in) |
| `buffer_matmul_ksplit512_w8_wide_bda_fp8_hw_fp16` | 512-lane WIDE + BDA |

Measured on Ada, 896x4864, min-of-3, all with the correctness check passing.
The software row is the same binary and archive with the five `_fp8_hw_` keys
stripped, so the two differ only in the decode instructions:

| route | GB/s |
|---|---|
| fp8 software decode | 38.0 |
| fp8 hardware, WIDE (E4M3) | 111.0 |
| fp8 hardware, WIDE (E5M2) | 127.9 |
| fp8 hardware, WIDE + BDA | 107.8 |
| fp8 hardware, 512-lane WIDE | 138.5 |
| int8 (reference) | 87.4 |

The hardware decode is ~3x the software one and ahead of int8: one vector
conversion beats int8's per-element XOR+convert. E5M2 runs ~8-12% ahead of
E4M3 on the same kernel; both grids compile into the same module and are
selected by `uPC.e5m2`, which the host sets from the weight tensor's
`elem_kind` — i.e. from the `.vkopbin`, exactly like the format a model was
quantized with.

### The three gates, and what each one costs

Both must hold, or the op keeps the software decode — a slowdown, never a wrong
answer.

1. **Device**: `shader_float8` enabled on the logical device. What this needs is
   a build against headers that carry `VkPhysicalDeviceShaderFloat8FeaturesEXT`
   — measured on b3, the extension enables fine under the **system 1.3.275
   loader** as well as the SDK's 1.4.363 one, because it is an extension and not
   gated on the instance's apiVersion. The `FeatureDescriptor` is behind
   `#ifdef VK_EXT_shader_float8`, which is the real gate: with 1.3.275 headers
   the whole block compiles out and the software decode stays.
2. **Build**: the archive carries the key. CMake probes this glslc for
   `GL_EXT_float_e4m3` at configure time (`VKOP_GLSLC_HAS_FLOAT8`) and skips the
   modules when the compiler is too old. `VKOP_ENABLE_FLOAT8=OFF` disables them
   outright.
3. **Model**: the `.vkopbin` was converted against an archive that had the key.
   A model converted earlier just lacks it; the host latches that per op and
   does not warn (it is the expected state on an old build).

### The fp8 cooperative matrix is NOT wired up

`shader_float8_coopmat` is probed and enabled when the device reports it, and
the 16x16x32 E4M3 config is listed on Ada (see `tools/float8_probe`), but no
kernel uses it. The weight-only route accumulates through a per-column fp32
scale that folds out of the K loop, so the MMA would need the scale applied in
the epilogue and a W8A8 activation quantisation on top — the same shape as the
existing `matmul_coop_i8.comp`, not a drop-in. The decode win above is
independent of it and already landed.

### What a converter must emit

Nothing new. The route keys on `elem_kind() == kFloat8E4M3FN` (or E5M2) on the
weight tensor, which the converter already sets. `fp8` / `fp8_e5m2` in the push
constants are unchanged; only the module the host binds differs.

