# vk_ppu_icd — a compute-only Vulkan ICD backed by CUDA/HGGC

A standalone Vulkan Installable Client Driver for the Alibaba T-Head PPU-ZW810E
("真武 810E"). The device has no Vulkan support of its own but exposes a working
CUDA-compatible stack (`libcuda.so.1` from the PPU SDK). This ICD presents it to
Vulkan applications as a compute-only device.

This directory is deliberately **not** part of the enclosing repository's CMake
project. It builds out of tree.

```
cmake -S tools/vulkan-icd-ppu -B tools/vulkan-icd-ppu/build
cmake --build tools/vulkan-icd-ppu/build -j
```

## Design: record, then replay

The one decision everything else follows from:

* `vkCmd*` **records** an operation into a per-command-buffer list. It never
  calls CUDA. (`src/command.cpp`)
* `vkQueueSubmit` **replays** that list onto the device's single `CUstream`:
  dispatch → `cuLaunchKernel`, copy → `cuMemcpyDtoDAsync`, fill →
  `cuMemsetD32Async`, update → `cuMemcpyHtoDAsync`. (`src/submit.cpp`)
* Because all work is ordered on one stream, `vkCmdPipelineBarrier` needs no
  device-side action; it is recorded for accounting only.
* Fences and binary semaphores are `CUevent`s. Waiting on a semaphore is
  `cuStreamWaitEvent`.

The CUDA driver is loaded with `dlopen("libcuda.so.1")` and the ~45 entry points
we need are resolved by `dlsym` (`src/runtime.hpp`). Nothing links against the
CUDA SDK, so the same binary builds on hosts without CUDA headers.

## Memory model (measured on the real device)

The accelerator exposes two distinct kinds of memory, and the ICD maps each to
a Vulkan memory type:

| Vulkan type | Flags | Backed by | Notes |
|---|---|---|---|
| 0 | `DEVICE_LOCAL` | `cuMemAlloc` | Not CPU-accessible; dereferencing it from the host segfaults. |
| 1 | `DEVICE_LOCAL \| HOST_VISIBLE \| HOST_COHERENT` | `cuMemHostAlloc(DEVICEMAP)` | The returned host pointer **is** a valid device pointer (unified VA), so `vkMapMemory` is free and kernels can use it directly. |

`vkAllocateMemory` therefore creates exactly one CUDA allocation, and
`vkMapMemory` on type 1 hands back the same pointer the kernel sees.

Allocation of type 1 costs roughly 1.6 s/GiB on this hardware, which is a
property of the driver, not of this ICD.

## Capabilities

Advertised honestly — see `src/instance.cpp`:

* One queue family: `VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT`, one queue.
* `VK_PHYSICAL_DEVICE_TYPE_OTHER`, API version 1.1.
* **No image support of any kind.** `vkGetPhysicalDeviceFormatProperties`
  reports zero features for every format, and
  `vkGetPhysicalDeviceImageFormatProperties` returns
  `VK_ERROR_FORMAT_NOT_SUPPORTED`.
* Two memory heaps: heap 0 is the accelerator's DEVICE_LOCAL memory, heap 1 is
  ordinary host RAM. Both memory types point at heap 0 (type 1 is
  `cuMemHostAlloc(DEVICEMAP)`, device memory with a CPU-visible address). Two
  heaps, rather than one, is what tells an application this is *not* a unified
  memory device, which is the truth.
* Limits that a compute consumer depends on: `maxPushConstantsSize = 128`
  (the largest operator parameter block), `maxUniformBufferRange = 64 KiB`,
  `maxStorageBufferRange = 4 GiB - 1`.
* Default subgroup: size 32, compute stage only, `BASIC | VOTE | BALLOT |
  SHUFFLE | ARITHMETIC`. `ARITHMETIC` rests on measured hardware:
  `__reduce_{add,min,max,and,or,xor}_sync` are single `v.red.*` instructions and
  float add/min/max/mul reduce exactly through shuffles. No `QUAD` bit (not
  measured), no clustered ops.
* Extensions:
  * Instance: `VK_KHR_get_physical_device_properties2` (the whole
    `vkGetPhysicalDevice*Properties2` family is implemented; without this name
    `vulkaninfo` never calls it).
  * Device: `VK_KHR_buffer_device_address`, `VK_KHR_maintenance3`,
    `VK_KHR_storage_buffer_storage_class`, `VK_KHR_shader_float16_int8`,
    `VK_KHR_16bit_storage`, `VK_KHR_8bit_storage`,
    `VK_EXT_subgroup_size_control` (min = max = 32),
    `VK_KHR_shader_integer_dot_product` (present with **every acceleration bit
    false**: `__dp4a` is four `v.madl.i32` ops, not an instruction),
    `VK_KHR_workgroup_memory_explicit_layout`, `VK_EXT_tooling_info`,
    `VK_KHR_timeline_semaphore` (one stream replays every op in order, so a
    counter value is promoted only when the CUDA event recorded at that
    submission completes — a wait observes real completion).
* Deliberately **not** advertised, each verified or rejected above:
  `VK_KHR_cooperative_matrix` / `VK_NV_cooperative_matrix` (the SPIR-V emitter
  has no cooperative-matrix lowering and fails on every `matmul_coop*` shader),
  `VK_KHR_shader_subgroup_rotate` (no rotate semantics), `VK_EXT_shader_atomic_float`
  / `VK_NV_shader_atomic_float16_vector` (not verified),
  `VK_KHR_zero_initialize_workgroup_memory` (measured: shared memory retains
  the previous block's residue instead of being zeroed).
* Timestamp queries via CUDA events (see `docs/entry-points.md`).
* Device extensions: no external memory/fence/semaphore handles, no push
  descriptors.

## The SPIR-V boundary: real compilation

`vkCreateComputePipelines` compiles SPIR-V. The chain is

```
SPIR-V --[emit, in process]--> LLVM 13 text IR --[llc -march=ppu]--> .o
       --[cuModuleLoadData]--> CUmodule
```

* **The emitter is linked in.** `spirv-emit/` is a library form of the
  standalone `emit` tool (the tool is now a thin `main()` over the same code,
  so the two cannot drift). It is a pure function of the SPIR-V words: it needs
  no files, and takes ~0.7 ms for a small shader and ~2 ms for the largest one
  in vkop's archive. A process per shader would add fork/exec/dynamic-link cost
  for nothing, and would turn a fail-loud reason into a string on a pipe.
* **llc is spawned, not linked.** It is the whole PPU backend (an 84 MB LLVM
  build) and only ships as a tool. `src/spirv_compile.cpp` uses `posix_spawnp`
  with an argv built in-process -- never `system()`, which would put a shell and
  its quoting between us and the compiler, report only the shell's exit status,
  and open an injection surface. The child's stderr is drained through a pipe
  concurrently with feeding stdin, so a 460 KiB module does not deadlock a
  full pipe buffer, and a lowering failure is reported with llc's own message.
* **Fail loud.** A shader the emitter cannot lower makes
  `vkCreateComputePipelines` return `VK_ERROR_INVALID_SHADER_NV` and log the
  entry point plus the exact instruction (`unsupported: GroupNonUniformFMax`).
  Nothing is ever returned as a plausibly-wrong module. `vkQueueSubmit` then
  never has an unlaunchable pipeline to trip over.

The kernel keeps the SPIR-V entry-point name, so `cuModuleGetFunction` finds
exactly the `pName` the application passed to
`VkPipelineShaderStageCreateInfo`.

### Caching

Compilation is cached twice:

* in memory: `(SPIR-V bytes, entry name) -> CUmodule`, reference counted. The
  key material is retained and compared, so a hash collision is a cache miss,
  never a wrong module.
* on disk: the object file, keyed by the same hash, so the next process reuses
  it too. The directory is `$VK_PPU_CACHE_DIR`, else
  `$XDG_CACHE_HOME/vk_ppu_icd`, else `$HOME/.cache/vk_ppu_icd`, else
  `/tmp/vk_ppu_icd-cache-$UID`.

Measured on the device: a cold compile of vkop's real `buffer_softmax_sg` is
**38.8 ms** (emit 2.2 ms, llc 36.0 ms); a warm one is a hash lookup.

### Where llc comes from

`$VK_PPU_LLC`, then `llc` on `PATH`, then the documented SDK locations
(`/usr/local/PPU_SDK/CUDA_SDK/bin/llc`, `/usr/local/PPU_SDK/bin/llc`). When it
cannot be found the compilation fails by name -- `vkCreateComputePipelines`
returns an error saying so, and nothing crashes.

### Coverage

Measured over the 348-entry `vkop_shaders.bin` in a vkop build tree, and every
step below was run, not inferred:

* 124 of 348 emit;
* 124 of 124 emitted modules pass `llc -march=ppu -filetype=obj`;
* 124 of 124 objects pass `cuModuleLoad` **and** `cuModuleGetFunction("main")`
  on the device.

The other 224 fail loud, each naming its blocker. The largest groups are
`VK_EXT_buffer_reference` shaders (`_bda`, ~132: a
`PhysicalStorageBuffer` access chain does not yet resolve), the image backend
(~43, irrelevant to a device with no images), cooperative matrix (~42) and
`VectorShuffle` in fp16 matmul (6). Four shaders are verified numerically
end-to-end (softmax, layernorm, rmsnorm, argmax in the emitter workstream), and
this ICD was verified end-to-end on vkop's own `buffer_softmax_sg`.

### Cubin escape hatch

For end-to-end testing before the compiler lands, `vkCreateShaderModule` also
accepts a **cubin** (ELF produced by `nvcc`/`hgcc`). Its block size defaults to
256 threads and can be overridden with `VK_PPU_CUBIN_BLOCK`. This path exists so
the queue/submit/descriptor machinery can be exercised against real kernels.

## Implemented vs. stubbed

See `docs/entry-points.md` for the full table.

## Environment variables

| Variable | Effect |
|---|---|
| `VK_PPU_DEBUG=1` | Log every ICD call to stderr. |
| `VK_PPU_DEVICE_INDEX=N` | Expose only CUDA device N (default: all devices). |
| `VK_PPU_SYNTHETIC_DEVICE=1` | On a host with no accelerator, report a clearly-labelled placeholder physical device so the loader handshake can be exercised. `vkCreateDevice` on it creates a null backend that refuses all memory allocation. |
| `VK_PPU_CUBIN_BLOCK=N` | Block size for cubin-based pipelines (default 256). |
| `VK_PPU_LLC=PATH` | The `llc` used to lower emitted IR to a PPU object. Default: `llc` on `PATH`, then the SDK's bin directory. |
| `VK_PPU_CACHE_DIR=DIR` | Where compiled object files are cached. Default under `$XDG_CACHE_HOME`/`$HOME`/`/tmp`. |

## Verification

Local (no accelerator; exercises the loader handshake):

```
cmake -S . -B build && cmake --build build -j
VK_ICD_FILENAMES=$PWD/build/ppu_icd.json VK_PPU_SYNTHETIC_DEVICE=1 vulkaninfo --summary
VK_ICD_FILENAMES=$PWD/build/ppu_icd.json VK_PPU_SYNTHETIC_DEVICE=1 ./build/ppu_roundtrip
```

On the PPU host (real round trip, needs `source /usr/local/PPU_SDK/envsetup.sh`):

```
nvcc -arch=sm_80 -cubin -o vecadd.cubin vecadd.cu
VK_ICD_FILENAMES=ppu.json VK_PPU_DEVICE_INDEX=1 ./ppu_roundtrip vecadd.cubin
```

`ppu.json` is the manifest written next to the library by CMake; point it at the
built `libvk_ppu_icd.so`.

The test prints one line per check and a final count. It exits non-zero on any
failure. `scripts/` holds the helpers used to build and run it on the device.
