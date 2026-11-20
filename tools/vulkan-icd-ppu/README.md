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
* Extensions: `VK_KHR_buffer_device_address`, `VK_KHR_maintenance3`,
  `VK_KHR_storage_buffer_storage_class`, `VK_KHR_shader_float16_int8`,
  `VK_EXT_tooling_info`.
* Timestamp queries via CUDA events (see `docs/entry-points.md`).
* Device extensions: no external memory/fence/semaphore handles, no timeline
  semaphores, no push descriptors.

## The SPIR-V boundary

`ppu::compile_spirv_to_ppu_module()` (`src/pipeline.cpp`) is the single
hand-off point for the SPIR-V → PPU code generation workstream. It currently
returns `CUDA_ERROR_NOT_SUPPORTED`. Replace its body and pipelines built from
SPIR-V will work; nothing else needs to change.

Until then:

* `vkCreateShaderModule` accepts SPIR-V and extracts `LocalSize` from
  `OpExecutionMode`.
* `vkCreateComputePipelines` returns a handle but logs that compilation is
  unimplemented; the pipeline cannot execute.
* `vkQueueSubmit` refuses to launch such a pipeline and returns
  `VK_ERROR_DEVICE_LOST` rather than pretending success.

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
