// Minimal, header-free CUDA driver API surface used by the PPU Vulkan ICD.
//
// We deliberately declare only the ~40 entry points we need instead of
// including <cuda.h>. That keeps this ICD buildable on hosts (like the local
// dev box) that have the PPU SDK but no CUDA headers, and it shields us from
// the HGGC shim's header version skew.
//
// Everything is resolved with dlopen("libcuda.so.1") + dlsym at first use.
#ifndef VK_PPU_RUNTIME_HPP
#define VK_PPU_RUNTIME_HPP

#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#define PPU_INTERNAL __declspec(dllexport)
#else
#define PPU_INTERNAL __attribute__((visibility("default")))
#endif

namespace ppu {

using CUresult = int;
using CUdevice = int;
using CUcontext = void *;
using CUstream = void *;
using CUmodule = void *;
using CUfunction = void *;
using CUevent = void *;
using CUdeviceptr = unsigned long long;

constexpr CUresult CUDA_SUCCESS = 0;
constexpr CUresult CUDA_ERROR_INVALID_VALUE = 1;
constexpr CUresult CUDA_ERROR_NOT_READY = 600;
constexpr CUresult CUDA_ERROR_NOT_FOUND = 500;
constexpr CUresult CUDA_ERROR_NOT_SUPPORTED = 801;
constexpr CUresult CUDA_ERROR_INVALID_SOURCE = 300;

// CU_MEMHOSTALLOC_* flags (cuda.h).
constexpr unsigned CU_MEMHOSTALLOC_PORTABLE = 0x01;
constexpr unsigned CU_MEMHOSTALLOC_DEVICEMAP = 0x02;

// Selected CU_DEVICE_ATTRIBUTE_* values we surface to Vulkan.
constexpr int CU_DEV_ATTR_MAX_THREADS_PER_BLOCK = 1;
constexpr int CU_DEV_ATTR_MAX_BLOCK_DIM_X = 2;
constexpr int CU_DEV_ATTR_MAX_SHARED_MEMORY_PER_BLOCK = 8;
constexpr int CU_DEV_ATTR_MAX_REGISTERS_PER_BLOCK = 12;
constexpr int CU_DEV_ATTR_CLOCK_RATE = 13;
constexpr int CU_DEV_ATTR_MULTIPROCESSOR_COUNT = 16;
constexpr int CU_DEV_ATTR_INTEGRATED = 18;
constexpr int CU_DEV_ATTR_CAN_MAP_HOST_MEMORY = 19;
constexpr int CU_DEV_ATTR_GLOBAL_MEMORY_BUS_WIDTH = 37;
constexpr int CU_DEV_ATTR_L2_CACHE_SIZE = 38;
constexpr int CU_DEV_ATTR_MAX_THREADS_PER_MP = 39;
constexpr int CU_DEV_ATTR_UNIFIED_ADDRESSING = 41;
constexpr int CU_DEV_ATTR_COMPUTE_CAPABILITY_MAJOR = 75;
constexpr int CU_DEV_ATTR_COMPUTE_CAPABILITY_MINOR = 76;
constexpr int CU_DEV_ATTR_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97;
constexpr int CU_DEV_ATTR_VMM_SUPPORTED = 102;

// All the driver entry points the ICD will ever call.
#define PPU_CU_FUNCS(X)                                                        \
  X(cuInit, CUresult, unsigned)                                                \
  X(cuDeviceGetCount, CUresult, int *)                                         \
  X(cuDeviceGet, CUresult, CUdevice *, int)                                    \
  X(cuDeviceGetName, CUresult, char *, int, CUdevice)                          \
  X(cuDeviceGetAttribute, CUresult, int *, int, CUdevice)                      \
  X(cuDeviceTotalMem_v2, CUresult, size_t *, CUdevice)                         \
  X(cuCtxCreate_v2, CUresult, CUcontext *, unsigned, CUdevice)                 \
  X(cuCtxDestroy_v2, CUresult, CUcontext)                                      \
  X(cuCtxSetCurrent, CUresult, CUcontext)                                      \
  X(cuStreamCreate, CUresult, CUstream *, unsigned)                            \
  X(cuStreamDestroy_v2, CUresult, CUstream)                                    \
  X(cuStreamSynchronize, CUresult, CUstream)                                   \
  X(cuStreamWaitEvent, CUresult, CUstream, CUevent, unsigned)                  \
  X(cuMemAlloc_v2, CUresult, CUdeviceptr *, size_t)                            \
  X(cuMemFree_v2, CUresult, CUdeviceptr)                                       \
  X(cuMemHostAlloc, CUresult, void **, size_t, unsigned)                       \
  X(cuMemHostGetDevicePointer_v2, CUresult, CUdeviceptr *, void *, unsigned)   \
  X(cuMemFreeHost, CUresult, void *)                                           \
  X(cuMemGetInfo_v2, CUresult, size_t *, size_t *)                             \
  X(cuMemcpyHtoD_v2, CUresult, CUdeviceptr, const void *, size_t)              \
  X(cuMemcpyDtoH_v2, CUresult, void *, CUdeviceptr, size_t)                    \
  X(cuMemcpyDtoD_v2, CUresult, CUdeviceptr, CUdeviceptr, size_t)               \
  X(cuMemcpyHtoDAsync_v2, CUresult, CUdeviceptr, const void *, size_t, CUstream) \
  X(cuMemcpyDtoDAsync_v2, CUresult, CUdeviceptr, CUdeviceptr, size_t, CUstream)  \
  X(cuMemsetD8_v2, CUresult, CUdeviceptr, unsigned char, size_t)               \
  X(cuMemsetD16_v2, CUresult, unsigned short, CUdeviceptr, size_t)             \
  X(cuMemsetD32_v2, CUresult, CUdeviceptr, unsigned, size_t)                   \
  X(cuMemsetD8Async, CUresult, CUdeviceptr, unsigned char, size_t, CUstream)   \
  X(cuMemsetD32Async, CUresult, CUdeviceptr, unsigned, size_t, CUstream)       \
  X(cuEventCreate, CUresult, CUevent *, unsigned)                              \
  X(cuEventDestroy_v2, CUresult, CUevent)                                      \
  X(cuEventRecord, CUresult, CUevent, CUstream)                                \
  X(cuEventQuery, CUresult, CUevent)                                           \
  X(cuEventSynchronize, CUresult, CUevent)                                     \
  X(cuEventElapsedTime, CUresult, float *, CUevent, CUevent)                   \
  X(cuModuleLoad, CUresult, CUmodule *, const char *)                          \
  X(cuModuleLoadData, CUresult, CUmodule *, const void *)                      \
  X(cuModuleUnload, CUresult, CUmodule)                                        \
  X(cuModuleGetFunction, CUresult, CUfunction *, CUmodule, const char *)       \
  X(cuLaunchKernel, CUresult, CUfunction, unsigned, unsigned, unsigned,        \
    unsigned, unsigned, unsigned, unsigned, CUstream, void **, void **)        \
  X(cuGetErrorName, CUresult, CUresult, const char **)                         \
  X(cuGetErrorString, CUresult, CUresult, const char **)

// One field per entry point above, all nullable.
struct Driver {
#define PPU_DECLARE_FN(name, ret, ...) ret (*name)(__VA_ARGS__) = nullptr;
  PPU_CU_FUNCS(PPU_DECLARE_FN)
#undef PPU_DECLARE_FN
  bool loaded = false;
  const char *loadError = nullptr;
};

// Resolve libcuda.so.1 once per process. Returns the process-wide driver.
// Thread-safe. If the library or a required symbol is missing, returns a
// Driver whose `loaded` is false and whose `loadError` explains why; callers
// must degrade gracefully rather than crash.
Driver &driver();

// Human-readable name for a CUresult, e.g. "CUDA_ERROR_INVALID_CONTEXT (201)".
// Never returns nullptr.
const char *cu_strerror(CUresult r);

}  // namespace ppu

#endif  // VK_PPU_RUNTIME_HPP
