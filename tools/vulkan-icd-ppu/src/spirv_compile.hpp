// SPIR-V -> PPU module compilation for the Vulkan ICD.
//
// The chain is:
//
//   SPIR-V --[emit, in-process]--> LLVM 13 text IR --[llc -march=ppu]--> .o
//          --[cuModuleLoadData]--> CUmodule
//
// Two halves, deliberately different:
//
//   * The emitter is a library (spirv-emit/) linked into the ICD. It is pure
//     computation over the SPIR-V words, takes microseconds, and needs no
//     external state -- so a process per shader would be pure overhead.
//   * llc is the PPU backend itself and only ships as a command-line tool, so
//     it is fork/exec'd (never system(): no shell, no injection, a real exit
//     status, and full control over the pipes).
//
// Both halves are cached by content hash: vkCreateComputePipelines is called
// once per operator (vkop builds 351 shaders at start-up) and applications do
// re-create identical pipelines, so a cache miss -- not a pipeline creation --
// is what should cost a compiler run.
#ifndef VK_PPU_SPIRV_COMPILE_HPP
#define VK_PPU_SPIRV_COMPILE_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "icd_common.hpp"

namespace ppu {

// One kernel parameter, in the emitted function's signature order.
struct KernelArgBinding {
  uint32_t set = 0;
  uint32_t binding = 0;
  bool push_constant = false;
};

struct CompiledKernel {
  CUmodule module = nullptr;
  std::vector<KernelArgBinding> args;
  uint32_t local_size[3] = {1, 1, 1};
  std::string entry_name;
};

// Compile `spirv` and resolve `entry_point`, caching both the object file and
// the loaded module. `context` scopes the CUmodule cache: module handles are
// only valid in the context that loaded them.
//
// On success `out` is filled and holds one cache reference, released with
// release_compiled_kernel(). On failure returns a CUresult other than
// CUDA_SUCCESS and writes a human-readable explanation into `error` naming the
// entry point and, when the emitter rejected it, the exact unsupported
// instruction. Never returns a partially usable module.
CUresult compile_spirv_to_ppu_kernel(CUcontext context, const uint32_t *spirv,
                                     size_t word_count, const char *entry_point,
                                     CompiledKernel *out, std::string *error);

void release_compiled_kernel(const CompiledKernel &kernel);

// Drop every cached module belonging to `context` (vkDestroyDevice calls this
// before cuCtxDestroy_v2, so no cached CUmodule outlives its context).
void purge_spirv_cache(CUcontext context);

// The llc executable this process will use, or nullptr when it cannot be
// found. Resolution order: $VK_PPU_LLC, then "llc" on PATH, then the SDK's
// default location. `why` (optional) receives the search that was performed,
// which is what a caller puts in the error it reports.
const char *llc_path(std::string *why);

// Exposed for tests: the cache key for (SPIR-V, entry point).
uint64_t spirv_cache_key(const uint32_t *spirv, size_t word_count, const char *entry_point);

}  // namespace ppu

#endif  // VK_PPU_SPIRV_COMPILE_HPP
