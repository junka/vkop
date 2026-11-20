#include "runtime.hpp"

#include <dlfcn.h>

#include <cstdio>
#include <mutex>

namespace ppu {
namespace {

Driver &the_driver() {
  static Driver d;
  return d;
}

std::once_flag g_once;

void load_driver(Driver &d) {
  // RTLD_NOW so a half-broken shim is reported at load time, not mid-render.
  void *h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  if (h == nullptr) {
    d.loadError = "dlopen(libcuda.so.1) failed";
    std::fprintf(stderr, "[vk_ppu] %s: %s\n", d.loadError, dlerror());
    return;
  }

  bool missing = false;
#define PPU_RESOLVE_FN(name, ret, ...)                                         \
  d.name = reinterpret_cast<ret (*)(__VA_ARGS__)>(dlsym(h, #name));            \
  if (d.name == nullptr) {                                                    \
    std::fprintf(stderr, "[vk_ppu] libcuda.so.1 is missing symbol %s\n", #name); \
    d.loadError = "libcuda.so.1 missing a required symbol";                    \
    missing = true;                                                           \
  }
  PPU_CU_FUNCS(PPU_RESOLVE_FN)
#undef PPU_RESOLVE_FN

  // cuStreamDestroy_v2 / cuEventDestroy_v2 / cuMemFreeHost are convenience
  // only; tolerate their absence on a minimal shim.
  if (missing && d.cuInit && d.cuDeviceGet && d.cuMemAlloc_v2 &&
      d.cuLaunchKernel) {
    // Core is there; keep going but remember the partial load.
    d.loaded = true;
    return;
  }
  d.loaded = !missing;
  if (d.loaded) return;
}

}  // namespace

Driver &driver() {
  std::call_once(g_once, [] { load_driver(the_driver()); });
  return the_driver();
}

const char *cu_strerror(CUresult r) {
  static thread_local char buf[256];
  if (r == CUDA_SUCCESS) return "CUDA_SUCCESS";

  Driver &d = driver();
  const char *name = nullptr;
  if (d.cuGetErrorName && d.cuGetErrorName(r, &name) == CUDA_SUCCESS && name)
    return name;
  // cuGetErrorName may return something stale for unknown codes; build our own.
  std::snprintf(buf, sizeof buf, "CUDA error %d", r);
  return buf;
}

}  // namespace ppu
