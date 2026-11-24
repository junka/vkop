// POSIX calls the ctest probes rely on, for toolchains that do not ship them.
//
// The probes are small standalone programs (not gtest), so they never include
// setup.hpp -- every gtest TU does, and gets its setenv/unsetenv shim from
// there. MSVC has neither those nor posix_memalign, so this header supplies
// them; tests/CMakeLists.txt force-includes it (/FI) into each probe on MSVC.
// Nothing here is compiled on POSIX beyond the free pairing.
#ifndef VKOP_TESTS_PROBE_COMPAT_HPP
#define VKOP_TESTS_PROBE_COMPAT_HPP

#include <cstddef>
#include <cstdlib>

#ifdef _WIN32
#include <cerrno>
#include <malloc.h>

inline int setenv(const char *name, const char *value, int overwrite) {
    if (!overwrite && std::getenv(name) != nullptr) return 0;
    return _putenv_s(name, value);
}
inline int unsetenv(const char *name) { return _putenv_s(name, ""); }

// posix_memalign(3), backed by _aligned_malloc: the matching release is
// aligned_free_host(), not free(), because the two allocators differ.
inline int posix_memalign(void **out, std::size_t alignment,
                          std::size_t size) {
    *out = _aligned_malloc(size, alignment);
    return *out ? 0 : ENOMEM;
}
inline void aligned_free_host(void *p) { _aligned_free(p); }
#else
inline void aligned_free_host(void *p) { std::free(p); }
#endif

#endif
