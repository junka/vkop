// Copyright 2025 @junka
#include "vulkan/VulkanLib.hpp"

#include "include/logger.hpp"

namespace vkop {

VulkanLib::VulkanLib() {
#ifdef __APPLE__
    lib_ = dlopen("libvulkan.dylib", RTLD_LAZY | RTLD_LOCAL);
    if (!lib_)
        lib_ = dlopen("libvulkan.1.dylib", RTLD_LAZY | RTLD_LOCAL);
    if (!lib_)
        lib_ = dlopen("libMoltenVK.dylib", RTLD_NOW | RTLD_LOCAL);
    if (!lib_ && getenv("DYLD_FALLBACK_LIBRARY_PATH") == nullptr)
        lib_ = dlopen("/usr/local/lib/libvulkan.dylib", RTLD_NOW | RTLD_LOCAL);
#elif defined __linux__
    lib_ = dlopen("libvulkan.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!lib_)
        lib_ = dlopen("libvulkan.so", RTLD_LAZY | RTLD_LOCAL);
#elif defined(__MSYS__)
    lib_ = dlopen("vulkan-1", RTLD_LAZY | RTLD_LOCAL);
    if (!lib_) {
        lib_ =
            dlopen("/c/Windows/System32/vulkan-1.dll", RTLD_LAZY | RTLD_LOCAL);
    }
#elif defined(_WIN32)
    lib_ = LoadLibraryA("vulkan-1.dll");
    if (!lib_) {
        lib_ = LoadLibraryA("C:\\Windows\\System32\\vulkan-1.dll");
    }
#else
#error "Unsupported platform"
#endif
    if (!lib_) {
        LOG_ERROR("Failed to load vulkan library");
        return;
    }

#define PFN(name)                                                              \
    name = reinterpret_cast<PFN_##name>(get_proc_address(#name));              \
    assert((name) != nullptr);
    VK_FUNCTION_LIST
#undef PFN
}

VulkanLib::~VulkanLib() {
    if (lib_) {
#ifdef _WIN32
        FreeLibrary(static_cast<HINSTANCE>(lib_));
#else
        dlclose(lib_);
#endif
    }
}

void *VulkanLib::get_proc_address(const char *name) {
#ifdef _WIN32
    return reinterpret_cast<void *>(
        GetProcAddress(static_cast<HINSTANCE>(lib_), name));
#else
    return dlsym(lib_, name);
#endif
}

} // namespace vkop
