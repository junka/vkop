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
#elif defined(_WIN32) || defined(__MSYS__)
    lib_ = dlopen("vulkan-1", RTLD_LAZY | RTLD_LOCAL);
    if (!lib_) {
        lib_ =
            dlopen("/c/Windows/System32/vulkan-1.dll", RTLD_LAZY | RTLD_LOCAL);
    }
#else
#error "Unsupported platform"
#endif
    if (!lib_) {
        LOG_ERROR("Failed to load vulkan library");
        return;
    }

#define PFN(name)                                                              \
    name = reinterpret_cast<PFN_##name>(dlsym(lib_, #name));                   \
    assert((name) != nullptr);
    VK_FUNCTION_LIST
#undef PFN
}

VulkanLib::~VulkanLib() {
    if (lib_) {
        dlclose(lib_);
    }
}

} // namespace vkop
