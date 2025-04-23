#include "VulkanLib.hpp"
#include <dlfcn.h>

#include <iostream>

namespace vkop {

VulkanLib::VulkanLib() {
#ifdef __APPLE__
        lib = dlopen("libvulkan.dylib", RTLD_LAZY | RTLD_LOCAL);
        if (!lib)
            lib = dlopen("libvulkan.1.dylib", RTLD_LAZY | RTLD_LOCAL);
        if (!lib)
		    lib = dlopen("libMoltenVK.dylib", RTLD_NOW | RTLD_LOCAL);
        if (!lib && getenv("DYLD_FALLBACK_LIBRARY_PATH") == nullptr)
            lib = dlopen("/usr/local/lib/libvulkan.dylib", RTLD_NOW | RTLD_LOCAL);
#elif defined __linux__
        lib = dlopen("libvulkan.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (!lib)
            lib = dlopen("libvulkan.so", RTLD_LAZY | RTLD_LOCAL);
#endif
        if (!lib) {
            std::cerr << "Failed to load vulkan library ," << dlerror() << std::endl;
            return ;
        }

#define PFN(name) name = reinterpret_cast<PFN_##name>(dlsym(lib, #name));
            VK_FUNCTION_LIST
#undef PFN
}

VulkanLib::~VulkanLib() {
    if (lib) {
        dlclose(lib);
    }
}

}