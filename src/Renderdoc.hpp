#ifndef __VULKAN_RENDERDOC_H__
#define __VULKAN_RENDERDOC_H__

#include "renderdoc_app.h"
#include <stdexcept>
#include <vulkan/vulkan.hpp>
#include <dlfcn.h>
#include "logger.hpp"

namespace vkop {


class Renderdoc {
private:
    VkInstance m_instance;
    RENDERDOC_API_1_6_0 *rdoc_api = nullptr;
public:
    Renderdoc(VkInstance ins): m_instance(ins) {
        StartRenderDocCapture(m_instance);
    }
    ~Renderdoc() {
        EndRenderDocCapture(m_instance);
    }

    RENDERDOC_API_1_6_0 *GetRdocApi() {
        
        if (rdoc_api) return rdoc_api;
    
        // Based on https://renderdoc.org/docs/in_application_api.html.
    #ifdef __linux__
        void *mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD);
        if (!mod && getenv("DYLD_FALLBACK_LIBRARY_PATH") == nullptr) {
            mod = dlopen("/usr/local/lib/librenderdoc.so", RTLD_NOW | RTLD_NOLOAD);
        }
        if (!mod) {
            mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_LOCAL);
        }
        if (!mod) {
            LOG_ERROR("Failed to load librenderdoc");
            throw std::runtime_error("Failed to load librenderdoc");
        }
        pRENDERDOC_GetAPI RENDERDOC_GetAPI =
            reinterpret_cast<pRENDERDOC_GetAPI>(dlsym(mod, "RENDERDOC_GetAPI"));
        int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_6_0,
                                    reinterpret_cast<void **>(&rdoc_api));
        if (ret != 1) {
            LOG_WARN("RenderDoc initialization failed");
        }
    #elif defined(__APPLE__)
        
    #else
        #warning "RenderDoc integration not implemented on this platform";
    #endif
    
        return rdoc_api;
    }
    
    void StartRenderDocCapture(VkInstance instance) {
        auto *rdoc_api = GetRdocApi();
        if (!rdoc_api) return;

        void *device = RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(instance);
        rdoc_api->StartFrameCapture(device, nullptr);
    }
    
    void EndRenderDocCapture(VkInstance instance) {
        auto *rdoc_api = GetRdocApi();
        if (!rdoc_api) return;
    
        void *device = RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(instance);
        rdoc_api->EndFrameCapture(device, nullptr);
    }

};

}

#endif