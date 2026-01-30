// Copyright 2026 @junka, All rights reserved.
#ifndef SRC_VULKANCUDALAUNCH_HPP_
#define SRC_VULKANCUDALAUNCH_HPP_
#ifdef ENABLE_CUDA_LAUNCH

#include "vulkan/VulkanLib.hpp"

using CUresult = int;

using CUdevice = void *;
using CUcontext = void *;
using CUmodule = void *;
using CUfunction = void *;
using PFN_cuInit = CUresult (*)(unsigned int Flags);
using PFN_cuDeviceGet = CUresult (*)(CUdevice *device, int ordinal);
using PFN_cuDeviceGetAttribute = CUresult (*)(int *pi, int attrib,
                                              CUdevice dev);
using PFN_cuCtxCreate_v2 = CUresult (*)(CUcontext *pctx, unsigned int flags,
                                        CUdevice dev);
using PFN_cuModuleLoad = CUresult (*)(CUmodule *module, const char *fname);
using PFN_cuModuleGetFunction = CUresult (*)(CUfunction *hfunc, CUmodule hmod,
                                             const char *name);
namespace vkop {
class VulkanCudaLaunch {
  public:
    VulkanCudaLaunch(VkDevice device, const unsigned char *ptxdata,
                     uint32_t ptxlen);
    ~VulkanCudaLaunch();

    bool create_function(const char *function);

    void run(VkCommandBuffer cmdBuffer, int width, int height, int depth,
             int block_size);

  private:
    VkDevice device;
    VkCudaModuleNV vkCudaModule = VK_NULL_HANDLE;
    VkCudaFunctionNV vkCudaFunction;
    void *cudaLib = nullptr;

    PFN_cuInit cuInit = nullptr;
    PFN_cuDeviceGet cuDeviceGet = nullptr;
    PFN_cuDeviceGetAttribute cuDeviceGetAttribute = nullptr;
    PFN_cuCtxCreate_v2 cuCtxCreate = nullptr;
    PFN_cuModuleLoad cuModuleLoad = nullptr;
    PFN_cuModuleGetFunction cuModuleGetFunction = nullptr;
};

} // namespace vkop
#endif // ENABLE_CUDA_LAUNCH
#endif // SRC_VULKANCUDALAUNCH_HPP_