// Copyright (c) 2026, junka. All rights reserved.
#ifdef ENABLE_CUDA_LAUNCH
#include "vulkan/VulkanCudaLaunch.hpp"
#include "vulkan/VulkanInstance.hpp"

namespace vkop {

VulkanCudaLaunch::VulkanCudaLaunch(VkDevice device,
                                   const unsigned char *ptxdata,
                                   uint32_t ptxlen)
    : device(device) {

    const char *cudalibs[] = {"libcudart.so", "libcuda.so",
                              "libcudart.so.11.0"};

    for (const char *lib : cudalibs) {
        cudaLib = dlopen(lib, RTLD_LAZY);
        if (cudaLib)
            break;
    }

    if (!cudaLib) {
        printf("Can't find cuda library\n");
        return;
    }

    *reinterpret_cast<void **>(&cuInit) = dlsym(cudaLib, "cuInit");
    *reinterpret_cast<void **>(&cuDeviceGet) = dlsym(cudaLib, "cuDeviceGet");
    *reinterpret_cast<void **>(&cuDeviceGetAttribute) =
        dlsym(cudaLib, "cuDeviceGetAttribute");
    *reinterpret_cast<void **>(&cuCtxCreate) = dlsym(cudaLib, "cuCtxCreate_v2");
    *reinterpret_cast<void **>(&cuModuleLoad) = dlsym(cudaLib, "cuModuleLoad");
    *reinterpret_cast<void **>(&cuModuleGetFunction) =
        dlsym(cudaLib, "cuModuleGetFunction");

    VkCudaModuleCreateInfoNV create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_CUDA_MODULE_CREATE_INFO_NV;
    create_info.pNext = nullptr;
    create_info.pData = ptxdata;
    create_info.dataSize = static_cast<size_t>(ptxlen);

    auto vkCreateCudaModuleNV = reinterpret_cast<PFN_vkCreateCudaModuleNV>(
        vkGetInstanceProcAddr(VulkanInstance::getVulkanInstance().getInstance(),
                              "vkCreateCudaModuleNV"));
    vkCreateCudaModuleNV(device, &create_info, nullptr, &vkCudaModule);
}

VulkanCudaLaunch::~VulkanCudaLaunch() {
    if (vkCudaFunction != VK_NULL_HANDLE) {
        auto vkDestroyCudaFunctionNV =
            reinterpret_cast<PFN_vkDestroyCudaFunctionNV>(vkGetInstanceProcAddr(
                VulkanInstance::getVulkanInstance().getInstance(),
                "vkDestroyCudaFunctionNV"));
        vkDestroyCudaFunctionNV(device, vkCudaFunction, nullptr);
    }
    if (vkCudaModule != VK_NULL_HANDLE) {
        auto vkDestroyCudaModuleNV =
            reinterpret_cast<PFN_vkDestroyCudaModuleNV>(vkGetInstanceProcAddr(
                VulkanInstance::getVulkanInstance().getInstance(),
                "vkDestroyCudaModuleNV"));
        vkDestroyCudaModuleNV(device, vkCudaModule, nullptr);
    }
    dlclose(cudaLib);
}

// https://docs.vulkan.org/spec/latest/chapters/dispatch.html#cudadispatch
bool VulkanCudaLaunch::create_function(const char *function) {
    // CUmodule module;
    // CUresult cuResult = cuModuleLoad(&module, ptxFile);
    // if (cuResult != CUDA_SUCCESS)
    //     return false;

    // cuResult = cuModuleGetFunction(&cudaFunction, module, functionName);
    // if (cuResult != CUDA_SUCCESS)
    //     return false;
    if (!vkCudaModule)
        return false;

    VkCudaFunctionCreateInfoNV func_create_info{};
    func_create_info.sType = VK_STRUCTURE_TYPE_CUDA_FUNCTION_CREATE_INFO_NV;
    func_create_info.module = vkCudaModule;
    func_create_info.pName = function;

    auto vkCreateCudaFunctionNV = reinterpret_cast<PFN_vkCreateCudaFunctionNV>(
        vkGetInstanceProcAddr(VulkanInstance::getVulkanInstance().getInstance(),
                              "vkCreateCudaFunctionNV"));
    auto ret = vkCreateCudaFunctionNV(device, &func_create_info, nullptr,
                                      &vkCudaFunction);
    return ret == VK_SUCCESS;
    return true;
}

void VulkanCudaLaunch::run(VkCommandBuffer cmdBuffer, int width, int height,
                           int depth, int block_size) {
    if (vkCudaFunction == VK_NULL_HANDLE)
        return;

    const void *params[] = {/* 你的内核参数 */};

    VkCudaLaunchInfoNV launch_info{};
    launch_info.sType = VK_STRUCTURE_TYPE_CUDA_LAUNCH_INFO_NV;
    launch_info.function = vkCudaFunction;
    launch_info.gridDimX = width;
    launch_info.gridDimY = height;
    launch_info.gridDimZ = depth;
    launch_info.blockDimX = block_size;
    launch_info.blockDimY = block_size;
    launch_info.blockDimZ = 1;
    launch_info.sharedMemBytes = 0;
    launch_info.paramCount = 0;
    launch_info.pParams = params;
    launch_info.extraCount = 0;
    launch_info.pExtras = nullptr;

    auto vkCmdCudaLaunchKernelNV =
        reinterpret_cast<PFN_vkCmdCudaLaunchKernelNV>(vkGetInstanceProcAddr(
            VulkanInstance::getVulkanInstance().getInstance(),
            "vkCmdCudaLaunchKernelNV"));
    vkCmdCudaLaunchKernelNV(cmdBuffer, &launch_info);
}

} // namespace vkop

#endif
