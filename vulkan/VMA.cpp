// Copyright 2025 @junka
#ifdef USE_VMA

#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include "vulkan/vk_mem_alloc.h"

#include "vulkan/VMA.hpp"

#include <stdexcept>

namespace vkop {

VMA::VMA(VkPhysicalDevice physicalDevice, VkDevice device) {
    VmaVulkanFunctions vulkan_functions = {};
    vulkan_functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vulkan_functions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo allocator_info = {};
    allocator_info.physicalDevice = physicalDevice;
    allocator_info.device = device;
    allocator_info.instance = VulkanInstance::getVulkanInstance().getInstance();
    allocator_info.vulkanApiVersion = VK_API_VERSION_1_2;
    allocator_info.flags = VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT |
                           VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT |
                           VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT |
                           VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT |
                           VMA_ALLOCATOR_CREATE_AMD_DEVICE_COHERENT_MEMORY_BIT;
    allocator_info.pVulkanFunctions = &vulkan_functions;

    if (vmaCreateAllocator(&allocator_info, &allocator_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create VMA allocator");
    }
}

VMA::~VMA() { vmaDestroyAllocator(allocator_); }

VkResult VMA::createBuffer(VkBufferCreateInfo *bufferInfo,
                           struct VmaBuffer *buf) {
    VmaAllocationCreateInfo alloc_info = {};
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
    alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                       VMA_ALLOCATION_CREATE_MAPPED_BIT;

    buf->info = new VmaAllocationInfo();
    return vmaCreateBuffer(allocator_, bufferInfo, &alloc_info, &buf->buffer,
                           &buf->allocation, buf->info);
}

VkResult VMA::createImage(VkImageCreateInfo *imageInfo, struct VmaImage *img) {
    VmaAllocationCreateInfo alloc_info = {};
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
    img->info = new VmaAllocationInfo();
    return vmaCreateImage(allocator_, imageInfo, &alloc_info, &img->image,
                          &img->allocation, img->info);
}

void *VMA::getMappedMemory(struct VmaBuffer *buf) {
    return buf->info->pMappedData;
}

void *VMA::getMappedMemory(struct VmaImage *img) {
    return img->info->pMappedData;
}

void VMA::destroyBuffer(struct VmaBuffer *buf) {
    delete buf->info;
    vmaDestroyBuffer(allocator_, buf->buffer, buf->allocation);
}

void VMA::destroyImage(struct VmaImage *img) {
    delete img->info;
    vmaDestroyImage(allocator_, img->image, img->allocation);
}

} // namespace vkop
#endif // USE_VMA