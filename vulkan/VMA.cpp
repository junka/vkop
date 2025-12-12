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
    createPools();
}

VMA::~VMA() {
    vmaDestroyPool(allocator_, device_pool_);
    vmaDestroyPool(allocator_, storage_pool_);
    vmaDestroyPool(allocator_, staging_pool_);
    vmaDestroyAllocator(allocator_);
}

void VMA::createPools() {
    uint32_t mem_type_idx = 0;
    VkImageCreateInfo img_info{};
    img_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img_info.imageType = VK_IMAGE_TYPE_2D;
    img_info.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    img_info.extent = {256, 256, 1};
    img_info.mipLevels = 1;
    img_info.arrayLayers = 64;
    img_info.samples = VK_SAMPLE_COUNT_1_BIT;
    img_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    img_info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    img_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo alloc_info = {};
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    alloc_info.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    vmaFindMemoryTypeIndexForImageInfo(allocator_, &img_info, &alloc_info,
                                       &mem_type_idx);
    printf("Device pool memory type index: %d\n", mem_type_idx);
    VmaPoolCreateInfo device_pool_info = {};
    device_pool_info.memoryTypeIndex = mem_type_idx;
    device_pool_info.blockSize = 16 * 1024 * 1024; // 16MB per block
    device_pool_info.minBlockCount = 1;
    device_pool_info.maxBlockCount = 32; // up to 16 * n
    vmaCreatePool(allocator_, &device_pool_info, &device_pool_);

    VkBufferCreateInfo devbuf_info{};
    memset(&alloc_info, 0, sizeof(alloc_info));
    devbuf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    devbuf_info.size = 4096;
    devbuf_info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    devbuf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    alloc_info.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    vmaFindMemoryTypeIndexForBufferInfo(allocator_, &devbuf_info, &alloc_info,
                                        &mem_type_idx);
    printf("device local pool memory type index: %d\n", mem_type_idx);
    VmaPoolCreateInfo devicelocal_pool_info = {};
    devicelocal_pool_info.memoryTypeIndex = mem_type_idx;
    devicelocal_pool_info.flags = VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT;
    devicelocal_pool_info.blockSize = 16 * 1024 * 1024;
    devicelocal_pool_info.minBlockCount = 1;
    devicelocal_pool_info.maxBlockCount = 16;
    vmaCreatePool(allocator_, &devicelocal_pool_info, &storage_pool_);

    // Staging pool (linear, for uploads/downloads)
    VkBufferCreateInfo buf_info{};
    memset(&alloc_info, 0, sizeof(alloc_info));
    buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_info.size = 4096;
    buf_info.usage =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    alloc_info.flags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
        VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT |
        VMA_ALLOCATION_CREATE_MAPPED_BIT;
    vmaFindMemoryTypeIndexForBufferInfo(allocator_, &buf_info, &alloc_info,
                                        &mem_type_idx);
    printf("staing pool memory type index: %d\n", mem_type_idx);
    VmaPoolCreateInfo staging_pool_info = {};
    staging_pool_info.memoryTypeIndex = mem_type_idx;
    staging_pool_info.flags = VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT;
    staging_pool_info.blockSize = 16 * 1024 * 1024;
    staging_pool_info.minBlockCount = 1;
    staging_pool_info.maxBlockCount = 1;
    vmaCreatePool(allocator_, &staging_pool_info, &staging_pool_);
}

VkResult VMA::createBuffer(VkBufferCreateInfo *bufferInfo,
                           struct VmaBuffer *buf, bool local) {
    VmaAllocationCreateInfo alloc_info = {};
    if (local) {
        alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        alloc_info.pool = storage_pool_;
    } else {
        alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
        alloc_info.flags =
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT;
        alloc_info.pool = staging_pool_;
    }
    buf->allocator = allocator_;
    return vmaCreateBuffer(allocator_, bufferInfo, &alloc_info, &buf->buffer,
                           &buf->allocation, nullptr);
}

VkResult VMA::createImage(VkImageCreateInfo *imageInfo, struct VmaImage *img) {
    VmaAllocationCreateInfo alloc_info = {};
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    alloc_info.pool = device_pool_;
    img->allocator = allocator_;
    return vmaCreateImage(allocator_, imageInfo, &alloc_info, &img->image,
                          &img->allocation, nullptr);
}

void *VMA::getMappedMemory(struct VmaBuffer *buf) {
    VmaAllocationInfo info;
    vmaGetAllocationInfo(buf->allocator, buf->allocation, &info);
    return info.pMappedData;
}

void VMA::destroyBuffer(struct VmaBuffer *buf) {
    vmaDestroyBuffer(allocator_, buf->buffer, buf->allocation);
}

void VMA::destroyImage(struct VmaImage *img) {
    vmaDestroyImage(allocator_, img->image, img->allocation);
}

} // namespace vkop
#endif // USE_VMA