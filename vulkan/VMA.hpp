// Copyright 2025 @junka
#ifndef SRC_VMA_HPP_
#define SRC_VMA_HPP_

#include "vulkan/VulkanInstance.hpp"
#include "vulkan/VulkanLib.hpp"

#ifdef USE_VMA

struct VmaAllocation_T;
struct VmaAllocator_T;
struct VmaAllocationInfo;
struct VmaPool_T;

using VmaAllocator = struct VmaAllocator_T *;
using VmaAllocation = struct VmaAllocation_T *;
using VmaPool = struct VmaPool_T *;

namespace vkop {

class VMA {

  public:
    struct VmaBuffer {
        VkBuffer buffer;
        VmaAllocation allocation;
        VmaAllocator allocator;
    };
    struct VmaImage {
        VkImage image;
        VmaAllocation allocation;
        VmaAllocator allocator;
    };

    VMA(VkPhysicalDevice physicalDevice, VkDevice device);
    ~VMA();

    // Allocate a Vulkan buffer with memory
    VkResult createBuffer(VkBufferCreateInfo *bufferInfo, struct VmaBuffer *buf,
                          bool local);

    // Allocate a Vulkan image with memory
    VkResult createImage(VkImageCreateInfo *imageInfo, struct VmaImage *img);

    // Free allocated memory for a buffer
    void destroyBuffer(struct VmaBuffer *buf);

    // Free allocated memory for an image
    void destroyImage(struct VmaImage *img);

    static void *getMappedMemory(struct VmaBuffer *buf);

    // Bytes actually reserved for this image's allocation: the driver's padded
    // requirement, possibly rounded up again by the pool. Always >= the image's
    // packed texel count, so it is the only honest number for footprint
    // accounting (measured equal to the requirement on MoltenVK, but that is
    // the allocator's choice, not a guarantee).
    static uint64_t getAllocatedSize(const struct VmaImage *img);

    void getStats();

    VMA() = delete;
    VMA(const VMA &) = delete;
    VMA(const VMA &&) = delete;
    VMA &operator=(const VMA &) = delete;
    VMA &operator=(const VMA &&) = delete;

  private:
    VmaAllocator allocator_;
    VmaPool device_pool_ = VK_NULL_HANDLE;
    VmaPool storage_pool_ = VK_NULL_HANDLE;
    VmaPool staging_pool_ = VK_NULL_HANDLE;

    void createPools();
};

} // namespace vkop

#endif // USE_VMA

#endif // SRC_VMA_HPP_
