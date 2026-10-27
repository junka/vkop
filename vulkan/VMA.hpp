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
                          bool local, VkMemoryPropertyFlags requiredFlags = 0);

    // Allocate a Vulkan image with memory
    VkResult createImage(VkImageCreateInfo *imageInfo, struct VmaImage *img);

    // Free allocated memory for a buffer
    void destroyBuffer(struct VmaBuffer *buf);

    // Free allocated memory for an image
    void destroyImage(struct VmaImage *img);

    static void *getMappedMemory(struct VmaBuffer *buf);

    // Host -> device visibility for direct mapped access. The staging pool
    // can land on HOST_CACHED memory WITHOUT HOST_COHERENT (VMA's
    // HOST_ACCESS_RANDOM prefers cached; e.g. Intel exposes HOST_CACHED and
    // HOST_COHERENT as two separate types), where staging writes are not
    // automatically visible to device copies. No-op on HOST_COHERENT memory;
    // otherwise flushes exactly the [offset, offset+size) range.
    static void flushAllocation(struct VmaBuffer *buf, VkDeviceSize offset,
                                VkDeviceSize size);

    // Device -> host visibility for the same range after a D2H copy: must be
    // called after the copy's wait and before host reads of the mapped
    // pointer. No-op on HOST_COHERENT memory.
    static void invalidateAllocation(struct VmaBuffer *buf, VkDeviceSize offset,
                                     VkDeviceSize size);

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
