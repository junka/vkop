// Copyright 2025 @junka
#ifndef SRC_VMA_HPP_
#define SRC_VMA_HPP_

#include "vulkan/VulkanInstance.hpp"
#include "vulkan/VulkanLib.hpp"

#ifdef USE_VMA

struct VmaAllocation_T;
struct VmaAllocator_T;
struct VmaAllocationInfo;

using VmaAllocator = struct VmaAllocator_T *;
using VmaAllocation = struct VmaAllocation_T *;

namespace vkop {

class VMA {

  public:
    struct VmaBuffer {
        VkBuffer buffer;
        VmaAllocation allocation;
        VmaAllocationInfo *info;
    };
    struct VmaImage {
        VkImage image;
        VmaAllocation allocation;
        VmaAllocationInfo *info;
    };

    VMA(VkPhysicalDevice physicalDevice, VkDevice device);
    ~VMA();

    // Allocate a Vulkan buffer with memory
    VkResult createBuffer(VkBufferCreateInfo *bufferInfo,
                          struct VmaBuffer *buf);

    // Allocate a Vulkan image with memory
    VkResult createImage(VkImageCreateInfo *imageInfo, struct VmaImage *img);

    // Free allocated memory for a buffer
    void destroyBuffer(struct VmaBuffer *buf);

    // Free allocated memory for an image
    void destroyImage(struct VmaImage *img);

    static void *getMappedMemory(struct VmaBuffer *buf);
    static void *getMappedMemory(struct VmaImage *img);

    VMA() = delete;
    VMA(const VMA &) = delete;
    VMA(const VMA &&) = delete;
    VMA &operator=(const VMA &) = delete;
    VMA &operator=(const VMA &&) = delete;

  private:
    VmaAllocator allocator_;
};

} // namespace vkop

#endif // USE_VMA

#endif // SRC_VMA_HPP_
