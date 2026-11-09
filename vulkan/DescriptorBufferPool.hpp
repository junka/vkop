// Copyright 2026 @junka
#ifndef VULKAN_DESCRIPTORBUFFERPOOL_HPP_
#define VULKAN_DESCRIPTORBUFFERPOOL_HPP_

#include <vulkan/vulkan.h>

#include <map>
#include <memory>
#include <mutex>

namespace vkop {

class VulkanDevice;
class VulkanBuffer;
struct DescriptorBufferInfo;

// One host-writable, device-addressable buffer shared by every compute
// pipeline that uses VK_EXT_descriptor_buffer. Pipelines carve a region out of
// it instead of owning a buffer each, so a recorded command buffer binds the
// buffer once and every dispatch only records
// vkCmdSetDescriptorBufferOffsetsEXT.
//
// The allocation granularity is the device's descriptorBufferOffsetAlignment
// (64 on the tested drivers), so every block offset is a legal set offset.
class DescriptorBufferPool {
  public:
    ~DescriptorBufferPool();

    // Returns nullptr when the buffer cannot be created (no memory type, BAR
    // too small, mapping failed); callers then keep descriptor sets.
    static std::shared_ptr<DescriptorBufferPool>
    create(const std::shared_ptr<VulkanDevice> &device,
           const DescriptorBufferInfo *info, VkMemoryPropertyFlags memory_flags,
           VkDeviceSize capacity);

    // `offset` is returned aligned to `alignment` (the pool's own alignment
    // wins when it is larger). False when the pool is exhausted.
    bool alloc(VkDeviceSize size, VkDeviceSize alignment, VkDeviceSize &offset);
    void free(VkDeviceSize offset, VkDeviceSize size);

    VkBuffer buffer() const;
    VkDeviceAddress address() const { return address_; }
    void *map() const { return map_; }
    VkDeviceSize capacity() const { return capacity_; }

  private:
    DescriptorBufferPool(const std::shared_ptr<VulkanDevice> &device,
                         VkDeviceSize alignment);

    std::shared_ptr<VulkanDevice> device_;
    std::unique_ptr<VulkanBuffer> storage_;
    void *map_ = nullptr;
    VkDeviceAddress address_ = 0;
    VkDeviceSize capacity_ = 0;
    VkDeviceSize alignment_ = 1;
    VkDeviceSize bump_ = 0;
    // Free blocks, keyed by offset; sizes are alignment multiples. Coalesced
    // on free so a long-lived process that reloads models does not fragment.
    std::map<VkDeviceSize, VkDeviceSize> free_blocks_;
    std::mutex mutex_;
};

} // namespace vkop

#endif // VULKAN_DESCRIPTORBUFFERPOOL_HPP_
