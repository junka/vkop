// Copyright 2025 @junka
#ifndef SRC_VULKANRESOURCE_HPP_
#define SRC_VULKANRESOURCE_HPP_

#include "vulkan/VulkanDevice.hpp"
#include <memory>
#include <variant>

#define STORAGE VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
#define UNIFORM VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT
#define DESCRIPTOR_TYPE_STORAGE VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
#define DESCRIPTOR_TYPE_UNIFORM VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER

namespace vkop {
// Enum to represent the type of resource
enum class ResourceType { VK_IMAGE, VK_BUFFER, VK_BUFFER_VIEW };

// Abstract base class for Vulkan resources
class VulkanResource {
  public:
    explicit VulkanResource(std::shared_ptr<VulkanDevice> &vdev)
        : m_vdev_(vdev) {}

    virtual ~VulkanResource() {
#ifndef USE_VMA
        if (m_memory_ != VK_NULL_HANDLE)
            vkFreeMemory(m_vdev_->getLogicalDevice(), m_memory_, nullptr);
#endif
    }

    // Method to get the resource type
    virtual ResourceType getResourceType() const = 0;

    // Method to get descriptor info
    virtual std::variant<VkDescriptorImageInfo *, VkDescriptorBufferInfo *,
                         VkBufferView *>
    getDescriptorInfo() = 0;

#ifndef USE_VMA
    static int32_t
    findMemoryTypeFromProperties(uint32_t memoryTypeBits,
                                 VkPhysicalDeviceMemoryProperties properties,
                                 VkMemoryPropertyFlags requiredProperties) {
        // Among the types that satisfy the requirement, pick the best one
        // instead of the first: type order is the driver's, and on a ReBAR
        // dGPU the first HOST_VISIBLE|HOST_COHERENT type can be the
        // DEVICE_LOCAL BAR window -- uncached VRAM that is slow to read back
        // and eats video memory for what is only a staging buffer.
        //   - host-visible but not asked to be device-local (staging): prefer
        //     HOST_CACHED (readback memcpy), avoid DEVICE_LOCAL.
        //   - device-local and not asked to be host-visible: avoid
        //     HOST_VISIBLE (the BAR heap is the small one on a dGPU).
        //   - never pick exotic types the caller did not ask for.
        // On UMA every type is DEVICE_LOCAL, so the penalties tie and the
        // driver's order decides, exactly as before.
        constexpr VkMemoryPropertyFlags kExotic =
            VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT |
            VK_MEMORY_PROPERTY_PROTECTED_BIT
#ifdef VK_AMD_device_coherent_memory
            | VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD |
            VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD
#endif
            ;
        const bool want_host =
            (requiredProperties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
        const bool want_local =
            (requiredProperties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
        int32_t best = -1;
        int best_score = 0;
        for (uint32_t index = 0; index < properties.memoryTypeCount; ++index) {
            const VkMemoryPropertyFlags f =
                properties.memoryTypes[index].propertyFlags;
            if (!(memoryTypeBits & (1u << index)) ||
                (f & requiredProperties) != requiredProperties ||
                (f & kExotic & ~requiredProperties) != 0) {
                continue;
            }
            int score = 100;
            if (want_host && !want_local) {
                if (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
                    score += 10;
                if (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                    score -= 20;
            }
            if (want_local && !want_host &&
                (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                score -= 20;
            }
            if (score > best_score) {
                best_score = score;
                best = static_cast<int32_t>(index);
            }
        }
        assert(best >= 0 && "Did not find a suitable memory type");
        return best;
    }
    bool allocMemory(VkMemoryRequirements memoryRequirements,
                     VkMemoryPropertyFlags requiredProperties, int ext_fd = -1,
                     bool allow_buffer_device_address = false) {
        VkPhysicalDeviceMemoryProperties memory_properties;
        vkGetPhysicalDeviceMemoryProperties(m_vdev_->getPhysicalDevice(),
                                            &memory_properties);

        auto memory_type_index =
            findMemoryTypeFromProperties(memoryRequirements.memoryTypeBits,
                                         memory_properties, requiredProperties);
        if (memory_type_index < 0) {
            return false;
        }
        VkMemoryAllocateInfo allocate_info = {};
        allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocate_info.allocationSize = memoryRequirements.size;
        allocate_info.memoryTypeIndex = memory_type_index;
        void *pnext = nullptr;
#ifdef VK_KHR_external_memory_fd
        VkImportMemoryFdInfoKHR importfdinfo = {};
        importfdinfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
        importfdinfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        importfdinfo.fd = ext_fd;
        if (ext_fd != -1)
            pnext = &importfdinfo;
#endif
        // Only legal for BUFFER memory, and only when the bufferDeviceAddress
        // feature was enabled. VulkanImage shares this helper and must not
        // get the flag, so the caller opts in explicitly.
        VkMemoryAllocateFlagsInfo allocate_flags = {};
        if (allow_buffer_device_address &&
            m_vdev_->is_support_buffer_device_address()) {
            allocate_flags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
            allocate_flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
            allocate_flags.pNext = pnext;
            pnext = &allocate_flags;
        }
        allocate_info.pNext = pnext;
        return vkAllocateMemory(m_vdev_->getLogicalDevice(), &allocate_info,
                                nullptr, &m_memory_) == VK_SUCCESS;
    }

    VkDeviceMemory getMemory() const { return m_memory_; }
    uint64_t getOffset() const { return offset_; }

#endif

    VkDescriptorType getDescriptorType() const { return m_desc_type_; }

    VulkanResource() = delete;
    VulkanResource(const VulkanResource &buff) = delete;
    VulkanResource(const VulkanResource &&buff) = delete;
    VulkanResource &operator=(const VulkanResource &) = delete;
    VulkanResource &operator=(const VulkanResource &&) = delete;

  protected:
    std::shared_ptr<VulkanDevice> m_vdev_;
    VkDescriptorType m_desc_type_;

  private:
#ifndef USE_VMA
    VkDeviceMemory m_memory_ = VK_NULL_HANDLE;
    uint64_t offset_ = 0;
#endif
};

} // namespace vkop
#endif // SRC_VULKANRESOURCE_HPP_
