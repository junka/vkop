// Copyright 2025 @junka
#include "VulkanBuffer.hpp"
#include "VulkanLib.hpp"
#include "VulkanResource.hpp"
#include <stdexcept>
#include <utility>

namespace vkop {
VulkanBuffer::VulkanBuffer(std::shared_ptr<VulkanDevice> &vdev,
                           VkDeviceSize size, VkBufferUsageFlags usage,
                           VkMemoryPropertyFlags requireProperties, int ext_fd)
    : VulkanResource(vdev), m_size_(size) {
    createBuffer(usage, requireProperties);
#ifndef USE_VMA
#ifdef VK_KHR_get_memory_requirements2
    VkMemoryRequirements2 mem_requirements2 = {};
    mem_requirements2.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
    VkBufferMemoryRequirementsInfo2 buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2;
    buffer_info.buffer = m_buffer_;
    vkGetBufferMemoryRequirements2(m_vdev_->getLogicalDevice(), &buffer_info,
                                   &mem_requirements2);
    VkMemoryRequirements mem_requirements =
        mem_requirements2.memoryRequirements;
#else
    VkMemoryRequirements mem_requirements;
    vkGetBufferMemoryRequirements(m_vdev_->getLogicalDevice(), m_buffer_,
                                  &mem_requirements);
#endif
    allocMemory(mem_requirements, requireProperties, ext_fd,
                /*allow_buffer_device_address=*/true);
    vkBindBufferMemory(m_vdev_->getLogicalDevice(), m_buffer_, getMemory(), 0);
#else
    (void)ext_fd;
#endif
}

// Borrowed sub-range view: share the parent's VkBuffer, own nothing. m_vdev_
// comes from the parent (protected member accessed on a VulkanBuffer object,
// which the language allows from a VulkanBuffer member). No createBuffer, no
// allocMemory, no vkBindBufferMemory.
VulkanBuffer::VulkanBuffer(std::shared_ptr<VulkanBuffer> parent,
                           VkDeviceSize offset, VkDeviceSize range)
    : VulkanResource(parent->m_vdev_), m_size_(range),
      parent_(std::move(parent)), range_offset_(offset) {}

// Memory-type scan for the host-import ctor prefers a non-DEVICE_LOCAL
// HOST_VISIBLE|HOST_COHERENT type — on AMD/MoltenVK the DEVICE_LOCAL|HOST_VISIBLE
// type is in memoryTypeBits for every allocation but REJECTS host-pointer
// imports (VK_ERROR_INVALID_EXTERNAL_HANDLE), while on UMA (Intel iGPU) the
// coherent type IS DEVICE_LOCAL, so the non-DL pass must fall through to it.
static int32_t find_host_import_memory_type(std::shared_ptr<VulkanDevice> &vdev,
                                            uint32_t memory_type_bits) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(vdev->getPhysicalDevice(), &mp);
    for (int pass = 0; pass < 2; ++pass) {
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            if (!(memory_type_bits & (1u << i)))
                continue;
            VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
            bool coherent = (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
                            (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            bool non_dl = !(f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (coherent && (pass == 1 || non_dl))
                return static_cast<int32_t>(i);
        }
    }
    return -1;
}

VulkanBuffer::VulkanBuffer(std::shared_ptr<VulkanDevice> &vdev,
                           VkDeviceSize size, VkBufferUsageFlags usage,
                           void *host_ptr, size_t host_size)
    : VulkanResource(vdev), m_size_(size) {
    if (!host_ptr) {
        throw std::runtime_error("vkop: host-import buffer with null host ptr");
    }
    if (!vdev->is_support_host_import()) {
        throw std::runtime_error(
            "vkop: host-import buffer but VK_EXT_external_memory_host is not "
            "enabled on this device");
    }
    // Raw vkCreateBuffer — createBuffer() routes through VMA on the USE_VMA
    // build, and VMA cannot import foreign memory.
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    // Deliberately NOT adding VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT here.
    // VK_EXT_external_memory_host imports foreign memory, and whether the
    // driver accepts VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT on it is not
    // covered by the device-init probe (which allocates ordinary memory). A
    // rejected flag would fail the whole import and take the per-step input
    // path down with it, so host-imported buffers stay descriptor-only until
    // that combination is measured on every target.
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    // A buffer bound to imported host memory must declare that external handle
    // type at creation; vkBindBufferMemory otherwise violates
    // VUID-vkBindBufferMemory-memory-02985.
    VkExternalMemoryBufferCreateInfo external_info{};
    external_info.sType =
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    external_info.handleTypes =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    buffer_info.pNext = &external_info;
    auto ret = vkCreateBuffer(m_vdev_->getLogicalDevice(), &buffer_info,
                              nullptr, &ext_buffer_);
    if (ret != VK_SUCCESS) {
        throw std::runtime_error("vkop: host-import buffer creation failed "
                                 "size=" + std::to_string(size) +
                                 " vkret=" + std::to_string(ret));
    }
    if (usage & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) {
        m_desc_type_ = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    } else if (usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) {
        m_desc_type_ = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    } else if (usage & VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT) {
        m_desc_type_ = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
    } else if (usage & VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT) {
        m_desc_type_ = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
    }

    VkMemoryRequirements mem_requirements;
    vkGetBufferMemoryRequirements(m_vdev_->getLogicalDevice(), ext_buffer_,
                                  &mem_requirements);
    auto mem_type =
        find_host_import_memory_type(vdev, mem_requirements.memoryTypeBits);
    if (mem_type < 0) {
        vkDestroyBuffer(m_vdev_->getLogicalDevice(), ext_buffer_, nullptr);
        ext_buffer_ = VK_NULL_HANDLE;
        throw std::runtime_error(
            "vkop: no HOST_VISIBLE|HOST_COHERENT memory type accepts a host "
            "import on this device");
    }
    // Allocation size may exceed the requested buffer size (driver padding);
    // round to the import alignment and require the host block to cover it.
    VkDeviceSize align = vdev->host_import_alignment();
    VkDeviceSize alloc_size =
        ((mem_requirements.size + align - 1) / align) * align;
    if (static_cast<VkDeviceSize>(host_size) < alloc_size) {
        vkDestroyBuffer(m_vdev_->getLogicalDevice(), ext_buffer_, nullptr);
        ext_buffer_ = VK_NULL_HANDLE;
        throw std::runtime_error(
            "vkop: host-import block too small: need " +
            std::to_string(alloc_size) + " have " +
            std::to_string(host_size));
    }
    VkImportMemoryHostPointerInfoEXT import_info{};
    import_info.sType =
        VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT;
    import_info.handleType =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    import_info.pHostPointer = host_ptr;
    VkMemoryAllocateInfo allocate_info{};
    allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate_info.pNext = &import_info;
    allocate_info.allocationSize = alloc_size;
    allocate_info.memoryTypeIndex = static_cast<uint32_t>(mem_type);
    ret = vkAllocateMemory(m_vdev_->getLogicalDevice(), &allocate_info,
                           nullptr, &ext_memory_);
    if (ret != VK_SUCCESS) {
        vkDestroyBuffer(m_vdev_->getLogicalDevice(), ext_buffer_, nullptr);
        ext_buffer_ = VK_NULL_HANDLE;
        throw std::runtime_error("vkop: host-import vkAllocateMemory failed "
                                 "vkret=" + std::to_string(ret));
    }
    ret = vkBindBufferMemory(m_vdev_->getLogicalDevice(), ext_buffer_,
                             ext_memory_, 0);
    if (ret != VK_SUCCESS) {
        vkFreeMemory(m_vdev_->getLogicalDevice(), ext_memory_, nullptr);
        vkDestroyBuffer(m_vdev_->getLogicalDevice(), ext_buffer_, nullptr);
        ext_memory_ = VK_NULL_HANDLE;
        ext_buffer_ = VK_NULL_HANDLE;
        throw std::runtime_error("vkop: host-import vkBindBufferMemory "
                                 "failed vkret=" + std::to_string(ret));
    }
    host_imported_ = true;
}

VulkanBuffer::~VulkanBuffer() {
    if (host_imported_) {
        // vkFreeMemory releases the imported-memory wrapper only — the host
        // block itself belongs to the caller and outlives this object.
        if (ext_buffer_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(m_vdev_->getLogicalDevice(), ext_buffer_, nullptr);
        }
        if (ext_memory_ != VK_NULL_HANDLE) {
            vkFreeMemory(m_vdev_->getLogicalDevice(), ext_memory_, nullptr);
        }
        return;
    }
    if (parent_) {
        return; // borrowed: the parent owns and frees the VkBuffer/memory
    }
#ifndef USE_VMA
    if (m_buffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(m_vdev_->getLogicalDevice(), m_buffer_, nullptr);
    }
#else
    m_vdev_->getVMA()->destroyBuffer(&m_vma_buffer_);
#endif
}

VkBuffer VulkanBuffer::getBuffer() const {
    if (host_imported_) {
        return ext_buffer_;
    }
    if (parent_) {
        return parent_->getBuffer();
    }
#ifndef USE_VMA
    return m_buffer_;
#else
    return m_vma_buffer_.buffer;
#endif
}

VkDeviceAddress VulkanBuffer::getDeviceAddress() const {
    if (!hasDeviceAddress()) {
        throw std::runtime_error(
            "vkop: VulkanBuffer::getDeviceAddress on a buffer without "
            "VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT (device BDA "
            "unsupported, or a host-imported buffer)");
    }
    if (parent_) {
        // A view addresses the same allocation at its window offset. The
        // caller must have chosen a shader whose buffer_reference_align
        // divides this offset (the descriptor path has no such constraint).
        return parent_->getDeviceAddress() + range_offset_;
    }
    if (device_address_ == 0) {
        device_address_ = m_vdev_->getBufferDeviceAddress(getBuffer());
        if (device_address_ == 0) {
            throw std::runtime_error(
                "vkop: vkGetBufferDeviceAddress returned 0 for a buffer "
                "created without VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT");
        }
    }
    return device_address_;
}

void VulkanBuffer::createBuffer(VkBufferUsageFlags usage,
                                VkMemoryPropertyFlags requireProperties) {
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = m_size_;
    // One choke point for every device buffer (tensors, scratch, indirect
    // args): make it addressable only when the device actually enabled BDA.
    // On a non-BDA device the usage flags are byte-for-byte what they were.
    buffer_info.usage = usage;
    if (m_vdev_->is_support_buffer_device_address()) {
        buffer_info.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        bda_capable_ = true;
    }
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    buffer_info.queueFamilyIndexCount = 0;
    buffer_info.pQueueFamilyIndices = nullptr;
#ifdef USE_VMA
    auto ret = m_vdev_->getVMA()->createBuffer(
        &buffer_info, &m_vma_buffer_,
        (requireProperties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0,
        requireProperties);
#else
    if (requireProperties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
    }
    auto ret = vkCreateBuffer(m_vdev_->getLogicalDevice(), &buffer_info,
                              nullptr, &m_buffer_);
#endif
    if (ret != VK_SUCCESS) {
        throw std::runtime_error(
            "Failed to create buffer! size=" + std::to_string(m_size_) +
            " usage=" + std::to_string(usage) +
            " vkret=" + std::to_string(ret));
    }
    if (usage & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) {
        m_desc_type_ = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    } else if (usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) {
        m_desc_type_ = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    } else if (usage & VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT) {
        m_desc_type_ = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
    } else if (usage & VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT) {
        m_desc_type_ = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
    }
}

std::variant<VkDescriptorImageInfo *, VkDescriptorBufferInfo *, VkBufferView *>
VulkanBuffer::getDescriptorInfo() {
    VkBuffer buff = getBuffer();
    buffer_info_.buffer = buff;
    // A borrowed sub-range binds the parent's VkBuffer at a byte offset; the
    // shader's element 0 then maps to parent byte `range_offset_` with no
    // shader change (the descriptor offset is transparent). An owning buffer
    // binds from 0. The offset is ROOT-relative (abs_offset), because
    // getBuffer() returns the root VkBuffer — a view of a view would otherwise
    // drop every ancestor's offset.
    buffer_info_.offset = parent_ ? abs_offset() : 0;
    buffer_info_.range = m_size_;
    return &buffer_info_;
}

void VulkanBuffer::transitionBuffer(VkCommandBuffer commandBuffer,
                                    VkAccessFlags dstAccessMask,
                                    VkPipelineStageFlags src_stage,
                                    VkPipelineStageFlags dst_stage,
                                    VkDeviceSize size, VkDeviceSize offset) {
    // A borrowed sub-range view forwards barriers to the parent: the access
    // state that matters belongs to the one real allocation, so the producer's
    // next transition sees the prior read/write. Over-wide (whole parent) is
    // safe.
    if (parent_) {
        parent_->transitionBuffer(commandBuffer, dstAccessMask, src_stage,
                                  dst_stage, size, offset);
        return;
    }
    VkBuffer buff = getBuffer();
    VkBufferMemoryBarrier barrier;
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.pNext = nullptr;
    barrier.srcAccessMask = m_access_;
    barrier.dstAccessMask = dstAccessMask;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buff;
    barrier.offset = offset;
    barrier.size = size;

    m_access_ = dstAccessMask;

    vkCmdPipelineBarrier(commandBuffer, src_stage, dst_stage, 0, 0, nullptr, 1,
                         &barrier, 0, nullptr);
}

void VulkanBuffer::shaderWriteBarrier(VkCommandBuffer commandBuffer,
                                      VkDeviceSize size, VkDeviceSize offset) {
    if (parent_) {
        parent_->shaderWriteBarrier(commandBuffer, size, offset);
        return;
    }
    // Explicitly set srcAccess=SHADER_WRITE (prior compute-shader writes),
    // dstAccess=SHADER_READ (subsequent compute-shader reads). Does NOT rely
    // on m_access_ tracking (which doesn't capture shader writes).
    VkBuffer buff = getBuffer();
    VkBufferMemoryBarrier barrier;
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.pNext = nullptr;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buff;
    barrier.offset = offset;
    barrier.size = size;
    m_access_ = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1,
                         &barrier, 0, nullptr);
}

void VulkanBuffer::indirectReadBarrier(VkCommandBuffer commandBuffer,
                                       VkDeviceSize size, VkDeviceSize offset) {
    if (parent_) {
        parent_->indirectReadBarrier(commandBuffer, size, offset);
        return;
    }
    // shader-write -> indirect-command-read. The dispatch_from_shape pre-pass
    // (compute shader) wrote the VkDispatchIndirectCommand;
    // vkCmdDispatchIndirect reads it at the DRAW_INDIRECT stage.
    // src=COMPUTE/SHADER_WRITE, dst=DRAW_INDIRECT/INDIRECT_COMMAND_READ.
    VkBuffer buff = getBuffer();
    VkBufferMemoryBarrier barrier;
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.pNext = nullptr;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buff;
    barrier.offset = offset;
    barrier.size = size;
    m_access_ = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 0, nullptr, 1,
                         &barrier, 0, nullptr);
}

void VulkanBuffer::transferBarrier(VkCommandBuffer commandBuffer,
                                   VkAccessFlags dstAccessMask,
                                   VkDeviceSize size, VkDeviceSize offset) {
    VkPipelineStageFlags source_stage =
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkPipelineStageFlags destination_stage =
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    transitionBuffer(commandBuffer, dstAccessMask, source_stage,
                     destination_stage, size, offset);
}

void VulkanBuffer::transferReadBarrier(VkCommandBuffer commandBuffer,
                                       VkDeviceSize size, VkDeviceSize offset) {
    transitionBuffer(commandBuffer, VK_ACCESS_TRANSFER_READ_BIT,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                     size, offset);
}

void VulkanBuffer::transferWriteBarrier(VkCommandBuffer commandBuffer,
                                        VkDeviceSize size,
                                        VkDeviceSize offset) {
    transitionBuffer(commandBuffer, VK_ACCESS_TRANSFER_WRITE_BIT,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, size, offset);
}

void VulkanBuffer::readBarrier(VkCommandBuffer commandBuffer, VkDeviceSize size,
                               VkDeviceSize offset) {
    transitionBuffer(commandBuffer, VK_ACCESS_SHADER_READ_BIT,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, size, offset);
}

void VulkanBuffer::writeBarrier(VkCommandBuffer commandBuffer,
                                VkDeviceSize size, VkDeviceSize offset) {
    transitionBuffer(commandBuffer, VK_ACCESS_SHADER_WRITE_BIT,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, size, offset);
}

// void VulkanBuffer::shaderWriteBarrier(VkCommandBuffer commandBuffer,
//                                       VkDeviceSize size, VkDeviceSize offset)
//                                       {
//     VkBuffer buff = getBuffer();
//     VkBufferMemoryBarrier barrier;
//     barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
//     barrier.pNext = nullptr;
//     barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
//     barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
//     barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
//     barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
//     barrier.buffer = buff;
//     barrier.offset = offset;
//     barrier.size = size;

//     m_access_ = VK_ACCESS_SHADER_READ_BIT;

//     vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
//                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
//                          1, &barrier, 0, nullptr);
// }

void VulkanBuffer::copyBufferToStageBuffer(VkCommandBuffer commandBuffer,
                                           VkBuffer dstbuffer,
                                           VkDeviceSize dstoffset,
                                           VkDeviceSize size,
                                           VkDeviceSize offset) {
    if (parent_) {
        // Read the view's window: shift by range_offset_ and default the size
        // to the range so a WHOLE_SIZE caller reads exactly the view.
        VkDeviceSize sz = (size == VK_WHOLE_SIZE) ? m_size_ : size;
        parent_->copyBufferToStageBuffer(commandBuffer, dstbuffer, dstoffset,
                                         sz, offset + range_offset_);
        return;
    }
    VkAccessFlags old_access = m_access_;
    if (m_access_ != VK_ACCESS_TRANSFER_READ_BIT) {
        transferReadBarrier(commandBuffer, size, offset);
    }
    VkBuffer buffer = getBuffer();

    VkBufferCopy copy_region = {};
    copy_region.srcOffset = offset;
    copy_region.dstOffset = dstoffset;
    copy_region.size = size;

    vkCmdCopyBuffer(commandBuffer, buffer, dstbuffer, 1, &copy_region);

    transferBarrier(commandBuffer, old_access, size, offset);
}

void VulkanBuffer::copyStageBufferToBuffer(VkCommandBuffer commandBuffer,
                                           VkBuffer srcbuffer,
                                           VkDeviceSize srcoffset,
                                           VkDeviceSize size,
                                           VkDeviceSize offset) {
    if (parent_) {
        VkDeviceSize sz = (size == VK_WHOLE_SIZE) ? m_size_ : size;
        parent_->copyStageBufferToBuffer(commandBuffer, srcbuffer, srcoffset,
                                         sz, offset + range_offset_);
        return;
    }
    VkAccessFlags old_access = m_access_;
    if (m_access_ != VK_ACCESS_TRANSFER_WRITE_BIT) {
        transferWriteBarrier(commandBuffer, size, offset);
    }
    VkBuffer buffer = getBuffer();

    VkBufferCopy copy_region = {};
    copy_region.srcOffset = srcoffset;
    copy_region.dstOffset = offset;
    copy_region.size = size;

    vkCmdCopyBuffer(commandBuffer, srcbuffer, buffer, 1, &copy_region);

    transferBarrier(commandBuffer, old_access, size, offset);
}

void VulkanBuffer::fillBuffer(VkCommandBuffer commandBuffer, uint32_t value,
                              VkDeviceSize size, VkDeviceSize offset) {
    if (parent_) {
        VkDeviceSize sz = (size == VK_WHOLE_SIZE) ? m_size_ : size;
        parent_->fillBuffer(commandBuffer, value, sz, offset + range_offset_);
        return;
    }
    VkAccessFlags old_access = m_access_;
    if (m_access_ != VK_ACCESS_TRANSFER_WRITE_BIT) {
        transferWriteBarrier(commandBuffer, size, offset);
    }
    VkBuffer buffer = getBuffer();
    vkCmdFillBuffer(commandBuffer, buffer, offset, size, value);
    transferBarrier(commandBuffer, old_access, size, offset);
}

void VulkanBuffer::updateBuffer(VkCommandBuffer commandBuffer, const void *data,
                                VkDeviceSize size, VkDeviceSize offset) {
    if (parent_) {
        parent_->updateBuffer(commandBuffer, data, size,
                              offset + range_offset_);
        return;
    }
    // vkCmdUpdateBuffer copies host data directly into the buffer during
    // command recording (no staging buffer). Limited to 65536 bytes per call
    // (Vulkan spec); Shape outputs are 32-64 bytes, well within range. The
    // driver captures the data inline, so `data` need not survive past this
    // call.
    assert(size <= 65536);
    VkAccessFlags old_access = m_access_;
    if (m_access_ != VK_ACCESS_TRANSFER_WRITE_BIT) {
        transferWriteBarrier(commandBuffer, size, offset);
    }
    VkBuffer buffer = getBuffer();
    vkCmdUpdateBuffer(commandBuffer, buffer, offset, size, data);
    transferBarrier(commandBuffer, old_access, size, offset);
}

void VulkanBuffer::flushMappedRange(VkDeviceSize offset, VkDeviceSize size) {
    if (parent_) {
        parent_->flushMappedRange(offset + range_offset_, size);
        return;
    }
    // The import memory-type scan requires HOST_COHERENT, so host writes are
    // already device-visible; a flush would also deref getMemory(), which is
    // null on the import path.
    if (host_imported_) {
        return;
    }
#ifdef USE_VMA
    VMA::flushAllocation(getVmaBuffer(), offset, size);
#else
    VkMappedMemoryRange range = {};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = getMemory();
    range.offset = getOffset() + offset;
    range.size = size;
    vkFlushMappedMemoryRanges(m_vdev_->getLogicalDevice(), 1, &range);
#endif
}
} // namespace vkop
