// vulkan/VulkanBasePipeline.cpp
#include "VulkanBasePipeline.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>

#include "include/logger.hpp"
#include "vulkan/DescriptorWriteProf.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanLib.hpp"

namespace vkop {

namespace {
VkDeviceSize align_up(VkDeviceSize value, VkDeviceSize alignment) {
    if (alignment <= 1) {
        return value;
    }
    return (value + alignment - 1) / alignment * alignment;
}
} // namespace

VulkanBasePipeline::VulkanBasePipeline(VkDevice device,
                                       std::vector<VkDescriptorType> types)
    : m_device_(device), m_types_(std::move(types)) {
    // The registry lookup keeps the raw device handle API at every
    // construction site while still making the device-level capabilities
    // (descriptor indexing bits, descriptor-buffer entry points) reachable.
    vdev_ = VulkanDevice::fromLogicalDevice(device);
}

VulkanBasePipeline::~VulkanBasePipeline() {
    // Descriptor-buffer memory first: it references the device that the
    // VulkanBuffer keeps a non-owning alias to, and the VkBuffer must be gone
    // before the device itself is destroyed by its owner.
    descriptor_buffer_storage_.reset();
    if (m_pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device_, m_pipeline_, nullptr);
        m_pipeline_ = VK_NULL_HANDLE;
    }

    if (m_descriptorSetLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_device_, m_descriptorSetLayout_,
                                     nullptr);
        m_descriptorSetLayout_ = VK_NULL_HANDLE;
    }

    if (m_pipelineLayout_ != VK_NULL_HANDLE) {

        vkDestroyPipelineLayout(m_device_, m_pipelineLayout_, nullptr);
        m_pipelineLayout_ = VK_NULL_HANDLE;
    }

    if (m_descriptorPool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_device_, m_descriptorPool_, nullptr);
        m_descriptorPool_ = VK_NULL_HANDLE;
    }
}

void VulkanBasePipeline::enable_descriptor_buffer() {
    if (descriptor_buffer_ || m_types_.empty() || !vdev_) {
        return;
    }
    const DescriptorBufferInfo *info = vdev_->descriptor_buffer_info();
    if (!info) {
        return;
    }
    // The write path turns VkDescriptorBufferInfo/VkDescriptorImageInfo into
    // VkDescriptorGetInfoEXT; every other descriptor type (texel buffers in
    // particular) would need information the VkWriteDescriptorSet does not
    // carry here, so those pipelines keep descriptor sets.
    for (VkDescriptorType type : m_types_) {
        switch (type) {
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
            break;
        default:
            return;
        }
    }
    // Descriptor memory must be host-writable (updates are plain memcpys via
    // vkGetDescriptorEXT), so it lives in HOST_VISIBLE|HOST_COHERENT memory.
    // On a discrete GPU that memory sits behind PCIe and the GPU's descriptor
    // fetch may cost more than the descriptor-set path it replaces; default
    // that case to sets unless VKOP_DESCRIPTOR_BUFFER=1 opts in. UMA (iGPU,
    // llvmpipe) reads the same memory the CPU wrote.
    const char *force = std::getenv("VKOP_DESCRIPTOR_BUFFER");
    const bool forced = force != nullptr && force[0] == '1';
    if (!vdev_->is_unified_memory() && !forced) {
        return;
    }
    vdev_keep_ = std::shared_ptr<VulkanDevice>(vdev_, [](VulkanDevice *) {});
    descriptor_buffer_ = info;
    if (std::getenv("VKOP_DESCRIPTOR_BUFFER_DEBUG")) {
        std::fprintf(stderr,
                     "[descbuf] pipeline with %zu bindings uses a "
                     "descriptor buffer\n",
                     m_types_.size());
    }
}

void VulkanBasePipeline::createDescriptorSetLayout(VkShaderStageFlags flags,
                                                   bool update_after_bind) {
    auto bindings = allocDescriptorSetLayoutBindings(flags);

    // Per-binding flags. UPDATE_AFTER_BIND is only set for descriptor types
    // whose feature was actually enabled on the device (the old code asked
    // only about storage buffers, which is wrong for SplitImage/Conv2dImage);
    // PARTIALLY_BOUND lets the buffer shaders' optional bindings stay
    // unwritten, which is exactly how fillWriteDescriptorSets() binds them.
    std::vector<VkDescriptorBindingFlags> bindingFlags(bindings.size(), 0);
    any_uab_binding_ = false;
    if (update_after_bind && !descriptor_buffer_) {
        for (size_t i = 0; i < bindings.size(); i++) {
            if (vdev_ &&
                vdev_->supports_update_after_bind(bindings[i].descriptorType)) {
                bindingFlags[i] |= VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
                any_uab_binding_ = true;
            }
        }
    }
    const bool partially_bound =
        vdev_ && vdev_->is_support_descriptor_partially_bound();
    if (partially_bound) {
        for (auto &f : bindingFlags) {
            f |= VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
        }
    }

    VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo = {};
    if (any_uab_binding_ || partially_bound) {
        bindingFlagsInfo.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
        bindingFlagsInfo.bindingCount =
            static_cast<uint32_t>(bindingFlags.size());
        bindingFlagsInfo.pBindingFlags = bindingFlags.data();
    }

    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
    layout_info.pBindings = bindings.data();
    if (any_uab_binding_ || partially_bound) {
        layout_info.pNext = &bindingFlagsInfo;
    }
    if (any_uab_binding_) {
        // Required by the binding-level UPDATE_AFTER_BIND flags above.
        layout_info.flags =
            VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    }
#ifdef VK_EXT_descriptor_buffer
    if (descriptor_buffer_) {
        // Marks the layout as descriptor-buffer based. It then cannot be used
        // with vkAllocateDescriptorSets, which is why db mode also leaves the
        // UPDATE_AFTER_BIND pool flag off.
        layout_info.flags |=
            VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
    }
#endif
    if (vkCreateDescriptorSetLayout(m_device_, &layout_info, nullptr,
                                    &m_descriptorSetLayout_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout!");
    }

#ifdef VK_EXT_descriptor_buffer
    if (descriptor_buffer_) {
        try {
            VkDeviceSize layout_size = 0;
            descriptor_buffer_->getLayoutSize(m_device_, m_descriptorSetLayout_,
                                              &layout_size);
            descriptor_buffer_binding_offsets_.resize(m_types_.size());
            for (uint32_t i = 0; i < m_types_.size(); ++i) {
                descriptor_buffer_->getBindingOffset(
                    m_device_, m_descriptorSetLayout_, i,
                    &descriptor_buffer_binding_offsets_[i]);
            }
            descriptor_buffer_stride_ =
                align_up(layout_size, descriptor_buffer_->offsetAlignment);
            if (descriptor_buffer_stride_ == 0) {
                throw std::runtime_error("empty descriptor set layout");
            }
            // Descriptor memory is host-written, but the GPU also fetches from
            // it at dispatch time. A device-local+host-visible type (BAR on a
            // discrete GPU, the one UMA type on an iGPU) keeps that fetch
            // local; without it the driver reads descriptors from system RAM
            // over PCIe, which on b3 costs ~2.4 ms per Qwen3-VL round (same
            // GPU timestamps, slower wall). Fall back to plain host-visible
            // when the device exposes no such type (or the BAR is too small).
            VkMemoryPropertyFlags db_mem = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            const VkMemoryPropertyFlags db_want =
                db_mem | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            VkPhysicalDeviceMemoryProperties db_mem_props{};
            vkGetPhysicalDeviceMemoryProperties(vdev_->getPhysicalDevice(),
                                                &db_mem_props);
            for (uint32_t i = 0; i < db_mem_props.memoryTypeCount; ++i) {
                if ((db_mem_props.memoryTypes[i].propertyFlags & db_want) ==
                    db_want) {
                    db_mem = db_want;
                    break;
                }
            }
            descriptor_buffer_storage_ = std::make_unique<VulkanBuffer>(
                vdev_keep_, descriptor_buffer_stride_ * kDescriptorBufferSlots,
                VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT, db_mem);
            descriptor_buffer_map_ =
                descriptor_buffer_storage_->getMappedMemory();
            descriptor_buffer_address_ =
                descriptor_buffer_storage_->getDeviceAddress();
            if (!descriptor_buffer_map_ || descriptor_buffer_address_ == 0) {
                throw std::runtime_error("descriptor buffer is not mappable");
            }
            descriptor_buffer_slot_used_.assign(kDescriptorBufferSlots, 0);
        } catch (const std::exception &e) {
            // The layout above was created with
            // VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT, so it
            // is NOT usable with vkAllocateDescriptorSets. Destroy it and
            // rebuild without the flag before falling back to descriptor sets.
            vkDestroyDescriptorSetLayout(m_device_, m_descriptorSetLayout_,
                                         nullptr);
            m_descriptorSetLayout_ = VK_NULL_HANDLE;
            LOG_WARN("descriptor buffer for this pipeline failed: %s; "
                     "rebuilding the layout for descriptor sets",
                     e.what());
            descriptor_buffer_ = nullptr;
            descriptor_buffer_storage_.reset();
            descriptor_buffer_map_ = nullptr;
            descriptor_buffer_address_ = 0;
            descriptor_buffer_stride_ = 0;
            descriptor_buffer_binding_offsets_.clear();
            descriptor_buffer_slot_used_.clear();
            // Rebuild with the original request (the UAB path was skipped
            // because db mode was active). Terminates: descriptor_buffer_ is
            // null now.
            createDescriptorSetLayout(flags, update_after_bind);
        }
    }
#endif
}

std::vector<VkDescriptorSetLayoutBinding>
VulkanBasePipeline::allocDescriptorSetLayoutBindings(VkShaderStageFlags flags) {
    std::vector<VkDescriptorSetLayoutBinding> bindings(m_types_.size());
    for (int i = 0; i < static_cast<int>(m_types_.size()); i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = m_types_[i];
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = flags;
    }
    return bindings;
}

void VulkanBasePipeline::createDescriptorPool(bool update_after_bind) {
    if (descriptor_buffer_) {
        return; // no pools, no sets: allocDescriptorSets() hands out slots
    }
    // The pool flag must mirror the layout flag, and the layout only got it
    // when at least one binding could actually carry UPDATE_AFTER_BIND. The
    // caller's request alone is not enough (unsupported type bits).
    (void)update_after_bind;
    update_after_bind = any_uab_binding_;
    std::map<VkDescriptorType, int> type_counts;
    for (auto t : m_types_) {
        if (type_counts.find(t) == type_counts.end()) {
            type_counts[t] = 1;
        } else {
            type_counts[t]++;
        }
    }
    std::vector<VkDescriptorPoolSize> pool_sizes;
    for (auto &t : type_counts) {
        VkDescriptorPoolSize ps;
        ps.type = t.first;
        ps.descriptorCount = t.second * kInflight * 16;
        pool_sizes.push_back(ps);
    }

    VkDescriptorPoolCreateInfo descriptor_pool_create_info = {};
    descriptor_pool_create_info.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptor_pool_create_info.flags =
        VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    if (update_after_bind) {
        descriptor_pool_create_info.flags |=
            VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    }
    descriptor_pool_create_info.maxSets = kInflight * 16;
    descriptor_pool_create_info.poolSizeCount =
        static_cast<uint32_t>(pool_sizes.size());
    descriptor_pool_create_info.pPoolSizes = pool_sizes.data();
    if (vkCreateDescriptorPool(m_device_, &descriptor_pool_create_info, nullptr,
                               &m_descriptorPool_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor pool!");
    }
}

void VulkanBasePipeline::createPipelineLayout(VkShaderStageFlags flags,
                                              size_t pushconstant_size) {
    std::vector<VkPushConstantRange> push_constant_ranges;
    if (pushconstant_size > 0) {
        VkPushConstantRange push_constant_range{};
        push_constant_range.stageFlags = flags;
        push_constant_range.offset = 0;
        push_constant_range.size = static_cast<uint32_t>(pushconstant_size);
        push_constant_ranges.push_back(push_constant_range);
    }

    VkPipelineLayoutCreateInfo pipeline_layout_create_info = {};
    pipeline_layout_create_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    // A BDA pipeline has no descriptor bindings at all; give it zero set
    // layouts rather than one empty set, so dispatching without a
    // vkCmdBindDescriptorSets is unambiguously valid.
    if (m_types_.empty()) {
        pipeline_layout_create_info.setLayoutCount = 0;
        pipeline_layout_create_info.pSetLayouts = nullptr;
    } else {
        pipeline_layout_create_info.setLayoutCount = 1;
        pipeline_layout_create_info.pSetLayouts = &m_descriptorSetLayout_;
    }
    pipeline_layout_create_info.pushConstantRangeCount =
        static_cast<uint32_t>(push_constant_ranges.size());
    pipeline_layout_create_info.pPushConstantRanges =
        push_constant_ranges.empty() ? nullptr : push_constant_ranges.data();

    if (vkCreatePipelineLayout(m_device_, &pipeline_layout_create_info, nullptr,
                               &m_pipelineLayout_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout!");
    }
}

VkDescriptorSet VulkanBasePipeline::allocDescriptorSets() {
    if (descriptor_buffer_) {
        for (size_t i = 0; i < descriptor_buffer_slot_used_.size(); ++i) {
            if (descriptor_buffer_slot_used_[i] == 0) {
                descriptor_buffer_slot_used_[i] = 1;
                // Slot token, not a real VkDescriptorSet: 0 is reserved for
                // VK_NULL_HANDLE, so the token is index+1. Every consumer
                // (updateDescriptorSets, freeDescriptorSets,
                // VulkanCommandBuffer::bind) branches on usesDescriptorBuffer()
                // before touching it, so the fake handle never reaches Vulkan.
                return reinterpret_cast<VkDescriptorSet>(
                    static_cast<uintptr_t>(i + 1));
            }
        }
        throw std::runtime_error(
            "descriptor buffer slots exhausted (kInflight * 16)");
    }
    VkDescriptorSet descriptor_set;
    VkDescriptorSetAllocateInfo descriptor_set_allocate_info = {};
    descriptor_set_allocate_info.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptor_set_allocate_info.descriptorPool = m_descriptorPool_;
    descriptor_set_allocate_info.descriptorSetCount = 1;
    descriptor_set_allocate_info.pSetLayouts = &m_descriptorSetLayout_;

    const uint64_t t_prof0 = ds_prof_now();
    auto ret = vkAllocateDescriptorSets(
        m_device_, &descriptor_set_allocate_info, &descriptor_set);
    ds_prof_record("alloc", ds_prof_now() - t_prof0, 1);
    if (ret != VK_SUCCESS) {
        printf("allocate descriptor set fail %d\n", ret);
        throw std::runtime_error("Failed to allocate descriptor set!");
    }
    return descriptor_set;
}

void VulkanBasePipeline::freeDescriptorSets(VkDescriptorSet ds) {
    if (descriptor_buffer_) {
        const uintptr_t token = reinterpret_cast<uintptr_t>(ds);
        if (token > 0 && token <= descriptor_buffer_slot_used_.size()) {
            descriptor_buffer_slot_used_[token - 1] = 0;
        }
        return;
    }
    if (ds != VK_NULL_HANDLE) {
        const uint64_t t_prof0 = ds_prof_now();
        vkFreeDescriptorSets(m_device_, m_descriptorPool_, 1, &ds);
        ds_prof_record("free", ds_prof_now() - t_prof0, 1);
    }
}

void VulkanBasePipeline::updateDescriptorSets(
    const std::vector<VkWriteDescriptorSet> &writes) {
    if (descriptor_buffer_) {
        // The descriptor-buffer equivalent of vkUpdateDescriptorSets is a
        // host memcpy per binding (vkGetDescriptorEXT writes the same bytes a
        // set would hold), which is the point of the extension: no driver
        // call, no pool, no set object.
        const uint64_t t_prof0 = ds_prof_now();
        for (const auto &write : writes) {
            writeDescriptorBuffer(write);
        }
        ds_prof_record("update", ds_prof_now() - t_prof0,
                       static_cast<uint32_t>(writes.size()));
        return;
    }
    const uint64_t t_prof0 = ds_prof_now();
    vkUpdateDescriptorSets(m_device_, static_cast<uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);
    ds_prof_record("update", ds_prof_now() - t_prof0,
                   static_cast<uint32_t>(writes.size()));
}

VkDeviceSize
VulkanBasePipeline::descriptorBufferSlotOffset(VkDescriptorSet token) const {
    const uintptr_t raw = reinterpret_cast<uintptr_t>(token);
    if (raw == 0 || raw > descriptor_buffer_slot_used_.size()) {
        throw std::runtime_error(
            "descriptor buffer: invalid set token (wrong pipeline?)");
    }
    return static_cast<VkDeviceSize>(raw - 1) * descriptor_buffer_stride_;
}

void VulkanBasePipeline::writeDescriptorBuffer(
    const VkWriteDescriptorSet &write) {
#ifdef VK_EXT_descriptor_buffer
    if (!descriptor_buffer_ || !descriptor_buffer_map_ ||
        write.dstSet == VK_NULL_HANDLE) {
        return;
    }
    if (write.dstBinding >= descriptor_buffer_binding_offsets_.size()) {
        throw std::runtime_error(
            "descriptor buffer: write to undeclared binding " +
            std::to_string(write.dstBinding));
    }
    if (write.descriptorCount != 1 || write.dstArrayElement != 0) {
        throw std::runtime_error(
            "descriptor buffer: only single-element descriptor writes are "
            "supported");
    }
    const VkDeviceSize offset =
        descriptorBufferSlotOffset(write.dstSet) +
        descriptor_buffer_binding_offsets_[write.dstBinding];

    VkDescriptorGetInfoEXT get_info{};
    get_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT;
    get_info.type = write.descriptorType;

    VkDescriptorAddressInfoEXT address_info{};
    address_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT;
    address_info.format = VK_FORMAT_UNDEFINED;

    switch (write.descriptorType) {
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: {
        if (!write.pBufferInfo) {
            throw std::runtime_error(
                "descriptor buffer: buffer descriptor without pBufferInfo");
        }
        VkBufferDeviceAddressInfo bda_info{};
        bda_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        bda_info.buffer = write.pBufferInfo->buffer;
        address_info.address =
            descriptor_buffer_->getBufferAddress(m_device_, &bda_info) +
            write.pBufferInfo->offset;
        // VkDescriptorAddressInfoEXT has no VK_WHOLE_SIZE form; descriptor
        // writes here always carry the resource's actual range.
        address_info.range = write.pBufferInfo->range;
        if (write.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
            get_info.data.pStorageBuffer = &address_info;
        } else {
            get_info.data.pUniformBuffer = &address_info;
        }
        break;
    }
    case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
        get_info.data.pCombinedImageSampler = write.pImageInfo;
        break;
    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        get_info.data.pSampledImage = write.pImageInfo;
        break;
    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
        get_info.data.pStorageImage = write.pImageInfo;
        break;
    case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
    case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
        // Descriptor buffers express texel buffers as an address+format
        // (VkDescriptorAddressInfoEXT), not as a VkBufferView, and a
        // VkWriteDescriptorSet carries only the view handle. Pipelines that
        // declare these types stay on descriptor sets (see
        // enable_descriptor_buffer).
        throw std::runtime_error(
            "descriptor buffer: texel buffer bindings are not supported");
    default:
        throw std::runtime_error("descriptor buffer: unsupported descriptor "
                                 "type " +
                                 std::to_string(write.descriptorType));
    }

    const VkDeviceSize size = descriptor_buffer_->sizeFor(write.descriptorType);
    if (size == 0) {
        throw std::runtime_error(
            "descriptor buffer: no descriptor size for type " +
            std::to_string(write.descriptorType));
    }
    descriptor_buffer_->getDescriptor(
        m_device_, &get_info, size,
        static_cast<uint8_t *>(descriptor_buffer_map_) + offset);
#endif
}

VkBuffer VulkanBasePipeline::getDescriptorBuffer() const {
    return descriptor_buffer_storage_ ? descriptor_buffer_storage_->getBuffer()
                                      : VK_NULL_HANDLE;
}

void VulkanBasePipeline::bindDescriptorBuffer(VkCommandBuffer cmd) const {
#ifdef VK_EXT_descriptor_buffer
    if (!descriptor_buffer_ || !descriptor_buffer_storage_) {
        return;
    }
    VkDescriptorBufferBindingInfoEXT binding{};
    binding.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
    binding.address = descriptor_buffer_address_;
    binding.usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT;
    descriptor_buffer_->cmdBindBuffers(cmd, 1, &binding);
#else
    (void)cmd;
#endif
}

void VulkanBasePipeline::setDescriptorBufferOffsets(
    VkCommandBuffer cmd, VkDescriptorSet token) const {
#ifdef VK_EXT_descriptor_buffer
    if (!descriptor_buffer_) {
        return;
    }
    const uint32_t buffer_index = 0;
    const VkDeviceSize offset = descriptorBufferSlotOffset(token);
    descriptor_buffer_->cmdSetOffsets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                      m_pipelineLayout_, 0, 1, &buffer_index,
                                      &offset);
#else
    (void)cmd;
    (void)token;
#endif
}

} // namespace vkop
