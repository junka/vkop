// vulkan/VulkanBasePipeline.hpp
#ifndef __VULKANBASEPIPELINE_HPP__
#define __VULKANBASEPIPELINE_HPP__

#include "vulkan/VulkanDevice.hpp"
#include <memory>
#include <vector>

namespace vkop {

class VulkanBuffer;
class DescriptorBufferPool;

class VulkanBasePipeline {
  public:
    virtual ~VulkanBasePipeline();

  protected:
    VulkanBasePipeline(VkDevice device, std::vector<VkDescriptorType> types);

    // Common members
    VkDevice m_device_;
    std::vector<VkDescriptorType> m_types_;

    VkDescriptorSetLayout m_descriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline m_pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptorPool_ = VK_NULL_HANDLE;

    // Common methods
    void createDescriptorSetLayout(VkShaderStageFlags flags,
                                   bool update_after_bind = false);
    void createDescriptorPool(bool update_after_bind = false);
    void createPipelineLayout(VkShaderStageFlags flags,
                              size_t pushconstant_size = 0);

    std::vector<VkDescriptorSetLayoutBinding>
    allocDescriptorSetLayoutBindings(VkShaderStageFlags flags);

    // Switch this pipeline's descriptor backend to VK_EXT_descriptor_buffer.
    // Called by VulkanPipeline before the layout is built; a no-op when the
    // extension is unsupported, the pipeline declares no descriptor sets, or
    // the device is discrete and VKOP_DESCRIPTOR_BUFFER did not opt in. Once
    // enabled, every descriptor lives in one host-visible buffer owned here:
    // allocDescriptorSets() hands out slot tokens, updateDescriptorSets()
    // memcpys via vkGetDescriptorEXT, and a dispatch only records
    // vkCmdSetDescriptorBufferOffsetsEXT.
    void enable_descriptor_buffer();

  public:
    // Public interface
    VkPipeline getPipeline() const { return m_pipeline_; }
    VkPipelineLayout getPipelineLayout() const { return m_pipelineLayout_; }

    VkDescriptorSet allocDescriptorSets();
    void freeDescriptorSets(VkDescriptorSet ds);
    void updateDescriptorSets(const std::vector<VkWriteDescriptorSet> &writes);

    // Descriptor-buffer accessors for VulkanCommandBuffer::bind(). In
    // descriptor-set mode usesDescriptorBuffer() is false and these are unused.
    bool usesDescriptorBuffer() const { return descriptor_buffer_ != nullptr; }
    VkBuffer getDescriptorBuffer() const;
    // vkCmdBindDescriptorBuffersEXT(this buffer) at buffer index 0.
    void bindDescriptorBuffer(VkCommandBuffer cmd) const;
    // vkCmdSetDescriptorBufferOffsetsEXT for the set token at buffer index 0.
    void setDescriptorBufferOffsets(VkCommandBuffer cmd,
                                    VkDescriptorSet token) const;

  protected:
    VulkanDevice *vdev_ = nullptr;
    const DescriptorBufferInfo *descriptor_buffer_ = nullptr;
    // One slot per concurrent descriptor set, matching the old descriptor
    // pool's maxSets (kInflight * 16). Slots are allocated lazily and reused
    // after freeDescriptorSets().
    static constexpr size_t kDescriptorBufferSlots = kInflight * 16;
    // Region of the device-wide descriptor buffer owned by this pipeline. The
    // pool keeps its VkBuffer shared by every pipeline, so a recording binds
    // the buffer once and only changes offsets per dispatch.
    std::shared_ptr<DescriptorBufferPool> descriptor_buffer_pool_;
    VkDeviceSize descriptor_buffer_region_ = 0;
    VkDeviceSize descriptor_buffer_region_size_ = 0;
    void *descriptor_buffer_map_ = nullptr;
    VkDeviceSize descriptor_buffer_stride_ = 0;
    std::vector<VkDeviceSize> descriptor_buffer_binding_offsets_;
    std::vector<uint8_t> descriptor_buffer_slot_used_;
    // Set when createDescriptorSetLayout saw at least one binding that can
    // carry UPDATE_AFTER_BIND; controls the pool flag.
    bool any_uab_binding_ = false;

    VkDeviceSize descriptorBufferSlotOffset(VkDescriptorSet token) const;
    void writeDescriptorBuffer(const VkWriteDescriptorSet &write);
};

} // namespace vkop

#endif // __VULKANBASEPIPELINE_HPP__