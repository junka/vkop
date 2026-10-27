// Copyright 2025 @junka
#ifndef SRC_VULKANBUFFER_HPP_
#define SRC_VULKANBUFFER_HPP_

#include <variant>
#include <vulkan/vulkan.hpp>

#include "vulkan/VulkanResource.hpp"

namespace vkop {

class VulkanBuffer : public VulkanResource {
  public:
    VulkanBuffer(std::shared_ptr<VulkanDevice> &vdev, VkDeviceSize size,
                 VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                 int ext_fd = -1);

    // Borrowed sub-range ctor: this object OWNS NOTHING — it shares `parent`'s
    // VkBuffer and exposes only the byte window [offset, offset+range). The
    // descriptor it produces is bound at that offset (VkDescriptorBufferInfo
    // offset is transparent to the shader, so a consumer reads element 0 =
    // parent byte `offset` with no shader change). Used for zero-copy Slice/
    // Split views: the output aliases the producer's buffer at a byte offset
    // instead of moving the bytes. The parent is kept alive by shared_ptr.
    // All barriers and the view_aliased/view_writer bookkeeping forward to the
    // parent, so access tracking and the shape-pool write-site hazard logic
    // keep keying on the one real allocation.
    VulkanBuffer(std::shared_ptr<VulkanBuffer> parent, VkDeviceSize offset,
                 VkDeviceSize range);
    ~VulkanBuffer() override;

    VkBuffer getBuffer() const;
    ResourceType getResourceType() const override {
        return ResourceType::VK_BUFFER;
    }

    // True for a borrowed sub-range view (see the ctor above). Not the owner of
    // the VkBuffer; never map/free it.
    bool is_view_range() const { return parent_ != nullptr; }
    std::shared_ptr<VulkanBuffer> parent() const { return parent_; }
    // Byte offset of this view's window within its immediate parent (the
    // parent-relative offset). Forwarding the byte movers adds this up the
    // chain, so it must stay parent-relative.
    VkDeviceSize range_offset() const { return range_offset_; }
    // Byte offset within the ROOT allocation. getBuffer() recurses to the root
    // VkBuffer, so the descriptor offset must accumulate the chain too: a view
    // of a view (a contiguous Slice of a Split block) is otherwise bound short
    // by every ancestor's offset. Only the descriptor path needs this — the
    // movers adopt the root buffer but walk offsets incrementally.
    VkDeviceSize abs_offset() const {
        return range_offset_ + (parent_ ? parent_->abs_offset() : 0);
    }
    std::variant<VkDescriptorImageInfo *, VkDescriptorBufferInfo *,
                 VkBufferView *>
    getDescriptorInfo() override;

    VkDeviceSize getSize() const { return m_size_; }

    // True once some tensor handed this buffer's bytes to ANOTHER tensor object
    // as a view alias (Tensor::alias_storage_buffer, the Reshape/Squeeze/
    // Unsqueeze fast path). From then on at least two tensor objects address
    // the same bytes. Sticky: an alias cannot be revoked from the producer
    // side. A borrowed sub-range view reports the PARENT's flag, so the
    // write-site hazard check in Runtime::Run sees the shared bytes.
    bool view_aliased() const {
        return parent_ ? parent_->view_aliased() : view_aliased_;
    }
    void mark_view_aliased() {
        if (parent_) {
            parent_->mark_view_aliased();
        } else {
            view_aliased_ = true;
        }
    }

    // Global node index of the last op that WROTE these bytes (-1 = never).
    // With view_aliased it lets Runtime::Run tell the two cases apart at the
    // write site: the same node writing again is the LLM/DiT round loop
    // re-making the same tensor (the view re-aliases the same bytes every
    // round, so a fresh buffer there would be pure churn), while a DIFFERENT
    // node is the shape pool having recycled this tensor object to an unrelated
    // op — that write would clobber what the aliased view still reads, so the
    // recycler takes its own buffer instead.
    //
    // A borrowed sub-range view keeps its OWN marker: it is RE-ALIASED each
    // round, not written, so Runtime's post-execute set_gpu_buffer_view_writer
    // must not stomp the parent's writer (which belongs to the real producer).
    // Reading the view's writer falls back to the parent's when unset, so the
    // aliased-from-producer case is still detected on the first round.
    int32_t view_writer() const {
        if (!parent_)
            return view_writer_;
        return view_writer_ >= 0 ? view_writer_ : parent_->view_writer();
    }
    void set_view_writer(int32_t node_idx) { view_writer_ = node_idx; }

    void transferBarrier(VkCommandBuffer commandBuffer,
                         VkAccessFlags dstAccessMask,
                         VkDeviceSize size = VK_WHOLE_SIZE,
                         VkDeviceSize offset = 0);
    void transferWriteBarrier(VkCommandBuffer commandBuffer,
                              VkDeviceSize size = VK_WHOLE_SIZE,
                              VkDeviceSize offset = 0);
    void transferReadBarrier(VkCommandBuffer commandBuffer,
                             VkDeviceSize size = VK_WHOLE_SIZE,
                             VkDeviceSize offset = 0);
    void readBarrier(VkCommandBuffer commandBuffer,
                     VkDeviceSize size = VK_WHOLE_SIZE,
                     VkDeviceSize offset = 0);
    void writeBarrier(VkCommandBuffer commandBuffer,
                      VkDeviceSize size = VK_WHOLE_SIZE,
                      VkDeviceSize offset = 0);
    // Proper shader-write -> shader-read barrier. Unlike
    // writeBarrier+readBarrier (which rely on m_access_ tracking that doesn't
    // capture shader writes), this explicitly sets srcAccess=SHADER_WRITE to
    // flush prior compute-shader writes before a subsequent compute-shader
    // read.
    void shaderWriteBarrier(VkCommandBuffer commandBuffer,
                            VkDeviceSize size = VK_WHOLE_SIZE,
                            VkDeviceSize offset = 0);
    // shader-write -> indirect-command-read barrier. Used after a compute
    // shader (dispatch_from_shape.comp) writes a VkDispatchIndirectCommand
    // into an indirect buffer, before vkCmdDispatchIndirect reads it. The
    // dst stage is DRAW_INDIRECT (the pipeline stage that sources indirect
    // command data), dst access INDIRECT_COMMAND_READ. Without this, the
    // indirect dispatch may read stale/unflushed bytes from the pre-pass.
    void indirectReadBarrier(VkCommandBuffer commandBuffer,
                             VkDeviceSize size = VK_WHOLE_SIZE,
                             VkDeviceSize offset = 0);

    void copyBufferToStageBuffer(VkCommandBuffer commandBuffer,
                                 VkBuffer dstbuffer, VkDeviceSize dstoffset,
                                 VkDeviceSize size, VkDeviceSize offset = 0);

    // Fill the whole buffer (or a [offset, offset+size) region) with a 4-byte
    // pattern via vkCmdFillBuffer. Transitions to TRANSFER_WRITE before and
    // back to the prior access afterwards. Requires TRANSFER_DST usage.
    void fillBuffer(VkCommandBuffer commandBuffer, uint32_t value,
                    VkDeviceSize size = VK_WHOLE_SIZE, VkDeviceSize offset = 0);

    void copyStageBufferToBuffer(VkCommandBuffer commandBuffer,
                                 VkBuffer srcbuffer, VkDeviceSize srcoffset,
                                 VkDeviceSize size, VkDeviceSize offset = 0);

    // Copy up to 65536 bytes of host data directly into this buffer during
    // command recording via vkCmdUpdateBuffer — NO staging buffer, NO staging
    // pool, NO submit+wait. The driver copies the data inline into the command
    // buffer (or defers it), so `data` must remain valid only for the duration
    // of this call (not until submit). Used by CPU-only shape-meta producers
    // (Shape) whose tiny outputs (32-64 bytes) don't justify a synchronous
    // copyToGPU pipeline stall. Transitions to TRANSFER_WRITE then back to the
    // prior access (shader read).
    void updateBuffer(VkCommandBuffer commandBuffer, const void *data,
                      VkDeviceSize size, VkDeviceSize offset = 0);

    void *getMappedMemory() {
        if (parent_)
            return parent_->getMappedMemory();
#ifdef USE_VMA
        return VMA::getMappedMemory(&m_vma_buffer_);

#else
        if (data_)
            return data_;
        auto ret = vkMapMemory(m_vdev_->getLogicalDevice(), getMemory(),
                               getOffset(), m_size_, 0, &data_);
        if (ret != VK_SUCCESS) {
            return nullptr;
        }
        return data_;
#endif
    };

#ifdef USE_VMA
    VMA::VmaBuffer *getVmaBuffer() {
        return parent_ ? parent_->getVmaBuffer() : &m_vma_buffer_;
    }
#endif

    void unmapMemory() {
        if (parent_) {
            parent_->unmapMemory();
            return;
        }
        // VMA_ALLOCATION_CREATE_MAPPED_BIT already keeps the memory mapped, so
        // we don't need to unmap it.
#ifndef USE_VMA
        if (data_) {
            vkUnmapMemory(m_vdev_->getLogicalDevice(), getMemory());
            data_ = nullptr;
        }
#endif
    }

  private:
#ifndef USE_VMA
    VkBuffer m_buffer_ = VK_NULL_HANDLE;
#else
    VMA::VmaBuffer m_vma_buffer_;
#endif
    VkDeviceSize m_size_;
    VkAccessFlags m_access_ = 0;
    bool view_aliased_ = false;
    int32_t view_writer_ = -1;
    void *data_ = nullptr;
    // Borrowed sub-range state (null parent_ = this object owns its VkBuffer).
    std::shared_ptr<VulkanBuffer> parent_;
    VkDeviceSize range_offset_ = 0;

    VkDescriptorBufferInfo buffer_info_;

    void transitionBuffer(VkCommandBuffer commandBuffer,
                          VkAccessFlags dstAccessMask,
                          VkPipelineStageFlags src_stage,
                          VkPipelineStageFlags dst_stage,
                          VkDeviceSize size = VK_WHOLE_SIZE,
                          VkDeviceSize offset = 0);
    void createBuffer(VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags requireProperties);
    void createBufferView(VkFormat format, VkDeviceSize offset);
};
} // namespace vkop
#endif // SRC_VULKANBUFFER_HPP_
