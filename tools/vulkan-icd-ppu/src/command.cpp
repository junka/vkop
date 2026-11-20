// Command pools and command buffers. Every vkCmd* here only records; nothing
// touches CUDA until submit.cpp replays the list.
#include "icd_common.hpp"

#include <algorithm>
#include <cstring>

namespace ppu {

bool alloc_scratch(CommandBuffer_T *cb, VkDeviceSize bytes, ppu::CUdeviceptr *dev, void **host) {
  if (cb == nullptr || cb->scratch_host == nullptr) return false;
  VkDeviceSize aligned = (bytes + 255) & ~static_cast<VkDeviceSize>(255);
  if (cb->scratch_used + aligned > cb->scratch_size) return false;
  *dev = cb->scratch_dev + cb->scratch_used;
  *host = static_cast<uint8_t *>(cb->scratch_host) + cb->scratch_used;
  cb->scratch_used += aligned;
  return true;
}

void free_scratch(CommandBuffer_T *cb) {
  if (cb == nullptr) return;
  ppu::Driver &d = driver();
  if (cb->scratch_host != nullptr && d.cuMemFreeHost) d.cuMemFreeHost(cb->scratch_host);
  cb->scratch_host = nullptr;
  cb->scratch_dev = 0;
  cb->scratch_size = 0;
  cb->scratch_used = 0;
}

}  // namespace ppu

namespace {

Device_T *as_device(VkDevice device) { return checked_device(device, __func__); }
CommandBuffer_T *as_cb(VkCommandBuffer cb) { return reinterpret_cast<CommandBuffer_T *>(cb); }

}  // namespace

// ---------------------------------------------------------------------------
// Pools
// ---------------------------------------------------------------------------
extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkCreateCommandPool(
    VkDevice device, const VkCommandPoolCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkCommandPool *pCommandPool) {
  (void)pAllocator;
  Device_T *dev = as_device(device);
  if (dev == nullptr || pCreateInfo == nullptr || pCommandPool == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  if (pCreateInfo->queueFamilyIndex != 0) {
    PPU_ERR("vkCreateCommandPool: only family 0 exists, requested %u",
            pCreateInfo->queueFamilyIndex);
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  auto *pool = new CommandPool_T();
  pool->device = dev;
  pool->family_index = pCreateInfo->queueFamilyIndex;
  pool->flags = pCreateInfo->flags;
  *pCommandPool = reinterpret_cast<VkCommandPool>(pool);
  return VK_SUCCESS;
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkDestroyCommandPool(VkDevice device,
                                                            VkCommandPool commandPool,
                                                            const VkAllocationCallbacks *pAllocator) {
  (void)pAllocator;
  (void)device;
  auto *pool = reinterpret_cast<CommandPool_T *>(commandPool);
  if (pool == nullptr) return;
  for (auto *cb : pool->buffers) {
    ppu::free_scratch(cb);
    delete cb;
  }
  delete pool;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkResetCommandPool(VkDevice device,
                                                              VkCommandPool commandPool,
                                                              VkCommandPoolResetFlags flags) {
  (void)flags;
  (void)device;
  auto *pool = reinterpret_cast<CommandPool_T *>(commandPool);
  if (pool == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
  for (auto *cb : pool->buffers) {
    cb->ops.clear();
    cb->recording = false;
    cb->ever_recorded = false;
    cb->pipeline = nullptr;
    cb->bound_sets.clear();
    cb->push_constants.clear();
    cb->scratch_used = 0;
  }
  return VK_SUCCESS;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkAllocateCommandBuffers(
    VkDevice device, const VkCommandBufferAllocateInfo *pAllocateInfo,
    VkCommandBuffer *pCommandBuffers) {
  Device_T *dev = as_device(device);
  if (dev == nullptr || pAllocateInfo == nullptr || pCommandBuffers == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  auto *pool = reinterpret_cast<CommandPool_T *>(pAllocateInfo->commandPool);
  if (pool == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
  if (pAllocateInfo->level != VK_COMMAND_BUFFER_LEVEL_PRIMARY) {
    // Secondary command buffers are not implemented; say so instead of
    // returning a handle that cannot work.
    PPU_ERR("vkAllocateCommandBuffers: only PRIMARY command buffers are supported");
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }

  ppu::Driver &d = ppu::driver();
  constexpr VkDeviceSize kScratch = 64 * 1024;
  for (uint32_t i = 0; i < pAllocateInfo->commandBufferCount; ++i) {
    auto *cb = new CommandBuffer_T();
    ppu::stamp(&cb->loader_data);
    cb->device = dev;
    cb->pool = pAllocateInfo->commandPool;
    // Pre-allocate the host-mapped scratch once per command buffer.
    void *host = nullptr;
    if (d.cuMemHostAlloc &&
        d.cuMemHostAlloc(&host, kScratch, ppu::CU_MEMHOSTALLOC_DEVICEMAP) == ppu::CUDA_SUCCESS &&
        host != nullptr) {
      ppu::CUdeviceptr dptr = 0;
      if (d.cuMemHostGetDevicePointer_v2(&dptr, host, 0) == ppu::CUDA_SUCCESS) {
        cb->scratch_host = host;
        cb->scratch_dev = dptr;
        cb->scratch_size = kScratch;
      } else {
        d.cuMemFreeHost(host);
      }
    }
    pool->buffers.push_back(cb);
    pCommandBuffers[i] = reinterpret_cast<VkCommandBuffer>(cb);
  }
  return VK_SUCCESS;
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkFreeCommandBuffers(
    VkDevice device, VkCommandPool commandPool, uint32_t commandBufferCount,
    const VkCommandBuffer *pCommandBuffers) {
  (void)device;
  auto *pool = reinterpret_cast<CommandPool_T *>(commandPool);
  for (uint32_t i = 0; i < commandBufferCount; ++i) {
    auto *cb = as_cb(pCommandBuffers[i]);
    if (cb == nullptr) continue;
    if (pool != nullptr) {
      auto it = std::find(pool->buffers.begin(), pool->buffers.end(), cb);
      if (it != pool->buffers.end()) pool->buffers.erase(it);
    }
    ppu::free_scratch(cb);
    delete cb;
  }
}

// ---------------------------------------------------------------------------
// Recording state
// ---------------------------------------------------------------------------
extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkBeginCommandBuffer(
    VkCommandBuffer commandBuffer, const VkCommandBufferBeginInfo *pBeginInfo) {
  auto *cb = as_cb(commandBuffer);
  if (cb == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
  if (cb->recording) return VK_ERROR_INITIALIZATION_FAILED;
  cb->ops.clear();
  cb->pipeline = nullptr;
  cb->bound_sets.clear();
  cb->push_constants.clear();
  cb->scratch_used = 0;
  cb->recording = true;
  cb->ever_recorded = true;
  (void)pBeginInfo;
  return VK_SUCCESS;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkEndCommandBuffer(VkCommandBuffer commandBuffer) {
  auto *cb = as_cb(commandBuffer);
  if (cb == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
  if (!cb->recording) return VK_ERROR_INITIALIZATION_FAILED;
  cb->recording = false;
  PPU_LOG("vkEndCommandBuffer: %zu op(s) recorded", cb->ops.size());
  return VK_SUCCESS;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkResetCommandBuffer(
    VkCommandBuffer commandBuffer, VkCommandBufferResetFlags flags) {
  (void)flags;
  auto *cb = as_cb(commandBuffer);
  if (cb == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
  cb->ops.clear();
  cb->recording = false;
  cb->pipeline = nullptr;
  cb->bound_sets.clear();
  cb->push_constants.clear();
  cb->scratch_used = 0;
  return VK_SUCCESS;
}

// ---------------------------------------------------------------------------
// Binding commands (recorded state)
// ---------------------------------------------------------------------------
extern "C" VKAPI_ATTR void VKAPI_CALL vkCmdBindPipeline(VkCommandBuffer commandBuffer,
                                                         VkPipelineBindPoint pipelineBindPoint,
                                                         VkPipeline pipeline) {
  auto *cb = as_cb(commandBuffer);
  if (cb == nullptr) return;
  if (pipelineBindPoint != VK_PIPELINE_BIND_POINT_COMPUTE) {
    PPU_ERR("vkCmdBindPipeline: only COMPUTE bind point is supported");
    return;
  }
  cb->pipeline = reinterpret_cast<ComputePipeline_T *>(pipeline);
  cb->active_layout = cb->pipeline ? cb->pipeline->layout : nullptr;
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkCmdBindDescriptorSets(
    VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
    VkPipelineLayout layout, uint32_t firstSet, uint32_t descriptorSetCount,
    const VkDescriptorSet *pDescriptorSets, uint32_t dynamicOffsetCount,
    const uint32_t *pDynamicOffsets) {
  auto *cb = as_cb(commandBuffer);
  if (cb == nullptr) return;
  (void)pipelineBindPoint;
  auto *pl = reinterpret_cast<PipelineLayout_T *>(layout);
  if (firstSet + descriptorSetCount > 8) return;
  if (cb->bound_sets.size() < firstSet + descriptorSetCount)
    cb->bound_sets.resize(firstSet + descriptorSetCount, nullptr);
  for (uint32_t i = 0; i < descriptorSetCount; ++i)
    cb->bound_sets[firstSet + i] = reinterpret_cast<DescriptorSet_T *>(pDescriptorSets[i]);
  cb->active_layout = pl ? pl : cb->active_layout;
  cb->dynamic_offsets.assign(pDynamicOffsets, pDynamicOffsets + dynamicOffsetCount);
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkCmdPushConstants(
    VkCommandBuffer commandBuffer, VkPipelineLayout layout, VkShaderStageFlags stageFlags,
    uint32_t offset, uint32_t size, const void *pValues) {
  auto *cb = as_cb(commandBuffer);
  if (cb == nullptr || pValues == nullptr) return;
  auto *pl = reinterpret_cast<PipelineLayout_T *>(layout);
  uint32_t total = offset + size;
  if (pl != nullptr) total = std::max(total, pl->push_constant_size);
  if (cb->push_constants.size() < total) cb->push_constants.resize(total, 0);
  std::memcpy(cb->push_constants.data() + offset, pValues, size);
  cb->active_layout = pl ? pl : cb->active_layout;
  (void)stageFlags;
}

// ---------------------------------------------------------------------------
// Transfer commands
// ---------------------------------------------------------------------------
extern "C" VKAPI_ATTR void VKAPI_CALL vkCmdCopyBuffer(VkCommandBuffer commandBuffer,
                                                       VkBuffer srcBuffer, VkBuffer dstBuffer,
                                                       uint32_t regionCount,
                                                       const VkBufferCopy *pRegions) {
  auto *cb = as_cb(commandBuffer);
  if (cb == nullptr) return;
  auto *src = reinterpret_cast<Buffer_T *>(srcBuffer);
  auto *dst = reinterpret_cast<Buffer_T *>(dstBuffer);
  if (src == nullptr || dst == nullptr || pRegions == nullptr) return;
  for (uint32_t i = 0; i < regionCount; ++i) {
    RecordedOp op;
    op.kind = RecordedOp::Kind::CopyBuffer;
    op.src_addr = ppu::buffer_address(src, pRegions[i].srcOffset);
    op.dst_addr = ppu::buffer_address(dst, pRegions[i].dstOffset);
    op.size = pRegions[i].size;
    cb->ops.push_back(std::move(op));
  }
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkCmdFillBuffer(VkCommandBuffer commandBuffer,
                                                       VkBuffer dstBuffer, VkDeviceSize dstOffset,
                                                       VkDeviceSize size, uint32_t data) {
  auto *cb = as_cb(commandBuffer);
  auto *dst = reinterpret_cast<Buffer_T *>(dstBuffer);
  if (cb == nullptr || dst == nullptr) return;
  if (size == VK_WHOLE_SIZE) size = dst->size - dstOffset;
  RecordedOp op;
  op.kind = RecordedOp::Kind::FillBuffer;
  op.fill_addr = ppu::buffer_address(dst, dstOffset);
  op.fill_value = data;
  op.size = size;
  cb->ops.push_back(std::move(op));
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkCmdUpdateBuffer(VkCommandBuffer commandBuffer,
                                                         VkBuffer dstBuffer,
                                                         VkDeviceSize dstOffset,
                                                         VkDeviceSize dataSize,
                                                         const void *pData) {
  auto *cb = as_cb(commandBuffer);
  auto *dst = reinterpret_cast<Buffer_T *>(dstBuffer);
  if (cb == nullptr || dst == nullptr || pData == nullptr) return;
  RecordedOp op;
  op.kind = RecordedOp::Kind::UpdateBuffer;
  op.update_addr = ppu::buffer_address(dst, dstOffset);
  const auto *bytes = static_cast<const uint8_t *>(pData);
  op.update_data.assign(bytes, bytes + dataSize);
  op.size = dataSize;
  cb->ops.push_back(std::move(op));
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkCmdCopyBufferToImage(
    VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkImage dstImage,
    VkImageLayout dstImageLayout, uint32_t regionCount, const VkBufferImageCopy *pRegions) {
  (void)commandBuffer;
  (void)srcBuffer;
  (void)dstImage;
  (void)dstImageLayout;
  (void)regionCount;
  (void)pRegions;
  PPU_ERR("vkCmdCopyBufferToImage: images are not supported by this device");
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkCmdCopyImageToBuffer(
    VkCommandBuffer commandBuffer, VkImage srcImage, VkImageLayout srcImageLayout,
    VkBuffer dstBuffer, uint32_t regionCount, const VkBufferImageCopy *pRegions) {
  (void)commandBuffer;
  (void)srcImage;
  (void)srcImageLayout;
  (void)dstBuffer;
  (void)regionCount;
  (void)pRegions;
  PPU_ERR("vkCmdCopyImageToBuffer: images are not supported by this device");
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------
extern "C" VKAPI_ATTR void VKAPI_CALL vkCmdDispatch(VkCommandBuffer commandBuffer,
                                                     uint32_t groupCountX, uint32_t groupCountY,
                                                     uint32_t groupCountZ) {
  auto *cb = as_cb(commandBuffer);
  if (cb == nullptr) return;
  if (cb->pipeline == nullptr) {
    PPU_ERR("vkCmdDispatch: no compute pipeline bound");
    return;
  }
  RecordedOp op;
  op.kind = RecordedOp::Kind::Dispatch;
  op.pipeline = cb->pipeline;
  op.group[0] = groupCountX;
  op.group[1] = groupCountY;
  op.group[2] = groupCountZ;
  // Snapshot descriptors and push constants now: bindings may change before
  // submit, but the dispatch must capture the state that was current here.
  op.args.assign(cb->pipeline->arg_slots.size(), 0ull);
  const auto &slots = cb->pipeline->arg_slots;
  for (size_t i = 0; i < slots.size(); ++i) {
    uint64_t addr = 0;
    const auto &slot = slots[i];
    if (slot.set < cb->bound_sets.size()) {
      DescriptorSet_T *set = cb->bound_sets[slot.set];
      if (set != nullptr && set->layout != nullptr) {
        size_t base = 0;
        for (const auto &b : set->layout->bindings) {
          if (b.binding == slot.binding) {
            size_t index = base;  // element 0
            if (index < set->descriptors.size()) {
              const Descriptor_T &desc = set->descriptors[index];
              if (desc.buffer != nullptr)
                addr = ppu::buffer_address(desc.buffer, desc.offset);
            }
            break;
          }
          base += b.count;
        }
      }
    }
    op.args[i] = addr;
  }
  if (cb->pipeline->has_push_constants && !cb->push_constants.empty()) {
    ppu::CUdeviceptr dev = 0;
    void *host = nullptr;
    if (ppu::alloc_scratch(cb, cb->push_constants.size(), &dev, &host)) {
      std::memcpy(host, cb->push_constants.data(), cb->push_constants.size());
      op.push_constant_addr = dev;
    } else {
      PPU_ERR("vkCmdDispatch: push-constant scratch exhausted");
    }
  }
  cb->ops.push_back(std::move(op));
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkCmdDispatchIndirect(VkCommandBuffer commandBuffer,
                                                             VkBuffer buffer,
                                                             VkDeviceSize offset) {
  auto *cb = as_cb(commandBuffer);
  if (cb == nullptr) return;
  if (cb->pipeline == nullptr) {
    PPU_ERR("vkCmdDispatchIndirect: no compute pipeline bound");
    return;
  }
  auto *buf = reinterpret_cast<Buffer_T *>(buffer);
  if (buf == nullptr) return;

  // Reuse the same descriptor snapshot logic as vkCmdDispatch.
  RecordedOp op;
  op.kind = RecordedOp::Kind::DispatchIndirect;
  op.pipeline = cb->pipeline;
  op.indirect_addr = ppu::buffer_address(buf, offset);
  op.args.assign(cb->pipeline->arg_slots.size(), 0ull);
  const auto &slots = cb->pipeline->arg_slots;
  for (size_t i = 0; i < slots.size(); ++i) {
    const auto &slot = slots[i];
    if (slot.set < cb->bound_sets.size()) {
      DescriptorSet_T *set = cb->bound_sets[slot.set];
      if (set != nullptr && set->layout != nullptr) {
        size_t base = 0;
        for (const auto &b : set->layout->bindings) {
          if (b.binding == slot.binding) {
            if (base < set->descriptors.size() && set->descriptors[base].buffer != nullptr)
              op.args[i] = ppu::buffer_address(set->descriptors[base].buffer,
                                               set->descriptors[base].offset);
            break;
          }
          base += b.count;
        }
      }
    }
  }
  if (cb->pipeline->has_push_constants && !cb->push_constants.empty()) {
    ppu::CUdeviceptr dev = 0;
    void *host = nullptr;
    if (ppu::alloc_scratch(cb, cb->push_constants.size(), &dev, &host)) {
      std::memcpy(host, cb->push_constants.data(), cb->push_constants.size());
      op.push_constant_addr = dev;
    }
  }
  cb->ops.push_back(std::move(op));
}

// ---------------------------------------------------------------------------
// Barriers
// ---------------------------------------------------------------------------
extern "C" VKAPI_ATTR void VKAPI_CALL vkCmdPipelineBarrier(
    VkCommandBuffer commandBuffer, VkPipelineStageFlags srcStageMask,
    VkPipelineStageFlags dstStageMask, VkDependencyFlags dependencyFlags,
    uint32_t memoryBarrierCount, const VkMemoryBarrier *pMemoryBarriers,
    uint32_t bufferMemoryBarrierCount,
    const VkBufferMemoryBarrier *pBufferMemoryBarriers, uint32_t imageMemoryBarrierCount,
    const VkImageMemoryBarrier *pImageMemoryBarriers) {
  (void)srcStageMask;
  (void)dstStageMask;
  (void)dependencyFlags;
  (void)memoryBarrierCount;
  (void)pMemoryBarriers;
  (void)pBufferMemoryBarriers;
  (void)pImageMemoryBarriers;
  if (imageMemoryBarrierCount != 0) {
    PPU_ERR("vkCmdPipelineBarrier: image barriers are not supported");
    return;
  }
  auto *cb = as_cb(commandBuffer);
  if (cb == nullptr) return;
  // All device work is replayed in order on a single CUDA stream, which
  // already provides the ordering a barrier would request. Record it anyway
  // so the submit path can account for it.
  RecordedOp op;
  op.kind = RecordedOp::Kind::Barrier;
  cb->ops.push_back(std::move(op));
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkCmdPipelineBarrier2(
    VkCommandBuffer commandBuffer, const VkDependencyInfo *pDependencyInfo) {
  if (pDependencyInfo == nullptr) return;
  if (pDependencyInfo->imageMemoryBarrierCount != 0) {
    PPU_ERR("vkCmdPipelineBarrier2: image barriers are not supported");
    return;
  }
  // All work replays in order on one CUDA stream, so the barrier is already
  // satisfied; record it for accounting.
  auto *cb = as_cb(commandBuffer);
  if (cb == nullptr) return;
  RecordedOp op;
  op.kind = RecordedOp::Kind::Barrier;
  cb->ops.push_back(std::move(op));
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkCmdWriteTimestamp(VkCommandBuffer commandBuffer,
                                                           VkPipelineStageFlagBits pipelineStage,
                                                           VkQueryPool queryPool,
                                                           uint32_t query) {
  (void)commandBuffer;
  (void)pipelineStage;
  (void)queryPool;
  (void)query;
  PPU_ERR("vkCmdWriteTimestamp: timestamps are not supported (timestampValidBits=0)");
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkCmdSetEvent(VkCommandBuffer commandBuffer, VkEvent event,
                                                     VkPipelineStageFlags stageMask) {
  (void)commandBuffer;
  (void)event;
  (void)stageMask;
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkCmdWaitEvents(
    VkCommandBuffer commandBuffer, uint32_t eventCount, const VkEvent *pEvents,
    VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask,
    uint32_t memoryBarrierCount, const VkMemoryBarrier *pMemoryBarriers,
    uint32_t bufferMemoryBarrierCount,
    const VkBufferMemoryBarrier *pBufferMemoryBarriers, uint32_t imageMemoryBarrierCount,
    const VkImageMemoryBarrier *pImageMemoryBarriers) {
  (void)commandBuffer;
  (void)eventCount;
  (void)pEvents;
  (void)srcStageMask;
  (void)dstStageMask;
  (void)memoryBarrierCount;
  (void)pMemoryBarriers;
  (void)bufferMemoryBarrierCount;
  (void)pBufferMemoryBarriers;
  (void)imageMemoryBarrierCount;
  (void)pImageMemoryBarriers;
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkCmdResetEvent(VkCommandBuffer commandBuffer,
                                                       VkEvent event,
                                                       VkPipelineStageFlags stageMask) {
  (void)commandBuffer;
  (void)event;
  (void)stageMask;
}
