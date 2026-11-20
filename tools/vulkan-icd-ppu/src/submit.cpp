// vkQueueSubmit: replay each recorded command buffer onto the CUDA stream.
#include "icd_common.hpp"

#include <algorithm>
#include <cstring>

namespace {

using ppu::CUDA_SUCCESS;

Device_T *as_device(VkDevice d) { return reinterpret_cast<Device_T *>(d); }

// Launch geometry for one dispatch. CUDA takes a grid of *blocks*, and the
// SPIR-V LocalSize is the block size, so Vulkan workgroup counts map through
// unchanged.
struct LaunchShape {
  unsigned grid[3];
  unsigned block[3];
};

LaunchShape shape_for(const RecordedOp &op) {
  LaunchShape s{};
  s.block[0] = op.pipeline->local_size[0];
  s.block[1] = op.pipeline->local_size[1];
  s.block[2] = op.pipeline->local_size[2];
  if (op.kind == RecordedOp::Kind::DispatchIndirect) {
    // CUDA has no indirect launch in the base driver API here; the group
    // counts live in device memory. We refuse rather than guessing.
    for (int i = 0; i < 3; ++i) s.grid[i] = 0;
  } else {
    for (int i = 0; i < 3; ++i) s.grid[i] = std::max(1u, op.group[i]);
  }
  for (int i = 0; i < 3; ++i) s.block[i] = std::max(1u, s.block[i]);
  return s;
}

bool replay_op(Device_T *dev, RecordedOp &op) {
  ppu::Driver &d = ppu::driver();

  switch (op.kind) {
    case RecordedOp::Kind::Dispatch: {
      if (op.pipeline == nullptr || op.pipeline->function == nullptr) {
        PPU_ERR("submit: dispatch with no compiled kernel (SPIR-V compilation unimplemented)");
        return false;
      }
      LaunchShape s = shape_for(op);
      std::vector<void *> kernel_args;
      kernel_args.reserve(op.args.size() + 1);
      // cuLaunchKernel wants an array of pointers to each argument value.
      for (auto &a : op.args) kernel_args.push_back(&a);
      void *push_ptr = op.push_constant_addr != 0 ? &op.push_constant_addr : nullptr;
      if (op.pipeline->has_push_constants) kernel_args.push_back(&push_ptr);

      ppu::CUresult rc = d.cuLaunchKernel(op.pipeline->function, s.grid[0], s.grid[1],
                                          s.grid[2], s.block[0], s.block[1], s.block[2],
                                          0, dev->stream, kernel_args.data(), nullptr);
      if (rc != CUDA_SUCCESS) {
        PPU_ERR("cuLaunchKernel failed: %s", ppu::cu_strerror(rc));
        return false;
      }
      return true;
    }
    case RecordedOp::Kind::DispatchIndirect: {
      // Not supported: requires device-side group counts, which this driver
      // stack cannot feed to cuLaunchKernel.
      PPU_ERR("submit: vkCmdDispatchIndirect is not implemented on this backend");
      return false;
    }
    case RecordedOp::Kind::CopyBuffer: {
      ppu::CUresult rc =
          d.cuMemcpyDtoDAsync_v2(op.dst_addr, op.src_addr, static_cast<size_t>(op.size),
                                 dev->stream);
      if (rc != CUDA_SUCCESS) {
        PPU_ERR("cuMemcpyDtoDAsync failed: %s", ppu::cu_strerror(rc));
        return false;
      }
      return true;
    }
    case RecordedOp::Kind::FillBuffer: {
      VkDeviceSize remaining = op.size;
      uint64_t addr = op.fill_addr;
      ppu::CUresult rc = CUDA_SUCCESS;
      // cuMemsetD32 needs a 4-byte-aligned address and a word count; handle
      // the leading bytes, then the word body, then the tail.
      uint64_t misalign = addr & 3;
      if (misalign != 0 && remaining > 0) {
        uint32_t lead = static_cast<uint32_t>(std::min<VkDeviceSize>(4 - misalign, remaining));
        uint8_t bytes[4];
        for (uint32_t i = 0; i < lead; ++i)
          bytes[i] = static_cast<uint8_t>((op.fill_value >> (8 * (misalign + i))) & 0xFF);
        rc = d.cuMemcpyHtoDAsync_v2(addr, bytes, lead, dev->stream);
        addr += lead;
        remaining -= lead;
      }
      VkDeviceSize words = remaining / 4;
      if (rc == CUDA_SUCCESS && words > 0) {
        rc = d.cuMemsetD32Async(addr, op.fill_value, static_cast<size_t>(words), dev->stream);
        addr += words * 4;
        remaining -= words * 4;
      }
      if (rc == CUDA_SUCCESS && remaining > 0) {
        uint8_t bytes[3] = {static_cast<uint8_t>(op.fill_value & 0xFF),
                            static_cast<uint8_t>((op.fill_value >> 8) & 0xFF),
                            static_cast<uint8_t>((op.fill_value >> 16) & 0xFF)};
        rc = d.cuMemcpyHtoDAsync_v2(addr, bytes, static_cast<size_t>(remaining), dev->stream);
      }
      if (rc != CUDA_SUCCESS) {
        PPU_ERR("fill failed: %s", ppu::cu_strerror(rc));
        return false;
      }
      return true;
    }
    case RecordedOp::Kind::UpdateBuffer: {
      // Buffer memory may be device-only, so copy through the stream rather
      // than the host pointer. The bytes live in the command buffer, which is
      // still alive for the lifetime of the recorded op.
      ppu::CUresult rc = d.cuMemcpyHtoDAsync_v2(
          op.update_addr, op.update_data.data(), op.update_data.size(), dev->stream);
      if (rc != CUDA_SUCCESS) {
        PPU_ERR("vkCmdUpdateBuffer replay failed: %s", ppu::cu_strerror(rc));
        return false;
      }
      return true;
    }
    case RecordedOp::Kind::Barrier:
      // Single-stream replay already orders everything.
      return true;
  }
  return true;
}

}  // namespace

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkQueueSubmit(VkQueue queue, uint32_t submitCount,
                                                         const VkSubmitInfo *pSubmits,
                                                         VkFence fence) {
  if (queue == VK_NULL_HANDLE) return VK_ERROR_DEVICE_LOST;
  auto *q = reinterpret_cast<Queue_T *>(queue);
  Device_T *dev = q->device;
  if (dev == nullptr || !dev->valid) return VK_ERROR_DEVICE_LOST;
  if (!ppu::bind_context(dev)) return VK_ERROR_DEVICE_LOST;

  ppu::Driver &d = ppu::driver();
  auto *fence_obj = reinterpret_cast<Fence_T *>(fence);
  // The fence becomes unsignalled now and is signalled once the replayed work
  // has been enqueued (and thus ordered) on the stream.
  if (fence_obj != nullptr) fence_obj->signaled = false;

  bool ok = true;
  for (uint32_t i = 0; i < submitCount && ok; ++i) {
    const VkSubmitInfo &si = pSubmits[i];

    // Wait semaphores: the recorded CUDA event must complete before our work.
    for (uint32_t s = 0; s < si.waitSemaphoreCount && ok; ++s) {
      auto *sem = reinterpret_cast<Semaphore_T *>(si.pWaitSemaphores[s]);
      if (sem == nullptr) continue;
      if (!sem->recorded) {
        PPU_ERR("vkQueueSubmit: waiting on a semaphore that was never signalled");
        ok = false;
        break;
      }
      ppu::CUresult rc = d.cuStreamWaitEvent(dev->stream, sem->event, 0);
      if (rc != CUDA_SUCCESS) {
        PPU_ERR("cuStreamWaitEvent failed: %s", ppu::cu_strerror(rc));
        ok = false;
      }
    }
    if (!ok) break;

    for (uint32_t c = 0; c < si.commandBufferCount && ok; ++c) {
      auto *cb = reinterpret_cast<CommandBuffer_T *>(si.pCommandBuffers[c]);
      if (cb == nullptr) continue;
      for (auto &op : cb->ops) {
        if (!replay_op(dev, op)) {
          ok = false;
          break;
        }
      }
    }
    if (!ok) break;

    // Signal semaphores: record an event at the current tip of the stream.
    for (uint32_t s = 0; s < si.signalSemaphoreCount && ok; ++s) {
      auto *sem = reinterpret_cast<Semaphore_T *>(si.pSignalSemaphores[s]);
      if (sem == nullptr) continue;
      ppu::CUresult rc = d.cuEventRecord(sem->event, dev->stream);
      if (rc != CUDA_SUCCESS) {
        PPU_ERR("cuEventRecord(semaphore) failed: %s", ppu::cu_strerror(rc));
        ok = false;
        break;
      }
      sem->recorded = true;
    }
  }

  if (!ok) {
    // Make sure a fence waiting on this submission does not hang forever.
    if (fence_obj != nullptr) {
      d.cuStreamSynchronize(dev->stream);
      d.cuEventRecord(fence_obj->event, dev->stream);
      d.cuStreamSynchronize(dev->stream);
      fence_obj->signaled = true;
    }
    return VK_ERROR_DEVICE_LOST;
  }

  if (fence_obj != nullptr) {
    ppu::CUresult rc = d.cuEventRecord(fence_obj->event, dev->stream);
    if (rc != CUDA_SUCCESS) {
      PPU_ERR("cuEventRecord(fence) failed: %s", ppu::cu_strerror(rc));
      return VK_ERROR_DEVICE_LOST;
    }
  }
  return VK_SUCCESS;
}

// ---------------------------------------------------------------------------
// Fences
// ---------------------------------------------------------------------------
extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkCreateFence(VkDevice device,
                                                         const VkFenceCreateInfo *pCreateInfo,
                                                         const VkAllocationCallbacks *pAllocator,
                                                         VkFence *pFence) {
  (void)pAllocator;
  Device_T *dev = as_device(device);
  if (dev == nullptr || pCreateInfo == nullptr || pFence == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  if (!ppu::bind_context(dev)) return VK_ERROR_DEVICE_LOST;

  ppu::Driver &d = ppu::driver();
  auto *fence = new Fence_T();
  fence->device = dev;
  ppu::CUresult rc = d.cuEventCreate(&fence->event, 0);
  if (rc != CUDA_SUCCESS) {
    delete fence;
    return VK_ERROR_DEVICE_LOST;
  }
  fence->signaled = (pCreateInfo->flags & VK_FENCE_CREATE_SIGNALED_BIT) != 0;
  if (fence->signaled) {
    d.cuEventRecord(fence->event, dev->stream);
    d.cuStreamSynchronize(dev->stream);
  }
  *pFence = reinterpret_cast<VkFence>(fence);
  return VK_SUCCESS;
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkDestroyFence(VkDevice device, VkFence fence,
                                                      const VkAllocationCallbacks *pAllocator) {
  (void)pAllocator;
  auto *f = reinterpret_cast<Fence_T *>(fence);
  if (f == nullptr) return;
  ppu::Driver &d = ppu::driver();
  if (d.cuEventDestroy_v2) d.cuEventDestroy_v2(f->event);
  delete f;
  (void)device;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkGetFenceStatus(VkDevice device, VkFence fence) {
  (void)device;
  auto *f = reinterpret_cast<Fence_T *>(fence);
  if (f == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
  if (f->signaled) return VK_SUCCESS;
  ppu::Driver &d = ppu::driver();
  ppu::CUresult rc = d.cuEventQuery(f->event);
  if (rc == ppu::CUDA_SUCCESS) {
    f->signaled = true;
    return VK_SUCCESS;
  }
  if (rc == ppu::CUDA_ERROR_NOT_READY) return VK_NOT_READY;
  PPU_ERR("vkGetFenceStatus: cuEventQuery failed: %s", ppu::cu_strerror(rc));
  return VK_ERROR_DEVICE_LOST;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkWaitForFences(VkDevice device, uint32_t fenceCount,
                                                           const VkFence *pFences,
                                                           VkBool32 waitAll, uint64_t timeout) {
  (void)device;
  (void)timeout;  // Treated as infinite; honest given the stream-based model.
  ppu::Driver &d = ppu::driver();
  uint32_t done = 0;
  for (uint32_t i = 0; i < fenceCount; ++i) {
    auto *f = reinterpret_cast<Fence_T *>(pFences[i]);
    if (f == nullptr) continue;
    if (f->signaled) {
      done++;
      continue;
    }
    ppu::CUresult rc = d.cuEventSynchronize(f->event);
    if (rc == CUDA_SUCCESS) {
      f->signaled = true;
      done++;
    } else if (!waitAll) {
      return VK_SUCCESS;
    }
  }
  if (waitAll && done < fenceCount) return VK_TIMEOUT;
  return VK_SUCCESS;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkResetFences(VkDevice device, uint32_t fenceCount,
                                                         const VkFence *pFences) {
  (void)device;
  for (uint32_t i = 0; i < fenceCount; ++i) {
    auto *f = reinterpret_cast<Fence_T *>(pFences[i]);
    if (f != nullptr) f->signaled = false;
  }
  return VK_SUCCESS;
}

// ---------------------------------------------------------------------------
// Semaphores
// ---------------------------------------------------------------------------
extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkCreateSemaphore(
    VkDevice device, const VkSemaphoreCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkSemaphore *pSemaphore) {
  (void)pAllocator;
  Device_T *dev = as_device(device);
  if (dev == nullptr || pCreateInfo == nullptr || pSemaphore == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  if (!ppu::bind_context(dev)) return VK_ERROR_DEVICE_LOST;

  ppu::Driver &d = ppu::driver();
  auto *sem = new Semaphore_T();
  sem->device = dev;
  ppu::CUresult rc = d.cuEventCreate(&sem->event, 0);
  if (rc != CUDA_SUCCESS) {
    delete sem;
    return VK_ERROR_DEVICE_LOST;
  }
  *pSemaphore = reinterpret_cast<VkSemaphore>(sem);
  return VK_SUCCESS;
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkDestroySemaphore(VkDevice device, VkSemaphore semaphore,
                                                          const VkAllocationCallbacks *pAllocator) {
  (void)device;
  (void)pAllocator;
  auto *sem = reinterpret_cast<Semaphore_T *>(semaphore);
  if (sem == nullptr) return;
  ppu::Driver &d = ppu::driver();
  if (d.cuEventDestroy_v2) d.cuEventDestroy_v2(sem->event);
  delete sem;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkGetSemaphoreCounterValue(VkDevice device,
                                                                     VkSemaphore semaphore,
                                                                     uint64_t *pValue) {
  (void)device;
  auto *sem = reinterpret_cast<Semaphore_T *>(semaphore);
  if (sem == nullptr || pValue == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
  // Timeline semaphores are not supported (we do not advertise the feature).
  (void)sem;
  return VK_ERROR_FEATURE_NOT_PRESENT;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkWaitSemaphores(
    VkDevice device, const VkSemaphoreWaitInfo *pWaitInfo, uint64_t timeout) {
  (void)device;
  (void)pWaitInfo;
  (void)timeout;
  return VK_ERROR_FEATURE_NOT_PRESENT;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkSignalSemaphore(
    VkDevice device, const VkSemaphoreSignalInfo *pSignalInfo) {
  (void)device;
  (void)pSignalInfo;
  return VK_ERROR_FEATURE_NOT_PRESENT;
}
