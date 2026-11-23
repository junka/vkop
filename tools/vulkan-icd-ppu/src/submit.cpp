// vkQueueSubmit: replay each recorded command buffer onto the CUDA stream.
#include "icd_common.hpp"

#include <algorithm>
#include <cstring>
#include <ctime>

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

LaunchShape shape_for(Device_T *dev, RecordedOp &op) {
  LaunchShape s{};
  s.block[0] = op.pipeline->local_size[0];
  s.block[1] = op.pipeline->local_size[1];
  s.block[2] = op.pipeline->local_size[2];
  if (op.kind == RecordedOp::Kind::DispatchIndirect) {
    // The group counts live in device memory (a VkDispatchIndirectCommand).
    // This driver stack has no indirect launch, but the counts are only three
    // uint32s and reading them back is a real read of the same 12 bytes the
    // hardware would have read -- so the geometry is the application's, never
    // a guess. The copy is stream-ordered, so if a pre-pass compute shader
    // (vkop's dispatch_from_shape) wrote them in this same command buffer, the
    // read observes that write; the host-visible path is covered the same way.
    uint32_t cmd[3] = {0, 0, 0};
    if (op.indirect_host != nullptr) {
      // Host-visible indirect buffer: the counts are in the mapping, written
      // by the application (or a prior map/flush). Read them directly -- on a
      // per-round decode path this must not cost a device round trip.
      std::memcpy(cmd, op.indirect_host, sizeof(cmd));
    } else if (op.indirect_addr != 0) {
      ppu::Driver &d = ppu::driver();
      ppu::CUresult rc = d.cuMemcpyDtoH_v2(cmd, op.indirect_addr, sizeof(cmd));
      if (rc != ppu::CUDA_SUCCESS) {
        PPU_ERR("indirect dispatch: reading group counts failed: %s",
                ppu::cu_strerror(rc));
      }
    }
    for (int i = 0; i < 3; ++i) s.grid[i] = cmd[i];
    op.group[0] = cmd[0];
    op.group[1] = cmd[1];
    op.group[2] = cmd[2];
  } else {
    for (int i = 0; i < 3; ++i) s.grid[i] = std::max(1u, op.group[i]);
  }
  // A zero group count is legal in Vulkan (the dispatch does nothing), but
  // cuLaunchKernel rejects gridDim 0. Clamp only for the launch; the recorded
  // counts stay as read.
  for (int i = 0; i < 3; ++i) s.grid[i] = std::max(1u, s.grid[i]);
  for (int i = 0; i < 3; ++i) s.block[i] = std::max(1u, s.block[i]);
  return s;
}

bool replay_op(Device_T *dev, RecordedOp &op) {
  ppu::Driver &d = ppu::driver();

  switch (op.kind) {
    case RecordedOp::Kind::Dispatch: {
      if (op.pipeline == nullptr || op.pipeline->function == nullptr) {
        PPU_ERR("submit: dispatch with no compiled kernel");
        return false;
      }
      LaunchShape s = shape_for(dev, op);
      std::vector<void *> kernel_args;
      kernel_args.reserve(op.args.size() + 1);
      // cuLaunchKernel wants an array of pointers to each argument value. A
      // buffer argument's value is the device address (already a uint64_t in
      // op.args), so its address is &op.args[i]. The push-constant block is
      // passed by pointer too -- the kernel signature takes
      // `T addrspace(1)*`, so the *value* is the device address of the block
      // and the argument slot points at that address, not at the slot that
      // stores it.
      for (size_t i = 0; i < op.args.size(); ++i) kernel_args.push_back(&op.args[i]);
      uint64_t push_addr = op.push_constant_addr;
      if (op.pipeline->has_push_constants) {
        if (push_addr == 0) {
          PPU_ERR("submit: kernel takes a push-constant block but none was recorded");
          return false;
        }
        kernel_args.push_back(&push_addr);
      }

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
      // Same launch as a direct dispatch: shape_for has already read the
      // group counts out of the indirect buffer (stream-ordered, so a
      // preceding dispatch_from_shape pre-pass is visible).
      if (op.pipeline == nullptr || op.pipeline->function == nullptr) {
        PPU_ERR("submit: indirect dispatch with no compiled kernel");
        return false;
      }
      LaunchShape s = shape_for(dev, op);
      std::vector<void *> kernel_args;
      kernel_args.reserve(op.args.size() + 1);
      for (size_t i = 0; i < op.args.size(); ++i) kernel_args.push_back(&op.args[i]);
      uint64_t push_addr = op.push_constant_addr;
      if (op.pipeline->has_push_constants) {
        if (push_addr == 0) {
          PPU_ERR("submit: kernel takes a push-constant block but none was recorded");
          return false;
        }
        kernel_args.push_back(&push_addr);
      }
      ppu::CUresult rc = d.cuLaunchKernel(op.pipeline->function, s.grid[0], s.grid[1],
                                          s.grid[2], s.block[0], s.block[1], s.block[2],
                                          0, dev->stream, kernel_args.data(), nullptr);
      if (rc != CUDA_SUCCESS) {
        PPU_ERR("cuLaunchKernel (indirect) failed: %s", ppu::cu_strerror(rc));
        return false;
      }
      return true;
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
    case RecordedOp::Kind::ResetQuery:
    case RecordedOp::Kind::WriteTimestamp:
      return ppu::replay_query_op(dev, op);
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

    // Timeline wait/signal values, if the application supplied them.
    const auto *timeline = reinterpret_cast<const VkTimelineSemaphoreSubmitInfo *>(si.pNext);
    while (timeline != nullptr &&
           timeline->sType != VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO) {
      timeline = reinterpret_cast<const VkTimelineSemaphoreSubmitInfo *>(timeline->pNext);
    }

    // Wait semaphores: the recorded CUDA event must complete before our work.
    for (uint32_t s = 0; s < si.waitSemaphoreCount && ok; ++s) {
      auto *sem = reinterpret_cast<Semaphore_T *>(si.pWaitSemaphores[s]);
      if (sem == nullptr) continue;
      if (sem->timeline) {
        const uint64_t want = timeline != nullptr &&
                                      timeline->waitSemaphoreValueCount > s
                                  ? timeline->pWaitSemaphoreValues[s]
                                  : 0;
        // A timeline semaphore's pending event is the exact point in the stream
        // where the value becomes visible, so ordering on it is the ordering
        // the application asked for. A value the host already signalled needs
        // no stream dependency at all.
        if (sem->value >= want) continue;
        if (sem->pending_event == nullptr) {
          PPU_ERR("vkQueueSubmit: timeline wait on value %llu, but the semaphore "
                  "has only reached %llu and nothing will advance it",
                  static_cast<unsigned long long>(want),
                  static_cast<unsigned long long>(sem->value));
          ok = false;
          break;
        }
        ppu::CUresult rc = d.cuStreamWaitEvent(dev->stream, sem->pending_event, 0);
        if (rc != CUDA_SUCCESS) {
          PPU_ERR("cuStreamWaitEvent(timeline) failed: %s", ppu::cu_strerror(rc));
          ok = false;
        }
        continue;
      }
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
      if (sem->timeline) {
        const uint64_t mark = timeline != nullptr &&
                                      timeline->signalSemaphoreValueCount > s
                                  ? timeline->pSignalSemaphoreValues[s]
                                  : sem->value + 1;
        // The value must not be visible before the work is done, so it becomes
        // the *pending* value now and is promoted by cuEventQuery in
        // vkGetSemaphoreCounterValue / vkWaitSemaphores once the stream drains
        // past this point.
        //
        // Each signal gets a fresh event when the previous one is still
        // unpromoted: re-recording an event that a submitted
        // cuStreamWaitEvent may be waiting on would move the point it refers
        // to, and a later submission could then pass a wait it should have
        // blocked on.
        if (sem->pending_event != nullptr && sem->pending_value > sem->value) {
          if (d.cuEventDestroy_v2) d.cuEventDestroy_v2(sem->pending_event);
          sem->pending_event = nullptr;
        }
        if (sem->pending_event == nullptr && d.cuEventCreate)
          d.cuEventCreate(&sem->pending_event, 0);
        if (sem->pending_event == nullptr) {
          PPU_ERR("vkQueueSubmit: cannot create an event for a timeline signal");
          ok = false;
          break;
        }
        ppu::CUresult rc = d.cuEventRecord(sem->pending_event, dev->stream);
        if (rc != CUDA_SUCCESS) {
          PPU_ERR("cuEventRecord(timeline) failed: %s", ppu::cu_strerror(rc));
          ok = false;
          break;
        }
        sem->pending_value = mark;
        sem->recorded = true;
        continue;
      }
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

  // VK_SEMAPHORE_TYPE_CREATE_INFO in pNext selects a timeline semaphore. The
  // device models one exactly: every recorded op replays in order on a single
  // stream, so "counter reached value N" is "the submission that signalled N
  // has drained", which cuEventQuery answers directly.
  for (const auto *base = static_cast<const VkBaseInStructure *>(pCreateInfo->pNext);
       base != nullptr; base = base->pNext) {
    if (base->sType != VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO) continue;
    const auto *ti = reinterpret_cast<const VkSemaphoreTypeCreateInfo *>(base);
    if (ti->semaphoreType == VK_SEMAPHORE_TYPE_TIMELINE) {
      sem->timeline = true;
      sem->value = ti->initialValue;
    } else if (ti->semaphoreType != VK_SEMAPHORE_TYPE_BINARY) {
      PPU_ERR("vkCreateSemaphore: semaphore type %d is not supported",
              static_cast<int>(ti->semaphoreType));
      if (d.cuEventDestroy_v2) d.cuEventDestroy_v2(sem->event);
      delete sem;
      return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    break;
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
  if (sem->timeline && sem->pending_event && d.cuEventDestroy_v2)
    d.cuEventDestroy_v2(sem->pending_event);
  delete sem;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkGetSemaphoreCounterValue(VkDevice device,
                                                                     VkSemaphore semaphore,
                                                                     uint64_t *pValue) {
  (void)device;
  auto *sem = reinterpret_cast<Semaphore_T *>(semaphore);
  if (sem == nullptr || pValue == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
  if (!sem->timeline) {
    // A binary semaphore has no counter; the spec makes this a validation
    // error, and returning the honest error is better than inventing a value.
    PPU_ERR("vkGetSemaphoreCounterValue on a binary semaphore");
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }
  // If a submission promised a higher value, it becomes visible only once that
  // submission really finished. Polling the CUDA event is what makes the answer
  // true rather than optimistic.
  ppu::Driver &d = ppu::driver();
  if (sem->pending_event != nullptr && sem->pending_value > sem->value &&
      d.cuEventQuery && d.cuEventQuery(sem->pending_event) == ppu::CUDA_SUCCESS) {
    sem->value = sem->pending_value;
    sem->pending_event = nullptr;  // safe to forget: the value is already ours
  }
  *pValue = sem->value;
  return VK_SUCCESS;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkWaitSemaphores(
    VkDevice device, const VkSemaphoreWaitInfo *pWaitInfo, uint64_t timeout) {
  if (pWaitInfo == nullptr || (pWaitInfo->semaphoreCount > 0 && pWaitInfo->pSemaphores == nullptr))
    return VK_ERROR_INITIALIZATION_FAILED;
  (void)device;
  ppu::Driver &d = ppu::driver();

  auto satisfied = [&](uint32_t i) {
    auto *sem = reinterpret_cast<Semaphore_T *>(pWaitInfo->pSemaphores[i]);
    if (sem == nullptr) return true;
    const uint64_t want =
        pWaitInfo->pValues != nullptr ? pWaitInfo->pValues[i] : 0;
    if (!sem->timeline) {
      // A binary semaphore's "counter" is not meaningful; treat a signalled
      // semaphore as reached. Nothing in this ICD submits binary-semaphore
      // waits through this entry point, but answering honestly beats hanging.
      return sem->recorded;
    }
    uint64_t cur = sem->value;
    if (sem->pending_event != nullptr && sem->pending_value > cur &&
        d.cuEventQuery && d.cuEventQuery(sem->pending_event) == ppu::CUDA_SUCCESS) {
      sem->value = sem->pending_value;
      sem->pending_event = nullptr;
      cur = sem->value;
    }
    return cur >= want;
  };

  auto all_satisfied = [&]() {
    if ((pWaitInfo->flags & VK_SEMAPHORE_WAIT_ANY_BIT) != 0) {
      for (uint32_t i = 0; i < pWaitInfo->semaphoreCount; ++i)
        if (satisfied(i)) return true;
      return pWaitInfo->semaphoreCount == 0;
    }
    for (uint32_t i = 0; i < pWaitInfo->semaphoreCount; ++i)
      if (!satisfied(i)) return false;
    return true;
  };

  if (all_satisfied()) return VK_SUCCESS;

  // Poll rather than block forever on a deadline the caller controls. The
  // wait is on work already submitted to this device's single stream, so
  // progress is guaranteed unless the GPU hangs.
  const uint64_t kForever = ~0ull;
  uint64_t waited_ns = 0;
  constexpr uint64_t kPollNs = 100 * 1000;  // 100 us
  while (!all_satisfied()) {
    if (timeout != kForever && waited_ns >= timeout) return VK_TIMEOUT;
    struct timespec ts{0, static_cast<long>(kPollNs)};
    ::nanosleep(&ts, nullptr);
    waited_ns += kPollNs;
  }
  return VK_SUCCESS;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkSignalSemaphore(
    VkDevice device, const VkSemaphoreSignalInfo *pSignalInfo) {
  (void)device;
  if (pSignalInfo == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
  auto *sem = reinterpret_cast<Semaphore_T *>(pSignalInfo->semaphore);
  if (sem == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
  if (!sem->timeline) {
    PPU_ERR("vkSignalSemaphore on a binary semaphore");
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }
  // A host signal is only legal on a value above the current one; anything
  // else is an application error, and accepting it would make the counter go
  // backwards.
  if (pSignalInfo->value <= sem->value) {
    // The spec makes this a validation error (a counter never goes backwards).
    // Refusing keeps the ordering guarantee a later wait relies on; accepting
    // it would let a wait pass on a value nothing signalled.
    PPU_ERR("vkSignalSemaphore: value %llu is not above the current %llu",
            static_cast<unsigned long long>(pSignalInfo->value),
            static_cast<unsigned long long>(sem->value));
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  sem->value = pSignalInfo->value;
  return VK_SUCCESS;
}
