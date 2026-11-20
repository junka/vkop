// Query pools: timestamps measured with CUDA events.
//
// Vulkan's timestamp queries are a pair of 64-bit values whose difference is a
// duration in units of limits.timestampPeriod nanoseconds. The CUDA driver
// measures exactly that between two events on the same context
// (cuEventElapsedTime), so a pool here is:
//
//   * one long-lived `base` event, recorded at the current tip of the stream
//     the first time a query in the pool is written after a reset, and
//   * one event per query slot, recorded at that query's position in the
//     replay order.
//
// A slot's value is the nanoseconds elapsed from `base` to its own event. Both
// ends of a begin/end pair therefore share one time origin, and a consumer's
// subtraction yields the time between the two points even when it only ever
// sees the difference. timestampPeriod is 1.0, so the value is already in
// nanoseconds.
//
// Everything here reports unavailable (VK_NOT_READY / zeroed) until the events
// have actually completed; no value is ever synthesized.
#include "icd_common.hpp"

#include <cmath>
#include <cstring>

namespace {

Device_T *as_device(VkDevice device) { return checked_device(device, __func__); }

ppu::CUevent make_event(Device_T *dev) {
  ppu::Driver &d = ppu::driver();
  if (d.cuEventCreate == nullptr) return nullptr;
  ppu::CUevent ev = nullptr;
  // flags 0 = timing enabled; the default-created event supports
  // cuEventElapsedTime, which a pool without timing could not serve.
  if (d.cuEventCreate(&ev, 0) != ppu::CUDA_SUCCESS) return nullptr;
  (void)dev;
  return ev;
}

void destroy_event(ppu::CUevent ev) {
  ppu::Driver &d = ppu::driver();
  if (ev != nullptr && d.cuEventDestroy_v2 != nullptr) d.cuEventDestroy_v2(ev);
}

}  // namespace

namespace ppu {

bool replay_query_op(Device_T *dev, const RecordedOp &op) {
  QueryPool_T *pool = op.query_pool;
  if (pool == nullptr) return false;
  Driver &d = driver();

  if (op.kind == RecordedOp::Kind::ResetQuery) {
    for (uint32_t i = 0; i < op.query_count; ++i) {
      QuerySlot &slot = pool->slots[op.query_first + i];
      slot.available = false;
      slot.ns = 0;
    }
    // The next timestamp in this pool anchors a fresh time origin, so a value
    // never measures from a reset that happened arbitrarily long ago.
    pool->base_recorded = false;
    return true;
  }

  if (op.kind != RecordedOp::Kind::WriteTimestamp) return false;
  if (op.query_first >= pool->count) return false;
  QuerySlot &slot = pool->slots[op.query_first];
  if (slot.event == nullptr) {
    slot.event = make_event(dev);
    if (slot.event == nullptr) {
      PPU_ERR("timestamp query %u: cuEventCreate failed", op.query_first);
      return false;
    }
  }
  if (pool->base == nullptr) {
    pool->base = make_event(dev);
    if (pool->base == nullptr) {
      PPU_ERR("timestamp query pool: cuEventCreate(base) failed");
      return false;
    }
  }
  if (!pool->base_recorded) {
    if (d.cuEventRecord(pool->base, dev->stream) != CUDA_SUCCESS) return false;
    pool->base_recorded = true;
  }
  slot.available = false;
  slot.ns = 0;
  if (d.cuEventRecord(slot.event, dev->stream) != CUDA_SUCCESS) {
    PPU_ERR("timestamp query %u: cuEventRecord failed", op.query_first);
    return false;
  }
  return true;
}

bool query_value(QueryPool_T *pool, uint32_t index, uint64_t *out_ns) {
  if (pool == nullptr || index >= pool->count || out_ns == nullptr) return false;
  QuerySlot &slot = pool->slots[index];
  if (slot.available) {
    *out_ns = slot.ns;
    return true;
  }
  if (slot.event == nullptr || pool->base == nullptr || !pool->base_recorded) return false;

  Driver &d = driver();
  // Both events must have completed for elapsed time to be defined.
  if (d.cuEventQuery(pool->base) != CUDA_SUCCESS) return false;
  if (d.cuEventQuery(slot.event) != CUDA_SUCCESS) return false;
  float ms = 0.0f;
  if (d.cuEventElapsedTime(&ms, pool->base, slot.event) != CUDA_SUCCESS) return false;
  if (ms < 0.0f) return false;
  slot.ns = static_cast<uint64_t>(std::llround(static_cast<double>(ms) * 1.0e6));
  slot.available = true;
  *out_ns = slot.ns;
  return true;
}

}  // namespace ppu

// ---------------------------------------------------------------------------
// Pool lifecycle
// ---------------------------------------------------------------------------
extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkCreateQueryPool(
    VkDevice device, const VkQueryPoolCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkQueryPool *pQueryPool) {
  (void)pAllocator;
  Device_T *dev = as_device(device);
  if (dev == nullptr || pCreateInfo == nullptr || pQueryPool == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  if (pCreateInfo->queryCount == 0) return VK_ERROR_INITIALIZATION_FAILED;

  if (pCreateInfo->queryType != VK_QUERY_TYPE_TIMESTAMP) {
    // Occlusion/performance/pipeline-statistics queries describe graphics work
    // this device cannot do. Say so rather than return a pool that answers 0.
    PPU_ERR("vkCreateQueryPool: query type %d is not supported (timestamps only)",
            static_cast<int>(pCreateInfo->queryType));
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }
  if (dev->phys == nullptr || !dev->phys->timestamps) {
    PPU_ERR("vkCreateQueryPool: timestamps are not available on this device");
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }

  auto *pool = new QueryPool_T();
  pool->device = dev;
  pool->type = pCreateInfo->queryType;
  pool->count = pCreateInfo->queryCount;
  pool->slots.resize(pool->count);
  *pQueryPool = reinterpret_cast<VkQueryPool>(pool);
  return VK_SUCCESS;
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkDestroyQueryPool(
    VkDevice device, VkQueryPool queryPool, const VkAllocationCallbacks *pAllocator) {
  (void)device;
  (void)pAllocator;
  auto *pool = reinterpret_cast<QueryPool_T *>(queryPool);
  if (pool == nullptr) return;
  for (auto &slot : pool->slots) destroy_event(slot.event);
  destroy_event(pool->base);
  delete pool;
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkResetQueryPool(VkDevice device, VkQueryPool queryPool,
                                                        uint32_t firstQuery,
                                                        uint32_t queryCount) {
  (void)device;
  auto *pool = reinterpret_cast<QueryPool_T *>(queryPool);
  if (pool == nullptr) return;
  if (firstQuery >= pool->count || firstQuery + queryCount > pool->count) {
    PPU_ERR("vkResetQueryPool: range %u+%u out of bounds (%u queries)", firstQuery,
            queryCount, pool->count);
    return;
  }
  for (uint32_t i = 0; i < queryCount; ++i) {
    QuerySlot &slot = pool->slots[firstQuery + i];
    slot.available = false;
    slot.ns = 0;
  }
  pool->base_recorded = false;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkGetQueryPoolResults(
    VkDevice device, VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount,
    size_t dataSize, void *pData, VkDeviceSize stride, VkQueryResultFlags flags) {
  Device_T *dev = as_device(device);
  auto *pool = reinterpret_cast<QueryPool_T *>(queryPool);
  if (dev == nullptr || pool == nullptr || pData == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  if (firstQuery >= pool->count || firstQuery + queryCount > pool->count)
    return VK_ERROR_INITIALIZATION_FAILED;
  if (pool->type != VK_QUERY_TYPE_TIMESTAMP) return VK_ERROR_FEATURE_NOT_PRESENT;
  if (!ppu::bind_context(dev)) return VK_ERROR_DEVICE_LOST;

  const bool want_wait = (flags & VK_QUERY_RESULT_WAIT_BIT) != 0;
  const bool with_availability = (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) != 0;
  const bool want_64 = (flags & VK_QUERY_RESULT_64_BIT) != 0;
  const VkDeviceSize value_size = want_64 ? sizeof(uint64_t) : sizeof(uint32_t);
  // Lazily materialise every slot that has completed since last time.
  unsigned char *base = static_cast<unsigned char *>(pData);
  VkResult result = VK_SUCCESS;
  for (uint32_t i = 0; i < queryCount; ++i) {
    const uint32_t index = firstQuery + i;
    uint64_t value = 0;
    bool ok = ppu::query_value(pool, index, &value);
    if (!ok && want_wait) {
      // Wait for the value to become available. The recorded work is all on
      // one FIFO stream, so draining it (rather than synchronising the slot's
      // own event, which may not have been replayed yet) guarantees every
      // submission recorded so far is complete before we look again.
      ppu::Driver &d = ppu::driver();
      if (d.cuStreamSynchronize != nullptr) d.cuStreamSynchronize(dev->stream);
      ok = ppu::query_value(pool, index, &value);
    }
    unsigned char *dst = base + stride * i;
    if (want_64)
      std::memcpy(dst, &value, sizeof(uint64_t));
    else {
      uint32_t v32 = static_cast<uint32_t>(value);
      std::memcpy(dst, &v32, sizeof(uint32_t));
    }
    if (with_availability) {
      uint32_t avail = ok ? 1u : 0u;
      std::memcpy(dst + value_size, &avail, sizeof(uint32_t));
    }
    if (!ok) result = VK_NOT_READY;  // never DEVICE_LOST: an unwritten query
                                     // is "not available", not a device fault
  }
  return result;
}
