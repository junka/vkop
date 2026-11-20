// Device creation, memory objects and buffers.
#include "icd_common.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "spirv_compile.hpp"

namespace ppu {

bool bind_context(Device_T *device) {
  if (device == nullptr || device->magic != kDeviceMagic || device->ctx == nullptr)
    return false;
  Driver &d = driver();
  if (d.cuCtxSetCurrent == nullptr) return false;
  return d.cuCtxSetCurrent(device->ctx) == CUDA_SUCCESS;
}

uint64_t buffer_address(const Buffer_T *buffer, VkDeviceSize offset) {
  if (buffer == nullptr || buffer->memory == nullptr) return 0;
  return buffer->memory->device_ptr + buffer->memory_offset + offset;
}

}  // namespace ppu

namespace {

using ppu::CUDA_SUCCESS;

Device_T *as_device(VkDevice device) { return checked_device(device, __func__); }

// Give the process-level driver a chance to report why a call cannot work.
VkResult device_failure(Device_T *dev, const char *what, ppu::CUresult rc) {
  PPU_ERR("%s failed: %s", what, ppu::cu_strerror(rc));
  (void)dev;
  return VK_ERROR_DEVICE_LOST;
}

}  // namespace

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------
extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkCreateDevice(
    VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkDevice *pDevice) {
  (void)pAllocator;
  if (physicalDevice == VK_NULL_HANDLE || pCreateInfo == nullptr || pDevice == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  PhysicalDevice_T *pd = reinterpret_cast<PhysicalDevice_T *>(physicalDevice);
  if (pd->synthetic || pd->cu_index < 0) {
    if (pd->synthetic) {
      // Synthetic bring-up device (VK_PPU_SYNTHETIC_DEVICE=1): there is no
      // accelerator, so this is a null backend. Object lifecycle and queries
      // work so the loader handshake can be exercised; every operation that
      // would need real hardware reports FEATURE_NOT_PRESENT.
      auto *dev = new Device_T();
      ppu::stamp(&dev->loader_data);
      dev->phys = pd;
      dev->valid = false;
      auto *queue = new Queue_T();
      ppu::stamp(&queue->loader_data);
      queue->device = dev;
      dev->queue = queue;
      *pDevice = reinterpret_cast<VkDevice>(dev);
      PPU_WARN("vkCreateDevice on the SYNTHETIC device: null backend, execution disabled");
      return VK_SUCCESS;
    }
    PPU_ERR("vkCreateDevice: no usable accelerator device");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  ppu::Driver &d = ppu::driver();
  if (!d.loaded || d.cuCtxCreate_v2 == nullptr) {
    PPU_ERR("vkCreateDevice: CUDA driver unavailable");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  // Reject queue requests we cannot satisfy rather than silently ignoring them.
  for (uint32_t i = 0; i < pCreateInfo->queueCreateInfoCount; ++i) {
    const auto &qci = pCreateInfo->pQueueCreateInfos[i];
    if (qci.queueFamilyIndex != 0) {
      PPU_ERR("vkCreateDevice: only queue family 0 exists, requested %u",
              qci.queueFamilyIndex);
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (qci.queueCount != 1) {
      PPU_ERR("vkCreateDevice: family 0 has exactly one queue, requested %u",
              qci.queueCount);
      return VK_ERROR_INITIALIZATION_FAILED;
    }
  }

  auto *dev = new Device_T();
  ppu::stamp(&dev->loader_data);
  dev->phys = pd;

  ppu::CUdevice cu_dev = 0;
  if (d.cuDeviceGet(&cu_dev, pd->cu_index) != CUDA_SUCCESS) {
    delete dev;
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  ppu::CUresult rc = d.cuCtxCreate_v2(&dev->ctx, 0, cu_dev);
  if (rc != CUDA_SUCCESS) {
    PPU_ERR("cuCtxCreate failed: %s", ppu::cu_strerror(rc));
    delete dev;
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  rc = d.cuStreamCreate(&dev->stream, 0);
  if (rc != CUDA_SUCCESS) {
    PPU_ERR("cuStreamCreate failed: %s", ppu::cu_strerror(rc));
    d.cuCtxDestroy_v2(dev->ctx);
    delete dev;
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  auto *queue = new Queue_T();
  ppu::stamp(&queue->loader_data);
  queue->device = dev;
  queue->family_index = 0;
  queue->queue_index = 0;
  dev->queue = queue;
  dev->valid = true;

  *pDevice = reinterpret_cast<VkDevice>(dev);
  PPU_LOG("vkCreateDevice on %s (cu device %d)", pd->name, pd->cu_index);
  return VK_SUCCESS;
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkDestroyDevice(VkDevice device,
                                                      const VkAllocationCallbacks *pAllocator) {
  (void)pAllocator;
  Device_T *dev = as_device(device);
  if (dev == nullptr) return;
  ppu::Driver &d = ppu::driver();
  if (dev->stream && d.cuStreamSynchronize) d.cuStreamSynchronize(dev->stream);
  if (dev->stream && d.cuStreamDestroy_v2) d.cuStreamDestroy_v2(dev->stream);
  // Cached CUmodules belong to this context; unload them while it is still
  // current, then destroy the context. (Otherwise cuModuleUnload -- which the
  // cache would attempt on the next compile -- has no context to act on.)
  if (dev->ctx) ppu::purge_spirv_cache(dev->ctx);
  if (dev->ctx && d.cuCtxDestroy_v2) d.cuCtxDestroy_v2(dev->ctx);
  delete dev->queue;
  delete dev;
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkGetDeviceQueue(VkDevice device, uint32_t queueFamilyIndex,
                                                       uint32_t queueIndex, VkQueue *pQueue) {
  Device_T *dev = as_device(device);
  if (pQueue == nullptr) return;
  if (dev == nullptr || queueFamilyIndex != 0 || queueIndex != 0) {
    PPU_ERR("vkGetDeviceQueue: invalid family %u index %u", queueFamilyIndex, queueIndex);
    *pQueue = VK_NULL_HANDLE;
    return;
  }
  *pQueue = reinterpret_cast<VkQueue>(dev->queue);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkDeviceWaitIdle(VkDevice device) {
  Device_T *dev = as_device(device);
  if (dev == nullptr) return VK_ERROR_DEVICE_LOST;
  ppu::Driver &d = ppu::driver();
  if (!dev->valid) return VK_ERROR_DEVICE_LOST;
  if (!ppu::bind_context(dev)) return VK_ERROR_DEVICE_LOST;
  ppu::CUresult rc = d.cuStreamSynchronize(dev->stream);
  if (rc != CUDA_SUCCESS) return device_failure(dev, "cuStreamSynchronize", rc);
  return VK_SUCCESS;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkQueueWaitIdle(VkQueue queue) {
  if (queue == VK_NULL_HANDLE) return VK_ERROR_DEVICE_LOST;
  return vkDeviceWaitIdle(reinterpret_cast<VkDevice>(
      reinterpret_cast<Queue_T *>(queue)->device));
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------
extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkAllocateMemory(
    VkDevice device, const VkMemoryAllocateInfo *pAllocateInfo,
    const VkAllocationCallbacks *pAllocator, VkDeviceMemory *pMemory) {
  (void)pAllocator;
  Device_T *dev = as_device(device);
  if (dev == nullptr || pAllocateInfo == nullptr || pMemory == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  if (!dev->valid || dev->ctx == nullptr) {
    PPU_ERR("vkAllocateMemory: synthetic/null device cannot back memory");
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }
  if (pAllocateInfo->memoryTypeIndex >= ppu::kMemoryTypeCount) {
    PPU_ERR("vkAllocateMemory: bad memory type %u", pAllocateInfo->memoryTypeIndex);
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }
  if (!ppu::bind_context(dev)) return VK_ERROR_DEVICE_LOST;

  ppu::Driver &d = ppu::driver();
  auto *mem = new DeviceMemory_T();
  mem->device = dev;
  mem->size = pAllocateInfo->allocationSize;
  mem->type_index = pAllocateInfo->memoryTypeIndex;

  if (pAllocateInfo->memoryTypeIndex == ppu::kMemoryTypeHost) {
    mem->kind = MemoryKind::HostMapped;
    void *host = nullptr;
    // DEVICEMAP yields a pointer usable from both CPU and device.
    ppu::CUresult rc = d.cuMemHostAlloc(&host, std::max<VkDeviceSize>(mem->size, 1),
                                        ppu::CU_MEMHOSTALLOC_DEVICEMAP);
    if (rc != CUDA_SUCCESS || host == nullptr) {
      PPU_ERR("cuMemHostAlloc(%llu) failed: %s", (unsigned long long)mem->size,
              ppu::cu_strerror(rc));
      delete mem;
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    mem->host_ptr = host;
    ppu::CUdeviceptr devptr = 0;
    rc = d.cuMemHostGetDevicePointer_v2(&devptr, host, 0);
    if (rc != CUDA_SUCCESS) {
      d.cuMemFreeHost(host);
      delete mem;
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    mem->device_ptr = devptr;
  } else {
    mem->kind = MemoryKind::Device;
    ppu::CUdeviceptr p = 0;
    ppu::CUresult rc = d.cuMemAlloc_v2(&p, std::max<VkDeviceSize>(mem->size, 1));
    if (rc != CUDA_SUCCESS) {
      PPU_ERR("cuMemAlloc(%llu) failed: %s", (unsigned long long)mem->size,
              ppu::cu_strerror(rc));
      delete mem;
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    mem->device_ptr = p;
  }

  *pMemory = reinterpret_cast<VkDeviceMemory>(mem);
  PPU_LOG("vkAllocateMemory type=%u size=%llu obj=%p -> ptr=0x%llx", mem->type_index,
          (unsigned long long)mem->size, (void *)mem, mem->device_ptr);
  return VK_SUCCESS;
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkFreeMemory(VkDevice device, VkDeviceMemory memory,
                                                   const VkAllocationCallbacks *pAllocator) {
  (void)pAllocator;
  (void)device;
  auto *mem = reinterpret_cast<DeviceMemory_T *>(memory);
  if (mem == nullptr) return;
  if (mem->magic != kMemoryMagic) {
    PPU_ERR("vkFreeMemory: %p is not a memory object from this driver", (void *)memory);
    return;
  }
  ppu::Driver &d = ppu::driver();
  if (mem->device) ppu::bind_context(mem->device);
  if (mem->kind == MemoryKind::HostMapped) {
    if (mem->host_ptr && d.cuMemFreeHost) d.cuMemFreeHost(mem->host_ptr);
  } else if (d.cuMemFree_v2) {
    d.cuMemFree_v2(mem->device_ptr);
  }
  delete mem;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkMapMemory(VkDevice device, VkDeviceMemory memory,
                                                      VkDeviceSize offset, VkDeviceSize size,
                                                      VkMemoryMapFlags flags, void **ppData) {
  (void)flags;
  (void)device;
  auto *mem = reinterpret_cast<DeviceMemory_T *>(memory);
  if (mem == nullptr || ppData == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
  if (mem->kind != MemoryKind::HostMapped || mem->host_ptr == nullptr) {
    // Device-only memory cannot be mapped; be explicit rather than lying.
    PPU_ERR("vkMapMemory on non host-visible memory type");
    return VK_ERROR_MEMORY_MAP_FAILED;
  }
  if (size == VK_WHOLE_SIZE) size = mem->size - offset;
  if (offset + size > mem->size) return VK_ERROR_MEMORY_MAP_FAILED;

  mem->mapped = true;
  mem->mapped_offset = offset;
  mem->mapped_size = size;
  *ppData = static_cast<uint8_t *>(mem->host_ptr) + offset;
  return VK_SUCCESS;
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkUnmapMemory(VkDevice device, VkDeviceMemory memory) {
  (void)device;
  auto *mem = reinterpret_cast<DeviceMemory_T *>(memory);
  if (mem == nullptr) return;
  mem->mapped = false;
  mem->mapped_offset = 0;
  mem->mapped_size = 0;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkFlushMappedMemoryRanges(
    VkDevice device, uint32_t memoryRangeCount, const VkMappedMemoryRange *pMemoryRanges) {
  (void)device;
  (void)memoryRangeCount;
  // Memory from the host-visible type is HOST_COHERENT, so there is nothing to
  // flush. Validate the handles so callers get honest errors for bad input.
  for (uint32_t i = 0; i < memoryRangeCount; ++i) {
    auto *mem = reinterpret_cast<DeviceMemory_T *>(pMemoryRanges[i].memory);
    if (mem == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    if (mem->kind != MemoryKind::HostMapped) {
      PPU_ERR("vkFlushMappedMemoryRanges on non host-visible memory");
      return VK_ERROR_MEMORY_MAP_FAILED;
    }
  }
  return VK_SUCCESS;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkInvalidateMappedMemoryRanges(
    VkDevice device, uint32_t memoryRangeCount, const VkMappedMemoryRange *pMemoryRanges) {
  (void)device;
  for (uint32_t i = 0; i < memoryRangeCount; ++i) {
    auto *mem = reinterpret_cast<DeviceMemory_T *>(pMemoryRanges[i].memory);
    if (mem == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    if (mem->kind != MemoryKind::HostMapped) {
      PPU_ERR("vkInvalidateMappedMemoryRanges on non host-visible memory");
      return VK_ERROR_MEMORY_MAP_FAILED;
    }
  }
  return VK_SUCCESS;
}

// ---------------------------------------------------------------------------
// Buffers
// ---------------------------------------------------------------------------
extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkCreateBuffer(VkDevice device,
                                                          const VkBufferCreateInfo *pCreateInfo,
                                                          const VkAllocationCallbacks *pAllocator,
                                                          VkBuffer *pBuffer) {
  (void)pAllocator;
  Device_T *dev = as_device(device);
  if (dev == nullptr || pCreateInfo == nullptr || pBuffer == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  if (pCreateInfo->size == 0) return VK_ERROR_INITIALIZATION_FAILED;

  auto *buf = new Buffer_T();
  buf->device = dev;
  buf->size = pCreateInfo->size;
  buf->usage = pCreateInfo->usage;
  buf->flags = pCreateInfo->flags;
  *pBuffer = reinterpret_cast<VkBuffer>(buf);
  return VK_SUCCESS;
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkDestroyBuffer(VkDevice device, VkBuffer buffer,
                                                       const VkAllocationCallbacks *pAllocator) {
  (void)device;
  (void)pAllocator;
  delete reinterpret_cast<Buffer_T *>(buffer);
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkGetBufferMemoryRequirements(
    VkDevice device, VkBuffer buffer, VkMemoryRequirements *pMemoryRequirements) {
  (void)device;
  auto *buf = reinterpret_cast<Buffer_T *>(buffer);
  if (buf == nullptr || pMemoryRequirements == nullptr) return;
  pMemoryRequirements->size = buf->size;
  pMemoryRequirements->alignment = ppu::kBufferAlignment;
  // Device and host memory types are both valid storage for a buffer.
  pMemoryRequirements->memoryTypeBits = (1u << ppu::kMemoryTypeDevice) |
                                        (1u << ppu::kMemoryTypeHost);
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkGetBufferMemoryRequirements2(
    VkDevice device, const VkBufferMemoryRequirementsInfo2 *pInfo,
    VkMemoryRequirements2 *pMemoryRequirements) {
  if (pInfo == nullptr || pMemoryRequirements == nullptr) return;
  vkGetBufferMemoryRequirements(device, pInfo->buffer, &pMemoryRequirements->memoryRequirements);
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkGetDeviceBufferMemoryRequirements(
    VkDevice device, const VkDeviceBufferMemoryRequirements *pInfo,
    VkMemoryRequirements2 *pMemoryRequirements) {
  if (pInfo == nullptr || pMemoryRequirements == nullptr) return;
  auto *dev = as_device(device);
  (void)dev;
  const VkBufferCreateInfo *ci = pInfo->pCreateInfo;
  pMemoryRequirements->memoryRequirements.size = ci ? ci->size : 0;
  pMemoryRequirements->memoryRequirements.alignment = ppu::kBufferAlignment;
  pMemoryRequirements->memoryRequirements.memoryTypeBits =
      (1u << ppu::kMemoryTypeDevice) | (1u << ppu::kMemoryTypeHost);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkBindBufferMemory(VkDevice device, VkBuffer buffer,
                                                              VkDeviceMemory memory,
                                                              VkDeviceSize memoryOffset) {
  Device_T *dev = as_device(device);
  (void)dev;
  auto *buf = reinterpret_cast<Buffer_T *>(buffer);
  auto *mem = reinterpret_cast<DeviceMemory_T *>(memory);
  if (buf == nullptr || mem == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
  if (memoryOffset % ppu::kBufferAlignment != 0) {
    PPU_ERR("vkBindBufferMemory: offset %llu is not %llu-aligned",
            (unsigned long long)memoryOffset, (unsigned long long)ppu::kBufferAlignment);
    return VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS;
  }
  if (memoryOffset + buf->size > mem->size) {
    PPU_ERR("vkBindBufferMemory: buffer (%llu) does not fit in memory (%llu) at offset %llu",
            (unsigned long long)buf->size, (unsigned long long)mem->size,
            (unsigned long long)memoryOffset);
    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
  }
  buf->memory = mem;
  buf->memory_offset = memoryOffset;
  return VK_SUCCESS;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkBindBufferMemory2(
    VkDevice device, uint32_t bindInfoCount, const VkBindBufferMemoryInfo *pBindInfos) {
  if (pBindInfos == nullptr && bindInfoCount != 0) return VK_ERROR_INITIALIZATION_FAILED;
  for (uint32_t i = 0; i < bindInfoCount; ++i) {
    VkResult r = vkBindBufferMemory(device, pBindInfos[i].buffer, pBindInfos[i].memory,
                                     pBindInfos[i].memoryOffset);
    if (r != VK_SUCCESS) return r;
  }
  return VK_SUCCESS;
}

extern "C" VKAPI_ATTR VkDeviceAddress VKAPI_CALL vkGetBufferDeviceAddress(
    VkDevice device, const VkBufferDeviceAddressInfo *pInfo) {
  Device_T *dev = as_device(device);
  (void)dev;
  if (pInfo == nullptr) return 0;
  auto *buf = reinterpret_cast<Buffer_T *>(pInfo->buffer);
  if (buf == nullptr || buf->memory == nullptr) return 0;
  return ppu::buffer_address(buf, 0);
}

extern "C" VKAPI_ATTR VkDeviceAddress VKAPI_CALL vkGetBufferDeviceAddressKHR(
    VkDevice device, const VkBufferDeviceAddressInfo *pInfo) {
  return vkGetBufferDeviceAddress(device, pInfo);
}

extern "C" VKAPI_ATTR uint64_t VKAPI_CALL vkGetBufferOpaqueCaptureAddress(
    VkDevice device, const VkBufferDeviceAddressInfo *pInfo) {
  (void)device;
  (void)pInfo;
  return 0;
}

extern "C" VKAPI_ATTR uint64_t VKAPI_CALL vkGetDeviceMemoryOpaqueCaptureAddress(
    VkDevice device, const VkDeviceMemoryOpaqueCaptureAddressInfo *pInfo) {
  (void)device;
  (void)pInfo;
  return 0;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkGetMemoryFdKHR(VkDevice device,
                                                           const VkMemoryGetFdInfoKHR *pGetFdInfo,
                                                           int *pFd) {
  (void)device;
  (void)pGetFdInfo;
  (void)pFd;
  return VK_ERROR_FEATURE_NOT_PRESENT;
}
