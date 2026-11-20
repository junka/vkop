// Instance, physical-device and loader-handshake entry points.
#include "icd_common.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unistd.h>

namespace ppu {

bool debug_enabled() {
  static int cached = -1;
  if (cached < 0) {
    const char *v = std::getenv("VK_PPU_DEBUG");
    cached = (v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0) ? 1 : 0;
  }
  return cached != 0;
}

void stamp(VK_LOADER_DATA *data) { set_loader_magic_value(data); }

namespace {

// Host RAM in bytes, as the amount userspace can actually allocate. Prefers
// /proc/meminfo's MemTotal (which excludes kernel-reserved memory) and falls
// back to sysconf's raw physical page count. Zero when unknown; callers then
// fall back to VK_WHOLE_SIZE rather than inventing a number.
size_t host_ram_bytes() {
  size_t mem_total = 0;
  if (std::FILE *f = std::fopen("/proc/meminfo", "r")) {
    char line[256];
    unsigned long long kb = 0;
    while (std::fgets(line, sizeof line, f) != nullptr) {
      if (std::sscanf(line, "MemTotal: %llu kB", &kb) == 1) {
        mem_total = static_cast<size_t>(kb) * 1024u;
        break;
      }
    }
    std::fclose(f);
  }
  long pages = sysconf(_SC_PHYS_PAGES);
  long page_size = sysconf(_SC_PAGESIZE);
  size_t raw = (pages > 0 && page_size > 0)
                   ? static_cast<size_t>(pages) * static_cast<size_t>(page_size)
                   : 0;
  // Never report more than userspace can allocate.
  if (mem_total != 0 && raw != 0) return std::min(mem_total, raw);
  return mem_total != 0 ? mem_total : raw;
}

// Process-wide CUDA discovery, performed once.
struct Enumeration {
  std::once_flag once;
  bool cuda_ok = false;
  std::vector<PhysicalDevice_T *> devices;
};

Enumeration &enumeration() {
  static Enumeration e;
  return e;
}

void read_device_info(PhysicalDevice_T *pd, int index) {
  Driver &d = driver();

  ppu::CUdevice dev = 0;
  if (d.cuDeviceGet(&dev, index) != CUDA_SUCCESS) return;

  // Host RAM, reported as the capacity of the non-device-local heap. The
  // accelerator is an add-in card, so this is the system memory the host can
  // stage through, not a second accelerator partition.
  pd->host_memory = host_ram_bytes();

  // Timestamps are only usable if the backend can measure them. Both symbols
  // are part of the CUDA driver API the shim exposes; a shim that has one but
  // not the other would produce garbage, so require both.
  pd->timestamps = d.cuEventCreate != nullptr && d.cuEventRecord != nullptr &&
                   d.cuEventElapsedTime != nullptr && d.cuEventQuery != nullptr &&
                   d.cuEventSynchronize != nullptr;

  char name[256] = {0};
  if (d.cuDeviceGetName(name, sizeof name, dev) == CUDA_SUCCESS)
    std::snprintf(pd->name, sizeof pd->name, "%s", name);

  size_t total = 0;
  if (d.cuDeviceTotalMem_v2(&total, dev) == CUDA_SUCCESS)
    pd->total_memory = total;

  auto attr = [&](int id, uint32_t fallback) -> uint32_t {
    int v = 0;
    if (d.cuDeviceGetAttribute(&v, id, dev) == CUDA_SUCCESS && v >= 0)
      return static_cast<uint32_t>(v);
    return fallback;
  };

  pd->max_threads_per_block = attr(CU_DEV_ATTR_MAX_THREADS_PER_BLOCK, 1024);
  pd->max_threads_per_mp = attr(CU_DEV_ATTR_MAX_THREADS_PER_MP, 2048);
  pd->mp_count = attr(CU_DEV_ATTR_MULTIPROCESSOR_COUNT, 1);
  pd->clock_mhz = attr(CU_DEV_ATTR_CLOCK_RATE, 0) / 1000;
  pd->bus_width_bits = attr(CU_DEV_ATTR_GLOBAL_MEMORY_BUS_WIDTH, 0);
  pd->l2_cache_size = attr(CU_DEV_ATTR_L2_CACHE_SIZE, 0);
  pd->max_shared_memory = attr(CU_DEV_ATTR_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, 48u * 1024);
  if (pd->max_shared_memory == 0)
    pd->max_shared_memory = attr(CU_DEV_ATTR_MAX_SHARED_MEMORY_PER_BLOCK, 48u * 1024);
  pd->cc_major = static_cast<int>(attr(CU_DEV_ATTR_COMPUTE_CAPABILITY_MAJOR, 0));
  pd->cc_minor = static_cast<int>(attr(CU_DEV_ATTR_COMPUTE_CAPABILITY_MINOR, 0));

  // The CUDA shim exposes no cuDeviceGetUuid here, so derive a stable 16-byte
  // identifier from the device index and model number.
  std::memset(pd->uuid, 0, sizeof pd->uuid);
  pd->uuid[0] = 'P';
  pd->uuid[1] = 'P';
  pd->uuid[2] = 'U';
  pd->uuid[3] = static_cast<uint8_t>(index & 0xFF);
  pd->uuid[4] = 0x81;
  pd->uuid[5] = 0x0E;
}

void enumerate_once() {
  Enumeration &e = enumeration();
  Driver &d = driver();
  if (!d.loaded) {
    PPU_WARN("CUDA driver unavailable: %s", d.loadError ? d.loadError : "unknown");
    return;
  }
  if (d.cuInit(0) != CUDA_SUCCESS) {
    PPU_WARN("cuInit failed; no accelerator devices will be reported");
    return;
  }
  int count = 0;
  if (d.cuDeviceGetCount(&count) != CUDA_SUCCESS || count <= 0) {
    PPU_WARN("cuDeviceGetCount reported %d devices", count);
    return;
  }
  // Only honour devices the user allows. VK_PPU_DEVICE_INDEX selects one.
  int wanted = -1;
  if (const char *env = std::getenv("VK_PPU_DEVICE_INDEX"))
    wanted = std::atoi(env);

  for (int i = 0; i < count; ++i) {
    if (wanted >= 0 && i != wanted) continue;
    auto *pd = new PhysicalDevice_T();
    stamp(&pd->loader_data);
    pd->cu_index = i;
    read_device_info(pd, i);
    e.devices.push_back(pd);
  }
  e.cuda_ok = !e.devices.empty();
  PPU_LOG("enumerated %zu device(s) from %d CUDA device(s)", e.devices.size(), count);
}

}  // namespace

namespace {

// A clearly-labelled placeholder physical device for exercising the loader
// handshake and the property/feature/memory queries on a host with no usable
// accelerator. It is only reported when VK_PPU_SYNTHETIC_DEVICE=1 is set, it
// says so in its device name, and vkCreateDevice rejects it. This is for
// bring-up testing; it never pretends to be able to execute anything.
PhysicalDevice_T *make_synthetic_device() {
  auto *pd = new PhysicalDevice_T();
  stamp(&pd->loader_data);
  pd->synthetic = true;
  pd->cu_index = -1;
  std::snprintf(pd->name, sizeof pd->name, "PPU (synthetic: no accelerator present)");
  std::memset(pd->uuid, 0, sizeof pd->uuid);
  std::snprintf(reinterpret_cast<char *>(pd->uuid), sizeof pd->uuid, "PPU-synthetic");

  // Plausible values so queries have something coherent to return.
  pd->total_memory = 96ull << 30;
  pd->host_memory = host_ram_bytes() ? host_ram_bytes() : (32ull << 30);
  pd->timestamps = false;  // no accelerator, so no events to measure
  pd->max_threads_per_block = 1024;
  pd->max_threads_per_mp = 2048;
  pd->mp_count = 64;
  pd->clock_mhz = 1700;
  pd->bus_width_bits = 6144;
  pd->l2_cache_size = 8u << 20;
  pd->max_shared_memory = 256 * 1024;
  pd->cc_major = 8;
  pd->cc_minor = 0;
  return pd;
}

bool synthetic_requested() {
  const char *v = std::getenv("VK_PPU_SYNTHETIC_DEVICE");
  return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
}

}  // namespace

std::vector<PhysicalDevice_T *> enumerate_physical_devices(Instance_T *instance) {
  Enumeration &e = enumeration();
  std::call_once(e.once, enumerate_once);
  if (e.devices.empty()) {
    // No accelerator, no real device. Reporting zero devices is the honest
    // answer; the synthetic placeholder is available for handshake testing
    // only, and only on request.
    if (synthetic_requested()) {
      static PhysicalDevice_T *placeholder = nullptr;
      if (placeholder == nullptr) placeholder = make_synthetic_device();
      placeholder->instance = instance;
      return {placeholder};
    }
    return {};
  }
  for (auto *pd : e.devices) pd->instance = instance;
  return e.devices;
}

}  // namespace ppu

// ---------------------------------------------------------------------------
// Loader interface
// ---------------------------------------------------------------------------
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vk_icdNegotiateLoaderICDInterfaceVersion(uint32_t *pVersion) {
  if (pVersion == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
  // We implement GetInstanceProcAddr (>=1), interface negotiation (>=2),
  // GetPhysicalDeviceProcAddr (>=4) and honour the app's apiVersion (>=5).
  constexpr uint32_t kSupported = 5;
  if (*pVersion > kSupported) *pVersion = kSupported;
  if (*pVersion < 2) PPU_WARN("loader interface version %u is too old", *pVersion);
  return VK_SUCCESS;
}

extern "C" VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vk_icdGetInstanceProcAddr(VkInstance instance, const char *pName) {
  (void)instance;
  return ppu::ppu_lookup_instance_proc(pName);
}

extern "C" VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vk_icdGetPhysicalDeviceProcAddr(VkInstance instance, const char *pName) {
  (void)instance;
  return ppu::ppu_lookup_instance_proc(pName);
}

// ---------------------------------------------------------------------------
// Instance
// ---------------------------------------------------------------------------
extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceVersion(uint32_t *pApiVersion) {
  if (pApiVersion == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
  *pApiVersion = VK_API_VERSION_1_1;
  return VK_SUCCESS;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkCreateInstance(
    const VkInstanceCreateInfo *pCreateInfo, const VkAllocationCallbacks *pAllocator,
    VkInstance *pInstance) {
  (void)pAllocator;
  if (pCreateInfo == nullptr || pInstance == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  auto *inst = new Instance_T();
  ppu::stamp(&inst->loader_data);
  inst->api_version = pCreateInfo->pApplicationInfo != nullptr
                          ? pCreateInfo->pApplicationInfo->apiVersion
                          : VK_API_VERSION_1_0;
  *pInstance = reinterpret_cast<VkInstance>(inst);
  PPU_LOG("vkCreateInstance apiVersion=%u.%u.%u",
          VK_API_VERSION_MAJOR(inst->api_version),
          VK_API_VERSION_MINOR(inst->api_version),
          VK_API_VERSION_PATCH(inst->api_version));
  return VK_SUCCESS;
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkDestroyInstance(VkInstance instance,
                                                        const VkAllocationCallbacks *pAllocator) {
  (void)pAllocator;
  if (instance == VK_NULL_HANDLE) return;
  delete reinterpret_cast<Instance_T *>(instance);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(
    const char *pLayerName, uint32_t *pPropertyCount,
    VkExtensionProperties *pProperties) {
  (void)pLayerName;
  // One instance extension. VK_KHR_get_physical_device_properties2 is the
  // instance-level name under which the vkGetPhysicalDevice*Properties2 family
  // was introduced in Vulkan 1.0; this ICD implements that whole family (core
  // 1.1 plus every KHR alias), so advertising the name is accurate. It is also
  // the gate a consumer uses to decide whether chaining a property struct is
  // legal: vulkaninfo resolves vkGetPhysicalDeviceProperties2 either way, but
  // only calls it when this name is present, so without it the subgroup and
  // integer-dot-product blocks are never printed even though the ICD returns
  // them.
  static const VkExtensionProperties kInstanceExtensions[] = {
      {"VK_KHR_get_physical_device_properties2", 2},
  };
  constexpr uint32_t kCount = sizeof(kInstanceExtensions) / sizeof(kInstanceExtensions[0]);
  if (pPropertyCount == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
  if (pProperties == nullptr) {
    *pPropertyCount = kCount;
    return VK_SUCCESS;
  }
  uint32_t n = std::min(*pPropertyCount, kCount);
  for (uint32_t i = 0; i < n; ++i) pProperties[i] = kInstanceExtensions[i];
  *pPropertyCount = n;
  return (n < kCount) ? VK_INCOMPLETE : VK_SUCCESS;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(
    uint32_t *pPropertyCount, VkLayerProperties *pProperties) {
  if (pPropertyCount == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
  *pPropertyCount = 0;
  (void)pProperties;
  return VK_SUCCESS;
}

// ---------------------------------------------------------------------------
// Physical devices
// ---------------------------------------------------------------------------
extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkEnumeratePhysicalDevices(
    VkInstance instance, uint32_t *pPhysicalDeviceCount,
    VkPhysicalDevice *pPhysicalDevices) {
  if (instance == VK_NULL_HANDLE || pPhysicalDeviceCount == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  auto *inst = reinterpret_cast<Instance_T *>(instance);
  auto devices = ppu::enumerate_physical_devices(inst);

  if (pPhysicalDevices == nullptr) {
    *pPhysicalDeviceCount = static_cast<uint32_t>(devices.size());
    return VK_SUCCESS;
  }
  uint32_t n = std::min<uint32_t>(*pPhysicalDeviceCount,
                                  static_cast<uint32_t>(devices.size()));
  for (uint32_t i = 0; i < n; ++i)
    pPhysicalDevices[i] = reinterpret_cast<VkPhysicalDevice>(devices[i]);
  VkResult r = (n < devices.size()) ? VK_INCOMPLETE : VK_SUCCESS;
  *pPhysicalDeviceCount = n;
  return r;
}

namespace {

PhysicalDevice_T *as_phys(VkPhysicalDevice pd) {
  return reinterpret_cast<PhysicalDevice_T *>(pd);
}

void fill_limits(PhysicalDevice_T *pd, VkPhysicalDeviceLimits *l) {
  std::memset(l, 0, sizeof *l);
  // Only compute/transfer limits are meaningful; everything image- or
  // graphics-related stays zero, which is how we advertise "no images".
  l->maxImageDimension1D = 0;
  l->maxImageDimension2D = 0;
  l->maxImageDimension3D = 0;

  l->maxMemoryAllocationCount = 4096;
  l->maxSamplerAllocationCount = 0;
  l->bufferImageGranularity = ppu::kBufferAlignment;
  l->sparseAddressSpaceSize = 0;
  l->maxBoundDescriptorSets = 8;

  // Resource limits are constrained by what the replay path actually binds:
  // one 64-bit address per descriptor into a cuLaunchKernel argument list, in
  // layout order. 32 storage buffers + 32 uniform buffers per stage and 64 per
  // set are what the kernel-argument packing supports today.
  l->maxPerStageDescriptorSamplers = 0;
  l->maxPerStageDescriptorUniformBuffers = 32;
  l->maxPerStageDescriptorStorageBuffers = 32;
  l->maxPerStageDescriptorSampledImages = 0;
  l->maxPerStageDescriptorStorageImages = 0;
  l->maxPerStageDescriptorInputAttachments = 0;
  l->maxPerStageResources = 64;
  l->maxDescriptorSetSamplers = 0;
  l->maxDescriptorSetUniformBuffers = 64;
  l->maxDescriptorSetUniformBuffersDynamic = 8;
  l->maxDescriptorSetStorageBuffers = 64;
  l->maxDescriptorSetStorageBuffersDynamic = 8;
  l->maxDescriptorSetSampledImages = 0;
  l->maxDescriptorSetStorageImages = 0;
  l->maxDescriptorSetInputAttachments = 0;

  l->maxVertexInputAttributes = 0;
  l->maxVertexInputBindings = 0;
  l->maxVertexInputAttributeOffset = 0;
  l->maxVertexInputBindingStride = 0;
  l->maxVertexOutputComponents = 0;
  l->maxFragmentInputComponents = 0;
  l->maxFragmentOutputAttachments = 0;
  l->maxFragmentDualSrcAttachments = 0;
  l->maxFragmentCombinedOutputResources = 0;

  l->maxComputeSharedMemorySize = pd->max_shared_memory;
  // grid dimensions come from CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_*.
  l->maxComputeWorkGroupCount[0] = 2147483647u;
  l->maxComputeWorkGroupCount[1] = 65535u;
  l->maxComputeWorkGroupCount[2] = 65535u;
  l->maxComputeWorkGroupInvocations = pd->max_threads_per_block;
  l->maxComputeWorkGroupSize[0] = pd->max_threads_per_block;
  l->maxComputeWorkGroupSize[1] = pd->max_threads_per_block;
  l->maxComputeWorkGroupSize[2] = std::min<uint32_t>(pd->max_threads_per_block, 64);

  l->subPixelPrecisionBits = 0;
  l->subTexelPrecisionBits = 0;
  l->mipmapPrecisionBits = 0;

  // Descriptor/constant ranges. maxPushConstantsSize is 128 because the
  // largest parameter block an operator pushes is exactly that; a smaller
  // value makes the consumer clamp its range and drop every field past the
  // clamp. Uniform buffers are used for weights and small constants, so the
  // range follows the Vulkan minimum of 64 KiB; storage buffers stay at the
  // 4 GiB - 1 ceiling the address space supports.
  l->maxPushConstantsSize = ppu::kMaxPushConstantsSize;
  l->maxUniformBufferRange = static_cast<uint32_t>(ppu::kMaxUniformBufferRange);

  l->maxDrawIndexedIndexValue = 0;
  l->maxDrawIndirectCount = 0;
  l->maxSamplerLodBias = 0;
  l->maxSamplerAnisotropy = 0;
  l->maxViewports = 0;
  l->maxViewportDimensions[0] = 0;
  l->maxViewportDimensions[1] = 0;
  l->viewportBoundsRange[0] = 0;
  l->viewportBoundsRange[1] = 0;
  l->viewportSubPixelBits = 0;
  l->minMemoryMapAlignment = 64;
  l->minTexelBufferOffsetAlignment = 256;
  l->minUniformBufferOffsetAlignment = static_cast<VkDeviceSize>(ppu::kMinUniformBufferOffsetAlignment);
  l->minStorageBufferOffsetAlignment = static_cast<VkDeviceSize>(ppu::kMinStorageBufferOffsetAlignment);
  l->minTexelOffset = 0;
  l->maxTexelOffset = 0;
  l->minTexelGatherOffset = 0;
  l->maxTexelGatherOffset = 0;
  l->minInterpolationOffset = 0;
  l->maxInterpolationOffset = 0;
  l->subPixelInterpolationOffsetBits = 0;
  l->maxFramebufferWidth = 0;
  l->maxFramebufferHeight = 0;
  l->maxFramebufferLayers = 0;
  l->framebufferColorSampleCounts = 0;
  l->framebufferDepthSampleCounts = 0;
  l->framebufferStencilSampleCounts = 0;
  l->framebufferNoAttachmentsSampleCounts = 0;
  l->maxColorAttachments = 0;
  l->sampledImageColorSampleCounts = 0;
  l->sampledImageIntegerSampleCounts = 0;
  l->sampledImageDepthSampleCounts = 0;
  l->sampledImageStencilSampleCounts = 0;
  l->storageImageSampleCounts = VK_SAMPLE_COUNT_1_BIT;
  l->maxSampleMaskWords = 0;
  // Timestamps come from CUDA events, so the period is a fixed 1 ns per tick
  // (see kTimestampPeriodNs): the query value is a nanosecond delta, not a
  // hardware counter. This is only claimed when the backend can honour it;
  // the synthetic placeholder device leaves it at zero.
  l->timestampComputeAndGraphics = pd->timestamps ? VK_TRUE : VK_FALSE;
  l->timestampPeriod = pd->timestamps ? ppu::kTimestampPeriodNs : 0.0f;
  l->maxClipDistances = 0;
  l->maxCullDistances = 0;
  l->maxCombinedClipAndCullDistances = 0;
  l->discreteQueuePriorities = 1;
  l->pointSizeRange[0] = 0;
  l->pointSizeRange[1] = 0;
  l->lineWidthRange[0] = 0;
  l->lineWidthRange[1] = 0;
  l->pointSizeGranularity = 0;
  l->lineWidthGranularity = 0;

  l->maxStorageBufferRange = 0xFFFFFFFFu;
  l->nonCoherentAtomSize = 64;
}

void fill_features(VkPhysicalDeviceFeatures *f, bool usable) {
  std::memset(f, 0, sizeof *f);
  // Compute is the only thing this device does. Everything image, vertex,
  // tessellation or fragment related stays VK_FALSE.
  f->shaderInt64 = VK_TRUE;
  f->shaderInt16 = VK_TRUE;
  f->shaderFloat64 = VK_TRUE;
  f->shaderStorageImageWriteWithoutFormat = VK_FALSE;
  f->shaderStorageBufferArrayDynamicIndexing = VK_TRUE;
  f->shaderUniformBufferArrayDynamicIndexing = VK_TRUE;
  f->shaderSampledImageArrayDynamicIndexing = VK_FALSE;
  f->shaderStorageImageArrayDynamicIndexing = VK_FALSE;
  (void)usable;
}

// Fill the extension feature structs we can honestly claim. Walks the caller's
// pNext chain so we never write a struct they did not ask about.
//
// What is deliberately NOT here (each was considered and rejected):
//  - VkPhysicalDeviceCooperativeMatrixFeaturesKHR / ...NV: the SPIR-V emitter
//    this ICD is paired with does not implement cooperative matrix (it fails
//    loud on every matmul_coop* shader). Advertising the feature would let a
//    consumer select a shader class that can never compile.
//  - VkPhysicalDeviceShaderSubgroupRotateFeatures: the device has
//    shuffle.idx/up/down but no rotate semantics, and nothing below lowers a
//    rotate to three shuffles.
//  - VkPhysicalDeviceShaderAtomicFloatFeatures / ...AtomicFloat16VectorNV:
//    fp32/fp16 atomic add on storage or shared memory was never verified on
//    this hardware; a false "yes" here silently changes the split-K strategy.
void fill_features_pnext(void *pNext, bool usable) {
  for (auto *s = reinterpret_cast<VkBaseOutStructure *>(pNext); s != nullptr; s = s->pNext) {
    switch (s->sType) {
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES: {
        auto *f = reinterpret_cast<VkPhysicalDeviceVulkan11Features *>(s);
        f->storageBuffer16BitAccess = VK_TRUE;
        f->uniformAndStorageBuffer16BitAccess = VK_TRUE;
        f->storagePushConstant16 = VK_TRUE;
        f->multiview = VK_FALSE;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES: {
        // The standalone pre-1.1 spelling. A consumer at apiVersion 1.1 asks
        // with Vk11Features above; one that checks the extension by name asks
        // with this struct, so both answer the same.
        auto *f = reinterpret_cast<VkPhysicalDevice16BitStorageFeatures *>(s);
        f->storageBuffer16BitAccess = VK_TRUE;
        f->uniformAndStorageBuffer16BitAccess = VK_TRUE;
        f->storagePushConstant16 = VK_TRUE;
        f->storageInputOutput16 = VK_FALSE;  // no fragment stage exists
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES: {
        auto *f = reinterpret_cast<VkPhysicalDevice8BitStorageFeatures *>(s);
        f->storageBuffer8BitAccess = VK_TRUE;
        f->uniformAndStorageBuffer8BitAccess = VK_TRUE;
        f->storagePushConstant8 = VK_FALSE;  // push constants stay 32-bit here
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES: {
        // Both halves, honestly: fp16 arithmetic (__hmul/__hadd measured) and
        // 8-bit integer arithmetic (the int8 kernels rely on it).
        auto *f = reinterpret_cast<VkPhysicalDeviceShaderFloat16Int8Features *>(s);
        f->shaderFloat16 = VK_TRUE;
        f->shaderInt8 = VK_TRUE;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES: {
        // The device has exactly one subgroup size (32), so "control" cannot
        // change anything, but the feature is what lets a pipeline pin
        // requiredSubgroupSize -- and 32 is the only value the range permits,
        // so every legal request is already what the hardware runs.
        // computeFullSubgroups stays VK_FALSE: it promises that a pipeline may
        // set REQUIRE_FULL_SUBGROUPS with a local size that is NOT a multiple
        // of the subgroup size and still have every subgroup fully populated.
        // This ICD hands the local size straight to cuLaunchKernel, which
        // leaves a partial tail warp, and the SPIR-V emitter emits no
        // predicated round-up, so the promise would be false.
        auto *f = reinterpret_cast<VkPhysicalDeviceSubgroupSizeControlFeatures *>(s);
        f->subgroupSizeControl = VK_TRUE;
        f->computeFullSubgroups = VK_FALSE;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES: {
        // Permission to put OpCapability DotProduct in a module. It says
        // nothing about speed; the matching *properties* struct reports every
        // acceleration bit as false.
        auto *f = reinterpret_cast<VkPhysicalDeviceShaderIntegerDotProductFeatures *>(s);
        f->shaderIntegerDotProduct = VK_TRUE;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR: {
        // 256 KiB/block of shared memory, and the emitter emits addrspace(3).
        // Scalar block layout is already true in the Vulkan12 chain; 8/16-bit
        // access to that memory was measured working.
        auto *f = reinterpret_cast<VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR *>(s);
        f->workgroupMemoryExplicitLayout = VK_TRUE;
        f->workgroupMemoryExplicitLayoutScalarBlockLayout = VK_TRUE;
        f->workgroupMemoryExplicitLayout8BitAccess = VK_TRUE;
        f->workgroupMemoryExplicitLayout16BitAccess = VK_TRUE;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ZERO_INITIALIZE_WORKGROUP_MEMORY_FEATURES: {
        // VK_KHR_zero_initialize_workgroup_memory is deliberately NOT
        // advertised. Measured on this device: a kernel that reads a
        // `__shared__` array it never wrote sees 0xFFFFFFFF residue left by
        // the previous block, so the guarantee the feature name promises does
        // not hold in hardware. Backing it would require the SPIR-V emitter
        // (a separate workstream) to inject a zero-fill prologue for every
        // workgroup variable, and no such lowering exists today. Advertising
        // it would let a consumer drop an explicit initialisation it needs.
        // We still answer a consumer that chains this struct: false.
        auto *f = reinterpret_cast<VkPhysicalDeviceZeroInitializeWorkgroupMemoryFeatures *>(s);
        f->shaderZeroInitializeWorkgroupMemory = VK_FALSE;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES: {
        auto *f = reinterpret_cast<VkPhysicalDeviceVulkan12Features *>(s);
        f->bufferDeviceAddress = usable ? VK_TRUE : VK_FALSE;
        f->shaderInt8 = VK_TRUE;
        f->storageBuffer8BitAccess = VK_TRUE;
        f->uniformAndStorageBuffer8BitAccess = VK_TRUE;
        f->scalarBlockLayout = VK_TRUE;
        f->shaderFloat16 = VK_TRUE;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES: {
        auto *f = reinterpret_cast<VkPhysicalDeviceBufferDeviceAddressFeatures *>(s);
        f->bufferDeviceAddress = usable ? VK_TRUE : VK_FALSE;
        f->bufferDeviceAddressCaptureReplay = VK_FALSE;
        f->bufferDeviceAddressMultiDevice = VK_FALSE;
        break;
      }
      default:
        break;
    }
  }
}

}  // namespace

extern "C" VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties(
    VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties *pProperties) {
  if (pProperties == nullptr) return;
  PhysicalDevice_T *pd = as_phys(physicalDevice);
  VkPhysicalDeviceProperties p;
  std::memset(&p, 0, sizeof p);

  p.apiVersion = VK_API_VERSION_1_1;
  p.driverVersion = VK_MAKE_VERSION(0, 1, 0);
  // 0x1AE0 is not a registered PCI vendor id; this is a synthetic ICD.
  p.vendorID = 0x1AE0;
  p.deviceID = 0x810E;
  p.deviceType = VK_PHYSICAL_DEVICE_TYPE_OTHER;
  std::snprintf(p.deviceName, sizeof p.deviceName, "%s", pd->name);
  std::memcpy(p.pipelineCacheUUID, pd->uuid, VK_UUID_SIZE);

  // Spot values from the accelerator.
  p.limits.maxComputeWorkGroupInvocations = pd->max_threads_per_block;
  fill_limits(pd, &p.limits);
  p.limits.maxComputeWorkGroupInvocations = pd->max_threads_per_block;
  p.sparseProperties.residencyStandard2DBlockShape = VK_FALSE;
  *pProperties = p;
  (void)pd->total_memory;
}

namespace {

// The subgroup facts, in one place so the standalone 1.1 struct, the Vulkan 1.1
// aggregate struct and the extension spellings cannot drift apart.
struct SubgroupFacts {
  uint32_t size;
  VkShaderStageFlags stages;
  VkSubgroupFeatureFlags operations;
  VkBool32 quad_in_all_stages;
};

// What the device actually provides, measured with nvcc -keep + llvm-dis and
// confirmed with end-to-end kernels on the hardware:
//   BASIC    - gl_SubgroupSize etc. (warpSize printed on device = 32)
//   VOTE     - __any_sync/__all_sync -> ballot + emsk test (verified)
//   BALLOT   - llvm.ppu.ballot.i32 (verified)
//   SHUFFLE  - llvm.ppu.shuffle.{idx,up,down}.pred, including xor and the
//              width argument (both verified)
//   ARITHMETIC - __reduce_{add,min,max,and,or,xor}_sync lower to a SINGLE
//              hardware instruction (`v.red.add.u32`, `v.red.max.i32`,
//              `v.red.and.b32`, ...), not a shuffle loop. Vulkan requires the
//              bit to cover ADD/MUL/MIN/MAX for integer and float; integer is
//              native hardware and float add/max/min/mul via shuffles measured
//              numerically exact. See the ARITHMETIC caveat in the report:
//              the SPIR-V emitter must lower OpGroupNonUniformArithmetic to
//              llvm.ppu.redux.* / the shuffle sequence for this to hold.
// Deliberately absent:
//   QUAD_*    - no evidence for quad (2x2 lane) operations; not measured.
//   CLUSTERED - no cluster semantics on this hardware.
//   SHUFFLE_RELATIVE - never verified.
constexpr SubgroupFacts kSubgroupFacts = {
    ppu::kSubgroupSize,
    VK_SHADER_STAGE_COMPUTE_BIT,
    VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_VOTE_BIT |
        VK_SUBGROUP_FEATURE_BALLOT_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT |
        VK_SUBGROUP_FEATURE_ARITHMETIC_BIT,
    VK_FALSE,
};

}  // namespace

extern "C" VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties2(
    VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties2 *pProperties) {
  if (pProperties == nullptr) return;
  vkGetPhysicalDeviceProperties(physicalDevice, &pProperties->properties);
  PhysicalDevice_T *pd = as_phys(physicalDevice);
  for (auto *s = reinterpret_cast<VkBaseOutStructure *>(pProperties->pNext); s != nullptr;
       s = s->pNext) {
    switch (s->sType) {
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES: {
        auto *id = reinterpret_cast<VkPhysicalDeviceIDProperties *>(s);
        std::memset(id->deviceUUID, 0, VK_UUID_SIZE);
        std::memcpy(id->deviceUUID, pd->uuid, VK_UUID_SIZE);
        std::memset(id->driverUUID, 0, VK_UUID_SIZE);
        std::memcpy(id->driverUUID, pd->uuid, VK_UUID_SIZE);
        id->deviceLUIDValid = VK_FALSE;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES: {
        auto *m3 = reinterpret_cast<VkPhysicalDeviceMaintenance3Properties *>(s);
        m3->maxPerSetDescriptors = 1024;
        m3->maxMemoryAllocationSize = pd->total_memory ? pd->total_memory : (1ull << 32);
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES: {
        // Core Vulkan 1.1, standalone spelling. This is the struct the
        // reference consumer chains (VulkanDevice::getProperties).
        auto *sg = reinterpret_cast<VkPhysicalDeviceSubgroupProperties *>(s);
        sg->subgroupSize = kSubgroupFacts.size;
        sg->supportedStages = kSubgroupFacts.stages;
        sg->supportedOperations = kSubgroupFacts.operations;
        sg->quadOperationsInAllStages = kSubgroupFacts.quad_in_all_stages;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES: {
        // The Vulkan 1.1 aggregate struct. A consumer at apiVersion 1.1 (which
        // is what this ICD reports) may query this instead of the standalone
        // structs above, and vulkaninfo does exactly that -- without this case
        // no subgroup block is printed at all even though the standalone
        // struct carries the same values.
        auto *p11 = reinterpret_cast<VkPhysicalDeviceVulkan11Properties *>(s);
        std::memset(p11->deviceUUID, 0, VK_UUID_SIZE);
        std::memcpy(p11->deviceUUID, pd->uuid, VK_UUID_SIZE);
        std::memset(p11->driverUUID, 0, VK_UUID_SIZE);
        std::memcpy(p11->driverUUID, pd->uuid, VK_UUID_SIZE);
        std::memset(p11->deviceLUID, 0, VK_LUID_SIZE);
        p11->deviceNodeMask = 0;
        p11->deviceLUIDValid = VK_FALSE;
        p11->subgroupSize = kSubgroupFacts.size;
        p11->subgroupSupportedStages = kSubgroupFacts.stages;
        p11->subgroupSupportedOperations = kSubgroupFacts.operations;
        p11->subgroupQuadOperationsInAllStages = kSubgroupFacts.quad_in_all_stages;
        p11->pointClippingBehavior = VK_POINT_CLIPPING_BEHAVIOR_ALL_CLIP_PLANES;
        p11->maxMultiviewViewCount = 0;
        p11->maxMultiviewInstanceIndex = 0;
        p11->protectedNoFault = VK_FALSE;
        p11->maxPerSetDescriptors = 1024;
        p11->maxMemoryAllocationSize = pd->total_memory ? pd->total_memory : (1ull << 32);
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES: {
        // Vulkan 1.3 core, but this ICD reports apiVersion 1.1.0, so it is
        // reached through VK_EXT_subgroup_size_control. There is exactly one
        // subgroup size, so the range is degenerate: pinning 32 always works
        // and pinning anything else is a caller error.
        auto *ssc = reinterpret_cast<VkPhysicalDeviceSubgroupSizeControlProperties *>(s);
        ssc->minSubgroupSize = ppu::kSubgroupSize;
        ssc->maxSubgroupSize = ppu::kSubgroupSize;
        ssc->maxComputeWorkgroupSubgroups = ppu::kMaxComputeWorkgroupSubgroups;
        ssc->requiredSubgroupSizeStages = VK_SHADER_STAGE_COMPUTE_BIT;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES: {
        // Measured negative result: __dp4a does NOT map to a device
        // instruction. `nvcc -keep` + `llvm-dis` shows it expanded into four
        // `v.madl.i32` ops plus prmt byte-extracts in the PPU assembly, i.e.
        // pure emulation. Every acceleration bit is therefore VK_FALSE --
        // including integerDotProduct4x8BitPackedSignedAccelerated, which a
        // consumer might otherwise route an int8 GEMV to. All 30 fields are
        // written explicitly rather than relying on the caller's zeroing, so
        // the answer holds even for an uninitialised struct.
        auto *ip = reinterpret_cast<VkPhysicalDeviceShaderIntegerDotProductProperties *>(s);
        ip->integerDotProduct8BitUnsignedAccelerated = VK_FALSE;
        ip->integerDotProduct8BitSignedAccelerated = VK_FALSE;
        ip->integerDotProduct8BitMixedSignednessAccelerated = VK_FALSE;
        ip->integerDotProduct4x8BitPackedUnsignedAccelerated = VK_FALSE;
        ip->integerDotProduct4x8BitPackedSignedAccelerated = VK_FALSE;
        ip->integerDotProduct4x8BitPackedMixedSignednessAccelerated = VK_FALSE;
        ip->integerDotProduct16BitUnsignedAccelerated = VK_FALSE;
        ip->integerDotProduct16BitSignedAccelerated = VK_FALSE;
        ip->integerDotProduct16BitMixedSignednessAccelerated = VK_FALSE;
        ip->integerDotProduct32BitUnsignedAccelerated = VK_FALSE;
        ip->integerDotProduct32BitSignedAccelerated = VK_FALSE;
        ip->integerDotProduct32BitMixedSignednessAccelerated = VK_FALSE;
        ip->integerDotProduct64BitUnsignedAccelerated = VK_FALSE;
        ip->integerDotProduct64BitSignedAccelerated = VK_FALSE;
        ip->integerDotProduct64BitMixedSignednessAccelerated = VK_FALSE;
        ip->integerDotProductAccumulatingSaturating8BitUnsignedAccelerated = VK_FALSE;
        ip->integerDotProductAccumulatingSaturating8BitSignedAccelerated = VK_FALSE;
        ip->integerDotProductAccumulatingSaturating8BitMixedSignednessAccelerated = VK_FALSE;
        ip->integerDotProductAccumulatingSaturating4x8BitPackedUnsignedAccelerated = VK_FALSE;
        ip->integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated = VK_FALSE;
        ip->integerDotProductAccumulatingSaturating4x8BitPackedMixedSignednessAccelerated = VK_FALSE;
        ip->integerDotProductAccumulatingSaturating16BitUnsignedAccelerated = VK_FALSE;
        ip->integerDotProductAccumulatingSaturating16BitSignedAccelerated = VK_FALSE;
        ip->integerDotProductAccumulatingSaturating16BitMixedSignednessAccelerated = VK_FALSE;
        ip->integerDotProductAccumulatingSaturating32BitUnsignedAccelerated = VK_FALSE;
        ip->integerDotProductAccumulatingSaturating32BitSignedAccelerated = VK_FALSE;
        ip->integerDotProductAccumulatingSaturating32BitMixedSignednessAccelerated = VK_FALSE;
        ip->integerDotProductAccumulatingSaturating64BitUnsignedAccelerated = VK_FALSE;
        ip->integerDotProductAccumulatingSaturating64BitSignedAccelerated = VK_FALSE;
        ip->integerDotProductAccumulatingSaturating64BitMixedSignednessAccelerated = VK_FALSE;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_POINT_CLIPPING_PROPERTIES: {
        break;
      }
      default:
        break;
    }
  }
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures(
    VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures *pFeatures) {
  if (pFeatures == nullptr) return;
  PhysicalDevice_T *pd = as_phys(physicalDevice);
  fill_features(pFeatures, !pd->synthetic);
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures2(
    VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures2 *pFeatures) {
  if (pFeatures == nullptr) return;
  PhysicalDevice_T *pd = as_phys(physicalDevice);
  fill_features(&pFeatures->features, !pd->synthetic);
  fill_features_pnext(pFeatures->pNext, !pd->synthetic);
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceQueueFamilyProperties(
    VkPhysicalDevice physicalDevice, uint32_t *pQueueFamilyPropertyCount,
    VkQueueFamilyProperties *pQueueFamilyProperties) {
  PhysicalDevice_T *pd = as_phys(physicalDevice);
  if (pQueueFamilyPropertyCount == nullptr) return;
  if (pQueueFamilyProperties == nullptr) {
    *pQueueFamilyPropertyCount = 1;
    return;
  }
  if (*pQueueFamilyPropertyCount < 1) {
    *pQueueFamilyPropertyCount = 0;
    return;
  }
  VkQueueFamilyProperties q;
  std::memset(&q, 0, sizeof q);
  q.queueFlags = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
  q.queueCount = 1;
  q.timestampValidBits = (pd != nullptr && pd->timestamps) ? ppu::kTimestampValidBits : 0;
  q.minImageTransferGranularity = {0, 0, 0};
  pQueueFamilyProperties[0] = q;
  *pQueueFamilyPropertyCount = 1;
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceQueueFamilyProperties2(
    VkPhysicalDevice physicalDevice, uint32_t *pQueueFamilyPropertyCount,
    VkQueueFamilyProperties2 *pQueueFamilyProperties) {
  if (pQueueFamilyPropertyCount == nullptr) return;
  if (pQueueFamilyProperties == nullptr) {
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, pQueueFamilyPropertyCount, nullptr);
    return;
  }
  uint32_t n = *pQueueFamilyPropertyCount;
  std::vector<VkQueueFamilyProperties> tmp(std::max<uint32_t>(n, 1));
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &n, tmp.data());
  for (uint32_t i = 0; i < n; ++i) pQueueFamilyProperties[i].queueFamilyProperties = tmp[i];
  *pQueueFamilyPropertyCount = n;
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceMemoryProperties(
    VkPhysicalDevice physicalDevice, VkPhysicalDeviceMemoryProperties *pMemoryProperties) {
  PhysicalDevice_T *pd = as_phys(physicalDevice);
  if (pMemoryProperties == nullptr) return;
  VkPhysicalDeviceMemoryProperties m;
  std::memset(&m, 0, sizeof m);

  // Two heaps. The accelerator has device memory and the host has system
  // memory, and reporting them as one heap is the signal applications read as
  // "unified memory architecture" - which would then select backends this ICD
  // does not implement. Both memory types below still live in heap 0 because
  // both are backed by device memory (type 1 is cuMemHostAlloc(DEVICEMAP), a
  // device allocation with a CPU-visible address, not system memory).
  m.memoryHeapCount = ppu::kHeapCount;
  m.memoryHeaps[ppu::kHeapDevice].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
  m.memoryHeaps[ppu::kHeapDevice].size =
      pd != nullptr && pd->total_memory != 0 ? pd->total_memory : VK_WHOLE_SIZE;
  m.memoryHeaps[ppu::kHeapHost].flags = 0;
  m.memoryHeaps[ppu::kHeapHost].size =
      pd != nullptr && pd->host_memory != 0 ? pd->host_memory : VK_WHOLE_SIZE;

  m.memoryTypeCount = ppu::kMemoryTypeCount;

  // Type 0: device-local, not host visible (cuMemAlloc).
  m.memoryTypes[ppu::kMemoryTypeDevice].heapIndex = ppu::kHeapDevice;
  m.memoryTypes[ppu::kMemoryTypeDevice].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

  // Type 1: host-visible, coherent and device-local. This maps to
  // cuMemHostAlloc(DEVICEMAP), which on this hardware hands back a pointer
  // that is simultaneously a valid CPU address and a valid device address.
  m.memoryTypes[ppu::kMemoryTypeHost].heapIndex = ppu::kHeapDevice;
  m.memoryTypes[ppu::kMemoryTypeHost].propertyFlags =
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

  *pMemoryProperties = m;
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceMemoryProperties2(
    VkPhysicalDevice physicalDevice, VkPhysicalDeviceMemoryProperties2 *pMemoryProperties) {
  if (pMemoryProperties == nullptr) return;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &pMemoryProperties->memoryProperties);
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFormatProperties(
    VkPhysicalDevice physicalDevice, VkFormat format, VkFormatProperties *pFormatProperties) {
  (void)physicalDevice;
  (void)format;
  if (pFormatProperties == nullptr) return;
  // No image, texel-buffer or tiling support of any kind.
  std::memset(pFormatProperties, 0, sizeof *pFormatProperties);
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFormatProperties2(
    VkPhysicalDevice physicalDevice, VkFormat format, VkFormatProperties2 *pFormatProperties) {
  if (pFormatProperties == nullptr) return;
  vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &pFormatProperties->formatProperties);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceImageFormatProperties(
    VkPhysicalDevice physicalDevice, VkFormat format, VkImageType type,
    VkImageTiling tiling, VkImageUsageFlags usage, VkImageCreateFlags flags,
    VkImageFormatProperties *pImageFormatProperties) {
  (void)physicalDevice;
  (void)format;
  (void)type;
  (void)tiling;
  (void)usage;
  (void)flags;
  (void)pImageFormatProperties;
  return VK_ERROR_FORMAT_NOT_SUPPORTED;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceImageFormatProperties2(
    VkPhysicalDevice physicalDevice, const VkPhysicalDeviceImageFormatInfo2 *pImageFormatInfo,
    VkImageFormatProperties2 *pImageFormatProperties) {
  (void)physicalDevice;
  (void)pImageFormatInfo;
  (void)pImageFormatProperties;
  return VK_ERROR_FORMAT_NOT_SUPPORTED;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceExtensionProperties(
    VkPhysicalDevice physicalDevice, const char *pLayerName, uint32_t *pPropertyCount,
    VkExtensionProperties *pProperties) {
  (void)physicalDevice;
  if (pLayerName != nullptr) {
    // Layer-scoped extension queries are not ours to answer.
    if (pPropertyCount) *pPropertyCount = 0;
    return VK_SUCCESS;
  }
  // The device extension list. Every name here is backed by a feature or
  // property the ICD actually returns, and the matching KHR/EXT entry point
  // (where the extension defines one) is served from dispatch_table.cpp.
  //
  // Deliberately absent, and why:
  //   VK_KHR_cooperative_matrix / VK_NV_cooperative_matrix -- the paired
  //     SPIR-V emitter does not implement cooperative matrix and fails loud on
  //     every matmul_coop* shader, so advertising it would route a consumer to
  //     a path that cannot compile.
  //   VK_KHR_shader_subgroup_rotate -- the device has shuffle.idx/down/up but
  //     no rotate, and nothing lowers rotate to three shuffles.
  //   VK_EXT_shader_atomic_float / VK_NV_shader_atomic_float16_vector -- not
  //     verified on this hardware; a false yes would change split-K strategy.
  //   VK_KHR_zero_initialize_workgroup_memory -- measured NOT to hold: shared
  //     memory retains the previous block's 0xFFFFFFFF residue.
  static const VkExtensionProperties kExtensions[] = {
      {"VK_KHR_buffer_device_address", 1},
      {"VK_KHR_maintenance3", 1},
      {"VK_KHR_storage_buffer_storage_class", 1},
      {"VK_KHR_shader_float16_int8", VK_KHR_SHADER_FLOAT16_INT8_SPEC_VERSION},
      // Feature structs already answered TRUE in the Vulkan11/12 chains; the
      // names are what a consumer that checks the extension list (rather than
      // only the core feature bit) gates on.
      {"VK_KHR_16bit_storage", VK_KHR_16BIT_STORAGE_SPEC_VERSION},
      {"VK_KHR_8bit_storage", VK_KHR_8BIT_STORAGE_SPEC_VERSION},
      {"VK_EXT_subgroup_size_control", VK_EXT_SUBGROUP_SIZE_CONTROL_SPEC_VERSION},
      // Present with every acceleration bit false: it tells a consumer the
      // capability exists but is not accelerated, which is the truthful and
      // useful answer (see the properties case for the dp4a measurement).
      {"VK_KHR_shader_integer_dot_product", VK_KHR_SHADER_INTEGER_DOT_PRODUCT_SPEC_VERSION},
      {"VK_KHR_workgroup_memory_explicit_layout",
       VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_SPEC_VERSION},
      // vkGetPhysicalDeviceToolProperties is implemented (and answers with an
      // empty tool list: this ICD is a driver, not a tool).
      {"VK_EXT_tooling_info", 1},
  };
  constexpr uint32_t kCount = sizeof(kExtensions) / sizeof(kExtensions[0]);
  if (pPropertyCount == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
  if (pProperties == nullptr) {
    *pPropertyCount = kCount;
    return VK_SUCCESS;
  }
  uint32_t n = std::min(*pPropertyCount, kCount);
  for (uint32_t i = 0; i < n; ++i) pProperties[i] = kExtensions[i];
  VkResult r = (n < kCount) ? VK_INCOMPLETE : VK_SUCCESS;
  *pPropertyCount = n;
  return r;
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceSparseImageFormatProperties(
    VkPhysicalDevice physicalDevice, VkFormat format, VkImageType type,
    VkSampleCountFlagBits samples, VkImageUsageFlags usage, VkImageTiling tiling,
    uint32_t *pPropertyCount, VkSparseImageFormatProperties *pProperties) {
  (void)physicalDevice;
  (void)format;
  (void)type;
  (void)samples;
  (void)usage;
  (void)tiling;
  if (pPropertyCount) *pPropertyCount = 0;
  (void)pProperties;
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceSparseImageFormatProperties2(
    VkPhysicalDevice physicalDevice, const VkPhysicalDeviceSparseImageFormatInfo2 *pFormatInfo,
    uint32_t *pPropertyCount, VkSparseImageFormatProperties2 *pProperties) {
  (void)pFormatInfo;
  (void)pProperties;
  (void)physicalDevice;
  if (pPropertyCount) *pPropertyCount = 0;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceToolProperties(
    VkPhysicalDevice physicalDevice, uint32_t *pToolCount,
    VkPhysicalDeviceToolProperties *pToolProperties) {
  (void)physicalDevice;
  if (pToolCount == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
  // This ICD is a driver, not a tooling layer, so the tool list is empty. The
  // entry point exists because applications query it unconditionally at
  // startup; an honest empty list is the correct answer, not an error.
  *pToolCount = 0;
  (void)pToolProperties;
  return VK_SUCCESS;
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceExternalBufferProperties(
    VkPhysicalDevice physicalDevice, const VkPhysicalDeviceExternalBufferInfo *pExternalBufferInfo,
    VkExternalBufferProperties *pExternalBufferProperties) {
  (void)physicalDevice;
  (void)pExternalBufferInfo;
  if (pExternalBufferProperties == nullptr) return;
  // No external memory handles are supported (no graphics interop on HGGC).
  std::memset(&pExternalBufferProperties->externalMemoryProperties, 0,
              sizeof pExternalBufferProperties->externalMemoryProperties);
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceExternalFenceProperties(
    VkPhysicalDevice physicalDevice, const VkPhysicalDeviceExternalFenceInfo *pExternalFenceInfo,
    VkExternalFenceProperties *pExternalFenceProperties) {
  (void)physicalDevice;
  (void)pExternalFenceInfo;
  if (pExternalFenceProperties == nullptr) return;
  std::memset(&pExternalFenceProperties->exportFromImportedHandleTypes, 0,
              sizeof pExternalFenceProperties->exportFromImportedHandleTypes);
  pExternalFenceProperties->compatibleHandleTypes = 0;
  pExternalFenceProperties->externalFenceFeatures = 0;
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceExternalSemaphoreProperties(
    VkPhysicalDevice physicalDevice,
    const VkPhysicalDeviceExternalSemaphoreInfo *pExternalSemaphoreInfo,
    VkExternalSemaphoreProperties *pExternalSemaphoreProperties) {
  (void)physicalDevice;
  (void)pExternalSemaphoreInfo;
  if (pExternalSemaphoreProperties == nullptr) return;
  pExternalSemaphoreProperties->exportFromImportedHandleTypes = 0;
  pExternalSemaphoreProperties->compatibleHandleTypes = 0;
  pExternalSemaphoreProperties->externalSemaphoreFeatures = 0;
}
