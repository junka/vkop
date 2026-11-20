// Shared internal definitions for the PPU Vulkan ICD.
//
// Object model
// ------------
// Vulkan handles are real pointers to the structs below. Dispatchable handles
// (instance, physical device, device, queue, command buffer) begin with a
// VK_LOADER_DATA word that we stamp with ICD_LOADER_MAGIC so the loader can
// recognise them as ICD-owned.
//
// Execution model
// ---------------
// Every vkCmd* call only *records* an operation into the command buffer. No
// CUDA API is touched while recording. vkQueueSubmit replays the recorded list
// onto the device's single CUstream. See submit.cpp.
#ifndef VK_PPU_ICD_COMMON_HPP
#define VK_PPU_ICD_COMMON_HPP

#include <vulkan/vulkan.h>

#include <vulkan/vk_icd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "runtime.hpp"

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------
namespace ppu {

// Enabled by VK_PPU_DEBUG (any non-empty value other than "0").
PPU_INTERNAL bool debug_enabled();

#define PPU_LOG(...)                                                           \
  do {                                                                         \
    if (::ppu::debug_enabled()) {                                              \
      std::fprintf(stderr, "[vk_ppu] " __VA_ARGS__);                           \
      std::fputc('\n', stderr);                                                \
    }                                                                          \
  } while (0)

#define PPU_WARN(...)                                                          \
  do {                                                                         \
    std::fprintf(stderr, "[vk_ppu] warning: " __VA_ARGS__);                    \
    std::fputc('\n', stderr);                                                  \
  } while (0)

#define PPU_ERR(...)                                                           \
  do {                                                                         \
    std::fprintf(stderr, "[vk_ppu] error: " __VA_ARGS__);                      \
    std::fputc('\n', stderr);                                                  \
  } while (0)

// ---------------------------------------------------------------------------
// Constants advertised to applications
// ---------------------------------------------------------------------------

// We expose a single unified heap backed by the accelerator's memory.
constexpr uint32_t kMemoryTypeDevice = 0;  // DEVICE_LOCAL only
constexpr uint32_t kMemoryTypeHost = 1;    // HOST_VISIBLE|HOST_COHERENT|DEVICE_LOCAL
constexpr uint32_t kMemoryTypeCount = 2;
constexpr uint32_t kHeapCount = 1;

constexpr VkDeviceSize kBufferAlignment = 256;
constexpr VkDeviceSize kMinUniformBufferOffsetAlignment = 256;
constexpr VkDeviceSize kMinStorageBufferOffsetAlignment = 16;

}  // namespace ppu

// ---------------------------------------------------------------------------
// Objects
// ---------------------------------------------------------------------------
struct Instance_T;
struct PhysicalDevice_T;
struct Device_T;
struct Queue_T;
struct CommandBuffer_T;

struct Instance_T {
  VK_LOADER_DATA loader_data;
  uint32_t api_version = VK_API_VERSION_1_0;
};

struct PhysicalDevice_T {
  VK_LOADER_DATA loader_data;
  Instance_T *instance = nullptr;
  // Index into the CUDA device list. -1 for the synthetic placeholder device
  // that exists only so the loader handshake can be exercised where no
  // accelerator is usable.
  int cu_index = -1;
  bool synthetic = false;

  char name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE] = {};
  uint8_t uuid[VK_UUID_SIZE] = {};
  size_t total_memory = 0;
  size_t free_memory = 0;
  uint32_t max_threads_per_block = 1024;
  uint32_t max_threads_per_mp = 0;
  uint32_t mp_count = 0;
  uint32_t clock_mhz = 0;
  uint32_t bus_width_bits = 0;
  uint64_t l2_cache_size = 0;
  uint32_t max_shared_memory = 0;
  int cc_major = 0;
  int cc_minor = 0;
};

// Written at construction and validated on every API entry. If it ever fails,
// something overwrote the object and we want to know immediately rather than
// crash inside the driver.
constexpr uint64_t kDeviceMagic = 0x5050554349444556ull;  // "PPUCIDEV"

struct Device_T {
  VK_LOADER_DATA loader_data;
  uint64_t magic = kDeviceMagic;
  PhysicalDevice_T *phys = nullptr;

  ppu::CUcontext ctx = nullptr;
  ppu::CUstream stream = nullptr;
  Queue_T *queue = nullptr;

  // Iteration mutex: the CUDA context is made current on whichever thread
  // calls into us, and all substream work is serialised through one stream.
  bool valid = false;
};

struct Queue_T {
  VK_LOADER_DATA loader_data;
  Device_T *device = nullptr;
  uint32_t family_index = 0;
  uint32_t queue_index = 0;
};

// Validate a VkDevice handle on entry. A mismatched magic means the caller
// passed a handle from another object type or the library. Returning nullptr
// makes the entry point fail with VK_ERROR_DEVICE_LOST instead of letting a
// bad pointer reach the CUDA driver, where it would crash the process.
inline Device_T *checked_device(VkDevice handle, const char *where) {
  auto *dev = reinterpret_cast<Device_T *>(handle);
  if (dev == nullptr) return nullptr;
  if (dev->magic != kDeviceMagic) {
    PPU_ERR("%s: VkDevice handle %p is not a PPU device", where, (void *)dev);
    return nullptr;
  }
  return dev;
}

// --- memory ---------------------------------------------------------------
enum class MemoryKind { Device, HostMapped };

constexpr uint64_t kMemoryMagic = 0x5050554d454d4f52ull;  // "PPUMEMOR"

struct DeviceMemory_T {
  uint64_t magic = kMemoryMagic;
  Device_T *device = nullptr;
  MemoryKind kind = MemoryKind::Device;
  ppu::CUdeviceptr device_ptr = 0;
  void *host_ptr = nullptr;  // non-null only for HostMapped
  VkDeviceSize size = 0;
  VkDeviceSize mapped_offset = 0;
  VkDeviceSize mapped_size = 0;
  uint32_t type_index = 0;
  bool mapped = false;
};

// --- buffers --------------------------------------------------------------
struct Buffer_T {
  Device_T *device = nullptr;
  VkDeviceSize size = 0;
  VkBufferUsageFlags usage = 0;
  VkBufferCreateFlags flags = 0;
  DeviceMemory_T *memory = nullptr;
  VkDeviceSize memory_offset = 0;
};

// --- shaders & pipelines --------------------------------------------------
enum class ModuleFormat { Spirv, Cubin };

struct ShaderModule_T {
  Device_T *device = nullptr;
  ModuleFormat format = ModuleFormat::Spirv;
  std::vector<uint8_t> code;
  std::string entry_hint;
};

// --- descriptors ----------------------------------------------------------
struct DescriptorBinding {
  uint32_t binding = 0;
  VkDescriptorType type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  uint32_t count = 1;
  VkShaderStageFlags stages = 0;
};

struct DescriptorSetLayout_T {
  Device_T *device = nullptr;
  std::vector<DescriptorBinding> bindings;  // sorted by binding number
};

struct PipelineLayout_T {
  Device_T *device = nullptr;
  std::vector<DescriptorSetLayout_T *> set_layouts;
  uint32_t push_constant_size = 0;  // max(offset + size) over all ranges
  uint32_t push_constant_stages = 0;
};

struct ComputePipeline_T {
  Device_T *device = nullptr;
  PipelineLayout_T *layout = nullptr;

  ppu::CUmodule module = nullptr;
  ppu::CUfunction function = nullptr;
  uint32_t local_size[3] = {1, 1, 1};

  // Serialised kernel argument order, resolved from the pipeline layout.
  // Each entry is a descriptor binding that contributes one 64-bit pointer.
  struct ArgSlot {
    uint32_t set;
    uint32_t binding;
    VkDescriptorType type;
  };
  std::vector<ArgSlot> arg_slots;
  bool has_push_constants = false;
};

struct DescriptorSet_T;

struct DescriptorPool_T {
  Device_T *device = nullptr;
  uint32_t max_sets = 0;
  std::vector<DescriptorSet_T *> live_sets;
};

struct Descriptor_T {
  VkDescriptorType type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  Buffer_T *buffer = nullptr;
  VkDeviceSize offset = 0;
  VkDeviceSize range = 0;
};

struct DescriptorSet_T {
  Device_T *device = nullptr;
  DescriptorPool_T *pool = nullptr;
  DescriptorSetLayout_T *layout = nullptr;
  bool freed = false;
  // Keyed by (binding, array element); flattened in the layout's binding order.
  std::vector<Descriptor_T> descriptors;
};

// --- synchronisation ------------------------------------------------------
struct Fence_T {
  Device_T *device = nullptr;
  ppu::CUevent event = nullptr;
  bool signaled = true;  // Vulkan fences start signaled unless flagged otherwise
};

struct Semaphore_T {
  Device_T *device = nullptr;
  ppu::CUevent event = nullptr;
  bool recorded = false;  // has an event record been enqueued since last wait
};

// --- commands -------------------------------------------------------------
struct RecordedOp {
  enum class Kind {
    Dispatch,
    DispatchIndirect,
    CopyBuffer,
    FillBuffer,
    UpdateBuffer,
    Barrier,
  };

  Kind kind = Kind::Barrier;

  // Dispatch / DispatchIndirect
  ComputePipeline_T *pipeline = nullptr;
  uint32_t group[3] = {1, 1, 1};
  uint64_t indirect_addr = 0;  // device address of the 3x uint32 group count
  std::vector<uint64_t> args;  // 64-bit kernel arguments, in order
  uint64_t push_constant_addr = 0;

  // CopyBuffer
  uint64_t src_addr = 0;
  uint64_t dst_addr = 0;
  VkDeviceSize size = 0;

  // FillBuffer
  uint64_t fill_addr = 0;
  uint32_t fill_value = 0;

  // UpdateBuffer
  uint64_t update_addr = 0;
  std::vector<uint8_t> update_data;
};

struct CommandBuffer_T {
  VK_LOADER_DATA loader_data;
  Device_T *device = nullptr;
  VkCommandPool pool = VK_NULL_HANDLE;

  bool recording = false;
  bool ever_recorded = false;
  std::vector<RecordedOp> ops;

  // Bound state, captured while recording.
  ComputePipeline_T *pipeline = nullptr;
  std::vector<DescriptorSet_T *> bound_sets;
  std::vector<uint32_t> dynamic_offsets;
  PipelineLayout_T *active_layout = nullptr;
  std::vector<uint8_t> push_constants;

  // Host-mapped scratch used for push constants and indirect group counts.
  ppu::CUdeviceptr scratch_dev = 0;
  void *scratch_host = nullptr;
  VkDeviceSize scratch_size = 0;
  VkDeviceSize scratch_used = 0;
};

struct CommandPool_T {
  Device_T *device = nullptr;
  uint32_t family_index = 0;
  VkCommandPoolCreateFlags flags = 0;
  std::vector<CommandBuffer_T *> buffers;
};

// Helpers shared across translation units.
namespace ppu {

// --- dispatch table -------------------------------------------------------
PPU_INTERNAL PFN_vkVoidFunction ppu_lookup_instance_proc(const char *name);
PPU_INTERNAL PFN_vkVoidFunction ppu_lookup_device_proc(const char *name);

PPU_INTERNAL void stamp(VK_LOADER_DATA *data);

// Make this device's CUDA context current on the calling thread.
PPU_INTERNAL bool bind_context(Device_T *device);

// Host-mapped scratch allocation (also directly kernel-accessible).
PPU_INTERNAL bool alloc_scratch(CommandBuffer_T *cb, VkDeviceSize bytes,
                                ppu::CUdeviceptr *dev, void **host);
PPU_INTERNAL void free_scratch(CommandBuffer_T *cb);

PPU_INTERNAL uint64_t buffer_address(const Buffer_T *buffer, VkDeviceSize offset);

// The SPIR-V -> PPU code generation boundary. The compiler workstream drops
// its implementation in here; the default returns CUDA_ERROR_NOT_SUPPORTED.
PPU_INTERNAL CUresult compile_spirv_to_ppu_module(const uint32_t *spirv, size_t word_count,
                                                  const char *entry_point,
                                                  ppu::CUmodule *out_module);

// Extract the LocalSize execution mode from a SPIR-V module. Returns false if
// the module does not declare one (the spec default is 1,1,1).
PPU_INTERNAL bool spirv_local_size(const uint32_t *spirv, size_t word_count, uint32_t out[3]);

}  // namespace ppu

#endif  // VK_PPU_ICD_COMMON_HPP
