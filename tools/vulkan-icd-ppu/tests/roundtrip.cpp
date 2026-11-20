// End-to-end round-trip test for the PPU Vulkan ICD.
//
// Phase A (any host, no accelerator needed):
//   - loader handshake, instance, physical device enumeration
//   - the device must honestly advertise no image support
//
// Phase B (requires a real CUDA/PPU device; skipped cleanly otherwise):
//   - create device, buffer, allocate/bind memory, map, write, unmap
//   - verify the data survived via a second map
//   - run a vkCmdFillBuffer / vkCmdCopyBuffer round trip through the queue
//   - dispatch a cubin kernel (the escape hatch path) and check its output
#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <vector>

namespace {

int g_failures = 0;
int g_checks = 0;


#define CHECK(cond)                                                            \
  do {                                                                         \
    ++g_checks;                                                                \
    if (!(cond)) {                                                             \
      ++g_failures;                                                            \
      std::fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
    }                                                                          \
  } while (0)

#define CHECK_MSG(cond, ...)                                                   \
  do {                                                                         \
    ++g_checks;                                                                \
    if (!(cond)) {                                                             \
      ++g_failures;                                                            \
      std::fprintf(stderr, "  FAIL %s:%d: ", __FILE__, __LINE__);              \
      std::fprintf(stderr, __VA_ARGS__);                                       \
      std::fputc('\n', stderr);                                                \
    }                                                                          \
  } while (0)

void section(const char *name) { std::printf("\n== %s ==\n", name); }

// Minimal loader shim: we link against libvulkan.so via dlopen so the binary
// itself has no Vulkan link-time dependency (gong has libvulkan.so.1 but no
// libvulkan.so dev symlink).
struct Api {
  PFN_vkGetInstanceProcAddr GetInstanceProcAddr = nullptr;
#define VKFN(name) PFN_##name name = nullptr;
  VKFN(vkEnumerateInstanceVersion)
  VKFN(vkCreateInstance)
  VKFN(vkDestroyInstance)
  VKFN(vkEnumeratePhysicalDevices)
  VKFN(vkGetPhysicalDeviceProperties)
  VKFN(vkGetPhysicalDeviceMemoryProperties)
  VKFN(vkGetPhysicalDeviceFormatProperties)
  VKFN(vkGetPhysicalDeviceImageFormatProperties)
  VKFN(vkGetPhysicalDeviceQueueFamilyProperties)
  VKFN(vkCreateDevice)
  VKFN(vkDestroyDevice)
  VKFN(vkGetDeviceProcAddr)
  VKFN(vkGetDeviceQueue)
  VKFN(vkDeviceWaitIdle)
  VKFN(vkCreateBuffer)
  VKFN(vkDestroyBuffer)
  VKFN(vkGetBufferMemoryRequirements)
  VKFN(vkAllocateMemory)
  VKFN(vkFreeMemory)
  VKFN(vkBindBufferMemory)
  VKFN(vkMapMemory)
  VKFN(vkUnmapMemory)
  VKFN(vkCreateShaderModule)
  VKFN(vkDestroyShaderModule)
  VKFN(vkCreateDescriptorSetLayout)
  VKFN(vkDestroyDescriptorSetLayout)
  VKFN(vkCreateDescriptorPool)
  VKFN(vkDestroyDescriptorPool)
  VKFN(vkAllocateDescriptorSets)
  VKFN(vkUpdateDescriptorSets)
  VKFN(vkCreatePipelineLayout)
  VKFN(vkDestroyPipelineLayout)
  VKFN(vkCreateComputePipelines)
  VKFN(vkDestroyPipeline)
  VKFN(vkCreateCommandPool)
  VKFN(vkDestroyCommandPool)
  VKFN(vkAllocateCommandBuffers)
  VKFN(vkFreeCommandBuffers)
  VKFN(vkBeginCommandBuffer)
  VKFN(vkEndCommandBuffer)
  VKFN(vkResetCommandBuffer)
  VKFN(vkCmdBindPipeline)
  VKFN(vkCmdBindDescriptorSets)
  VKFN(vkCmdDispatch)
  VKFN(vkCmdCopyBuffer)
  VKFN(vkCmdFillBuffer)
  VKFN(vkCmdPipelineBarrier)
  VKFN(vkQueueSubmit)
  VKFN(vkQueueWaitIdle)
  VKFN(vkCreateFence)
  VKFN(vkDestroyFence)
  VKFN(vkWaitForFences)
  VKFN(vkResetFences)
  VKFN(vkGetBufferDeviceAddress)
  VKFN(vkGetFenceStatus)
#undef VKFN
};

// Resolve global entry points with a NULL instance, then the instance- and
// device-level ones once the instance exists. The loader only returns global
// functions for a NULL VkInstance.
bool load_global(void *lib, Api *api) {
  api->GetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
      reinterpret_cast<void *>(dlsym(lib, "vkGetInstanceProcAddr")));
  if (api->GetInstanceProcAddr == nullptr) {
    std::fprintf(stderr, "vkGetInstanceProcAddr not found in loader\n");
    return false;
  }
  api->vkEnumerateInstanceVersion = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
      api->GetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));
  api->vkCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(
      api->GetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance"));
  return api->vkEnumerateInstanceVersion != nullptr && api->vkCreateInstance != nullptr;
}

bool load_instance_level(Api *api, VkInstance instance) {
#define VKRESOLVE(name)                                                        \
  api->name = reinterpret_cast<PFN_##name>(api->GetInstanceProcAddr(instance, #name)); \
  if (api->name == nullptr) {                                                  \
    std::fprintf(stderr, "could not resolve %s\n", #name);                     \
    return false;                                                              \
  }
  VKRESOLVE(vkDestroyInstance)
  VKRESOLVE(vkEnumeratePhysicalDevices)
  VKRESOLVE(vkGetPhysicalDeviceProperties)
  VKRESOLVE(vkGetPhysicalDeviceMemoryProperties)
  VKRESOLVE(vkGetPhysicalDeviceFormatProperties)
  VKRESOLVE(vkGetPhysicalDeviceImageFormatProperties)
  VKRESOLVE(vkGetPhysicalDeviceQueueFamilyProperties)
  VKRESOLVE(vkCreateDevice)
  VKRESOLVE(vkDestroyDevice)
  VKRESOLVE(vkGetDeviceProcAddr)
  VKRESOLVE(vkGetDeviceQueue)
  VKRESOLVE(vkDeviceWaitIdle)
  VKRESOLVE(vkCreateBuffer)
  VKRESOLVE(vkDestroyBuffer)
  VKRESOLVE(vkGetBufferMemoryRequirements)
  VKRESOLVE(vkAllocateMemory)
  VKRESOLVE(vkFreeMemory)
  VKRESOLVE(vkBindBufferMemory)
  VKRESOLVE(vkMapMemory)
  VKRESOLVE(vkUnmapMemory)
  VKRESOLVE(vkCreateShaderModule)
  VKRESOLVE(vkDestroyShaderModule)
  VKRESOLVE(vkCreateDescriptorSetLayout)
  VKRESOLVE(vkDestroyDescriptorSetLayout)
  VKRESOLVE(vkCreateDescriptorPool)
  VKRESOLVE(vkDestroyDescriptorPool)
  VKRESOLVE(vkAllocateDescriptorSets)
  VKRESOLVE(vkUpdateDescriptorSets)
  VKRESOLVE(vkCreatePipelineLayout)
  VKRESOLVE(vkDestroyPipelineLayout)
  VKRESOLVE(vkCreateComputePipelines)
  VKRESOLVE(vkDestroyPipeline)
  VKRESOLVE(vkCreateCommandPool)
  VKRESOLVE(vkDestroyCommandPool)
  VKRESOLVE(vkAllocateCommandBuffers)
  VKRESOLVE(vkFreeCommandBuffers)
  VKRESOLVE(vkBeginCommandBuffer)
  VKRESOLVE(vkEndCommandBuffer)
  VKRESOLVE(vkResetCommandBuffer)
  VKRESOLVE(vkCmdBindPipeline)
  VKRESOLVE(vkCmdBindDescriptorSets)
  VKRESOLVE(vkCmdDispatch)
  VKRESOLVE(vkCmdCopyBuffer)
  VKRESOLVE(vkCmdFillBuffer)
  VKRESOLVE(vkCmdPipelineBarrier)
  VKRESOLVE(vkQueueSubmit)
  VKRESOLVE(vkQueueWaitIdle)
  VKRESOLVE(vkCreateFence)
  VKRESOLVE(vkDestroyFence)
  VKRESOLVE(vkWaitForFences)
  VKRESOLVE(vkResetFences)
  VKRESOLVE(vkGetBufferDeviceAddress)
  VKRESOLVE(vkGetFenceStatus)
#undef VKRESOLVE
  return true;
}

// Read a cubin if one was handed to us (argv[1]).
std::vector<uint8_t> read_file(const char *path) {
  std::vector<uint8_t> out;
  std::FILE *f = std::fopen(path, "rb");
  if (f == nullptr) return out;
  std::fseek(f, 0, SEEK_END);
  long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (n > 0) {
    out.resize(static_cast<size_t>(n));
    if (std::fread(out.data(), 1, out.size(), f) != out.size()) out.clear();
  }
  std::fclose(f);
  return out;
}

}  // namespace

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);  // unbuffered so a crash keeps the log
  const char *cubin_path = argc > 1 ? argv[1] : nullptr;

  void *lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
  if (lib == nullptr) lib = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
  if (lib == nullptr) {
    std::fprintf(stderr, "cannot dlopen libvulkan: %s\n", dlerror());
    return 77;  // skip
  }
  Api api;
  if (!load_global(lib, &api)) return 77;

  section("loader handshake");
  uint32_t loader_version = 0;
  VkResult r = api.vkEnumerateInstanceVersion(&loader_version);
  std::printf("  vkEnumerateInstanceVersion -> %d, version %u.%u.%u\n", r,
              VK_API_VERSION_MAJOR(loader_version), VK_API_VERSION_MINOR(loader_version),
              VK_API_VERSION_PATCH(loader_version));
  CHECK(r == VK_SUCCESS);

  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "ppu_roundtrip";
  app.apiVersion = VK_API_VERSION_1_1;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;

  VkInstance instance = VK_NULL_HANDLE;
  r = api.vkCreateInstance(&ici, nullptr, &instance);
  std::printf("  vkCreateInstance -> %d\n", r);
  CHECK_MSG(r == VK_SUCCESS, "vkCreateInstance returned %d", r);
  if (r != VK_SUCCESS) return 1;

  if (!load_instance_level(&api, instance)) {
    std::fprintf(stderr, "loader could not resolve instance-level entry points\n");
    return 1;
  }

  section("physical devices");
  uint32_t count = 0;
  r = api.vkEnumeratePhysicalDevices(instance, &count, nullptr);
  std::printf("  vkEnumeratePhysicalDevices count -> %d (result %d)\n", count, r);
  CHECK(r == VK_SUCCESS || r == VK_INCOMPLETE);
  CHECK(count >= 1);
  if (count == 0) {
    api.vkDestroyInstance(instance, nullptr);
    return 1;
  }
  std::vector<VkPhysicalDevice> phys(count);
  api.vkEnumeratePhysicalDevices(instance, &count, phys.data());

  VkPhysicalDeviceProperties props{};
  api.vkGetPhysicalDeviceProperties(phys[0], &props);
  std::printf("  device: '%s' type=%d api=%u.%u.%u vendor=0x%04x device=0x%04x\n",
              props.deviceName, static_cast<int>(props.deviceType),
              VK_API_VERSION_MAJOR(props.apiVersion), VK_API_VERSION_MINOR(props.apiVersion),
              VK_API_VERSION_PATCH(props.apiVersion), props.vendorID, props.deviceID);
  std::printf("  maxComputeWorkGroupInvocations=%u  maxComputeSharedMemorySize=%u\n",
              props.limits.maxComputeWorkGroupInvocations,
              props.limits.maxComputeSharedMemorySize);
  CHECK(props.limits.maxComputeWorkGroupInvocations >= 1);

  section("honest 'no images' capability");
  VkFormatProperties fp{};
  api.vkGetPhysicalDeviceFormatProperties(phys[0], VK_FORMAT_R8G8B8A8_UNORM, &fp);
  std::printf("  R8G8B8A8_UNORM linear=0x%x optimal=0x%x buffer=0x%x\n",
              fp.linearTilingFeatures, fp.optimalTilingFeatures, fp.bufferFeatures);
  CHECK(fp.linearTilingFeatures == 0);
  CHECK(fp.optimalTilingFeatures == 0);
  CHECK(fp.bufferFeatures == 0);

  VkImageFormatProperties ifp{};
  r = api.vkGetPhysicalDeviceImageFormatProperties(
      phys[0], VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
      VK_IMAGE_USAGE_STORAGE_BIT, 0, &ifp);
  std::printf("  vkGetPhysicalDeviceImageFormatProperties -> %d (expect -11 = FORMAT_NOT_SUPPORTED)\n", r);
  CHECK(r == VK_ERROR_FORMAT_NOT_SUPPORTED);

  section("queue families");
  uint32_t qf_count = 0;
  api.vkGetPhysicalDeviceQueueFamilyProperties(phys[0], &qf_count, nullptr);
  std::printf("  queue family count = %u\n", qf_count);
  CHECK(qf_count == 1);
  std::vector<VkQueueFamilyProperties> qf(qf_count);
  api.vkGetPhysicalDeviceQueueFamilyProperties(phys[0], &qf_count, qf.data());
  if (qf_count >= 1) {
    std::printf("  family 0: flags=0x%x (compute=%d transfer=%d) count=%u\n", qf[0].queueFlags,
                (qf[0].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0,
                (qf[0].queueFlags & VK_QUEUE_TRANSFER_BIT) != 0, qf[0].queueCount);
    CHECK((qf[0].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0);
    CHECK((qf[0].queueFlags & VK_QUEUE_TRANSFER_BIT) != 0);
    CHECK((qf[0].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0);
  }

  VkPhysicalDeviceMemoryProperties mp{};
  api.vkGetPhysicalDeviceMemoryProperties(phys[0], &mp);
  std::printf("  memory types = %u, heaps = %u\n", mp.memoryTypeCount, mp.memoryHeapCount);
  for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
    std::printf("    type %u: flags=0x%x heap=%u\n", i, mp.memoryTypes[i].propertyFlags,
                mp.memoryTypes[i].heapIndex);
  CHECK(mp.memoryTypeCount >= 1);

  section("device creation");
  float priority = 1.0f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = 0;
  qci.queueCount = 1;
  qci.pQueuePriorities = &priority;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;

  VkDevice device = VK_NULL_HANDLE;
  r = api.vkCreateDevice(phys[0], &dci, nullptr, &device);
  std::printf("  vkCreateDevice -> %d\n", r);
  if (r != VK_SUCCESS) {
    std::printf("  (no usable accelerator here: phase B skipped)\n");
    api.vkDestroyInstance(instance, nullptr);
    std::printf("\n%d checks, %d failure(s) [phase A only]\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
  }

  VkQueue queue = VK_NULL_HANDLE;
  api.vkGetDeviceQueue(device, 0, 0, &queue);
  CHECK(queue != VK_NULL_HANDLE);

  // ---- Phase B: buffer + memory round trip ------------------------------
  section("buffer memory round trip");
  constexpr VkDeviceSize kSize = 4096;
  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = kSize;
  bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
              VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkBuffer buffer = VK_NULL_HANDLE;
  r = api.vkCreateBuffer(device, &bci, nullptr, &buffer);
  CHECK_MSG(r == VK_SUCCESS, "vkCreateBuffer -> %d", r);

  VkMemoryRequirements req{};
  api.vkGetBufferMemoryRequirements(device, buffer, &req);
  std::printf("  buffer requirements: size=%llu align=%llu types=0x%x\n",
              (unsigned long long)req.size, (unsigned long long)req.alignment,
              req.memoryTypeBits);

  // Pick the HOST_VISIBLE type if available so we can map it.
  uint32_t type_index = 0;
  bool found_host = false;
  for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
    if ((req.memoryTypeBits & (1u << i)) == 0) continue;
    if (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
      type_index = i;
      found_host = true;
      break;
    }
  }
  CHECK_MSG(found_host, "no host-visible memory type accepted the buffer");

  VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.allocationSize = req.size;
  mai.memoryTypeIndex = type_index;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  r = api.vkAllocateMemory(device, &mai, nullptr, &memory);
  std::printf("  vkAllocateMemory(type %u) -> %d\n", type_index, r);
  if (r == VK_ERROR_FEATURE_NOT_PRESENT) {
    // The synthetic null backend (VK_PPU_SYNTHETIC_DEVICE) creates a VkDevice
    // but refuses to back memory. That is phase A only by construction.
    std::printf("  (device is a null backend: phase B skipped)\n");
    api.vkDestroyDevice(device, nullptr);
    api.vkDestroyInstance(instance, nullptr);
    std::printf("\n%d checks, %d failure(s) [phase A only]\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
  }
  CHECK_MSG(r == VK_SUCCESS, "vkAllocateMemory -> %d", r);

  r = api.vkBindBufferMemory(device, buffer, memory, 0);
  CHECK_MSG(r == VK_SUCCESS, "vkBindBufferMemory -> %d", r);

  // Map, write a pattern, unmap.
  static const char kPattern[] = "PPU-ICD round trip payload";
  void *mapped = nullptr;
  r = api.vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &mapped);
  std::printf("  vkMapMemory -> %d (ptr %p)\n", r, mapped);
  CHECK_MSG(r == VK_SUCCESS, "vkMapMemory -> %d", r);
  if (r == VK_SUCCESS) {
    std::memcpy(mapped, kPattern, sizeof kPattern);
    api.vkUnmapMemory(device, memory);
  }

  // Map again and confirm it survived the unmap.
  void *mapped2 = nullptr;
  r = api.vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &mapped2);
  CHECK(r == VK_SUCCESS);
  if (r == VK_SUCCESS) {
    bool same = std::memcmp(mapped2, kPattern, sizeof kPattern) == 0;
    std::printf("  re-read after unmap: '%s' (%s)\n", static_cast<char *>(mapped2),
                same ? "match" : "MISMATCH");
    CHECK(same);
    api.vkUnmapMemory(device, memory);
  }

  // ---- Phase B: submit path (fill + copy) -------------------------------
  section("recorded fill + copy through vkQueueSubmit");
  VkBufferCreateInfo dst_bci = bci;
  VkBuffer dst_buffer = VK_NULL_HANDLE;
  api.vkCreateBuffer(device, &dst_bci, nullptr, &dst_buffer);
  VkMemoryRequirements dst_req{};
  api.vkGetBufferMemoryRequirements(device, dst_buffer, &dst_req);
  VkMemoryAllocateInfo dst_mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  dst_mai.allocationSize = dst_req.size;
  dst_mai.memoryTypeIndex = type_index;
  VkDeviceMemory dst_memory = VK_NULL_HANDLE;
  r = api.vkAllocateMemory(device, &dst_mai, nullptr, &dst_memory);
  CHECK(r == VK_SUCCESS);
  CHECK(api.vkBindBufferMemory(device, dst_buffer, dst_memory, 0) == VK_SUCCESS);

  VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpci.queueFamilyIndex = 0;
  VkCommandPool pool = VK_NULL_HANDLE;
  CHECK(api.vkCreateCommandPool(device, &cpci, nullptr, &pool) == VK_SUCCESS);

  VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  CHECK(api.vkAllocateCommandBuffers(device, &cbai, &cmd) == VK_SUCCESS);

  VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  CHECK(api.vkBeginCommandBuffer(cmd, &cbbi) == VK_SUCCESS);
  api.vkCmdFillBuffer(cmd, dst_buffer, 0, 64, 0xA5A5A5A5u);
  VkBufferCopy region{};
  region.size = 32;
  api.vkCmdCopyBuffer(cmd, buffer, dst_buffer, 1, &region);
  api.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 0, nullptr, 0, nullptr);
  CHECK(api.vkEndCommandBuffer(cmd) == VK_SUCCESS);

  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence = VK_NULL_HANDLE;
  CHECK(api.vkCreateFence(device, &fci, nullptr, &fence) == VK_SUCCESS);

  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  r = api.vkQueueSubmit(queue, 1, &si, fence);
  std::printf("  vkQueueSubmit -> %d\n", r);
  CHECK_MSG(r == VK_SUCCESS, "vkQueueSubmit -> %d", r);

  r = api.vkWaitForFences(device, 1, &fence, VK_TRUE, 5000000000ull);
  std::printf("  vkWaitForFences -> %d\n", r);
  CHECK(r == VK_SUCCESS);

  void *dst_view = nullptr;
  r = api.vkMapMemory(device, dst_memory, 0, VK_WHOLE_SIZE, 0, &dst_view);
  CHECK(r == VK_SUCCESS);
  if (r == VK_SUCCESS) {
    const auto *words = static_cast<const uint32_t *>(dst_view);
    std::printf("  dst[0..3]=%08x %08x %08x %08x\n", words[0], words[1], words[2], words[3]);
    // Recorded order was: fill 64 bytes, then copy 32 bytes over the front.
    // So words 0..7 hold the copied source pattern and words 8..15 keep the
    // fill value, which also proves the two ops replayed in order.
    CHECK(words[8] == 0xA5A5A5A5u);
    CHECK(words[15] == 0xA5A5A5A5u);
    const char *copied = static_cast<const char *>(dst_view);
    CHECK(std::memcmp(copied, kPattern, 16) == 0);
    api.vkUnmapMemory(device, dst_memory);
  }

  // ---- Phase B: dispatch a cubin (SPIR-V compiler escape hatch) ----------
  if (cubin_path != nullptr) {
    section("compute dispatch via cubin");
    std::vector<uint8_t> cubin = read_file(cubin_path);
    std::printf("  cubin '%s': %zu bytes\n", cubin_path, cubin.size());
    if (!cubin.empty()) {
      VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
      smci.codeSize = cubin.size();
      smci.pCode = reinterpret_cast<const uint32_t *>(cubin.data());
      VkShaderModule module = VK_NULL_HANDLE;
      r = api.vkCreateShaderModule(device, &smci, nullptr, &module);
      std::printf("  vkCreateShaderModule -> %d\n", r);
      CHECK(r == VK_SUCCESS);

      VkDescriptorSetLayoutBinding binds[3]{};
      for (int i = 0; i < 3; ++i) {
        binds[i].binding = static_cast<uint32_t>(i);
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      }
      VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
      dslci.bindingCount = 3;
      dslci.pBindings = binds;
      VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
      CHECK(api.vkCreateDescriptorSetLayout(device, &dslci, nullptr, &dsl) == VK_SUCCESS);

      VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
      plci.setLayoutCount = 1;
      plci.pSetLayouts = &dsl;
      VkPipelineLayout pl = VK_NULL_HANDLE;
      CHECK(api.vkCreatePipelineLayout(device, &plci, nullptr, &pl) == VK_SUCCESS);

      VkComputePipelineCreateInfo cpci2{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
      cpci2.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      cpci2.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
      cpci2.stage.module = module;
      cpci2.stage.pName = "vecadd";
      cpci2.layout = pl;
      VkPipeline pipeline = VK_NULL_HANDLE;
      r = api.vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci2, nullptr, &pipeline);
      std::printf("  vkCreateComputePipelines -> %d\n", r);
      CHECK(r == VK_SUCCESS);

      // Three storage buffers of 1024 floats each. This matches the fixed
      // 1024-thread guard in the test cubin; with the documented 256-thread
      // cubin block size that is exactly four workgroups.
      constexpr uint32_t kElems = 1024;
      constexpr VkDeviceSize kBytes = kElems * sizeof(float);
      VkBuffer bufs[3] = {};
      VkDeviceMemory mems[3] = {};
      for (int i = 0; i < 3; ++i) {
        VkBufferCreateInfo b{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        b.size = kBytes;
        b.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        b.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        api.vkCreateBuffer(device, &b, nullptr, &bufs[i]);
        VkMemoryRequirements mr{};
        api.vkGetBufferMemoryRequirements(device, bufs[i], &mr);
        VkMemoryAllocateInfo ma{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ma.allocationSize = mr.size;
        // Pick a host-visible type: we fill these buffers from the CPU and read
        // the result back, so device-only memory would be unmappable.
        uint32_t ti = UINT32_MAX;
        for (uint32_t t = 0; t < mp.memoryTypeCount; ++t)
          if ((mr.memoryTypeBits & (1u << t)) &&
              (mp.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
            ti = t;
            break;
          }
        CHECK_MSG(ti != UINT32_MAX, "no host-visible memory type for compute buffer");
        ma.memoryTypeIndex = ti;
        r = api.vkAllocateMemory(device, &ma, nullptr, &mems[i]);
        CHECK_MSG(r == VK_SUCCESS, "compute memory allocation -> %d", r);
        if (r != VK_SUCCESS) continue;
        CHECK(api.vkBindBufferMemory(device, bufs[i], mems[i], 0) == VK_SUCCESS);
      }
      for (int i = 0; i < 2; ++i) {
        void *p = nullptr;
        r = api.vkMapMemory(device, mems[i], 0, VK_WHOLE_SIZE, 0, &p);
        CHECK_MSG(r == VK_SUCCESS && p != nullptr, "map compute input %d -> %d (%p)", i, r, p);
        if (r != VK_SUCCESS || p == nullptr) continue;
        auto *f = static_cast<float *>(p);
        for (uint32_t e = 0; e < kElems; ++e) f[e] = static_cast<float>(e) * (i + 1);
        api.vkUnmapMemory(device, mems[i]);
      }

      VkDescriptorPoolSize psz{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
      VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
      dpci.maxSets = 1;
      dpci.poolSizeCount = 1;
      dpci.pPoolSizes = &psz;
      VkDescriptorPool dp = VK_NULL_HANDLE;
      CHECK(api.vkCreateDescriptorPool(device, &dpci, nullptr, &dp) == VK_SUCCESS);
      VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
      dsai.descriptorPool = dp;
      dsai.descriptorSetCount = 1;
      dsai.pSetLayouts = &dsl;
      VkDescriptorSet ds = VK_NULL_HANDLE;
      CHECK(api.vkAllocateDescriptorSets(device, &dsai, &ds) == VK_SUCCESS);

      VkDescriptorBufferInfo dbi[3]{};
      VkWriteDescriptorSet writes[3]{};
      for (int i = 0; i < 3; ++i) {
        dbi[i].buffer = bufs[i];
        dbi[i].offset = 0;
        dbi[i].range = kBytes;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = ds;
        writes[i].dstBinding = static_cast<uint32_t>(i);
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &dbi[i];
      }
      api.vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);

      CHECK(api.vkResetCommandBuffer(cmd, 0) == VK_SUCCESS);
      CHECK(api.vkBeginCommandBuffer(cmd, &cbbi) == VK_SUCCESS);
      api.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
      api.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
      api.vkCmdDispatch(cmd, kElems / 256, 1, 1);
      CHECK(api.vkEndCommandBuffer(cmd) == VK_SUCCESS);

      api.vkResetFences(device, 1, &fence);
      r = api.vkQueueSubmit(queue, 1, &si, fence);
      CHECK(r == VK_SUCCESS);
      r = api.vkWaitForFences(device, 1, &fence, VK_TRUE, 5000000000ull);
      CHECK(r == VK_SUCCESS);

      void *out = nullptr;
      r = api.vkMapMemory(device, mems[2], 0, VK_WHOLE_SIZE, 0, &out);
      CHECK_MSG(r == VK_SUCCESS && out != nullptr, "map compute output -> %d (%p)", r, out);
      if (r != VK_SUCCESS || out == nullptr) {
        api.vkDestroyDevice(device, nullptr);
        return 1;
      }
      const auto *c = static_cast<const float *>(out);
      bool ok = true;
      for (uint32_t e = 0; e < kElems; ++e) {
        float expect = static_cast<float>(e) + static_cast<float>(e) * 2.0f;
        if (c[e] != expect) {
          std::printf("  MISMATCH at %u: got %f want %f\n", e, c[e], expect);
          ok = false;
          break;
        }
      }
      std::printf("  kernel result c[0..3]=%f %f %f %f (c[1023]=%f) -> %s\n", c[0], c[1], c[2], c[3], c[kElems-1],
                  ok ? "MATCH" : "MISMATCH");
      CHECK(ok);
      api.vkUnmapMemory(device, mems[2]);

      for (int i = 0; i < 3; ++i) {
        api.vkFreeMemory(device, mems[i], nullptr);
        api.vkDestroyBuffer(device, bufs[i], nullptr);
      }
      api.vkDestroyDescriptorPool(device, dp, nullptr);
      api.vkDestroyPipeline(device, pipeline, nullptr);
      api.vkDestroyPipelineLayout(device, pl, nullptr);
      api.vkDestroyDescriptorSetLayout(device, dsl, nullptr);
      api.vkDestroyShaderModule(device, module, nullptr);
    }
  }

  section("teardown");
  api.vkDestroyFence(device, fence, nullptr);
  api.vkDestroyBuffer(device, dst_buffer, nullptr);
  api.vkFreeMemory(device, dst_memory, nullptr);
  api.vkDestroyBuffer(device, buffer, nullptr);
  api.vkFreeMemory(device, memory, nullptr);
  api.vkDestroyCommandPool(device, pool, nullptr);
  CHECK(api.vkDeviceWaitIdle(device) == VK_SUCCESS);
  api.vkDestroyDevice(device, nullptr);
  api.vkDestroyInstance(instance, nullptr);

  std::printf("\n%d checks, %d failure(s)\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
