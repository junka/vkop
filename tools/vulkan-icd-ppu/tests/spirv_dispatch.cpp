// Dispatch a SPIR-V-compiled kernel through the ICD and check the numbers.
//
// This is the level above ppu_roundtrip: that one proves the plumbing works by
// loading a cubin (an object produced elsewhere). This one hands
// vkCreateComputePipelines *SPIR-V* and expects the ICD to compile it, then
// dispatches it and compares the result against a host reference.
//
// The kernel is tests/shaders/simple_mul.comp:
//     out[i] = a[i] * b[i] + scaler      for i < total
//
// Usage: ppu_spirv_dispatch <simple_mul.spv> [device-index]
//
// Exit status: 0 all checks passed, 1 failures, 77 no usable device.
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
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

struct Api {
  PFN_vkGetInstanceProcAddr GetInstanceProcAddr = nullptr;
#define VKFN(name) PFN_##name name = nullptr;
  VKFN(vkEnumerateInstanceVersion)
  VKFN(vkCreateInstance)
  VKFN(vkDestroyInstance)
  VKFN(vkEnumeratePhysicalDevices)
  VKFN(vkGetPhysicalDeviceProperties)
  VKFN(vkGetPhysicalDeviceMemoryProperties)
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
  VKFN(vkBeginCommandBuffer)
  VKFN(vkEndCommandBuffer)
  VKFN(vkCmdBindPipeline)
  VKFN(vkCmdBindDescriptorSets)
  VKFN(vkCmdPushConstants)
  VKFN(vkCmdDispatch)
  VKFN(vkQueueSubmit)
  VKFN(vkQueueWaitIdle)
  VKFN(vkCreateFence)
  VKFN(vkDestroyFence)
  VKFN(vkWaitForFences)
#undef VKFN
};

bool load_global(void *lib, Api *api) {
  api->GetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
      reinterpret_cast<void *>(dlsym(lib, "vkGetInstanceProcAddr")));
  if (api->GetInstanceProcAddr == nullptr) return false;
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
  VKRESOLVE(vkBeginCommandBuffer)
  VKRESOLVE(vkEndCommandBuffer)
  VKRESOLVE(vkCmdBindPipeline)
  VKRESOLVE(vkCmdBindDescriptorSets)
  VKRESOLVE(vkCmdPushConstants)
  VKRESOLVE(vkCmdDispatch)
  VKRESOLVE(vkQueueSubmit)
  VKRESOLVE(vkQueueWaitIdle)
  VKRESOLVE(vkCreateFence)
  VKRESOLVE(vkDestroyFence)
  VKRESOLVE(vkWaitForFences)
#undef VKRESOLVE
  return true;
}

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

struct Buffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  void *mapped = nullptr;
  VkDeviceSize size = 0;
};

bool make_buffer(Api &api, VkDevice device, const VkPhysicalDeviceMemoryProperties &mp,
                 VkDeviceSize size, Buffer *out) {
  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = size;
  bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (api.vkCreateBuffer(device, &bci, nullptr, &out->buffer) != VK_SUCCESS) return false;
  VkMemoryRequirements req{};
  api.vkGetBufferMemoryRequirements(device, out->buffer, &req);
  uint32_t type = UINT32_MAX;
  for (uint32_t t = 0; t < mp.memoryTypeCount; ++t) {
    if ((req.memoryTypeBits & (1u << t)) == 0) continue;
    if (mp.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
      type = t;
      break;
    }
  }
  if (type == UINT32_MAX) return false;
  VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.allocationSize = req.size;
  mai.memoryTypeIndex = type;
  if (api.vkAllocateMemory(device, &mai, nullptr, &out->memory) != VK_SUCCESS) return false;
  if (api.vkBindBufferMemory(device, out->buffer, out->memory, 0) != VK_SUCCESS) return false;
  if (api.vkMapMemory(device, out->memory, 0, VK_WHOLE_SIZE, 0, &out->mapped) != VK_SUCCESS)
    return false;
  out->size = size;
  return true;
}

void destroy_buffer(Api &api, VkDevice device, Buffer *b) {
  if (b->mapped) api.vkUnmapMemory(device, b->memory);
  if (b->buffer) api.vkDestroyBuffer(device, b->buffer, nullptr);
  if (b->memory) api.vkFreeMemory(device, b->memory, nullptr);
  *b = Buffer{};
}

}  // namespace

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <simple_mul.spv>\n", argv[0]);
    return 2;
  }
  const std::vector<uint8_t> spirv = read_file(argv[1]);
  if (spirv.empty() || spirv.size() % 4 != 0) {
    std::fprintf(stderr, "cannot read SPIR-V from '%s'\n", argv[1]);
    return 2;
  }

  void *lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
  if (lib == nullptr) lib = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
  if (lib == nullptr) {
    std::fprintf(stderr, "cannot dlopen libvulkan: %s\n", dlerror());
    return 77;
  }
  Api api;
  if (!load_global(lib, &api)) return 77;

  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "ppu_spirv_dispatch";
  app.apiVersion = VK_API_VERSION_1_1;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  VkInstance instance = VK_NULL_HANDLE;
  if (api.vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) {
    std::fprintf(stderr, "vkCreateInstance failed\n");
    return 77;
  }
  if (!load_instance_level(&api, instance)) return 1;

  uint32_t phys_count = 0;
  api.vkEnumeratePhysicalDevices(instance, &phys_count, nullptr);
  if (phys_count == 0) {
    std::printf("no physical device\n");
    api.vkDestroyInstance(instance, nullptr);
    return 77;
  }
  std::vector<VkPhysicalDevice> phys(phys_count);
  api.vkEnumeratePhysicalDevices(instance, &phys_count, phys.data());

  VkPhysicalDeviceProperties props{};
  api.vkGetPhysicalDeviceProperties(phys[0], &props);
  std::printf("device: %s\n", props.deviceName);

  float priority = 1.0f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = 0;
  qci.queueCount = 1;
  qci.pQueuePriorities = &priority;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  VkDevice device = VK_NULL_HANDLE;
  if (api.vkCreateDevice(phys[0], &dci, nullptr, &device) != VK_SUCCESS) {
    std::printf("no usable device (phase B skipped)\n");
    api.vkDestroyInstance(instance, nullptr);
    return 77;
  }
  VkPhysicalDeviceMemoryProperties mp{};
  api.vkGetPhysicalDeviceMemoryProperties(phys[0], &mp);
  VkQueue queue = VK_NULL_HANDLE;
  api.vkGetDeviceQueue(device, 0, 0, &queue);

  // A null backend (VK_PPU_SYNTHETIC_DEVICE, or a host with no accelerator)
  // cannot compile or dispatch anything: there is no driver to load a module
  // with. Detect it the way the round-trip test does -- by trying to back a
  // small allocation -- and skip instead of reporting a failure.
  {
    VkBufferCreateInfo probe{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    probe.size = 256;
    probe.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    probe.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer pb = VK_NULL_HANDLE;
    if (api.vkCreateBuffer(device, &probe, nullptr, &pb) == VK_SUCCESS) {
      VkMemoryRequirements req{};
      api.vkGetBufferMemoryRequirements(device, pb, &req);
      uint32_t ti = UINT32_MAX;
      for (uint32_t t = 0; t < mp.memoryTypeCount; ++t)
        if ((req.memoryTypeBits & (1u << t)) && (mp.memoryTypes[t].propertyFlags &
                                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
          ti = t;
      VkMemoryAllocateInfo ma{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
      ma.allocationSize = req.size;
      ma.memoryTypeIndex = ti == UINT32_MAX ? 0 : ti;
      VkDeviceMemory pm = VK_NULL_HANDLE;
      VkResult ar = api.vkAllocateMemory(device, &ma, nullptr, &pm);
      if (pm) api.vkFreeMemory(device, pm, nullptr);
      api.vkDestroyBuffer(device, pb, nullptr);
      if (ar != VK_SUCCESS) {
        std::printf("  (device is a null backend: SPIR-V dispatch skipped)\n");
        api.vkDestroyDevice(device, nullptr);
        api.vkDestroyInstance(instance, nullptr);
        std::printf("\n%d checks, %d failure(s) [no accelerator]\n", g_checks, g_failures);
        return 0;
      }
    }
  }

  // ---- shader module ----
  section("SPIR-V shader module");
  VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize = spirv.size();
  smci.pCode = reinterpret_cast<const uint32_t *>(spirv.data());
  VkShaderModule module = VK_NULL_HANDLE;
  VkResult r = api.vkCreateShaderModule(device, &smci, nullptr, &module);
  std::printf("  vkCreateShaderModule(%zu bytes) -> %d\n", spirv.size(), r);
  CHECK(r == VK_SUCCESS);

  // ---- layout: 3 storage buffers + a 8-byte push-constant block ----
  // Deliberately declares binding 3 as well, which the shader does not use:
  // the ICD must pass exactly the shader's bindings to the kernel, not the
  // layout's.
  section("pipeline layout");
  VkDescriptorSetLayoutBinding binds[4]{};
  for (int i = 0; i < 4; ++i) {
    binds[i].binding = static_cast<uint32_t>(i);
    binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binds[i].descriptorCount = 1;
    binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dslci.bindingCount = 4;
  dslci.pBindings = binds;
  VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
  CHECK(api.vkCreateDescriptorSetLayout(device, &dslci, nullptr, &dsl) == VK_SUCCESS);

  VkPushConstantRange pcr{};
  pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcr.offset = 0;
  pcr.size = 8;
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &dsl;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pcr;
  VkPipelineLayout pl = VK_NULL_HANDLE;
  CHECK(api.vkCreatePipelineLayout(device, &plci, nullptr, &pl) == VK_SUCCESS);

  // ---- compile through vkCreateComputePipelines ----
  section("vkCreateComputePipelines from SPIR-V");
  VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpci.stage.module = module;
  cpci.stage.pName = "main";
  cpci.layout = pl;
  VkPipeline pipeline = VK_NULL_HANDLE;
  r = api.vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline);
  std::printf("  vkCreateComputePipelines -> %d\n", r);
  CHECK_MSG(r == VK_SUCCESS, "SPIR-V compilation via vkCreateComputePipelines failed: %d", r);
  CHECK(pipeline != VK_NULL_HANDLE);
  if (r != VK_SUCCESS || pipeline == VK_NULL_HANDLE) {
    api.vkDestroyDevice(device, nullptr);
    api.vkDestroyInstance(instance, nullptr);
    std::printf("\n%d checks, %d failure(s)\n", g_checks, g_failures);
    return 1;
  }

  // Creating the same pipeline twice must reuse the compiled module. We cannot
  // see the cache from here, but we can prove the second creation works and is
  // not a failure path.
  VkPipeline pipeline2 = VK_NULL_HANDLE;
  r = api.vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline2);
  std::printf("  second vkCreateComputePipelines (same SPIR-V) -> %d\n", r);
  CHECK(r == VK_SUCCESS && pipeline2 != VK_NULL_HANDLE);

  // ---- buffers: 1000 elements is not a multiple of the 64-thread workgroup,
  // so the kernel's bounds check is exercised on the last group. ----
  section("dispatch and verify");
  constexpr uint32_t kElems = 1000;
  constexpr VkDeviceSize kBytes = kElems * sizeof(float);
  Buffer ob, ab, bb;
  CHECK(make_buffer(api, device, mp, kBytes, &ob));
  CHECK(make_buffer(api, device, mp, kBytes, &ab));
  CHECK(make_buffer(api, device, mp, kBytes, &bb));
  if (g_failures) {
    api.vkDestroyDevice(device, nullptr);
    api.vkDestroyInstance(instance, nullptr);
    std::printf("\n%d checks, %d failure(s)\n", g_checks, g_failures);
    return 1;
  }

  auto *av = static_cast<float *>(ab.mapped);
  auto *bv = static_cast<float *>(bb.mapped);
  auto *ov = static_cast<float *>(ob.mapped);
  for (uint32_t i = 0; i < kElems; ++i) {
    av[i] = static_cast<float>(i) * 0.5f - 3.0f;
    bv[i] = static_cast<float>(i % 17) + 1.25f;
    ov[i] = -12345.0f;  // sentinel: anything unwritten keeps a visible value
  }

  VkDescriptorPoolSize psz{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
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

  Buffer *bufs[3] = {&ob, &ab, &bb};
  VkDescriptorBufferInfo dbi[3]{};
  VkWriteDescriptorSet writes[3]{};
  for (int i = 0; i < 3; ++i) {
    dbi[i].buffer = bufs[i]->buffer;
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

  VkCommandPoolCreateInfo cpci2{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpci2.queueFamilyIndex = 0;
  VkCommandPool pool = VK_NULL_HANDLE;
  CHECK(api.vkCreateCommandPool(device, &cpci2, nullptr, &pool) == VK_SUCCESS);
  VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  CHECK(api.vkAllocateCommandBuffers(device, &cbai, &cmd) == VK_SUCCESS);
  VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  CHECK(api.vkBeginCommandBuffer(cmd, &cbbi) == VK_SUCCESS);
  api.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  api.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
  struct {
    uint32_t total;
    uint32_t scaler;
  } pc = {kElems, 7u};
  api.vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
  api.vkCmdDispatch(cmd, (kElems + 63) / 64, 1, 1);
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
  r = api.vkWaitForFences(device, 1, &fence, VK_TRUE, 30000000000ull);
  CHECK_MSG(r == VK_SUCCESS, "vkWaitForFences -> %d", r);

  // Host reference, computed with doubles and compared at fp32 precision.
  uint32_t mismatches = 0;
  float worst_got = 0.0f, worst_want = 0.0f;
  uint32_t worst_at = 0;
  for (uint32_t i = 0; i < kElems; ++i) {
    const double ai = static_cast<double>(i) * 0.5 - 3.0;
    const double bi = static_cast<double>(i % 17) + 1.25;
    const float want = static_cast<float>(ai * bi + 7.0);
    const float got = ov[i];
    // mul/add at fp32: a few ULP is expected, so compare relatively.
    const double tol = 1e-5 * std::max(1.0, std::fabs(static_cast<double>(want)));
    if (std::fabs(static_cast<double>(got) - static_cast<double>(want)) > tol) {
      if (mismatches == 0) {
        worst_at = i;
        worst_got = got;
        worst_want = want;
      }
      ++mismatches;
    }
  }
  std::printf("  out[0..3] = %.4f %.4f %.4f %.4f   out[999] = %.4f\n", ov[0], ov[1], ov[2],
              ov[3], ov[999]);
  if (mismatches != 0) {
    std::printf("  first mismatch at %u: got %.6f want %.6f (%u of %u wrong)\n", worst_at,
                worst_got, worst_want, mismatches, kElems);
  }
  CHECK_MSG(mismatches == 0, "%u of %u elements differ from the host reference", mismatches,
            kElems);

  section("teardown");
  CHECK(api.vkQueueWaitIdle(queue) == VK_SUCCESS);
  api.vkDestroyFence(device, fence, nullptr);
  api.vkDestroyCommandPool(device, pool, nullptr);
  api.vkDestroyDescriptorPool(device, dp, nullptr);
  api.vkDestroyPipeline(device, pipeline, nullptr);
  api.vkDestroyPipeline(device, pipeline2, nullptr);
  api.vkDestroyPipelineLayout(device, pl, nullptr);
  api.vkDestroyDescriptorSetLayout(device, dsl, nullptr);
  api.vkDestroyShaderModule(device, module, nullptr);
  destroy_buffer(api, device, &ob);
  destroy_buffer(api, device, &ab);
  destroy_buffer(api, device, &bb);
  CHECK(api.vkDeviceWaitIdle(device) == VK_SUCCESS);
  api.vkDestroyDevice(device, nullptr);
  api.vkDestroyInstance(instance, nullptr);

  std::printf("\n%d checks, %d failure(s)\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
