// Numeric probe: dispatch any SSBO kernel through the ICD and print the raw
// outputs, so a lowering can be checked against a host reference outside the
// program.
//
// Shape of the kernel it drives (same as tests/shaders/simple_mul.comp):
//   binding 0 writeonly float out[]   binding 1 readonly float in[]
//   binding 2 readonly float in2[]    push_constant { uint total; uint p1; }
//
// Inputs come from files so the caller picks exact values (that is the point:
// the cos() bug only shows on runtime values, not on constants llc folds).
//
// Usage:
//   ppu_numprobe <kernel.spv> <in.bin> [in2.bin] [--p1 N] [--total N] [--print N]
//                [--local N] [--fp16-out]
// Prints "out[i] = <value>" for the first --print elements (default 8), one per
// line, then a "RAW ..." line with all of them so a script can diff.
// --local sets the kernel's LocalSize.x (dispatch = ceil(threads/local)).
// --fp16-out reads the output as packed half2 words, one thread per word.
#include <vulkan/vulkan.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>

namespace {

struct Api {
  PFN_vkGetInstanceProcAddr GetInstanceProcAddr = nullptr;
#define VKFN(name) PFN_##name name = nullptr;
  VKFN(vkEnumerateInstanceVersion)
  VKFN(vkCreateInstance)
  VKFN(vkDestroyInstance)
  VKFN(vkEnumeratePhysicalDevices)
  VKFN(vkGetPhysicalDeviceProperties)
  VKFN(vkGetPhysicalDeviceMemoryProperties)
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

#define RESOLVE(g, n)                                                          \
  do {                                                                         \
    api.n = reinterpret_cast<PFN_##n>(g(#n));                                  \
    if (!api.n) return false;                                                  \
  } while (0)

bool load(Api &api, VkInstance inst, void *lib) {
  auto gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
      dlsym(lib, "vkGetInstanceProcAddr"));
  if (!gipa) return false;
  api.GetInstanceProcAddr = gipa;
  // Instance-level entry points are queried with the instance handle.
  auto g = [&](const char *n) { return gipa(inst, n); };
  RESOLVE(g, vkCreateInstance);
  RESOLVE(g, vkDestroyInstance);
  RESOLVE(g, vkEnumerateInstanceVersion);
  RESOLVE(g, vkEnumeratePhysicalDevices);
  RESOLVE(g, vkGetPhysicalDeviceProperties);
  RESOLVE(g, vkGetPhysicalDeviceMemoryProperties);
  RESOLVE(g, vkCreateDevice);
  RESOLVE(g, vkDestroyDevice);
  RESOLVE(g, vkGetDeviceProcAddr);
  RESOLVE(g, vkGetDeviceQueue);
  RESOLVE(g, vkDeviceWaitIdle);
  return true;
}

bool load_device(Api &api, VkDevice dev) {
  auto gipa = api.vkGetDeviceProcAddr;
  auto g = [&](const char *n) { return gipa(dev, n); };
  RESOLVE(g, vkCreateBuffer);
  RESOLVE(g, vkDestroyBuffer);
  RESOLVE(g, vkGetBufferMemoryRequirements);
  RESOLVE(g, vkAllocateMemory);
  RESOLVE(g, vkFreeMemory);
  RESOLVE(g, vkBindBufferMemory);
  RESOLVE(g, vkMapMemory);
  RESOLVE(g, vkUnmapMemory);
  RESOLVE(g, vkCreateShaderModule);
  RESOLVE(g, vkDestroyShaderModule);
  RESOLVE(g, vkCreateDescriptorSetLayout);
  RESOLVE(g, vkDestroyDescriptorSetLayout);
  RESOLVE(g, vkCreateDescriptorPool);
  RESOLVE(g, vkDestroyDescriptorPool);
  RESOLVE(g, vkAllocateDescriptorSets);
  RESOLVE(g, vkUpdateDescriptorSets);
  RESOLVE(g, vkCreatePipelineLayout);
  RESOLVE(g, vkDestroyPipelineLayout);
  RESOLVE(g, vkCreateComputePipelines);
  RESOLVE(g, vkDestroyPipeline);
  RESOLVE(g, vkCreateCommandPool);
  RESOLVE(g, vkDestroyCommandPool);
  RESOLVE(g, vkAllocateCommandBuffers);
  RESOLVE(g, vkBeginCommandBuffer);
  RESOLVE(g, vkEndCommandBuffer);
  RESOLVE(g, vkCmdBindPipeline);
  RESOLVE(g, vkCmdBindDescriptorSets);
  RESOLVE(g, vkCmdPushConstants);
  RESOLVE(g, vkCmdDispatch);
  RESOLVE(g, vkQueueSubmit);
  RESOLVE(g, vkQueueWaitIdle);
  RESOLVE(g, vkCreateFence);
  RESOLVE(g, vkDestroyFence);
  RESOLVE(g, vkWaitForFences);
  (void)dev;
  return true;
}

// Decode element `i` of an fp16 buffer packed two-per-uint-word. Written by
// hand (not __fp16) so the probe builds on a host toolchain with no fp16
// support and so a NaN/inf still prints instead of trapping.
double half_at(const void *base, uint32_t i) {
  const uint32_t *w = static_cast<const uint32_t *>(base);
  const uint32_t bits = (w[i >> 1] >> ((i & 1) * 16)) & 0xFFFFu;
  const int sign = (bits >> 15) & 1;
  const int exp = (bits >> 10) & 0x1F;
  const int mant = bits & 0x3FF;
  double v;
  if (exp == 0) {
    v = mant * (1.0 / 16777216.0);  // subnormal: mant * 2^-24
  } else if (exp == 31) {
    v = mant ? std::nan("") : HUGE_VAL;
  } else {
    v = (1.0 + mant / 1024.0) * std::ldexp(1.0, exp - 15);
  }
  return sign ? -v : v;
}

std::vector<uint8_t> read_file(const char *p) {
  std::vector<uint8_t> out;
  std::FILE *f = std::fopen(p, "rb");
  if (!f) return out;
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

struct Buf {
  VkBuffer b = VK_NULL_HANDLE;
  VkDeviceMemory m = VK_NULL_HANDLE;
  void *p = nullptr;
  VkDeviceSize size = 0;
};

bool make_buf(Api &api, VkDevice d, const VkPhysicalDeviceMemoryProperties &mp,
              VkDeviceSize size, Buf *o) {
  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = size;
  bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  if (api.vkCreateBuffer(d, &bci, nullptr, &o->b) != VK_SUCCESS) return false;
  VkMemoryRequirements req{};
  api.vkGetBufferMemoryRequirements(d, o->b, &req);
  uint32_t t = UINT32_MAX;
  for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
    if ((req.memoryTypeBits & (1u << i)) &&
        (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
      t = i;
  if (t == UINT32_MAX) return false;
  VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.allocationSize = req.size;
  mai.memoryTypeIndex = t;
  if (api.vkAllocateMemory(d, &mai, nullptr, &o->m) != VK_SUCCESS) return false;
  if (api.vkBindBufferMemory(d, o->b, o->m, 0) != VK_SUCCESS) return false;
  if (api.vkMapMemory(d, o->m, 0, VK_WHOLE_SIZE, 0, &o->p) != VK_SUCCESS) return false;
  o->size = size;
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: %s <kernel.spv> <in.bin> [in2.bin] [--p1 N] [--total N] "
                 "[--print N] [--split N]\n",
                 argv[0]);
    return 2;
  }
  const std::vector<uint8_t> spv = read_file(argv[1]);
  const std::vector<uint8_t> inb = read_file(argv[2]);
  std::vector<uint8_t> in2b;
  uint32_t p1 = 7, total = 0, nprint = 8, split = 0, local = 64;
  bool out_fp16 = false;
  for (int i = 3; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--p1" && i + 1 < argc) p1 = static_cast<uint32_t>(std::strtoul(argv[++i], 0, 10));
    else if (a == "--total" && i + 1 < argc) total = static_cast<uint32_t>(std::strtoul(argv[++i], 0, 10));
    else if (a == "--print" && i + 1 < argc) nprint = static_cast<uint32_t>(std::strtoul(argv[++i], 0, 10));
    else if (a == "--split" && i + 1 < argc) split = static_cast<uint32_t>(std::strtoul(argv[++i], 0, 10));
    else if (a == "--local" && i + 1 < argc) local = static_cast<uint32_t>(std::strtoul(argv[++i], 0, 10));
    else if (a == "--fp16-out") out_fp16 = true;
    else if (a.size() > 1 && a[0] != '-') in2b = read_file(a.c_str());
  }
  if (total == 0) total = static_cast<uint32_t>(inb.size() / sizeof(float));
  if (in2b.empty()) in2b = inb;
  // fp16 output is packed two halves per uint word, so the buffer holds
  // ceil(total/2) words and a thread owns one word (not one element).
  const uint32_t out_words = out_fp16 ? (total + 1) / 2 : total;
  const uint32_t threads = out_fp16 ? out_words : total;

  void *lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!lib) lib = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
  if (!lib) {
    std::fprintf(stderr, "cannot dlopen libvulkan\n");
    return 77;
  }
  Api api;
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "ppu_numprobe";
  app.apiVersion = VK_API_VERSION_1_1;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  // Instance-level procs come straight from the loader.
  auto gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
      dlsym(lib, "vkGetInstanceProcAddr"));
  api.vkCreateInstance =
      reinterpret_cast<PFN_vkCreateInstance>(gipa(nullptr, "vkCreateInstance"));
  VkInstance inst = VK_NULL_HANDLE;
  if (api.vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) {
    std::fprintf(stderr, "vkCreateInstance failed\n");
    return 77;
  }
  if (!load(api, inst, lib)) return 1;

  uint32_t n = 0;
  api.vkEnumeratePhysicalDevices(inst, &n, nullptr);
  std::vector<VkPhysicalDevice> phys(n);
  api.vkEnumeratePhysicalDevices(inst, &n, phys.data());
  if (n == 0) return 77;
  VkPhysicalDeviceProperties props{};
  api.vkGetPhysicalDeviceProperties(phys[0], &props);
  std::printf("device: %s\n", props.deviceName);

  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = 0;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  VkDevice dev = VK_NULL_HANDLE;
  if (api.vkCreateDevice(phys[0], &dci, nullptr, &dev) != VK_SUCCESS) return 77;
  if (!load_device(api, dev)) return 1;
  VkPhysicalDeviceMemoryProperties mp{};
  api.vkGetPhysicalDeviceMemoryProperties(phys[0], &mp);
  VkQueue q = VK_NULL_HANDLE;
  api.vkGetDeviceQueue(dev, 0, 0, &q);

  VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize = spv.size();
  smci.pCode = reinterpret_cast<const uint32_t *>(spv.data());
  VkShaderModule mod = VK_NULL_HANDLE;
  if (api.vkCreateShaderModule(dev, &smci, nullptr, &mod) != VK_SUCCESS) {
    std::fprintf(stderr, "vkCreateShaderModule failed\n");
    return 1;
  }

  VkDescriptorSetLayoutBinding binds[3]{};
  for (int i = 0; i < 3; ++i) {
    binds[i].binding = static_cast<uint32_t>(i);
    binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binds[i].descriptorCount = 1;
    binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo dslci{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dslci.bindingCount = 3;
  dslci.pBindings = binds;
  VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
  api.vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl);

  VkPushConstantRange pcr{};
  pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcr.size = 8;
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &dsl;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pcr;
  VkPipelineLayout pl = VK_NULL_HANDLE;
  api.vkCreatePipelineLayout(dev, &plci, nullptr, &pl);

  VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpci.stage.module = mod;
  cpci.stage.pName = "main";
  cpci.layout = pl;
  VkPipeline pipe = VK_NULL_HANDLE;
  VkResult r = api.vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe);
  std::printf("pipeline -> %d\n", r);
  if (r != VK_SUCCESS) return 1;

  const VkDeviceSize bytes = static_cast<VkDeviceSize>(total) * sizeof(float);
  // The fp16 output is packed two-per-word, so it needs only ceil(total/2)
  // words; size it that way but never smaller than one word.
  const VkDeviceSize out_bytes =
      out_fp16 ? static_cast<VkDeviceSize>(out_words) * sizeof(float) : bytes;
  Buf ob, ab, bb;
  if (!make_buf(api, dev, mp, out_bytes, &ob) || !make_buf(api, dev, mp, bytes, &ab) ||
      !make_buf(api, dev, mp, bytes, &bb)) {
    std::fprintf(stderr, "buffer setup failed\n");
    return 1;
  }
  std::memcpy(ab.p, inb.data(), std::min<size_t>(bytes, inb.size()));
  std::memcpy(bb.p, in2b.data(), std::min<size_t>(bytes, in2b.size()));
  std::memset(ob.p, 0, static_cast<size_t>(out_bytes));

  VkDescriptorPoolSize psz{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
  VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpci.maxSets = 1;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &psz;
  VkDescriptorPool dp = VK_NULL_HANDLE;
  api.vkCreateDescriptorPool(dev, &dpci, nullptr, &dp);
  VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsai.descriptorPool = dp;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &dsl;
  VkDescriptorSet ds = VK_NULL_HANDLE;
  api.vkAllocateDescriptorSets(dev, &dsai, &ds);

  Buf *bufs[3] = {&ob, &ab, &bb};
  const VkDeviceSize ranges[3] = {out_bytes, bytes, bytes};
  VkDescriptorBufferInfo dbi[3]{};
  VkWriteDescriptorSet wr[3]{};
  for (int i = 0; i < 3; ++i) {
    dbi[i].buffer = bufs[i]->b;
    dbi[i].range = ranges[i];
    wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wr[i].dstSet = ds;
    wr[i].dstBinding = static_cast<uint32_t>(i);
    wr[i].descriptorCount = 1;
    wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wr[i].pBufferInfo = &dbi[i];
  }
  api.vkUpdateDescriptorSets(dev, 3, wr, 0, nullptr);

  VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpi.queueFamilyIndex = 0;
  VkCommandPool pool = VK_NULL_HANDLE;
  api.vkCreateCommandPool(dev, &cpi, nullptr, &pool);
  VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  api.vkAllocateCommandBuffers(dev, &cbai, &cmd);
  VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  api.vkBeginCommandBuffer(cmd, &cbbi);
  api.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
  api.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0,
                              nullptr);
  struct {
    uint32_t total, p1;
  } pc = {total, p1};
  api.vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
  // --local is the kernel's LocalSize.x. Callers that pass it get the real
  // workgroup geometry; the default matches the 64 the simple_mul test uses.
  uint32_t work = split ? split : (threads + local - 1) / local;
  api.vkCmdDispatch(cmd, work, 1, 1);
  api.vkEndCommandBuffer(cmd);

  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence = VK_NULL_HANDLE;
  api.vkCreateFence(dev, &fci, nullptr, &fence);
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  r = api.vkQueueSubmit(q, 1, &si, fence);
  if (r != VK_SUCCESS) {
    std::fprintf(stderr, "vkQueueSubmit -> %d\n", r);
    return 1;
  }
  api.vkWaitForFences(dev, 1, &fence, VK_TRUE, ~0ull);

  const float *in = static_cast<const float *>(ab.p);
  const float *out = static_cast<const float *>(ob.p);
  uint32_t shown = nprint < total ? nprint : total;
  for (uint32_t i = 0; i < shown; ++i) {
    // fp16 output: element i is lane (i&1) of word (i>>1). Printing both the
    // decoded element and the raw word makes a half-written buffer visible:
    // unwritten words read as 0.0 while the written half keeps its value.
    double o = out_fp16 ? half_at(ob.p, i) : static_cast<double>(out[i]);
    std::printf("out[%u] = %.6f  (in %.6f)\n", i, o, static_cast<double>(in[i]));
  }
  std::printf("RAW");
  for (uint32_t i = 0; i < shown; ++i)
    std::printf(" %.6f",
                out_fp16 ? half_at(ob.p, i) : static_cast<double>(out[i]));
  std::printf("\n");
  return 0;
}
