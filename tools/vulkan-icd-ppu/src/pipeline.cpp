// Descriptor sets, pipeline layouts, shader modules and compute pipelines.
#include "icd_common.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace {

using ppu::CUDA_SUCCESS;

Device_T *as_device(VkDevice device) { return checked_device(device, __func__); }

VkResult device_lost(const char *what, ppu::CUresult rc) {
  PPU_ERR("%s failed: %s", what, ppu::cu_strerror(rc));
  return VK_ERROR_DEVICE_LOST;
}

// SPIR-V module magic, little-endian.
constexpr uint32_t kSpirvMagic = 0x07230203u;
// ELF magic, for cubins produced by nvcc/hgcc.
constexpr uint32_t kElfMagic = 0x464C457Fu;

}  // namespace

// ---------------------------------------------------------------------------
// SPIR-V local size extraction
// ---------------------------------------------------------------------------
namespace ppu {

// Walks the instruction stream looking for OpEntryPoint (15) and OpExecutionMode
// (16) with LocalSize (17). Only the first entry point is considered.
bool spirv_local_size(const uint32_t *spirv, size_t word_count, uint32_t out[3]) {
  out[0] = out[1] = out[2] = 1;
  if (spirv == nullptr || word_count < 5) return false;
  if (spirv[0] != kSpirvMagic) return false;

  uint32_t entry_id = 0;
  for (size_t i = 5; i < word_count;) {
    uint32_t instruction = spirv[i];
    uint16_t opcode = instruction & 0xFFFF;
    uint16_t length = (instruction >> 16) & 0xFFFF;
    if (length == 0 || i + length > word_count) break;
    if (opcode == 15 /* OpEntryPoint */ && length >= 4) {
      entry_id = spirv[i + 2];
    }
    if (opcode == 16 /* OpExecutionMode */ && length >= 6 && spirv[i + 1] == entry_id &&
        spirv[i + 2] == 17 /* LocalSize */) {
      out[0] = spirv[i + 3];
      out[1] = spirv[i + 4];
      out[2] = spirv[i + 5];
      return true;
    }
    i += length;
  }
  return false;
}

// Default implementation. The SPIR-V -> PPU compiler workstream replaces this.
CUresult compile_spirv_to_ppu_module(const uint32_t *spirv, size_t word_count,
                                     const char *entry_point, ppu::CUmodule *out_module) {
  (void)spirv;
  (void)word_count;
  (void)entry_point;
  (void)out_module;
  return CUDA_ERROR_NOT_SUPPORTED;
}

}  // namespace ppu

// ---------------------------------------------------------------------------
// Descriptor set layouts
// ---------------------------------------------------------------------------
extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkCreateDescriptorSetLayout(
    VkDevice device, const VkDescriptorSetLayoutCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkDescriptorSetLayout *pSetLayout) {
  (void)pAllocator;
  Device_T *dev = as_device(device);
  if (dev == nullptr || pCreateInfo == nullptr || pSetLayout == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  auto *layout = new DescriptorSetLayout_T();
  layout->device = dev;
  for (uint32_t i = 0; i < pCreateInfo->bindingCount; ++i) {
    const VkDescriptorSetLayoutBinding &b = pCreateInfo->pBindings[i];
    if (b.descriptorType != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER &&
        b.descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
      // Images and samplers are out of scope for this device.
      PPU_ERR("vkCreateDescriptorSetLayout: descriptor type %d is not supported",
              static_cast<int>(b.descriptorType));
      delete layout;
      return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    DescriptorBinding db;
    db.binding = b.binding;
    db.type = b.descriptorType;
    db.count = b.descriptorCount ? b.descriptorCount : 1;
    db.stages = b.stageFlags;
    layout->bindings.push_back(db);
  }
  std::sort(layout->bindings.begin(), layout->bindings.end(),
            [](const DescriptorBinding &a, const DescriptorBinding &b) {
              return a.binding < b.binding;
            });
  *pSetLayout = reinterpret_cast<VkDescriptorSetLayout>(layout);
  return VK_SUCCESS;
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkDestroyDescriptorSetLayout(
    VkDevice device, VkDescriptorSetLayout descriptorSetLayout,
    const VkAllocationCallbacks *pAllocator) {
  (void)device;
  (void)pAllocator;
  delete reinterpret_cast<DescriptorSetLayout_T *>(descriptorSetLayout);
}

// ---------------------------------------------------------------------------
// Descriptor pools and sets
// ---------------------------------------------------------------------------
extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkCreateDescriptorPool(
    VkDevice device, const VkDescriptorPoolCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkDescriptorPool *pDescriptorPool) {
  (void)pAllocator;
  Device_T *dev = as_device(device);
  if (dev == nullptr || pCreateInfo == nullptr || pDescriptorPool == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  // Reject sampler/image pool sizes up front: we cannot honour them.
  for (uint32_t i = 0; i < pCreateInfo->poolSizeCount; ++i) {
    VkDescriptorType t = pCreateInfo->pPoolSizes[i].type;
    if (t != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER && t != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
      if (pCreateInfo->pPoolSizes[i].descriptorCount != 0) {
        PPU_ERR("vkCreateDescriptorPool: descriptor type %d is not supported",
                static_cast<int>(t));
        return VK_ERROR_FEATURE_NOT_PRESENT;
      }
    }
  }

  auto *pool = new DescriptorPool_T();
  pool->device = dev;
  pool->max_sets = pCreateInfo->maxSets;
  *pDescriptorPool = reinterpret_cast<VkDescriptorPool>(pool);
  return VK_SUCCESS;
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkDestroyDescriptorPool(
    VkDevice device, VkDescriptorPool descriptorPool, const VkAllocationCallbacks *pAllocator) {
  (void)device;
  (void)pAllocator;
  auto *pool = reinterpret_cast<DescriptorPool_T *>(descriptorPool);
  if (pool == nullptr) return;
  for (auto *set : pool->live_sets) delete set;
  delete pool;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkResetDescriptorPool(
    VkDevice device, VkDescriptorPool descriptorPool, VkDescriptorPoolResetFlags flags) {
  (void)device;
  (void)flags;
  auto *pool = reinterpret_cast<DescriptorPool_T *>(descriptorPool);
  if (pool == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
  for (auto *set : pool->live_sets) delete set;
  pool->live_sets.clear();
  return VK_SUCCESS;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkAllocateDescriptorSets(
    VkDevice device, const VkDescriptorSetAllocateInfo *pAllocateInfo,
    VkDescriptorSet *pDescriptorSets) {
  Device_T *dev = as_device(device);
  if (dev == nullptr || pAllocateInfo == nullptr || pDescriptorSets == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  auto *pool = reinterpret_cast<DescriptorPool_T *>(pAllocateInfo->descriptorPool);
  if (pool == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
  if (pool->live_sets.size() + pAllocateInfo->descriptorSetCount > pool->max_sets) {
    PPU_ERR("vkAllocateDescriptorSets: pool exhausted (%zu live, max %u)",
            pool->live_sets.size(), pool->max_sets);
    return VK_ERROR_OUT_OF_POOL_MEMORY;
  }

  for (uint32_t i = 0; i < pAllocateInfo->descriptorSetCount; ++i) {
    auto *layout = reinterpret_cast<DescriptorSetLayout_T *>(pAllocateInfo->pSetLayouts[i]);
    auto *set = new DescriptorSet_T();
    set->device = dev;
    set->pool = pool;
    set->layout = layout;
    if (layout != nullptr) {
      size_t total = 0;
      for (const auto &b : layout->bindings) total += b.count;
      set->descriptors.resize(total);
    }
    pool->live_sets.push_back(set);
    pDescriptorSets[i] = reinterpret_cast<VkDescriptorSet>(set);
  }
  return VK_SUCCESS;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkFreeDescriptorSets(
    VkDevice device, VkDescriptorPool descriptorPool, uint32_t descriptorSetCount,
    const VkDescriptorSet *pDescriptorSets) {
  (void)device;
  auto *pool = reinterpret_cast<DescriptorPool_T *>(descriptorPool);
  if (pool == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
  for (uint32_t i = 0; i < descriptorSetCount; ++i) {
    auto *set = reinterpret_cast<DescriptorSet_T *>(pDescriptorSets[i]);
    auto it = std::find(pool->live_sets.begin(), pool->live_sets.end(), set);
    if (it != pool->live_sets.end()) {
      pool->live_sets.erase(it);
      delete set;
    }
  }
  return VK_SUCCESS;
}

namespace {

// Locate the descriptor slot for (binding, element) in a layout.
Descriptor_T *find_descriptor(DescriptorSet_T *set, uint32_t binding, uint32_t element) {
  if (set == nullptr || set->layout == nullptr) return nullptr;
  size_t base = 0;
  for (const auto &b : set->layout->bindings) {
    if (b.binding == binding) {
      if (element >= b.count) return nullptr;
      size_t index = base + element;
      return index < set->descriptors.size() ? &set->descriptors[index] : nullptr;
    }
    base += b.count;
  }
  return nullptr;
}

// Apply one VkWriteDescriptorSet to a set. Shared by vkUpdateDescriptorSets
// and the update-template path so both produce identical binding state.
void apply_descriptor_write(DescriptorSet_T *set, const VkWriteDescriptorSet &w) {
  for (uint32_t e = 0; e < w.descriptorCount; ++e) {
    Descriptor_T *slot = find_descriptor(set, w.dstBinding, w.dstArrayElement + e);
    if (slot == nullptr) {
      PPU_ERR("descriptor write: binding %u element %u not in layout", w.dstBinding,
              w.dstArrayElement + e);
      continue;
    }
    slot->type = w.descriptorType;
    if (w.pBufferInfo != nullptr) {
      slot->buffer = reinterpret_cast<Buffer_T *>(w.pBufferInfo[e].buffer);
      slot->offset = w.pBufferInfo[e].offset;
      slot->range = w.pBufferInfo[e].range;
    } else {
      slot->buffer = nullptr;
      slot->offset = 0;
      slot->range = 0;
    }
  }
}

}  // namespace

extern "C" VKAPI_ATTR void VKAPI_CALL vkUpdateDescriptorSets(
    VkDevice device, uint32_t descriptorWriteCount, const VkWriteDescriptorSet *pDescriptorWrites,
    uint32_t descriptorCopyCount, const VkCopyDescriptorSet *pDescriptorCopies) {
  (void)device;
  for (uint32_t i = 0; i < descriptorWriteCount; ++i) {
    auto *set = reinterpret_cast<DescriptorSet_T *>(pDescriptorWrites[i].dstSet);
    apply_descriptor_write(set, pDescriptorWrites[i]);
  }
  for (uint32_t i = 0; i < descriptorCopyCount; ++i) {
    const VkCopyDescriptorSet &c = pDescriptorCopies[i];
    auto *src = reinterpret_cast<DescriptorSet_T *>(c.srcSet);
    auto *dst = reinterpret_cast<DescriptorSet_T *>(c.dstSet);
    for (uint32_t e = 0; e < c.descriptorCount; ++e) {
      Descriptor_T *s = find_descriptor(src, c.srcBinding, c.srcArrayElement + e);
      Descriptor_T *d = find_descriptor(dst, c.dstBinding, c.dstArrayElement + e);
      if (s != nullptr && d != nullptr) *d = *s;
    }
  }
}

// ---------------------------------------------------------------------------
// Descriptor update templates
// ---------------------------------------------------------------------------
extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkCreateDescriptorUpdateTemplate(
    VkDevice device, const VkDescriptorUpdateTemplateCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkDescriptorUpdateTemplate *pDescriptorUpdateTemplate) {
  (void)pAllocator;
  Device_T *dev = as_device(device);
  if (dev == nullptr || pCreateInfo == nullptr || pDescriptorUpdateTemplate == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  if (pCreateInfo->templateType == VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS_KHR) {
    // Push-descriptor templates are only meaningful alongside
    // VK_KHR_push_descriptor, which this ICD does not advertise (its shaders
    // take the classic bind path). Refuse rather than accept a template whose
    // later use would silently do nothing.
    PPU_ERR("vkCreateDescriptorUpdateTemplate: push-descriptor templates are not "
            "supported (VK_KHR_push_descriptor is not advertised)");
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }

  auto *tmpl = new DescriptorUpdateTemplate_T();
  tmpl->device = dev;
  tmpl->template_type = pCreateInfo->templateType;
  tmpl->set_layout = reinterpret_cast<DescriptorSetLayout_T *>(pCreateInfo->descriptorSetLayout);
  tmpl->pipeline_layout = reinterpret_cast<PipelineLayout_T *>(pCreateInfo->pipelineLayout);
  tmpl->set = pCreateInfo->set;

  if (tmpl->set_layout == nullptr) {
    PPU_ERR("vkCreateDescriptorUpdateTemplate: a descriptor set layout is required");
    delete tmpl;
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  for (uint32_t i = 0; i < pCreateInfo->descriptorUpdateEntryCount; ++i) {
    const VkDescriptorUpdateTemplateEntry &e = pCreateInfo->pDescriptorUpdateEntries[i];
    if (e.descriptorType != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER &&
        e.descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
      PPU_ERR("vkCreateDescriptorUpdateTemplate: descriptor type %d is not supported",
              static_cast<int>(e.descriptorType));
      delete tmpl;
      return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    DescriptorUpdateTemplateEntry entry;
    entry.dst_binding = e.dstBinding;
    entry.dst_array_element = e.dstArrayElement;
    entry.descriptor_count = e.descriptorCount;
    entry.type = e.descriptorType;
    entry.offset = e.offset;
    entry.stride = e.stride;
    tmpl->entries.push_back(entry);
  }
  *pDescriptorUpdateTemplate = reinterpret_cast<VkDescriptorUpdateTemplate>(tmpl);
  return VK_SUCCESS;
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkDestroyDescriptorUpdateTemplate(
    VkDevice device, VkDescriptorUpdateTemplate descriptorUpdateTemplate,
    const VkAllocationCallbacks *pAllocator) {
  (void)device;
  (void)pAllocator;
  delete reinterpret_cast<DescriptorUpdateTemplate_T *>(descriptorUpdateTemplate);
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkUpdateDescriptorSetWithTemplate(
    VkDevice device, VkDescriptorSet descriptorSet,
    VkDescriptorUpdateTemplate descriptorUpdateTemplate, const void *pData) {
  (void)device;
  auto *set = reinterpret_cast<DescriptorSet_T *>(descriptorSet);
  auto *tmpl = reinterpret_cast<const DescriptorUpdateTemplate_T *>(descriptorUpdateTemplate);
  if (set == nullptr || tmpl == nullptr || pData == nullptr) return;

  for (const auto &e : tmpl->entries) {
    for (uint32_t i = 0; i < e.descriptor_count; ++i) {
      const size_t off = e.offset + e.stride * i;
      VkDescriptorBufferInfo bi{};
      std::memcpy(&bi, static_cast<const uint8_t *>(pData) + off, sizeof bi);
      Descriptor_T *slot = find_descriptor(set, e.dst_binding, e.dst_array_element + i);
      if (slot == nullptr) {
        PPU_ERR("vkUpdateDescriptorSetWithTemplate: binding %u element %u not in layout",
                e.dst_binding, e.dst_array_element + i);
        continue;
      }
      slot->type = e.type;
      slot->buffer = reinterpret_cast<Buffer_T *>(bi.buffer);
      slot->offset = bi.offset;
      slot->range = bi.range;
    }
  }
}

// ---------------------------------------------------------------------------
// Pipeline layout
// ---------------------------------------------------------------------------
extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkCreatePipelineLayout(
    VkDevice device, const VkPipelineLayoutCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkPipelineLayout *pPipelineLayout) {
  (void)pAllocator;
  Device_T *dev = as_device(device);
  if (dev == nullptr || pCreateInfo == nullptr || pPipelineLayout == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  auto *layout = new PipelineLayout_T();
  layout->device = dev;
  for (uint32_t i = 0; i < pCreateInfo->setLayoutCount; ++i)
    layout->set_layouts.push_back(
        reinterpret_cast<DescriptorSetLayout_T *>(pCreateInfo->pSetLayouts[i]));

  for (uint32_t i = 0; i < pCreateInfo->pushConstantRangeCount; ++i) {
    const VkPushConstantRange &r = pCreateInfo->pPushConstantRanges[i];
    layout->push_constant_size = std::max(layout->push_constant_size, r.offset + r.size);
    layout->push_constant_stages |= r.stageFlags;
  }
  *pPipelineLayout = reinterpret_cast<VkPipelineLayout>(layout);
  return VK_SUCCESS;
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkDestroyPipelineLayout(
    VkDevice device, VkPipelineLayout pipelineLayout, const VkAllocationCallbacks *pAllocator) {
  (void)device;
  (void)pAllocator;
  delete reinterpret_cast<PipelineLayout_T *>(pipelineLayout);
}

// ---------------------------------------------------------------------------
// Shader modules
// ---------------------------------------------------------------------------
extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkCreateShaderModule(
    VkDevice device, const VkShaderModuleCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkShaderModule *pShaderModule) {
  (void)pAllocator;
  Device_T *dev = as_device(device);
  if (dev == nullptr || pCreateInfo == nullptr || pShaderModule == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  if (pCreateInfo->codeSize == 0 || pCreateInfo->pCode == nullptr)
    return VK_ERROR_INVALID_SHADER_NV;

  auto *mod = new ShaderModule_T();
  mod->device = dev;
  const auto *bytes = reinterpret_cast<const uint8_t *>(pCreateInfo->pCode);
  mod->code.assign(bytes, bytes + pCreateInfo->codeSize);

  uint32_t magic = 0;
  std::memcpy(&magic, bytes, std::min<size_t>(sizeof magic, pCreateInfo->codeSize));
  if (magic == kSpirvMagic) {
    mod->format = ModuleFormat::Spirv;
  } else if (magic == kElfMagic) {
    // Accept nvcc/hgcc cubins. This is a deliberate escape hatch so the
    // SPIR-V compiler workstream can be exercised end-to-end with a real
    // kernel before it lands.
    mod->format = ModuleFormat::Cubin;
  } else {
    PPU_ERR("vkCreateShaderModule: unrecognised code magic 0x%08x", magic);
    delete mod;
    return VK_ERROR_INVALID_SHADER_NV;
  }
  *pShaderModule = reinterpret_cast<VkShaderModule>(mod);
  return VK_SUCCESS;
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkDestroyShaderModule(VkDevice device,
                                                            VkShaderModule shaderModule,
                                                            const VkAllocationCallbacks *pAllocator) {
  (void)device;
  (void)pAllocator;
  delete reinterpret_cast<ShaderModule_T *>(shaderModule);
}

// ---------------------------------------------------------------------------
// Compute pipelines
// ---------------------------------------------------------------------------
namespace {

// Resolve which descriptor bindings become kernel arguments, in a stable
// order: (set, binding) ascending, array elements in order.
void build_arg_slots(ComputePipeline_T *pipeline) {
  pipeline->arg_slots.clear();
  PipelineLayout_T *layout = pipeline->layout;
  if (layout == nullptr) return;
  for (uint32_t s = 0; s < layout->set_layouts.size(); ++s) {
    DescriptorSetLayout_T *sl = layout->set_layouts[s];
    if (sl == nullptr) continue;
    for (const auto &b : sl->bindings) {  // already sorted by binding
      for (uint32_t e = 0; e < b.count; ++e)
        pipeline->arg_slots.push_back({s, b.binding, b.type});
    }
  }
  pipeline->has_push_constants = layout->push_constant_size > 0;
}

}  // namespace

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkCreateComputePipelines(
    VkDevice device, VkPipelineCache pipelineCache, uint32_t createInfoCount,
    const VkComputePipelineCreateInfo *pCreateInfos,
    const VkAllocationCallbacks *pAllocator, VkPipeline *pPipelines) {
  (void)pipelineCache;
  (void)pAllocator;
  Device_T *dev = as_device(device);
  if (dev == nullptr || pPipelines == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
  if (!ppu::bind_context(dev)) return VK_ERROR_DEVICE_LOST;

  ppu::Driver &d = ppu::driver();
  VkResult first_error = VK_SUCCESS;

  for (uint32_t i = 0; i < createInfoCount; ++i) {
    pPipelines[i] = VK_NULL_HANDLE;
    const VkComputePipelineCreateInfo &ci = pCreateInfos[i];
    if (ci.stage.stage != VK_SHADER_STAGE_COMPUTE_BIT) {
      PPU_ERR("vkCreateComputePipelines: stage %d is not a compute stage",
              static_cast<int>(ci.stage.stage));
      if (first_error == VK_SUCCESS) first_error = VK_ERROR_FEATURE_NOT_PRESENT;
      continue;
    }
    auto *shader = reinterpret_cast<ShaderModule_T *>(ci.stage.module);
    if (shader == nullptr) {
      if (first_error == VK_SUCCESS) first_error = VK_ERROR_INITIALIZATION_FAILED;
      continue;
    }

    auto *pipeline = new ComputePipeline_T();
    pipeline->device = dev;
    pipeline->layout = reinterpret_cast<PipelineLayout_T *>(ci.layout);
    build_arg_slots(pipeline);

    const char *entry = ci.stage.pName != nullptr ? ci.stage.pName : "main";

    if (shader->format == ModuleFormat::Cubin) {
      ppu::CUmodule module = nullptr;
      ppu::CUresult rc = d.cuModuleLoadData(&module, shader->code.data());
      if (rc != CUDA_SUCCESS) {
        PPU_ERR("cuModuleLoadData(cubin) failed: %s", ppu::cu_strerror(rc));
        delete pipeline;
        if (first_error == VK_SUCCESS) first_error = device_lost("cuModuleLoadData", rc);
        continue;
      }
      ppu::CUfunction fn = nullptr;
      rc = d.cuModuleGetFunction(&fn, module, entry);
      if (rc != CUDA_SUCCESS) {
        PPU_ERR("cuModuleGetFunction('%s') failed: %s", entry, ppu::cu_strerror(rc));
        d.cuModuleUnload(module);
        delete pipeline;
        if (first_error == VK_SUCCESS) first_error = VK_ERROR_INVALID_SHADER_NV;
        continue;
      }
      pipeline->module = module;
      pipeline->function = fn;
      // A cubin carries no Vulkan LocalSize, so the block size has to come
      // from somewhere. Use a documented default (overridable per process with
      // VK_PPU_CUBIN_BLOCK) so vkCmdDispatch's group counts mean workgroups, as
      // applications expect. Once SPIR-V compilation lands, LocalSize is read
      // from the module and this path is only used by tests.
      uint32_t block = 256;
      if (const char *env = std::getenv("VK_PPU_CUBIN_BLOCK")) {
        long v = std::strtol(env, nullptr, 10);
        if (v > 0 && v <= 1024) block = static_cast<uint32_t>(v);
      }
      pipeline->local_size[0] = block;
      pipeline->local_size[1] = 1;
      pipeline->local_size[2] = 1;
    } else {
      // SPIR-V: the compiler workstream owns this boundary.
      uint32_t local[3] = {1, 1, 1};
      const auto *words = reinterpret_cast<const uint32_t *>(shader->code.data());
      size_t word_count = shader->code.size() / sizeof(uint32_t);
      ppu::spirv_local_size(words, word_count, local);
      pipeline->local_size[0] = local[0];
      pipeline->local_size[1] = local[1];
      pipeline->local_size[2] = local[2];

      ppu::CUmodule module = nullptr;
      ppu::CUresult rc = ppu::compile_spirv_to_ppu_module(words, word_count, entry, &module);
      if (rc != CUDA_SUCCESS) {
        PPU_ERR("vkCreateComputePipelines: SPIR-V -> PPU compilation is not implemented "
                "(entry '%s', local size %ux%ux%u). Pipelines are created unusable; "
                "dispatch will fail.",
                entry, local[0], local[1], local[2]);
        // The pipeline handle is still returned so applications can complete
        // their object lifecycle; dispatch reports the failure honestly.
        pipeline->module = nullptr;
        pipeline->function = nullptr;
        *pPipelines = reinterpret_cast<VkPipeline>(pipeline);
        continue;
      }
      pipeline->module = module;
      ppu::CUfunction fn = nullptr;
      if (d.cuModuleGetFunction(&fn, module, entry) == CUDA_SUCCESS)
        pipeline->function = fn;
    }

    *pPipelines = reinterpret_cast<VkPipeline>(pipeline);
  }

  return first_error;
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkDestroyPipeline(VkDevice device, VkPipeline pipeline,
                                                         const VkAllocationCallbacks *pAllocator) {
  (void)device;
  (void)pAllocator;
  auto *p = reinterpret_cast<ComputePipeline_T *>(pipeline);
  if (p == nullptr) return;
  ppu::Driver &d = ppu::driver();
  if (p->module && d.cuModuleUnload) d.cuModuleUnload(p->module);
  delete p;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkCreatePipelineCache(
    VkDevice device, const VkPipelineCacheCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkPipelineCache *pPipelineCache) {
  (void)device;
  (void)pCreateInfo;
  (void)pAllocator;
  if (pPipelineCache == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
  *pPipelineCache = VK_NULL_HANDLE;
  return VK_SUCCESS;
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkDestroyPipelineCache(VkDevice device,
                                                              VkPipelineCache pipelineCache,
                                                              const VkAllocationCallbacks *pAllocator) {
  (void)device;
  (void)pipelineCache;
  (void)pAllocator;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkGetPipelineCacheData(VkDevice device,
                                                                  VkPipelineCache pipelineCache,
                                                                  size_t *pDataSize, void *pData) {
  (void)device;
  (void)pipelineCache;
  (void)pData;
  if (pDataSize) *pDataSize = 0;
  return VK_SUCCESS;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkMergePipelineCaches(
    VkDevice device, VkPipelineCache dstCache, uint32_t srcCacheCount,
    const VkPipelineCache *pSrcCaches) {
  (void)device;
  (void)dstCache;
  (void)srcCacheCount;
  (void)pSrcCaches;
  return VK_SUCCESS;
}
