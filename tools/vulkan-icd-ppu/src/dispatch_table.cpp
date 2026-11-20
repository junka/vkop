// Dispatch table.
//
// Every entry point the ICD answers lives in one table, keyed by the name an
// application may pass to vkGetInstanceProcAddr / vkGetDeviceProcAddr.
// Anything requiring images, graphics, queries or shader interop is answered by
// an honest stub that returns VK_ERROR_FEATURE_NOT_PRESENT - never by a call
// that pretends to succeed.
#include "icd_common.hpp"

#include <cstring>
#include <vector>

// Entry points implemented across the other translation units, where the
// exported symbol name is exactly the Vulkan entry point name.
#define PPU_INSTANCE_ENTRY_POINTS(X)                                           \
  X(vkEnumerateInstanceVersion)                                                \
  X(vkCreateInstance)                                                          \
  X(vkDestroyInstance)                                                         \
  X(vkEnumerateInstanceExtensionProperties)                                    \
  X(vkEnumerateInstanceLayerProperties)                                        \
  X(vkEnumeratePhysicalDevices)                                                \
  X(vkGetPhysicalDeviceProperties)                                             \
  X(vkGetPhysicalDeviceProperties2)                                            \
  X(vkGetPhysicalDeviceFeatures)                                               \
  X(vkGetPhysicalDeviceFeatures2)                                              \
  X(vkGetPhysicalDeviceQueueFamilyProperties)                                  \
  X(vkGetPhysicalDeviceQueueFamilyProperties2)                                 \
  X(vkGetPhysicalDeviceMemoryProperties)                                       \
  X(vkGetPhysicalDeviceMemoryProperties2)                                      \
  X(vkGetPhysicalDeviceFormatProperties)                                       \
  X(vkGetPhysicalDeviceFormatProperties2)                                      \
  X(vkGetPhysicalDeviceImageFormatProperties)                                  \
  X(vkGetPhysicalDeviceImageFormatProperties2)                                 \
  X(vkEnumerateDeviceExtensionProperties)                                      \
  X(vkGetPhysicalDeviceSparseImageFormatProperties)                            \
  X(vkGetPhysicalDeviceSparseImageFormatProperties2)                           \
  X(vkGetPhysicalDeviceExternalBufferProperties)                               \
  X(vkGetPhysicalDeviceExternalFenceProperties)                                \
  X(vkGetPhysicalDeviceExternalSemaphoreProperties)                            \
  X(vkCreateDevice)                                                            \
  X(vkDestroyDevice)                                                           \
  X(vkGetDeviceQueue)                                                          \
  X(vkDeviceWaitIdle)                                                          \
  X(vkQueueWaitIdle)                                                           \
  X(vkAllocateMemory)                                                          \
  X(vkFreeMemory)                                                              \
  X(vkMapMemory)                                                               \
  X(vkUnmapMemory)                                                             \
  X(vkFlushMappedMemoryRanges)                                                 \
  X(vkInvalidateMappedMemoryRanges)                                            \
  X(vkCreateBuffer)                                                            \
  X(vkDestroyBuffer)                                                           \
  X(vkGetBufferMemoryRequirements)                                             \
  X(vkGetBufferMemoryRequirements2)                                            \
  X(vkGetDeviceBufferMemoryRequirements)                                       \
  X(vkBindBufferMemory)                                                        \
  X(vkBindBufferMemory2)                                                       \
  X(vkGetBufferDeviceAddress)                                                  \
  X(vkGetBufferOpaqueCaptureAddress)                                           \
  X(vkGetDeviceMemoryOpaqueCaptureAddress)                                     \
  X(vkGetMemoryFdKHR)                                                          \
  X(vkCreateDescriptorSetLayout)                                               \
  X(vkDestroyDescriptorSetLayout)                                              \
  X(vkCreateDescriptorPool)                                                    \
  X(vkDestroyDescriptorPool)                                                   \
  X(vkResetDescriptorPool)                                                     \
  X(vkAllocateDescriptorSets)                                                  \
  X(vkFreeDescriptorSets)                                                      \
  X(vkUpdateDescriptorSets)                                                    \
  X(vkCreatePipelineLayout)                                                    \
  X(vkDestroyPipelineLayout)                                                   \
  X(vkCreateShaderModule)                                                      \
  X(vkDestroyShaderModule)                                                     \
  X(vkCreateComputePipelines)                                                  \
  X(vkDestroyPipeline)                                                         \
  X(vkCreatePipelineCache)                                                     \
  X(vkDestroyPipelineCache)                                                    \
  X(vkGetPipelineCacheData)                                                    \
  X(vkMergePipelineCaches)                                                     \
  X(vkCreateCommandPool)                                                       \
  X(vkDestroyCommandPool)                                                      \
  X(vkResetCommandPool)                                                        \
  X(vkAllocateCommandBuffers)                                                  \
  X(vkFreeCommandBuffers)                                                      \
  X(vkBeginCommandBuffer)                                                      \
  X(vkEndCommandBuffer)                                                        \
  X(vkResetCommandBuffer)                                                      \
  X(vkCmdBindPipeline)                                                         \
  X(vkCmdBindDescriptorSets)                                                   \
  X(vkCmdPushConstants)                                                        \
  X(vkCmdCopyBuffer)                                                           \
  X(vkCmdFillBuffer)                                                           \
  X(vkCmdUpdateBuffer)                                                         \
  X(vkCmdCopyBufferToImage)                                                    \
  X(vkCmdCopyImageToBuffer)                                                    \
  X(vkCmdDispatch)                                                             \
  X(vkCmdDispatchIndirect)                                                     \
  X(vkCmdPipelineBarrier)                                                      \
  X(vkCmdPipelineBarrier2)                                                     \
  X(vkCmdWriteTimestamp)                                                       \
  X(vkCmdSetEvent)                                                             \
  X(vkCmdWaitEvents)                                                           \
  X(vkCmdResetEvent)                                                           \
  X(vkQueueSubmit)                                                             \
  X(vkCreateFence)                                                             \
  X(vkDestroyFence)                                                            \
  X(vkGetFenceStatus)                                                          \
  X(vkWaitForFences)                                                           \
  X(vkResetFences)                                                             \
  X(vkCreateSemaphore)                                                         \
  X(vkDestroySemaphore)                                                        \
  X(vkGetSemaphoreCounterValue)                                                \
  X(vkWaitSemaphores)                                                          \
  X(vkSignalSemaphore)                                                         \
  X(vkGetDeviceProcAddr)

namespace {

constexpr VkResult kUnsupported = VK_ERROR_FEATURE_NOT_PRESENT;

// ---------------------------------------------------------------------------
// Honest stubs. Each returns an error rather than a fake success.
// ---------------------------------------------------------------------------
#define PPU_STUB_WARN(what)                                                    \
  PPU_ERR("%s: not supported by this device (compute + transfer only)", what)

// --- images ---------------------------------------------------------------
VkResult stub_CreateImage(VkDevice, const VkImageCreateInfo *, const VkAllocationCallbacks *,
                          VkImage *pImage) {
  if (pImage) *pImage = VK_NULL_HANDLE;
  PPU_STUB_WARN("vkCreateImage");
  return kUnsupported;
}
void stub_DestroyImage(VkDevice, VkImage, const VkAllocationCallbacks *) {}
VkResult stub_BindImageMemory(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize) {
  return kUnsupported;
}
void stub_GetImageMemoryRequirements(VkDevice, VkImage, VkMemoryRequirements *pReq) {
  if (pReq) std::memset(pReq, 0, sizeof *pReq);
}
VkResult stub_GetImageMemoryRequirements2(VkDevice, const VkImageMemoryRequirementsInfo2 *,
                                          VkMemoryRequirements2 *pReq) {
  if (pReq) std::memset(&pReq->memoryRequirements, 0, sizeof pReq->memoryRequirements);
  return kUnsupported;
}
VkResult stub_CreateImageView(VkDevice, const VkImageViewCreateInfo *,
                              const VkAllocationCallbacks *, VkImageView *pView) {
  if (pView) *pView = VK_NULL_HANDLE;
  return kUnsupported;
}
void stub_DestroyImageView(VkDevice, VkImageView, const VkAllocationCallbacks *) {}
VkResult stub_CreateSampler(VkDevice, const VkSamplerCreateInfo *,
                            const VkAllocationCallbacks *, VkSampler *pSampler) {
  if (pSampler) *pSampler = VK_NULL_HANDLE;
  return kUnsupported;
}
void stub_DestroySampler(VkDevice, VkSampler, const VkAllocationCallbacks *) {}
VkResult stub_CreateSamplerYcbcrConversion(VkDevice,
                                           const VkSamplerYcbcrConversionCreateInfo *,
                                           const VkAllocationCallbacks *,
                                           VkSamplerYcbcrConversion *pConversion) {
  if (pConversion) *pConversion = VK_NULL_HANDLE;
  return kUnsupported;
}
void stub_DestroySamplerYcbcrConversion(VkDevice, VkSamplerYcbcrConversion,
                                        const VkAllocationCallbacks *) {}
VkResult stub_GetImageSubresourceLayout(VkDevice, VkImage, const VkImageSubresource *,
                                        VkSubresourceLayout *pLayout) {
  if (pLayout) std::memset(pLayout, 0, sizeof *pLayout);
  return kUnsupported;
}
VkResult stub_CreateBufferView(VkDevice, const VkBufferViewCreateInfo *,
                               const VkAllocationCallbacks *, VkBufferView *pView) {
  if (pView) *pView = VK_NULL_HANDLE;
  return kUnsupported;
}
void stub_DestroyBufferView(VkDevice, VkBufferView, const VkAllocationCallbacks *) {}

// --- graphics -------------------------------------------------------------
VkResult stub_CreateGraphicsPipelines(VkDevice, VkPipelineCache, uint32_t count,
                                      const VkGraphicsPipelineCreateInfo *,
                                      const VkAllocationCallbacks *, VkPipeline *pPipelines) {
  for (uint32_t i = 0; i < count && pPipelines; ++i) pPipelines[i] = VK_NULL_HANDLE;
  PPU_STUB_WARN("vkCreateGraphicsPipelines");
  return kUnsupported;
}
VkResult stub_CreateRenderPass(VkDevice, const VkRenderPassCreateInfo *,
                               const VkAllocationCallbacks *, VkRenderPass *p) {
  if (p) *p = VK_NULL_HANDLE;
  return kUnsupported;
}
void stub_DestroyRenderPass(VkDevice, VkRenderPass, const VkAllocationCallbacks *) {}
VkResult stub_CreateFramebuffer(VkDevice, const VkFramebufferCreateInfo *,
                                const VkAllocationCallbacks *, VkFramebuffer *p) {
  if (p) *p = VK_NULL_HANDLE;
  return kUnsupported;
}
void stub_DestroyFramebuffer(VkDevice, VkFramebuffer, const VkAllocationCallbacks *) {}

// --- queries --------------------------------------------------------------
VkResult stub_CreateQueryPool(VkDevice, const VkQueryPoolCreateInfo *,
                              const VkAllocationCallbacks *, VkQueryPool *p) {
  if (p) *p = VK_NULL_HANDLE;
  return kUnsupported;
}
void stub_DestroyQueryPool(VkDevice, VkQueryPool, const VkAllocationCallbacks *) {}
void stub_ResetQueryPool(VkDevice, VkQueryPool, uint32_t, uint32_t) {}

// --- events ---------------------------------------------------------------
VkResult stub_CreateEvent(VkDevice, const VkEventCreateInfo *, const VkAllocationCallbacks *,
                          VkEvent *p) {
  if (p) *p = VK_NULL_HANDLE;
  return kUnsupported;
}
void stub_DestroyEvent(VkDevice, VkEvent, const VkAllocationCallbacks *) {}
VkResult stub_GetEventStatus(VkDevice, VkEvent) { return kUnsupported; }
VkResult stub_SetEvent(VkDevice, VkEvent) { return kUnsupported; }
VkResult stub_ResetEvent(VkDevice, VkEvent) { return kUnsupported; }

// --- table ----------------------------------------------------------------
using PfnVoid = PFN_vkVoidFunction;

struct Entry {
  const char *name;
  PfnVoid fn;
};

#define PPU_ENTRY(name) Entry{#name, reinterpret_cast<PfnVoid>(name)},
#define PPU_ALIAS(alias, target) Entry{#alias, reinterpret_cast<PfnVoid>(target)},

const std::vector<Entry> &table() {
  static const std::vector<Entry> entries = {
      PPU_INSTANCE_ENTRY_POINTS(PPU_ENTRY)

      // Loader interface (queryable via GetInstanceProcAddr from v7 on).
      PPU_ENTRY(vk_icdNegotiateLoaderICDInterfaceVersion)
      PPU_ENTRY(vk_icdGetInstanceProcAddr)
      PPU_ENTRY(vk_icdGetPhysicalDeviceProcAddr)

      // KHR aliases for the Vulkan 1.1 core functions we implement.
      PPU_ALIAS(vkGetPhysicalDeviceProperties2KHR, vkGetPhysicalDeviceProperties2)
      PPU_ALIAS(vkGetPhysicalDeviceFeatures2KHR, vkGetPhysicalDeviceFeatures2)
      PPU_ALIAS(vkGetPhysicalDeviceQueueFamilyProperties2KHR,
                vkGetPhysicalDeviceQueueFamilyProperties2)
      PPU_ALIAS(vkGetPhysicalDeviceMemoryProperties2KHR, vkGetPhysicalDeviceMemoryProperties2)
      PPU_ALIAS(vkGetPhysicalDeviceFormatProperties2KHR, vkGetPhysicalDeviceFormatProperties2)
      PPU_ALIAS(vkGetPhysicalDeviceImageFormatProperties2KHR,
                vkGetPhysicalDeviceImageFormatProperties2)
      PPU_ALIAS(vkGetBufferMemoryRequirements2KHR, vkGetBufferMemoryRequirements2)
      PPU_ALIAS(vkBindBufferMemory2KHR, vkBindBufferMemory2)
      PPU_ALIAS(vkCmdPipelineBarrier2KHR, vkCmdPipelineBarrier2)

      // Buffer device address is a core 1.2 feature here; expose both spellings.
      PPU_ALIAS(vkGetBufferDeviceAddressKHR, vkGetBufferDeviceAddress)
      PPU_ALIAS(vkGetDeviceBufferMemoryRequirementsKHR, vkGetDeviceBufferMemoryRequirements)

      // Honest stubs, under their real Vulkan names.
      PPU_ALIAS(vkCreateImage, stub_CreateImage)
      PPU_ALIAS(vkDestroyImage, stub_DestroyImage)
      PPU_ALIAS(vkBindImageMemory, stub_BindImageMemory)
      PPU_ALIAS(vkGetImageMemoryRequirements, stub_GetImageMemoryRequirements)
      PPU_ALIAS(vkGetImageMemoryRequirements2, stub_GetImageMemoryRequirements2)
      PPU_ALIAS(vkGetImageMemoryRequirements2KHR, stub_GetImageMemoryRequirements2)
      PPU_ALIAS(vkCreateImageView, stub_CreateImageView)
      PPU_ALIAS(vkDestroyImageView, stub_DestroyImageView)
      PPU_ALIAS(vkCreateSampler, stub_CreateSampler)
      PPU_ALIAS(vkDestroySampler, stub_DestroySampler)
      PPU_ALIAS(vkCreateSamplerYcbcrConversion, stub_CreateSamplerYcbcrConversion)
      PPU_ALIAS(vkDestroySamplerYcbcrConversion, stub_DestroySamplerYcbcrConversion)
      PPU_ALIAS(vkGetImageSubresourceLayout, stub_GetImageSubresourceLayout)
      PPU_ALIAS(vkCreateBufferView, stub_CreateBufferView)
      PPU_ALIAS(vkDestroyBufferView, stub_DestroyBufferView)
      PPU_ALIAS(vkCreateGraphicsPipelines, stub_CreateGraphicsPipelines)
      PPU_ALIAS(vkCreateRenderPass, stub_CreateRenderPass)
      PPU_ALIAS(vkDestroyRenderPass, stub_DestroyRenderPass)
      PPU_ALIAS(vkCreateFramebuffer, stub_CreateFramebuffer)
      PPU_ALIAS(vkDestroyFramebuffer, stub_DestroyFramebuffer)
      PPU_ALIAS(vkCreateQueryPool, stub_CreateQueryPool)
      PPU_ALIAS(vkDestroyQueryPool, stub_DestroyQueryPool)
      PPU_ALIAS(vkResetQueryPool, stub_ResetQueryPool)
      PPU_ALIAS(vkCreateEvent, stub_CreateEvent)
      PPU_ALIAS(vkDestroyEvent, stub_DestroyEvent)
      PPU_ALIAS(vkGetEventStatus, stub_GetEventStatus)
      PPU_ALIAS(vkSetEvent, stub_SetEvent)
      PPU_ALIAS(vkResetEvent, stub_ResetEvent)
  };
  return entries;
}

const Entry *find(const char *name) {
  if (name == nullptr) return nullptr;
  for (const auto &e : table())
    if (std::strcmp(e.name, name) == 0) return &e;
  return nullptr;
}

}  // namespace

namespace ppu {

PPU_INTERNAL PFN_vkVoidFunction ppu_lookup_instance_proc(const char *name) {
  const Entry *e = find(name);
  if (e == nullptr) {
    PPU_LOG("GetInstanceProcAddr(%s) -> not handled by this ICD", name ? name : "(null)");
    return nullptr;
  }
  return e->fn;
}

PPU_INTERNAL PFN_vkVoidFunction ppu_lookup_device_proc(const char *name) {
  return ppu_lookup_instance_proc(name);
}

}  // namespace ppu

extern "C" VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device,
                                                                       const char *pName) {
  (void)device;
  return ppu::ppu_lookup_device_proc(pName);
}
