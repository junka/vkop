# Entry point status

## Implemented

These are called and verified by `tests/roundtrip.cpp` (60 checks) on a
real PPU-ZW810E via `libvulkan.so.1`.

* `vkEnumerateInstanceVersion`
* `vkCreateInstance`
* `vkDestroyInstance`
* `vkEnumerateInstanceExtensionProperties` (`VK_KHR_get_physical_device_properties2`)
* `vkEnumerateInstanceLayerProperties`
* `vkEnumeratePhysicalDevices`
* `vkGetPhysicalDeviceProperties`
* `vkGetPhysicalDeviceProperties2`
* `vkGetPhysicalDeviceFeatures`
* `vkGetPhysicalDeviceFeatures2`
* `vkGetPhysicalDeviceQueueFamilyProperties`
* `vkGetPhysicalDeviceQueueFamilyProperties2`
* `vkGetPhysicalDeviceMemoryProperties`
* `vkGetPhysicalDeviceMemoryProperties2`
* `vkGetPhysicalDeviceFormatProperties`
* `vkGetPhysicalDeviceFormatProperties2`
* `vkGetPhysicalDeviceImageFormatProperties`
* `vkGetPhysicalDeviceImageFormatProperties2`
* `vkEnumerateDeviceExtensionProperties`
* `vkGetPhysicalDeviceSparseImageFormatProperties`
* `vkGetPhysicalDeviceSparseImageFormatProperties2`
* `vkGetPhysicalDeviceExternalBufferProperties`
* `vkGetPhysicalDeviceExternalFenceProperties`
* `vkGetPhysicalDeviceExternalSemaphoreProperties`
* `vkGetPhysicalDeviceToolProperties` (answers with an empty tool list: this
  is a driver, not a tooling layer) — advertised as `VK_EXT_tooling_info`;
  the `...EXT` spelling is also served
* `vkCreateDevice`
* `vkDestroyDevice`
* `vkGetDeviceQueue`
* `vkGetDeviceProcAddr`
* `vkDeviceWaitIdle`
* `vkQueueWaitIdle`
* `vkAllocateMemory`
* `vkFreeMemory`
* `vkMapMemory`
* `vkUnmapMemory`
* `vkFlushMappedMemoryRanges`
* `vkInvalidateMappedMemoryRanges`
* `vkCreateBuffer`
* `vkDestroyBuffer`
* `vkGetBufferMemoryRequirements`
* `vkGetBufferMemoryRequirements2`
* `vkGetDeviceBufferMemoryRequirements`
* `vkBindBufferMemory`
* `vkBindBufferMemory2`
* `vkGetBufferDeviceAddress`
* `vkGetBufferOpaqueCaptureAddress`
* `vkGetDeviceMemoryOpaqueCaptureAddress`
* `vkCreateDescriptorSetLayout`
* `vkDestroyDescriptorSetLayout`
* `vkCreateDescriptorPool`
* `vkDestroyDescriptorPool`
* `vkResetDescriptorPool`
* `vkAllocateDescriptorSets`
* `vkFreeDescriptorSets`
* `vkUpdateDescriptorSets`
* `vkCreateDescriptorUpdateTemplate`
* `vkDestroyDescriptorUpdateTemplate`
* `vkUpdateDescriptorSetWithTemplate`
* `vkCreateQueryPool` (timestamps only; other query types are rejected with
  `VK_ERROR_FEATURE_NOT_PRESENT`)
* `vkDestroyQueryPool`
* `vkResetQueryPool`
* `vkGetQueryPoolResults`
* `vkCmdResetQueryPool`
* `vkCmdWriteTimestamp`
* `vkCreatePipelineLayout`
* `vkDestroyPipelineLayout`
* `vkCreateShaderModule`
* `vkDestroyShaderModule`
* `vkCreateComputePipelines`
* `vkDestroyPipeline`
* `vkCreatePipelineCache`
* `vkDestroyPipelineCache`
* `vkGetPipelineCacheData`
* `vkMergePipelineCaches`
* `vkCreateCommandPool`
* `vkDestroyCommandPool`
* `vkResetCommandPool`
* `vkAllocateCommandBuffers`
* `vkFreeCommandBuffers`
* `vkBeginCommandBuffer`
* `vkEndCommandBuffer`
* `vkResetCommandBuffer`
* `vkCmdBindPipeline`
* `vkCmdBindDescriptorSets`
* `vkCmdPushConstants`
* `vkCmdCopyBuffer`
* `vkCmdFillBuffer`
* `vkCmdUpdateBuffer`
* `vkCmdDispatch`
* `vkCmdPipelineBarrier`
* `vkCmdPipelineBarrier2`
* `vkQueueSubmit`
* `vkCreateFence`
* `vkDestroyFence`
* `vkGetFenceStatus`
* `vkWaitForFences`
* `vkResetFences`
* `vkCreateSemaphore`
* `vkDestroySemaphore`
* `vk_icdNegotiateLoaderICDInterfaceVersion`
* `vk_icdGetInstanceProcAddr`
* `vk_icdGetPhysicalDeviceProcAddr`

Aliases also served:

* `vkGetPhysicalDeviceProperties2KHR`, `vkGetPhysicalDeviceFeatures2KHR`,
  `vkGetPhysicalDeviceQueueFamilyProperties2KHR`, `vkGetPhysicalDeviceMemoryProperties2KHR`,
  `vkGetPhysicalDeviceFormatProperties2KHR`, `vkGetPhysicalDeviceImageFormatProperties2KHR`,
  `vkGetBufferMemoryRequirements2KHR`, `vkBindBufferMemory2KHR`,
  `vkGetDeviceBufferMemoryRequirementsKHR`, `vkGetBufferDeviceAddressKHR`,
  `vkCmdPipelineBarrier2KHR`, `vkGetImageMemoryRequirements2KHR`

## Stubbed

These return `VK_ERROR_FEATURE_NOT_PRESENT` (or the format-specific error), never
a fake success:

* `vkCreateImage`
* `vkDestroyImage`
* `vkBindImageMemory`
* `vkGetImageMemoryRequirements`
* `vkGetImageMemoryRequirements2`
* `vkCreateImageView`
* `vkDestroyImageView`
* `vkCreateSampler`
* `vkDestroySampler`
* `vkCreateSamplerYcbcrConversion`
* `vkDestroySamplerYcbcrConversion`
* `vkGetImageSubresourceLayout`
* `vkCreateBufferView`
* `vkDestroyBufferView`
* `vkCreateGraphicsPipelines`
* `vkCreateRenderPass`
* `vkDestroyRenderPass`
* `vkCreateFramebuffer`
* `vkDestroyFramebuffer`
* `vkCreateEvent`
* `vkDestroyEvent`
* `vkGetEventStatus`
* `vkSetEvent`
* `vkResetEvent`

## Explicit errors in otherwise-implemented calls

* `vkCmdCopyBufferToImage / vkCmdCopyImageToBuffer` — logs an error, records nothing (images unsupported)
* `vkCreateQueryPool` with a non-TIMESTAMP type — `VK_ERROR_FEATURE_NOT_PRESENT`
* `vkCreateDescriptorUpdateTemplate` with `..._TYPE_PUSH_DESCRIPTORS_KHR` —
  `VK_ERROR_FEATURE_NOT_PRESENT` (VK_KHR_push_descriptor is not advertised)
* `vkGetBufferDeviceAddressKHR and other KHR aliases` — same implementation as the core entry point
* `vkGetMemoryFdKHR` — VK_ERROR_FEATURE_NOT_PRESENT (no external handles)
* `vkCmdSetEvent` / `vkCmdWaitEvents` / `vkCmdResetEvent` — no-ops; single-stream replay
  already orders all work, and `vkCreateEvent` is unsupported so these cannot be reached
  with a valid event anyway
* `vkGetSemaphoreCounterValue / vkWaitSemaphores / vkSignalSemaphore` — VK_ERROR_FEATURE_NOT_PRESENT (no timeline semaphores)

## Partial: records but cannot execute

* `vkCmdDispatchIndirect` — recording succeeds, but `vkQueueSubmit` fails the
  submission with `VK_ERROR_DEVICE_LOST`. The base CUDA driver API here has no
  indirect launch that takes device-resident group counts, so rather than
  silently dispatch the wrong geometry it refuses.

## Behaviour worth knowing

* `vkCreateComputePipelines` with SPIR-V returns a handle but logs that SPIR-V → PPU
  compilation is unimplemented. `vkQueueSubmit` then fails the submission with
  `VK_ERROR_DEVICE_LOST` instead of launching nothing silently.
* `vkAllocateCommandBuffers` rejects `SECONDARY` buffers.
* `vkCreateDescriptorSetLayout` / `vkCreateDescriptorPool` reject sampler and image
  descriptor types.
* `vkCreateDevice` rejects queue families other than 0 and queue counts other than 1.
* `vkMapMemory` on device-only memory (type 0) returns `VK_ERROR_MEMORY_MAP_FAILED`.
* Timestamp queries measure `cuEventElapsedTime` between a per-pool origin event
  and a per-write event, and report nanoseconds (timestampPeriod = 1.0). The
  value is a delta from that origin, not a hardware counter, which is exactly
  what a begin/end subtraction needs. When the backend cannot create or time
  events (`cuEventElapsedTime` missing), `timestampValidBits` is 0 and
  `vkCreateQueryPool` returns `VK_ERROR_FEATURE_NOT_PRESENT`.
