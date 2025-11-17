// Copyright 2025 @junka
#ifndef SRC_VULKANLIB_HPP_
#define SRC_VULKANLIB_HPP_

#include <vulkan/vulkan.hpp>

namespace vkop {

#define VK_FUNCTION_LIST                                                       \
    PFN(vkEnumerateInstanceVersion)                                            \
    PFN(vkEnumerateInstanceLayerProperties)                                    \
    PFN(vkCreateInstance)                                                      \
    PFN(vkEnumerateInstanceExtensionProperties)                                \
    PFN(vkGetInstanceProcAddr)                                                 \
    PFN(vkMapMemory)                                                           \
    PFN(vkUnmapMemory)                                                         \
    PFN(vkGetBufferMemoryRequirements)                                         \
    PFN(vkGetPhysicalDeviceMemoryProperties)                                   \
    PFN(vkAllocateMemory)                                                      \
    PFN(vkAllocateCommandBuffers)                                              \
    PFN(vkBindBufferMemory)                                                    \
    PFN(vkCmdBindPipeline)                                                     \
    PFN(vkCmdDispatch)                                                         \
    PFN(vkCmdWriteTimestamp)                                                   \
    PFN(vkCmdBindDescriptorSets)                                               \
    PFN(vkCmdResetQueryPool)                                                   \
    PFN(vkBeginCommandBuffer)                                                  \
    PFN(vkEndCommandBuffer)                                                    \
    PFN(vkQueueSubmit)                                                         \
    PFN(vkQueueWaitIdle)                                                       \
    PFN(vkCreateBuffer)                                                        \
    PFN(vkCreateQueryPool)                                                     \
    PFN(vkCreateDescriptorPool)                                                \
    PFN(vkAllocateDescriptorSets)                                              \
    PFN(vkUpdateDescriptorSets)                                                \
    PFN(vkCreateCommandPool)                                                   \
    PFN(vkResetCommandPool)                                                    \
    PFN(vkCreateComputePipelines)                                              \
    PFN(vkCreateDevice)                                                        \
    PFN(vkGetDeviceQueue)                                                      \
    PFN(vkCreateDescriptorSetLayout)                                           \
    PFN(vkCreatePipelineLayout)                                                \
    PFN(vkDestroyBuffer)                                                       \
    PFN(vkDestroyQueryPool)                                                    \
    PFN(vkDestroyDescriptorPool)                                               \
    PFN(vkDestroyPipeline)                                                     \
    PFN(vkDestroyPipelineLayout)                                               \
    PFN(vkDestroyDescriptorSetLayout)                                          \
    PFN(vkDestroyDevice)                                                       \
    PFN(vkDestroyInstance)                                                     \
    PFN(vkGetQueryPoolResults)                                                 \
    PFN(vkCreateShaderModule)                                                  \
    PFN(vkDestroyShaderModule)                                                 \
    PFN(vkDestroyCommandPool)                                                  \
    PFN(vkFreeMemory)                                                          \
    PFN(vkGetPhysicalDeviceQueueFamilyProperties)                              \
    PFN(vkGetPhysicalDeviceProperties)                                         \
    PFN(vkGetPhysicalDeviceProperties2)                                        \
    PFN(vkEnumeratePhysicalDevices)                                            \
    PFN(vkEnumerateDeviceExtensionProperties)                                  \
    PFN(vkResetCommandBuffer)                                                  \
    PFN(vkFreeCommandBuffers)                                                  \
    PFN(vkGetPhysicalDeviceFeatures)                                           \
    PFN(vkGetPhysicalDeviceFeatures2)                                          \
    PFN(vkBindBufferMemory2)                                                   \
    PFN(vkCreateImage)                                                         \
    PFN(vkGetImageMemoryRequirements)                                          \
    PFN(vkDestroyImage)                                                        \
    PFN(vkBindImageMemory)                                                     \
    PFN(vkCreateImageView)                                                     \
    PFN(vkDestroyImageView)                                                    \
    PFN(vkCreateSampler)                                                       \
    PFN(vkDestroySampler)                                                      \
    PFN(vkCmdPipelineBarrier)                                                  \
    PFN(vkGetImageSubresourceLayout)                                           \
    PFN(vkCmdCopyBufferToImage)                                                \
    PFN(vkCmdCopyImageToBuffer)                                                \
    PFN(vkCmdCopyBuffer)                                                       \
    PFN(vkCmdCopyImage)                                                        \
    PFN(vkFreeDescriptorSets)                                                  \
    PFN(vkCreateDescriptorUpdateTemplate)                                      \
    PFN(vkResetQueryPool)                                                      \
    PFN(vkGetImageMemoryRequirements2)                                         \
    PFN(vkGetBufferMemoryRequirements2)                                        \
    PFN(vkGetImageSparseMemoryRequirements2)                                   \
    PFN(vkGetDeviceProcAddr)                                                   \
    PFN(vkBindImageMemory2)                                                    \
    PFN(vkInvalidateMappedMemoryRanges)                                        \
    PFN(vkFlushMappedMemoryRanges)                                             \
    PFN(vkDeviceWaitIdle)                                                      \
    PFN(vkWaitForFences)                                                       \
    PFN(vkDestroyFence)                                                        \
    PFN(vkCreateFence)

class VulkanLib {
  public:
    static VulkanLib &getVulkanLib() {
        static VulkanLib instance;
        return instance;
    }

    VulkanLib();
    ~VulkanLib();

    VulkanLib(const VulkanLib &) = delete;
    VulkanLib &operator=(const VulkanLib &) = delete;

#define PFN(name) PFN_##name name;
    VK_FUNCTION_LIST
#undef PFN
  private:
    void *lib_;
};

// this file should be included in all vulkanXX.cpp files
#define PFN(name) static PFN_##name name = VulkanLib::getVulkanLib().name;
VK_FUNCTION_LIST
#undef PFN

} // namespace vkop

#endif // SRC_VULKANLIB_HPP_
