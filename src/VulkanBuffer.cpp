#include "VulkanLib.hpp"
#include "VulkanResource.hpp"
#include "VulkanBuffer.hpp"
#include <stdexcept>

namespace vkop {
VulkanBuffer::VulkanBuffer(VkPhysicalDevice physicalDevice, const uint32_t queueFamilyIndex, VkDevice device, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags requireProperties)
    : VulkanResource(physicalDevice, queueFamilyIndex, device), m_buffer(VK_NULL_HANDLE), m_size(size) {

    createBuffer(size, usage);
#ifdef VK_KHR_get_memory_requirements2
    VkMemoryRequirements2 memRequirements2 = {};
    memRequirements2.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
    VkBufferMemoryRequirementsInfo2 bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2;
    bufferInfo.buffer = m_buffer;
    vkGetBufferMemoryRequirements2(m_device, &bufferInfo, &memRequirements2);
    VkMemoryRequirements memRequirements = memRequirements2.memoryRequirements;
#else
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(m_device, m_buffer, &memRequirements);
#endif
    allocMemory(memRequirements, requireProperties);

    vkBindBufferMemory(m_device, m_buffer, getMemory(), 0);
}

VulkanBuffer::~VulkanBuffer() {
    cleanup();
}

VkBuffer VulkanBuffer::getBuffer() const {
    return m_buffer;
}


void VulkanBuffer::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bufferInfo.queueFamilyIndexCount = 1;
    bufferInfo.pQueueFamilyIndices = &m_queueFamilyIndex;

    if (vkCreateBuffer(m_device, &bufferInfo, nullptr, &m_buffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create buffer!");
    }
}


std::variant<VkDescriptorImageInfo, VkDescriptorBufferInfo> VulkanBuffer::getDescriptorInfo() const {
    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = m_buffer;
    bufferInfo.offset = 0;
    bufferInfo.range = m_size;
    return bufferInfo;
}

void VulkanBuffer::cleanup() {
    if (m_buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(m_device, m_buffer, nullptr);
    }
}

} // namespace vkop