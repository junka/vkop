// Copyright 2025 @junka
#include "vulkan/VulkanDebug.hpp"

namespace vkop {

std::atomic<VkDevice> VulkanDebug::s_device_{VK_NULL_HANDLE};

} // namespace vkop
