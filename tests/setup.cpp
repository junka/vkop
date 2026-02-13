/* Copyright (c) 2026 junka. All Rights Reserved. */
#include "setup.hpp"

namespace vkop {
namespace tests {

std::shared_ptr<vkop::VulkanDevice> vkop::tests::TestEnv::dev_ = nullptr;
std::shared_ptr<vkop::VulkanCommandPool> vkop::tests::TestEnv::cmdpool_ = nullptr;
bool vkop::tests::TestEnv::initialized_ = false;

} // namespace tests
} // namespace vkop