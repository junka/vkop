/* Copyright (c) 2026 junka. All Rights Reserved.
 *
 * Out-of-line definitions for TestEnv's statics, split out of setup.cpp so a
 * non-gtest probe can link them without dragging in gtest's main(). setup.cpp
 * also seeds the dev shader archive and defines main(); this file defines only
 * the storage the header declares.
 */
#include "setup.hpp"

namespace vkop {
namespace tests {

std::shared_ptr<VulkanDevice> TestEnv::dev_ = nullptr;
std::shared_ptr<VulkanCommandPool> TestEnv::cmdpool_ = nullptr;
bool TestEnv::initialized_ = false;

} // namespace tests
} // namespace vkop
