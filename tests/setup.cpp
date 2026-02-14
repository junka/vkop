/* Copyright (c) 2026 junka. All Rights Reserved. */
#include "setup.hpp"

namespace vkop {
namespace tests {


std::shared_ptr<VulkanDevice> TestEnv::dev_ = nullptr;
std::shared_ptr<VulkanCommandPool> TestEnv::cmdpool_ = nullptr;
bool TestEnv::initialized_ = false;

} // namespace tests
} // namespace vkop

::testing::Environment* const kVkopenv = 
    ::testing::AddGlobalTestEnvironment(new vkop::tests::TestEnv);


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}