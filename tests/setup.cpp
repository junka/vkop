/* Copyright (c) 2026 junka. All Rights Reserved. */
#include "setup.hpp"

#include "include/dev_shaders.hpp"

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
    // Every test in this binary builds its ops directly (no model file), so on
    // the default stripped libvkop they need the build tree's archive: see
    // include/dev_shaders.hpp. Done before gtest init so the env is in place
    // for whatever the first test does.
    vkop::dev::seed_shader_archive_from_build_tree();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}