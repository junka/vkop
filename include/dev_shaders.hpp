// Copyright 2025 @junka
#ifndef INCLUDE_DEV_SHADERS_HPP_
#define INCLUDE_DEV_SHADERS_HPP_

#include <cstdlib>
#include <unistd.h>

namespace vkop {
namespace dev {

// Point a development binary that builds ops outside Runtime::LoadModel at
// this build tree's shader archive.
//
// libvkop is built with VKOP_STRIP_BUILTIN_SHADERS=ON by default, so its own
// `*_spv[]` arrays are 1-byte placeholders: every shader byte comes from an
// archive. A model carries its own (onnx2vkop embeds one), but the unit tests'
// torch fixtures, the probes and bench_matmul construct operators straight
// through ops::create_from_type(), so they have no model to take shaders from
// and die on the stub array instead. Operator falls back to
// ShaderArchive::process_default() in exactly that case, and this seeds that
// hook's VKOP_SHADER_ARCHIVE from the path CMake baked in
// (VKOP_DEV_SHADER_ARCHIVE), so `./tests/all_tests` just works after a build.
//
// Two deliberate limits. An already-set VKOP_SHADER_ARCHIVE always wins, so a
// session can still point at another archive. And this is dev-only: shipping
// binaries (llm_driver, vkbench, a converted model's consumer) do NOT include
// this header, because there the model's embedded subset must be the source --
// a baked-in build-tree path would both mask a broken embedding and make the
// binary silently read a file the deployment doesn't have.
inline void seed_shader_archive_from_build_tree() {
#ifdef VKOP_DEV_SHADER_ARCHIVE
    if (std::getenv("VKOP_SHADER_ARCHIVE") == nullptr &&
        ::access(VKOP_DEV_SHADER_ARCHIVE, R_OK) == 0) {
        ::setenv("VKOP_SHADER_ARCHIVE", VKOP_DEV_SHADER_ARCHIVE, 1);
    }
#endif
}

} // namespace dev
} // namespace vkop

#endif // INCLUDE_DEV_SHADERS_HPP_
