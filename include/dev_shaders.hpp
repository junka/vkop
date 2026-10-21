// Copyright 2025 @junka
#ifndef INCLUDE_DEV_SHADERS_HPP_
#define INCLUDE_DEV_SHADERS_HPP_

#include "core/ShaderArchive.hpp"

namespace vkop {
namespace dev {

// Point a development binary that builds ops outside Runtime::LoadModel at
// this build tree's shader archive.
//
// libvkop ships no built-in SPIR-V: its own `*_spv[]` arrays are 1-byte
// placeholders, so every shader byte comes from an archive. A model carries its
// own (onnx2vkop always embeds one), but the unit
// tests' torch fixtures, the probes and bench_matmul construct operators
// straight through ops::create_from_type(), so they have no model to take
// shaders from and die on the stub array instead. Operator falls back to
// ShaderArchive::process_default() in exactly that case, and this installs
// that fallback from the path CMake baked in (VKOP_DEV_SHADER_ARCHIVE), so
// `./tests/all_tests` just works after a build.
//
// Dev-only, and that is the whole point: shipping binaries (llm_driver,
// vkbench, a converted model's consumer) do NOT include this header, so the
// model's embedded subset is the only source they can have. Nothing an
// operator can set at run time -- no environment variable, no CLI flag --
// can substitute a different set of kernels, which is what makes a profiled
// run the same thing as a deployed run.
inline void seed_shader_archive_from_build_tree() {
#ifdef VKOP_DEV_SHADER_ARCHIVE
    vkop::core::ShaderArchive::set_process_default(VKOP_DEV_SHADER_ARCHIVE);
#endif
}

} // namespace dev
} // namespace vkop

#endif // INCLUDE_DEV_SHADERS_HPP_
