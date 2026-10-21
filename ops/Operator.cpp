// Copyright 2025 @junka
// Out-of-line definitions for Operator's methods that need the full
// ShaderArchive type. Operator.hpp forward-declares the class so ops headers
// don't have to pull in model/load.hpp; anything that dereferences the
// archive pointer lives here.

#include "ops/Operator.hpp"

#include <cstdio>
#include <cstdlib>

#include "core/ShaderArchive.hpp"
#include "core/ShaderKeys.hpp"

namespace vkop {
namespace ops {

namespace {
// The archive this op should consult: the pointer Runtime set when it built
// the op from a model file, otherwise the process-wide one dev binaries
// install (include/dev_shaders.hpp). Null when neither applies -- an op built
// with no model and no dev archive, which then fails by name.
const vkop::core::ShaderArchive *
effective_archive(const vkop::core::ShaderArchive *own) {
    return own ? own : vkop::core::ShaderArchive::process_default();
}

// The failure every unresolved lookup produces. libvkop links in no SPIR-V, so
// the only fix is a model whose archive covers the key.
[[noreturn]] void missing_shader(const std::string &name,
                                 const std::string &key, const char *what) {
    throw std::runtime_error(
        "vkop: " + name + " needs the shader " + what + " '" + key +
        "' and nothing supplied it: this libvkop carries no built-in SPIR-V "
        "and the archive the .vkopbin carries does not have that key. "
        "Reconvert the model with onnx2vkop -- it embeds every key its kernel "
        "plan lists -- against a complete vkop_shaders.bin "
        "(--master-archive).");
}
} // namespace

void Operator::resolve_primary_spv() {
    const std::string key =
        core::shader_key_for(type_, uses_buffer_backend(), fp16_);
    if (key.empty()) {
        // Host-only op (Shape, SqueezeUnsqueeze): no pipeline to build.
        return;
    }
    const auto *archive = effective_archive(shader_archive_);
    if (archive) {
        auto [spv, len] = archive->lookup(key);
        if (spv) {
            override_spv(spv, len);
            if (std::getenv("VKOP_SHADER_DEBUG")) {
                std::fprintf(stderr,
                             "[shaders] self-resolved key=%s len=%u op=%s\n",
                             key.c_str(), len, name_.c_str());
            }
            return;
        }
        if (std::getenv("VKOP_SHADER_DEBUG")) {
            std::fprintf(stderr,
                         "[shaders] archive has no key=%s for op=%s -- this "
                         "archive does not cover the op\n",
                         key.c_str(), name_.c_str());
        }
    }
    missing_shader(name_.empty() ? std::string(convert_optype_to_string(type_))
                                 : name_,
                   key, "variant");
}

std::pair<const uint32_t *, uint32_t>
Operator::resolve_spv(const std::string &key) const {
    const auto *archive = effective_archive(shader_archive_);
    if (archive) {
        auto r = archive->lookup(key);
        if (r.first) {
            return r;
        }
    }
    missing_shader(name_.empty() ? std::string(convert_optype_to_string(type_))
                                 : name_,
                   key, "variant");
}

} // namespace ops
} // namespace vkop
