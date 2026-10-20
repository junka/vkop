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
// with no model and no dev archive, whose stub then fails by name.
const vkop::core::ShaderArchive *
effective_archive(const vkop::core::ShaderArchive *own) {
    return own ? own : vkop::core::ShaderArchive::process_default();
}
} // namespace

bool Operator::resolve_stubbed_primary_spv() {
    const auto *archive = effective_archive(shader_archive_);
    if (!archive) {
        return false;
    }
    const std::string key =
        core::shader_key_for(type_, uses_buffer_backend(), fp16_);
    if (key.empty()) {
        // Host-only op (Shape, SqueezeUnsqueeze) or a variant the key table
        // does not name: nothing to resolve, keep the caller's throw.
        return false;
    }
    auto [spv, len] = archive->lookup(key);
    if (!spv) {
        if (std::getenv("VKOP_SHADER_DEBUG")) {
            std::fprintf(stderr,
                         "[shaders] archive has no key=%s for op=%s — this "
                         "archive does not cover the op\n",
                         key.c_str(), name_.c_str());
        }
        return false;
    }
    override_spv(spv, len);
    if (std::getenv("VKOP_SHADER_DEBUG")) {
        std::fprintf(stderr, "[shaders] self-resolved key=%s len=%u op=%s\n",
                     key.c_str(), len, name_.c_str());
    }
    return spv_len_ > 0;
}

std::pair<const uint32_t *, uint32_t>
Operator::resolve_spv(const std::string &key, const unsigned char *embedded,
                      unsigned int embedded_len) const {
    const auto *archive = effective_archive(shader_archive_);
    if (archive) {
        auto r = archive->lookup(key);
        if (r.first) {
            return r;
        }
    }
    // A stub is what VKOP_STRIP_BUILTIN_SHADERS=ON leaves behind (spv_to_c.py
    // writes `X_spv[1] = {0}` with a zero length), and it is not a shader: the
    // caller would build a zero-length module and fail inside VulkanPipeline
    // with no hint as to which key went missing. Every in-tree caller feeds
    // the pair straight into a pipeline, so failing here is strictly clearer.
    if (embedded == nullptr || embedded_len == 0) {
        throw std::runtime_error(
            "vkop: " +
            (name_.empty() ? std::string(convert_optype_to_string(type_))
                           : name_) +
            " needs the shader variant '" + key +
            "' and nothing supplied it: this libvkop's copy is a stub "
            "(VKOP_STRIP_BUILTIN_SHADERS=ON is the default) and the archive "
            "the .vkopbin carries does not have that key. Reconvert the model "
            "with onnx2vkop -- it embeds every key its kernel plan lists -- "
            "against a complete vkop_shaders.bin (--master-archive).");
    }
    return {reinterpret_cast<const uint32_t *>(
                const_cast<unsigned char *>(embedded)),
            embedded_len};
}

} // namespace ops
} // namespace vkop
