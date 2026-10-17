// Copyright 2025 @junka
// Out-of-line definitions for Operator's methods that need the full
// ShaderArchive type. Operator.hpp forward-declares the class so ops headers
// don't have to pull in model/load.hpp; anything that dereferences the
// archive pointer lives here.

#include "ops/Operator.hpp"

#include "core/ShaderArchive.hpp"

namespace vkop {
namespace ops {

std::pair<const uint32_t *, uint32_t>
Operator::resolve_spv(const std::string &key, const unsigned char *embedded,
                      unsigned int embedded_len) const {
    if (shader_archive_) {
        auto r = shader_archive_->lookup(key);
        if (r.first) {
            return r;
        }
    }
    return {reinterpret_cast<const uint32_t *>(
                const_cast<unsigned char *>(embedded)),
            embedded_len};
}

} // namespace ops
} // namespace vkop
