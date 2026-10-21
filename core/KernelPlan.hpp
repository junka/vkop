// Copyright 2025 @junka
#ifndef CORE_KERNEL_PLAN_HPP_
#define CORE_KERNEL_PLAN_HPP_

#include <string>
#include <vector>

namespace vkop {
namespace core {

// The kernel selection for one op_type, as the converter wrote it into the
// .vkopbin (Model.kernel_plans, referenced by Node.plan).
//
// Before this existed the runtime *derived* each op's archive key from its own
// OpType -> name table (core/ShaderKeys.hpp) while the converter derived the
// same key independently to decide what to embed (shader_map.py). Two
// derivations of one decision, and since libvkop carries no built-in SPIR-V a
// disagreement is not a graceful fallback to a linked-in C array any more, it
// is a hard failure when the pipeline is created. Reading the answer out of the
// file removes the second derivation.
//
// The four key fields are post-fallback, i.e. exactly what shader_key_for()
// returns for that (backend, precision) pair: the backend's own basename when
// the op has one, otherwise the other backend's (with that backend's prefix),
// and the fp16 spelling only when the op is in DUAL_FP16_SHADERS — so an op
// without a dual build repeats its fp32 key in the *_fp16 field rather than
// leaving it empty. An empty field means the op genuinely has no shader.
//
// This mirrors `shader_map.plan_for_op` field for field, and
// tests/KernelPlanTest.cpp checks a converted model's rows against
// shader_key_for() so the two sides cannot drift.
struct KernelPlan {
    std::string op_type;
    std::string image;
    std::string image_fp16;
    std::string buffer;
    std::string buffer_fp16;
    // Secondary pipelines the op may build by literal key through
    // Operator::resolve_spv() (Matmul's coop/coop16/coop_i8/pack, the int64
    // twin of Expand/Concat/Gather/Equal/Where, ...). Over-inclusive on
    // purpose: which one runs is a device-feature or input-dtype question, so
    // the list is "what the embedded archive was built from", not a promise
    // that any particular entry is used.
    std::vector<std::string> variants;

    // The archive key for this request, or "" when the op has no shader at
    // all (which is what keeps host-only ops on their ctor-assigned nullptr).
    const std::string &key(bool backend_buffer, int fp16) const {
        return backend_buffer ? (fp16 ? buffer_fp16 : buffer)
                              : (fp16 ? image_fp16 : image);
    }

    bool has_variant(const std::string &k) const {
        for (const auto &v : variants) {
            if (v == k) {
                return true;
            }
        }
        return false;
    }

    // True when no key at all is recorded, i.e. the row is useless and the
    // caller should fall back to the built-in table.
    bool empty() const {
        return image.empty() && image_fp16.empty() && buffer.empty() &&
               buffer_fp16.empty();
    }
};

} // namespace core
} // namespace vkop

#endif // CORE_KERNEL_PLAN_HPP_
