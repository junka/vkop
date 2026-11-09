// Runtime's per-op shader dtype decision (core::op_shader_fp16).
//
// Every op whose buffer kernel has an fp16 and an fp32 build must choose that
// build from the tensor it is actually handed, not from the Runtime's
// precision_. The two packings are not interchangeable: fp16 stores two halfs
// per 32-bit word, fp32 one float, so building the wrong one reads the other
// layout. GlobalAveragePool's thread count also follows the same choice, which
// makes a mismatch an out-of-bounds SSBO write rather than merely wrong
// numbers -- and vkbench loads fp16 CNN models with precision 0 fixed, so that
// is a real path, not a hypothetical.
//
// Plain main (no gtest, no torch, no Vulkan): the decision reads dtype() only,
// so it needs no device, which is exactly why it is a free function.

#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include "core/Tensor.hpp"
#include "core/runtime.hpp"
#include "ops/Ops.hpp"

namespace {

int failures = 0;

void check(bool ok, const char *what) {
    std::printf("%s: %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) {
        ++failures;
    }
}

} // namespace

int main() {
    using vkop::core::op_shader_fp16;
    using vkop::core::Tensor;
    using vkop::ops::OpType;

    const std::vector<int> shape = {1, 3, 8, 8};
    auto f16 = std::make_shared<Tensor<uint16_t>>(shape);
    auto f32 = std::make_shared<Tensor<float>>(shape);

    // Regression: GAP must follow its data even when precision_ says otherwise.
    check(op_shader_fp16(OpType::GLOBALAVERAGEPOOL, f16.get(), f16.get(), 0) ==
              1,
          "GAP fp16 data + precision 0 -> fp16 build");
    check(op_shader_fp16(OpType::GLOBALAVERAGEPOOL, f32.get(), f32.get(), 1) ==
              0,
          "GAP fp32 data + precision 1 -> fp32 build");

    // The pre-existing data-driven ops keep doing the same.
    check(op_shader_fp16(OpType::ADD, f16.get(), f16.get(), 0) == 1,
          "Add fp16 data + precision 0 -> fp16 build");
    check(op_shader_fp16(OpType::ADD, f32.get(), f32.get(), 1) == 0,
          "Add fp32 data + precision 1 -> fp32 build");

    // FUSED_ELEMWISE is the one op keyed on the OUTPUT (the terminal of a
    // Cast-crossing chain), so a mixed pair must not follow the input.
    check(op_shader_fp16(OpType::FUSED_ELEMWISE, f16.get(), f32.get(), 1) == 0,
          "FusedElemwise follows output (fp32) over input (fp16)");
    check(op_shader_fp16(OpType::FUSED_ELEMWISE, f32.get(), f16.get(), 0) == 1,
          "FusedElemwise follows output (fp16) over input (fp32)");
    check(op_shader_fp16(OpType::FUSED_ELEMWISE, f32.get(), nullptr, 1) == 0,
          "FusedElemwise with no output -> fp32 build");

    // No data input: nothing to infer from, so precision is the answer.
    check(op_shader_fp16(OpType::GLOBALAVERAGEPOOL, nullptr, nullptr, 1) == 1,
          "no data input -> precision passthrough (1)");
    check(op_shader_fp16(OpType::GLOBALAVERAGEPOOL, nullptr, nullptr, 0) == 0,
          "no data input -> precision passthrough (0)");

    std::printf(failures ? "RESULT: %d FAILURE(S)\n" : "RESULT: all pass\n",
                failures);
    return failures ? 1 : 0;
}
