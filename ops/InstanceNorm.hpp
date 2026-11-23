// Copyright 2026 @junka
#ifndef OPS_INSTANCENORM_HPP_
#define OPS_INSTANCENORM_HPP_

#include "ops/BufferBase.hpp"
#include "ops/PimplFacade.hpp"

#include <memory>
#include <stdexcept>
#include <string>

namespace vkop {
namespace ops {

// Push constant for shaders/buffer/instance_norm.comp. 16-byte scalar prefix,
// then the left-aligned dims[8] shape (the buffer-backend convention, see
// shaders/common/buffer_common.comp).
struct alignas(16) InstanceNormPC {
    float eps;      // ONNX `epsilon` attribute
    int rank;       // input rank, 3..8: [N, C, *spatial]
    int outer_size; // N * C: one workgroup per (n, c) slice
    int channels;   // C = dims[1]; slice wg reads scale/bias[wg % C]
    int dims[8];    // full input shape, left-aligned
};
static_assert(sizeof(InstanceNormPC) == 48, "InstanceNormPC layout drift");
static_assert(sizeof(InstanceNormPC) <= 128, "InstanceNormPC PC overflow");

// Buffer-backend (SSBO) InstanceNormalization.
//
// ONNX InstanceNormalization is also what torch's nn.GroupNorm exports to:
//
//   GroupNorm(C, G) == Reshape [N,C,*spatial] -> [N,G,C/G*spatial]
//                      -> InstanceNormalization(scale=[G], bias=[G], eps)
//                      -> Reshape back -> Mul(weight[C]) -> Add(bias[C])
//
// (verified by exporting a tiny nn.GroupNorm at opset 17: the graph is exactly
// Constant/Reshape/Constant/Constant/InstanceNormalization/Shape/Reshape/Mul/
// Add, and the decomposition reproduces nn.GroupNorm to 4e-16 in float64).
// The Reshape makes the tensor rank-3 [N, G, spatial] with C' = G, so THIS op
// runs GroupNorm unchanged -- normalize over all spatial elements of each
// (n, g). No GroupNorm-specific path exists or is needed; the surrounding
// Reshapes are plain RESHAPE nodes and the per-channel Mul/Add are the
// existing MUL/ADD ops.
//
// Semantics per (n, c): mean and BIASED variance over the slice's spatial
// extent (divide by M, not M-1 -- ONNX's definition), then
//   y = scale[c] * (x - mean) / sqrt(var + eps) + bias[c].
//
// One workgroup per slice, dispatch (N*C, 1, 1). fp32 and fp16 are separate
// shader builds (packed half2 words in the fp16 one); both accumulate in fp32
// -- see the shader's header for why the variance is a second reduction
// around the computed mean rather than the fused E[x^2] - mean^2 form.
//
// Bindings: 0=out, 1=in(x), 2=scale, 3=bias. scale/bias are [C] in x's dtype.
class InstanceNormBuffer : public BufferFactory {
  public:
    explicit InstanceNormBuffer(int fp16)
        : BufferFactory(OpType::INSTANCENORM,
                        {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                         DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                        sizeof(InstanceNormPC), fp16),
          fp16_(fp16) {}

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        // ONNX spells it `epsilon`; accept `eps` too so a converter pass that
        // normalizes attribute names does not silently fall back to 1e-5.
        if (attributes.find("epsilon") != attributes.end()) {
            eps_ = std::stof(attributes.at("epsilon"));
        }
        if (attributes.find("eps") != attributes.end()) {
            eps_ = std::stof(attributes.at("eps"));
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto shape = inputs[0]->getShape();
        const int rank = static_cast<int>(shape.size());
        if (rank < 3 || rank > 8) {
            throw std::runtime_error(
                "InstanceNorm: input must be rank 3..8 [N, C, *spatial], got " +
                core::shape_to_string(shape));
        }
        const int channels = shape[1];
        if (channels <= 0) {
            throw std::runtime_error(
                "InstanceNorm: channel dim is empty in " +
                core::shape_to_string(shape));
        }
        const int batch = shape[0];
        const int outer_size = batch * channels;

        int inner_size = 1;
        for (int i = 2; i < rank; ++i) {
            inner_size *= shape[i];
        }
        if (inner_size <= 0) {
            throw std::runtime_error(
                "InstanceNorm: empty spatial extent in " +
                core::shape_to_string(shape));
        }

        // The shader build is fixed at construction, so the live dtype has to
        // match it: a packed half2 read as float bits (or the reverse) is
        // silent garbage.
        if ((inputs[0]->dtype() == typeid(uint16_t)) != (fp16_ != 0)) {
            throw std::runtime_error(
                "InstanceNorm: input dtype disagrees with the built " +
                std::string(fp16_ != 0 ? "fp16" : "fp32") + " variant");
        }
        // scale and bias are [C] in the same dtype as x. Reading a [C-1] or
        // [C*k] table would walk past its end (or mix channels), so the length
        // is checked rather than trusted.
        if (inputs.size() < 3) {
            throw std::runtime_error(
                "InstanceNorm: needs 3 inputs (x, scale, bias), got " +
                std::to_string(inputs.size()));
        }
        for (int i = 1; i <= 2; ++i) {
            const char *what = (i == 1) ? "scale" : "bias";
            if (inputs[i]->dtype() != inputs[0]->dtype()) {
                throw std::runtime_error(
                    std::string("InstanceNorm: ") + what +
                    " dtype differs from the input");
            }
            const int n = static_cast<int>(inputs[i]->size() /
                                           (fp16_ != 0 ? 2 : 4));
            if (n != channels) {
                throw std::runtime_error(
                    std::string("InstanceNorm: ") + what + " has " +
                    std::to_string(n) + " elements but the input has " +
                    std::to_string(channels) + " channels");
            }
        }
        if (fp16_ != 0 && (inner_size % 2) != 0) {
            // An odd slice starts mid-word: its last half would share a word
            // with the neighbouring slice, and two workgroups would then
            // read-modify-write the same word.
            throw std::runtime_error(
                "InstanceNorm fp16: odd spatial extent straddles packed "
                "half2 words");
        }

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->num_elements() != total_elems(shape)) {
                output->resize(shape);
            }
            bind_ssbo<T>(outputs[0], /*is_output=*/true);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            bind_ssbo<T>(inputs[0], /*is_output=*/false);
        });
        dispatch_by_dtype(inputs[1]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            bind_ssbo<T>(inputs[1], /*is_output=*/false);
        });
        dispatch_by_dtype(inputs[2]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            bind_ssbo<T>(inputs[2], /*is_output=*/false);
        });

        InstanceNormPC pc{};
        pc.eps = eps_;
        pc.rank = rank;
        pc.outer_size = outer_size;
        pc.channels = channels;
        fill_dims(pc.dims, shape);

        submit(&pc, outer_size, 1, 1);
    }

    int fp16_ = 0;
    float eps_ = 1e-5f;
};

// PIMPL façade. Buffer-only: ONNX InstanceNormalization reaches vkop through
// the buffer backend (the target device has no image support), and no image
// kernel exists for it, so asking for one is an error rather than a silent
// fallback to a different op's kernel.
class InstanceNorm : public PimplFacade {
  public:
    InstanceNorm(int fp16, bool backend_buffer)
        : PimplFacade(OpType::INSTANCENORM) {
        if (!backend_buffer) {
            throw std::runtime_error(
                "InstanceNorm is buffer-backend only (no image impl).");
        }
        impl_ = std::unique_ptr<Operator>(
            std::make_unique<InstanceNormBuffer>(fp16));
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_INSTANCENORM_HPP_
