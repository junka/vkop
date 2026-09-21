// Copyright 2026 @junka
#ifndef OPS_RMSNORM_HPP_
#define OPS_RMSNORM_HPP_

#include "ops/BufferBase.hpp"
#include "ops/PimplFacade.hpp"
#include <memory>

extern "C" {
extern unsigned char buffer_rmsnorm_spv[];
extern unsigned int buffer_rmsnorm_spv_len;
extern unsigned char buffer_rmsnorm_fp16_spv[];
extern unsigned int buffer_rmsnorm_fp16_spv_len;
}

namespace vkop {
namespace ops {

// Push constant for shaders/buffer/rmsnorm.comp.
struct alignas(16) RMSNormPC {
    float eps;
    int inner_size; // elements normalized together (head_dim, trailing dims)
    int outer_size; // independent normalization slices = total / inner_size
    int _pad;
};
static_assert(sizeof(RMSNormPC) <= 128, "RMSNormPC PC overflow");

// Buffer-backend (SSBO) RMSNorm. Replaces the 9-op ONNX decomposition
//   Cast(fp16->fp32) -> Pow(2) -> ReduceMean -> Add(eps) -> Sqrt -> Div(1,.)
//   -> Mul(x,.) -> Cast(->fp16) -> Mul(weight)
// with one dispatch computing  rms = mean(x^2);  out = x * rsqrt(rms+eps) * w.
//
// One workgroup per outer slice; the shader does a shared-memory tree reduce
// of sum(x^2) over inner_size elements, then normalizes. fp16 build reads
// packed half2 words and computes in fp32 registers (no host readback — the
// old LayerNorm fp16 path did copyToCPU every call, which would reintroduce
// the readback bottleneck RMSNorm runs 113×/decode round).
//
// Bindings: 0=out, 1=in(x), 2=weight. x and weight share the same dtype
// (fp16 for the LLM decode path; fp32 for tests).
class RMSNormBuffer : public BufferFactory {
  public:
    explicit RMSNormBuffer(int fp16)
        : BufferFactory(OpType::RMSNORM,
                        fp16 ? buffer_rmsnorm_fp16_spv : buffer_rmsnorm_spv,
                        fp16 ? buffer_rmsnorm_fp16_spv_len
                             : buffer_rmsnorm_spv_len,
                        std::vector<VkDescriptorType>{DESCRIPTOR_TYPE_STORAGE,
                                                      DESCRIPTOR_TYPE_STORAGE,
                                                      DESCRIPTOR_TYPE_STORAGE},
                        sizeof(RMSNormPC), fp16),
          fp16_(fp16) {
        update_after_bind_ = true;
    }

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("eps") != attributes.end()) {
            eps_ = std::stof(attributes.at("eps"));
        }
        if (attributes.find("axis") != attributes.end()) {
            axis_ = std::stol(attributes.at("axis"));
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto shape = inputs[0]->getShape();
        int total = total_elems(shape);
        int rank = static_cast<int>(shape.size());
        int ax = axis_;
        if (ax < 0)
            ax += rank;
        // Normalize over dims [axis, rank). inner_size = product of those.
        int inner_size = 1;
        for (int d = ax; d < rank; ++d) {
            inner_size *= shape[d];
        }
        int outer_size = total / inner_size;

        // Output has the same shape as x; resize if the runtime created it
        // with a placeholder (dynamic -1 -> 1 at load).
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto type_tag) {
            using T = decltype(type_tag);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->num_elements() != total) {
                output->resize(shape);
            }
            bind_ssbo<T>(outputs[0], /*is_output=*/true);
        });
        // x
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto type_tag) {
            using T = decltype(type_tag);
            bind_ssbo<T>(inputs[0], /*is_output=*/false);
        });
        // weight (1-D over inner_size, same dtype as x)
        dispatch_by_dtype(inputs[1]->dtype(), [&](auto type_tag) {
            using T = decltype(type_tag);
            bind_ssbo<T>(inputs[1], /*is_output=*/false);
        });

        RMSNormPC pc{};
        pc.eps = eps_;
        pc.inner_size = inner_size;
        pc.outer_size = outer_size;
        pc._pad = 0;
        // One workgroup per outer slice (the reduce is intra-slice shared mem).
        submit(&pc, outer_size, 1, 1);
    }

    int fp16_;
    float eps_ = 1e-6f;
    long axis_ = -1;
};

// PIMPL façade. Buffer-only (per the runtime-op authorization); the image
// backend is not implemented.
class RMSNorm : public PimplFacade {
  public:
    RMSNorm(int fp16, bool backend_buffer) : PimplFacade(OpType::RMSNORM) {
        if (!backend_buffer) {
            throw std::runtime_error(
                "RMSNorm is buffer-backend only (no image impl).");
        }
        impl_ =
            std::unique_ptr<Operator>(std::make_unique<RMSNormBuffer>(fp16));
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_RMSNORM_HPP_
