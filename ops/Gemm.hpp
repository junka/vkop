// Copyright 2025 @junka
#ifndef OPS_GEMM_HPP_
#define OPS_GEMM_HPP_

#include "ops/Conv2d.hpp"
#include "ops/Operator.hpp"
#include <numeric>
#include <string>

extern "C" {
extern unsigned char buffer_gemm_spv[];
extern unsigned int buffer_gemm_spv_len;
}
namespace vkop {
namespace ops {

namespace gemm {
struct alignas(16) GpuGemmParam {
    int M;
    int N;
    int K;
    // Defaulted: a Gemm without bias never assigns it, and an indeterminate
    // byte in the push range would make the shader add C[j] from a dummy
    // buffer.
    int has_bias = 0;
    int transA;
    int transB;
    float alpha;
    float beta;
    int fp16a;
    int fp16b;
    int fp16c;
    int fp16o;
    int activation;
    // 1 = B is a byte-packed int8 weight with a per-output-column fp32 scale at
    // binding 4. The kernel then ignores fp16b — the payload is neither format.
    int weight_int8 = 0;
};
} // namespace gemm

// SSBO-only op: Y = alpha * A' * B' + beta * C with optional fused
// activation. Supports per-tensor fp16/fp32 mixing, and int8 weight-only B
// (dequantized in the kernel).
class Gemm : public Operator {
  public:
    Gemm()
        : Operator(OpType::GEMM, buffer_gemm_spv, buffer_gemm_spv_len,
                   {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                    DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                    DESCRIPTOR_TYPE_STORAGE},
                   sizeof(gemm::GpuGemmParam)) {
        para_.alpha = 1.0F;
        para_.beta = 1.0F;
        para_.transA = 0;
        para_.transB = 0;
        para_.activation = static_cast<int>(conv2d::ActivationMode::NONE);
    }

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("alpha") != attributes.end()) {
            auto alpha = std::stof(attributes.at("alpha"));
            para_.alpha = alpha;
        }
        if (attributes.find("beta") != attributes.end()) {
            auto beta = std::stof(attributes.at("beta"));
            para_.beta = beta;
        }
        if (attributes.find("transA") != attributes.end()) {
            para_.transA = std::stol(attributes.at("transA"));
        }
        if (attributes.find("transB") != attributes.end()) {
            para_.transB = std::stol(attributes.at("transB"));
        }

        if (attributes.find("activation") != attributes.end()) {
            std::string activation = attributes.at("activation");
            if (activation == "Relu") {
                para_.activation =
                    static_cast<int>(conv2d::ActivationMode::RELU);
            } else if (activation == "Sigmoid") {
                para_.activation =
                    static_cast<int>(conv2d::ActivationMode::SIGMOID);
            } else if (activation == "Tanh") {
                para_.activation =
                    static_cast<int>(conv2d::ActivationMode::TANH);
            } else if (activation == "HardSwish") {
                para_.activation =
                    static_cast<int>(conv2d::ActivationMode::HARDSWISH);
            } else if (activation == "Mish") {
                para_.activation =
                    static_cast<int>(conv2d::ActivationMode::MISH);
            } else if (activation == "Relu6") {
                para_.activation =
                    static_cast<int>(conv2d::ActivationMode::RELU6);
            } else if (activation == "Swish") {
                para_.activation =
                    static_cast<int>(conv2d::ActivationMode::SWISH);
            } else {
                throw std::invalid_argument("Unsupported activation: " +
                                            activation);
            }
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        // Element formats and input slots. A, the bias C and the output must be
        // float or half; B (inputs[1]) may also be an int8 weight, whose
        // per-output-column dequant scale the optimizer appends as the LAST
        // input — the same convention Conv2d uses:
        //   fp32/fp16, no bias : [A, B]
        //   fp32/fp16, w/ bias : [A, B, C]
        //   int8,     no bias : [A, B_int8, scale]
        //   int8,     w/ bias : [A, B_int8, C, scale]
        // Keying the bias slot off inputs.size() alone would read the scale as
        // a bias, so both are derived from the weight's format, not its count.
        const bool weight_int8 =
            inputs[1]->elem_kind() == core::ElemKind::kInt8;
        const bool has_bias =
            weight_int8 ? (inputs.size() == 4) : (inputs.size() > 2);
        const size_t bias_index = 2;
        const size_t scale_index = weight_int8 ? (has_bias ? 3 : 2) : 0;
        if (inputs.size() > 4 || (!weight_int8 && inputs.size() > 3)) {
            throw std::runtime_error("vkop: Gemm got " +
                                     std::to_string(inputs.size()) +
                                     " inputs; expected [A, B], [A, B, C], "
                                     "[A, B_int8, scale] or [A, B_int8, C, "
                                     "scale]");
        }
        core::require_float_elem(inputs[0]->elem_kind(), "Gemm", "input 0");
        if (!weight_int8) {
            core::require_float_elem(inputs[1]->elem_kind(), "Gemm", "input 1");
        }
        if (has_bias) {
            core::require_float_elem(inputs[bias_index]->elem_kind(), "Gemm",
                                     "bias");
        }
        core::require_float_elem(outputs[0]->elem_kind(), "Gemm", "output");
        if (weight_int8 &&
            inputs[scale_index]->elem_kind() != core::ElemKind::kFloat32) {
            throw std::runtime_error(
                std::string(
                    "vkop: Gemm int8 dequant scale must be float32, got ") +
                core::elem_name(inputs[scale_index]->elem_kind()));
        }

        int m = inputs[0]->getShape()[0];
        int n = inputs[1]->getShape()[1];
        int k = inputs[0]->getShape()[1];
        if (para_.transA) {
            m = inputs[0]->getShape()[1];
            k = inputs[0]->getShape()[0];
        }
        if (para_.transB) {
            n = inputs[1]->getShape()[0];
            k = inputs[1]->getShape()[1];
        }
        // The scale indexes Y's columns, so its length IS N. A mismatch means
        // the weight was quantized along the wrong axis (a [N, K] weight read
        // as [K, N], which is what a transposed export produces), and no kernel
        // can recover the intended values afterwards. size() is bytes and the
        // scale was just proven float32, so 4 bytes per entry.
        if (weight_int8) {
            const size_t entries =
                inputs[scale_index]->size() /
                core::elem_bytes(core::ElemKind::kFloat32, 1);
            if (entries != static_cast<size_t>(n)) {
                throw std::runtime_error(
                    "vkop: Gemm int8 scale has " + std::to_string(entries) +
                    " entries but the output has " + std::to_string(n) +
                    " columns — the weight was quantized along the wrong axis");
            }
        }

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto t) {
            using T = decltype(t);
            auto outputptr = core::as_tensor<T>(outputs[0]);
            if (outputptr->size() == 0) {
                outputptr->resize(std::vector<int>{m, n});
            }
            auto output_buffer = outputptr->as_storage_buffer(m_dev_);
            objs_.emplace_back(output_buffer);
        });
        // bindings 1..4: A, B, C-or-dummy, scale-or-dummy — bound by slot, not
        // by walking inputs, because the int8 layout shifts the bias.
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto t) {
            using T = decltype(t);
            objs_.emplace_back(
                core::as_tensor<T>(inputs[0])->as_storage_buffer(m_dev_));
        });
        dispatch_by_dtype(inputs[1]->dtype(), [&](auto t) {
            using T = decltype(t);
            objs_.emplace_back(
                core::as_tensor<T>(inputs[1])->as_storage_buffer(m_dev_));
        });
        if (has_bias) {
            dispatch_by_dtype(inputs[bias_index]->dtype(), [&](auto t) {
                using T = decltype(t);
                objs_.emplace_back(core::as_tensor<T>(inputs[bias_index])
                                       ->as_storage_buffer(m_dev_));
            });
        } else {
            objs_.emplace_back(dummy_buffer_);
        }
        if (weight_int8) {
            dispatch_by_dtype(inputs[scale_index]->dtype(), [&](auto t) {
                using T = decltype(t);
                objs_.emplace_back(core::as_tensor<T>(inputs[scale_index])
                                       ->as_storage_buffer(m_dev_));
            });
        } else {
            objs_.emplace_back(dummy_buffer_);
        }

        para_.M = m;
        para_.N = n;
        para_.K = k;
        para_.has_bias = has_bias ? 1 : 0;
        para_.fp16a =
            inputs[0]->elem_kind() == core::ElemKind::kFloat16 ? 1 : 0;
        para_.fp16b =
            inputs[1]->elem_kind() == core::ElemKind::kFloat16 ? 1 : 0;
        para_.fp16c = has_bias && inputs[bias_index]->elem_kind() ==
                                      core::ElemKind::kFloat16
                          ? 1
                          : 0;
        para_.fp16o =
            outputs[0]->elem_kind() == core::ElemKind::kFloat16 ? 1 : 0;
        para_.weight_int8 = weight_int8 ? 1 : 0;

        if (para_.fp16o == 1) {
            submit(&para_, UP_DIV(n, 32), UP_DIV(m, 16), 1);
        } else {
            submit(&para_, UP_DIV(n, 16), UP_DIV(m, 16), 1);
        }
    }

    gemm::GpuGemmParam para_;
};

} // namespace ops
} // namespace vkop
#endif // OPS_GEMM_HPP_
