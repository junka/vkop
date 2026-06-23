// Copyright 2025 @junka
#ifndef OPS_MATMUL_HPP_
#define OPS_MATMUL_HPP_

#include "Operator.hpp"
extern "C" {
extern unsigned char matmul_spv[];
extern unsigned int matmul_spv_len;
extern unsigned char matmul_nv_spv[];
extern unsigned int matmul_nv_spv_len;
extern unsigned char matmul_coop_spv[];
extern unsigned int matmul_coop_spv_len;
}
namespace vkop {
namespace ops {

namespace matmul {

enum class Method {
    BASIC_ARITHMETIC = 0,
    VK_COOPERATE_MATRIX = 1,
    NV_TENSORCORE = 2,
};

struct alignas(16) GpuMatMulParam {
    int M;
    int N;
    int K;
    int C;
    int fp32;
};
} // namespace matmul
class MatMul : public Operator {
  public:
    MatMul(const MatMul &) = delete;
    MatMul &operator=(const MatMul &) = delete;
    MatMul(MatMul &&) = delete;
    MatMul &operator=(MatMul &&) = delete;

    explicit MatMul(int use_tensorcore = 0)
        : Operator(OpType::MATMUL,
                   use_tensorcore == 2
                       ? matmul_nv_spv
                       : (use_tensorcore == 1 ? matmul_coop_spv : matmul_spv),
                   use_tensorcore == 2
                       ? matmul_nv_spv_len
                       : (use_tensorcore == 1 ? matmul_coop_spv_len
                                              : matmul_spv_len),
                   {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER},
                   sizeof(matmul::GpuMatMulParam)) {
        method_ =
            use_tensorcore == 2
                ? matmul::Method::NV_TENSORCORE
                : (use_tensorcore == 1 ? matmul::Method::VK_COOPERATE_MATRIX
                                       : matmul::Method::BASIC_ARITHMETIC);
    };

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        int chan = inputs[0]->get_channel();
        int m = inputs[0]->get_height();
        int n = inputs[1]->get_width();
        int k = inputs[0]->get_width();
        int rank = inputs[0]->num_dims();
        auto shape = inputs[0]->getShape();
        shape[rank - 1] = n;
        shape[rank - 2] = m;
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto t) {
            using T = decltype(t);
            auto outputptr = core::as_tensor<T>(outputs[0]);
            if (outputptr->size() == 0) {
                outputptr->resize(shape);
            }
            auto output_image = outputptr->as_output_image(m_dev_, m_cmd_);
            objs_.emplace_back(output_image);
        });
        for (const auto &input : inputs) {
            dispatch_by_dtype(input->dtype(), [&](auto t) {
                using T = decltype(t);
                auto inputptr = core::as_tensor<T>(input);
                auto input_image = inputptr->as_input_image(m_dev_, m_cmd_);
                objs_.emplace_back(input_image);
                if (typeid(uint16_t) == typeid(T)) {
                    para_.fp32 = 0;
                } else if (typeid(float) == typeid(T)) {
                    para_.fp32 = 1;
                }
            });
        }
        para_.M = m;
        para_.N = n;
        para_.K = k;
        para_.C = chan;
        if (method_ == matmul::Method::VK_COOPERATE_MATRIX) {
            // coop kernel: 16x16 workgroup, 8 subgroups (2 M x 4 N),
            // workgroup output footprint = 16 (M) x 32 (N).
            submit(&para_, UP_DIV(n, 32), UP_DIV(m, 16), UP_DIV(chan, 4));
        } else if (method_ == matmul::Method::NV_TENSORCORE) {
            // nv kernel: same 16x16 workgroup / 8 subgroups / 16x32 footprint
            // as coop.
            submit(&para_, UP_DIV(n, 32), UP_DIV(m, 16), UP_DIV(chan, 4));
        } else {
            submit(&para_, UP_DIV(n, 16), UP_DIV(m, 16), UP_DIV(chan, 4));
        }
    }

    matmul::GpuMatMulParam para_;
    matmul::Method method_ = matmul::Method::BASIC_ARITHMETIC;
};

} // namespace ops
} // namespace vkop
#endif // OPS_MATMUL_HPP_
