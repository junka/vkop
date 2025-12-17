// Copyright 2025 @junka
#ifndef OPS_MATMUL_HPP_
#define OPS_MATMUL_HPP_

#include "Operator.hpp"

extern unsigned char matmul_spv[];
extern unsigned int matmul_spv_len;

namespace vkop {
namespace ops {

namespace matmul {
struct alignas(16) GpuMatMulParam {
    int M;
    int N;
    int K;
};
} // namespace matmul
class MatMul : public Operator {
  public:
    MatMul()
        : Operator(OpType::MATMUL, matmul_spv, matmul_spv_len,
                   sizeof(matmul::GpuMatMulParam)) {
        n_imgs_ = 3;
        types_ = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER};
        objs_.reserve(types_.size());
    };

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
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
            });
        }
        matmul::GpuMatMulParam para;
        para.M = m;
        para.N = n;
        para.K = k;

        submit(&para, UP_DIV(n, 16), UP_DIV(m, 16), 1);
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_MATMUL_HPP_
