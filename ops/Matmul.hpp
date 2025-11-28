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
        n_imgs_ = 0;
        types_ = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
        objs_.reserve(types_.size());
    };

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        int m = inputs[0]->getShape()[0];
        int n = inputs[1]->getShape()[1];
        int k = inputs[0]->getShape()[1];
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto t) {
            using T = decltype(t);
            auto outputptr = core::as_tensor<T>(outputs[0]);
            if (outputptr->size() == 0) {
                outputptr->resize(std::vector<int>{m, n});
            }
            auto output_buffer = outputptr->as_storage_buffer(m_dev_);
            objs_.emplace_back(output_buffer);
        });
        for (const auto &input : inputs) {
            dispatch_by_dtype(input->dtype(), [&](auto t) {
                using T = decltype(t);
                auto inputptr = core::as_tensor<T>(input);
                auto input_buffer = inputptr->as_storage_buffer(m_dev_);
                inputptr->copyToGPU(m_cmdpool_);
                objs_.emplace_back(input_buffer);
            });
        }
        matmul::GpuMatMulParam para;
        para.M = m;
        para.N = n;
        para.K = k;

        submit(&para, UP_DIV(n, 16), UP_DIV(m, 16), 1);
    }

  private:
};

} // namespace ops
} // namespace vkop
#endif // OPS_MATMUL_HPP_
