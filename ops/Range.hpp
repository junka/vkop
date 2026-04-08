// Copyright 2026 @junka
#ifndef OPS_RANGE_HPP_
#define OPS_RANGE_HPP_

#include "core/Tensor.hpp"
#include "ops/Operator.hpp"
#include <cmath>
#include <numeric>

extern "C" {
extern unsigned char range_spv[];
extern unsigned int range_spv_len;
}
namespace vkop {
namespace ops {
namespace range {
struct GpuRangeParam {
    bool fp16;
};
} // namespace range

class Range : public Operator {
  public:
    explicit Range()
        : Operator(OpType::RANGE, range_spv, range_spv_len,
                   {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                   sizeof(range::GpuRangeParam)) {
        param_.fp16 = false;
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {

        std::vector<int> out_shape = outputs[0]->getShape();
        if (out_shape.empty()) {
            // all tensor contains scalar value
            dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
                using T = decltype(dummy);
                auto input0 = core::as_tensor<T>(inputs[0]);
                input0->copyToCPU(m_cmdpool_);
                auto start = input0->at(0);
                auto input1 = core::as_tensor<T>(inputs[1]);
                input1->copyToCPU(m_cmdpool_);
                auto limit = input1->at(0);
                auto input2 = core::as_tensor<T>(inputs[2]);
                input2->copyToCPU(m_cmdpool_);
                auto delta = input2->at(0);
                int inums = static_cast<int>(
                    std::ceil((limit - start) / std::abs(delta)));
                out_shape.push_back(inums);
            });
        }

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->size() == 0) {
                output->resize(out_shape);
            }
            auto output_buffer = output->as_storage_buffer(m_dev_, m_cmd_);
            objs_.emplace_back(output_buffer);
            if (typeid(uint16_t) == typeid(T)) {
                param_.fp16 = true;
            }
        });

        for (const auto &in : inputs) {
            dispatch_by_dtype(in->dtype(), [&](auto dummy) {
                using T = decltype(dummy);
                auto input = core::as_tensor<T>(in);
                auto input_buffer = input->as_storage_buffer(m_dev_, m_cmd_);
                objs_.emplace_back(input_buffer);
            });
        }

        auto total_size = std::accumulate(out_shape.begin(), out_shape.end(), 1,
                                          std::multiplies<>());
        submit(&param_, UP_DIV(total_size, 256), 1, 1);
    }

    range::GpuRangeParam param_;
};

} // namespace ops
} // namespace vkop
#endif // OPS_RANGE_HPP_
