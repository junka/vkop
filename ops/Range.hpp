// Copyright 2026 @junka
#ifndef OPS_RANGE_HPP_
#define OPS_RANGE_HPP_

#include "core/Tensor.hpp"
#include "ops/BufferBase.hpp"
#include "ops/Operator.hpp"
#include <cmath>
#include <numeric>

extern "C" {
extern unsigned char buffer_range_spv[];
extern unsigned int buffer_range_spv_len;
}
namespace vkop {
namespace ops {

namespace range {
struct GpuRangeParam {
    bool fp16;
};
} // namespace range

// SSBO-only op: generates a 1-D sequence [start, start+delta, ...].
class Range : public Operator {
  public:
    explicit Range()
        : Operator(OpType::RANGE, buffer_range_spv, buffer_range_spv_len,
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

        // int64 range runs on the CPU (all 6 instances are part of the shape
        // meta-chain, e.g. position indices). Inputs are CPU-resident scalars
        // during recording. int64 arithmetic: delta may be negative; count =
        // max(0, ceil((limit-start)/delta)) with the div rounding away from
        // zero per ONNX (delta and limit-start always share sign in practice).
        if (inputs[0]->dtype() == typeid(int64_t)) {
            auto start = core::as_tensor<int64_t>(inputs[0])->at(0);
            auto limit = core::as_tensor<int64_t>(inputs[1])->at(0);
            auto delta = core::as_tensor<int64_t>(inputs[2])->at(0);
            int64_t range = limit - start;
            int64_t inums = 0;
            if (range == 0) {
                inums = 0;
            } else if (range > 0) {
                inums = (delta > 0) ? (range + delta - 1) / delta : 0;
            } else {
                inums = (delta < 0) ? (range - delta - 1) / delta : 0;
            }
            out_shape = {static_cast<int>(inums)};
            std::vector<int64_t> out(static_cast<size_t>(inums));
            for (int64_t i = 0; i < inums; ++i) {
                out[static_cast<size_t>(i)] = start + i * delta;
            }
            auto output = core::as_tensor<int64_t>(outputs[0]);
            if (output->size() == 0) {
                output->resize(out_shape);
            }
            output->fillToCPU(out);
            objs_.emplace_back(output->as_storage_buffer(m_dev_, m_cmd_));
            output->copyToGPU(m_cmdpool_, out.data());
            return;
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
