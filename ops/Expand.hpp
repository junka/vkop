// Copyright 2026 @junka
#ifndef OPS_EXPAND_HPP_
#define OPS_EXPAND_HPP_

#include "core/Tensor.hpp"
#include "ops/BufferBase.hpp"
#include "ops/Operator.hpp"
#include <cmath>
#include <numeric>

extern "C" {
extern unsigned char buffer_expand_spv[];
extern unsigned int buffer_expand_spv_len;
}
namespace vkop {
namespace ops {

namespace expand {
struct GpuExpandParam {
    ivec4 inshape;
    uint32_t shape_length;
    int fp16;
};
} // namespace expand

// SSBO-only op: broadcasts input to the given output shape.
class Expand : public Operator {
  public:
    explicit Expand()
        : Operator(OpType::EXPAND, buffer_expand_spv, buffer_expand_spv_len,
                   {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                   sizeof(expand::GpuExpandParam)) {
        param_.fp16 = 0;
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {

        std::vector<int> inshape = inputs[0]->getShape();
        std::vector<int> out_shape = outputs[0]->getShape();
        if (out_shape.size() == 0) {
            dispatch_by_dtype(inputs[1]->dtype(), [&](auto dummy) {
                using T = decltype(dummy);
                auto shape_input = core::as_tensor<T>(inputs[1]);
                shape_input->copyToCPU(m_cmdpool_);
                auto num = shape_input->size();
                out_shape.resize(num);
                for (int i = 0; i < num; ++i) {
                    out_shape[i] = static_cast<int>(shape_input->data()[i]);
                }
                shape_input->copyToGPU(m_cmdpool_);
            });
        }

        // int64 data: CPU broadcast (part of the shape meta-chain). The target
        // shape (inputs[1]) is the authoritative source — do NOT trust a
        // recycled output's stale shape. inputs[1] is CPU-resident here.
        if (inputs[0]->dtype() == typeid(int64_t)) {
            auto shape_input = core::as_tensor<int64_t>(inputs[1]);
            if (!shape_input->has_cpu_data()) {
                shape_input->copyToCPU(m_cmdpool_);
            }
            out_shape.resize(shape_input->num_elements());
            for (int i = 0; i < shape_input->num_elements(); ++i) {
                out_shape[i] = static_cast<int>((*shape_input)[i]);
            }
            int total = total_elems(out_shape);
            std::vector<int64_t> out(total);
            auto src = core::as_tensor<int64_t>(inputs[0]);
            printf("[expandint64] src cpu=%d gpu=%d size=%d inshape=[",
                   (int)src->has_cpu_data(), (int)src->has_gpu_buffer(),
                   (int)src->num_elements());
            for (auto d : inshape)
                printf("%d,", d);
            printf("] out_shape=[");
            for (auto d : out_shape)
                printf("%d,", d);
            printf("] total=%d\n", total);
            fflush(stdout);
            if (!src->has_cpu_data()) {
                src->copyToCPU(m_cmdpool_);
            }
            for (int i = 0; i < total; ++i) {
                out[i] = (*src)[broadcast_index(inshape, out_shape, i)];
            }
            auto output = core::as_tensor<int64_t>(outputs[0]);
            output->resize(out_shape);
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
                param_.fp16 = 1;
            }
        });

        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto input = core::as_tensor<T>(inputs[0]);
            auto input_buffer = input->as_storage_buffer(m_dev_, m_cmd_);
            objs_.emplace_back(input_buffer);
        });
        dispatch_by_dtype(inputs[1]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto shapeinput = core::as_tensor<T>(inputs[1]);
            auto input_buffer = shapeinput->as_storage_buffer(m_dev_, m_cmd_);
            objs_.emplace_back(input_buffer);
        });

        auto total_size = std::accumulate(out_shape.begin(), out_shape.end(), 1,
                                          std::multiplies<>());
        for (size_t i = 0; i < out_shape.size(); ++i) {
            if (static_cast<int>(inshape.size() - i - 1) >= 0) {
                param_.inshape[out_shape.size() - i - 1] =
                    inshape[inshape.size() - i - 1];
            } else {
                param_.inshape[out_shape.size() - i - 1] = 1;
            }
        }
        param_.shape_length = static_cast<uint32_t>(out_shape.size());
        submit(&param_, UP_DIV(total_size, 256), 1, 1);
    }

    expand::GpuExpandParam param_;
};

} // namespace ops
} // namespace vkop
#endif // OPS_EXPAND_HPP_
