// Copyright 2026 @junka
#ifndef OPS_WHERE_HPP_
#define OPS_WHERE_HPP_

#include "core/Tensor.hpp"
#include "ops/BufferBase.hpp"
#include "ops/Operator.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

extern "C" {
extern unsigned char buffer_where_spv[];
extern unsigned int buffer_where_spv_len;
}
namespace vkop {
namespace ops {

// SSBO-only op: no image path. Selects elements from X or Y based on a
// boolean condition buffer.
class Where : public Operator {
  public:
    explicit Where()
        : Operator(OpType::WHERE, buffer_where_spv, buffer_where_spv_len,
                   {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                   0) {}

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {

        std::vector<int> out_shape = outputs[0]->getShape();
        // The graph's recorded output shape is often stale for the int64
        // shape-meta chain (symbolic dims resolved to a max, not the runtime
        // value). For int64 Where, recompute the broadcasted output shape from
        // the authoritative inputs (cond/X/Y) — the CPU loop below needs the
        // true shape to broadcast against.
        if (inputs[0]->dtype() == typeid(int64_t)) {
            std::vector<int> bcast;
            for (const auto &in : inputs) {
                auto s = in->getShape();
                if (s.empty())
                    continue;
                if (bcast.empty()) {
                    bcast = s;
                } else {
                    size_t m = std::max(bcast.size(), s.size());
                    std::vector<int> nb(m, 1);
                    for (size_t i = 0; i < m; ++i) {
                        int a = (i < bcast.size()) ? bcast[bcast.size() - 1 - i]
                                                   : 1;
                        int b = (i < s.size()) ? s[s.size() - 1 - i] : 1;
                        nb[m - 1 - i] = std::max(a, b);
                    }
                    bcast = nb;
                }
            }
            if (!bcast.empty())
                out_shape = bcast;
        }
        if (out_shape.empty()) {
            auto inshape = inputs[0]->getShape();
            out_shape = inshape;
        }

        // int64 Where runs on the CPU (all 66 instances are part of the
        // shape meta-chain: cond = Equal int64, X = ConstantOfShape, Y =
        // Concat int64). cond, X, and Y are broadcast against out_shape;
        // cond nonzero selects X, else Y.
        if (inputs[0]->dtype() == typeid(int64_t)) {
            auto cond = core::as_tensor<int64_t>(inputs[0]);
            auto x = core::as_tensor<int64_t>(inputs[1]);
            auto y = core::as_tensor<int64_t>(inputs[2]);
            if (!cond->has_cpu_data()) {
                cond->copyToCPU(m_cmdpool_);
            }
            if (!x->has_cpu_data()) {
                x->copyToCPU(m_cmdpool_);
            }
            if (!y->has_cpu_data()) {
                y->copyToCPU(m_cmdpool_);
            }
            int total = total_elems(out_shape);
            std::vector<int64_t> out(total);
            for (int i = 0; i < total; ++i) {
                int64_t cv = (*cond)[broadcast_index(inputs[0]->getShape(),
                                                     out_shape, i)];
                out[i] = (cv != 0) ? (*x)[broadcast_index(inputs[1]->getShape(),
                                                          out_shape, i)]
                                   : (*y)[broadcast_index(inputs[2]->getShape(),
                                                          out_shape, i)];
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
            if (output->num_elements() != total_elems(out_shape)) {
                output->resize(out_shape);
            }
            auto output_buffer = output->as_storage_buffer(m_dev_, m_cmd_);
            objs_.emplace_back(output_buffer);
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
        submit(nullptr, UP_DIV(total_size, 256), 1, 1);
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_WHERE_HPP_
