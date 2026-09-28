// Copyright 2025 @junka
#ifndef OPS_REDUCEMEAN_HPP_
#define OPS_REDUCEMEAN_HPP_

#include "Operator.hpp"
#include "ops/BufferBase.hpp"
#include "ops/PimplFacade.hpp"
extern "C" {
extern unsigned char buffer_reduce_fp16_spv[];
extern unsigned int buffer_reduce_fp16_spv_len;
}
namespace vkop {
namespace ops {

// ReduceMean is a specialization of Reduce with op=MEAN. The Reduce shader
// already supports MEAN (reduce_op=5), so we just need to wire it up as a
// standalone ONNX operator.
class ReduceMeanBuffer : public BufferFactory {
  public:
    explicit ReduceMeanBuffer(int fp16)
        : BufferFactory(OpType::REDUCEMEAN,
                        fp16 ? buffer_reduce_fp16_spv : nullptr,
                        fp16 ? buffer_reduce_fp16_spv_len : 0,
                        {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                        sizeof(ReducePC), fp16) {}

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("axes") != attributes.end()) {
            axes_ = parse_attr_list<int>(attributes.at("axes"));
        }
        if (attributes.find("keepdims") != attributes.end()) {
            keepdims_ = std::stol(attributes.at("keepdims"));
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto in_shape = inputs[0]->getShape();
        int rank = static_cast<int>(in_shape.size());

        // Compute output shape by reducing specified axes
        std::vector<bool> is_red(rank, false);
        for (int ax : axes_) {
            int a = (ax < 0) ? ax + rank : ax;
            if (a >= 0 && a < rank)
                is_red[a] = true;
        }
        std::vector<int> out_shape;
        for (int i = 0; i < rank; ++i) {
            if (is_red[i]) {
                if (keepdims_ == 1)
                    out_shape.push_back(1);
            } else {
                out_shape.push_back(in_shape[i]);
            }
        }

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->num_elements() != total_elems(out_shape)) {
                output->resize(out_shape);
            }
            bind_ssbo<T>(outputs[0], /*is_output=*/true);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            bind_ssbo<T>(inputs[0], /*is_output=*/false);
        });

        // Build push constant with reduce_op=MEAN (5)
        ReducePC pc{};
        pc.rank = rank;
        fill_dims(pc.inDims, in_shape);
        std::vector<int> out_dims_full(rank);
        for (int i = 0; i < rank; ++i) {
            out_dims_full[i] = is_red[i] ? 1 : in_shape[i];
        }
        fill_dims(pc.outDims, out_dims_full);
        int axes_mask = 0;
        for (int ax : axes_) {
            int a = (ax < 0) ? ax + rank : ax;
            if (a >= 0 && a < rank)
                axes_mask |= (1 << a);
        }
        pc.axes_mask = axes_mask;
        pc.reduce_op = 5; // REDUCE_MEAN
        pc.keepdims = keepdims_;
        int out_total = total_elems(out_shape);
        int nthreads = (fp16_ != 0) ? (out_total + 1) / 2 : out_total;
        submit(&pc, UP_DIV(nthreads, 256), 1, 1);
    }

    std::vector<int> axes_;
    int keepdims_ = 1;
};

// PIMPL façade: always uses buffer SSBO impl (no image variant needed).
class ReduceMean : public PimplFacade {
  public:
    ReduceMean(int fp16, bool backend_buffer)
        : PimplFacade(OpType::REDUCEMEAN) {
        impl_ =
            std::unique_ptr<Operator>(std::make_unique<ReduceMeanBuffer>(fp16));
        (void)backend_buffer; // suppress unused warning - always uses buffer
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_REDUCEMEAN_HPP_
