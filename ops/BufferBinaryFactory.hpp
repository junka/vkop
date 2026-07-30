// Copyright 2025 @junka
#ifndef OPS_BUFFER_BINARY_FACTORY_HPP_
#define OPS_BUFFER_BINARY_FACTORY_HPP_

#include "ops/BufferBase.hpp"

// Shared base for elementwise binary buffer (SSBO) ops with ONNX
// right-aligned broadcasting + optional post-fuse activation. Mirrors the
// image path's BinaryFactory: each per-op leaf (AddBuffer, SubBuffer, ...)
// derives from this and passes its own `buffer_<op>_spv` symbol; the
// shared execute() binds the output + two input SSBOs, fills the
// BinaryElemPC push constant (broadcast shapes + activation + total), and
// dispatches one thread per output element (fp32) or per uint word (fp16).

namespace vkop {
namespace ops {

class BufferBinaryFactory : public BufferFactory {
  public:
    BufferBinaryFactory(OpType type, uint8_t *spv, uint32_t spv_len, int fp16,
                        BufferActivation activation = BufferActivation::NONE)
        : BufferFactory(type, spv, spv_len,
                        {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                         DESCRIPTOR_TYPE_STORAGE},
                        sizeof(BinaryElemPC), fp16),
          activation_(static_cast<int>(activation)) {}

  protected:
    static std::vector<int>
    computeBroadcastShape(const std::vector<int> &shape1,
                          const std::vector<int> &shape2) {
        size_t max_dims = std::max(shape1.size(), shape2.size());
        std::vector<int> result(max_dims, 1);
        for (int i = static_cast<int>(max_dims) - 1; i >= 0; --i) {
            int idx1 = i - (max_dims - shape1.size());
            int idx2 = i - (max_dims - shape2.size());
            int dim1 = (idx1 >= 0) ? shape1[idx1] : 1;
            int dim2 = (idx2 >= 0) ? shape2[idx2] : 1;
            if (dim1 == 1 || dim2 == 1) {
                result[i] = std::max(dim1, dim2);
            } else if (dim1 == dim2) {
                result[i] = dim1;
            } else {
                throw std::runtime_error("Shapes are not broadcast-compatible");
            }
        }
        return result;
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto shape_a = inputs[0]->getShape();
        auto shape_b = inputs[1]->getShape();
        auto out_shape = computeBroadcastShape(shape_a, shape_b);
        int total = total_elems(out_shape);

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->size() == 0) {
                output->resize(out_shape);
            }
            bind_ssbo<T>(outputs[0], /*is_output=*/true);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            bind_ssbo<T>(inputs[0], /*is_output=*/false);
        });
        dispatch_by_dtype(inputs[1]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            bind_ssbo<T>(inputs[1], /*is_output=*/false);
        });

        BinaryElemPC pc{};
        pc.rank = static_cast<int>(out_shape.size());
        fill_dims(pc.outDims, out_shape);
        fill_dims_broadcast(pc.in0Dims, shape_a, pc.rank);
        fill_dims_broadcast(pc.in1Dims, shape_b, pc.rank);
        pc.activation = activation_;
        pc.broadcast = (shape_a == out_shape && shape_b == out_shape) ? 0 : 1;
        pc.total = total;
        int nthreads = (fp16_ != 0) ? (total + 1) / 2 : total;
        submit(&pc, UP_DIV(nthreads, 256), 1, 1);
    }

    int activation_;
};

} // namespace ops
} // namespace vkop

#endif // OPS_BUFFER_BINARY_FACTORY_HPP_
