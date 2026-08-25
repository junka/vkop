// Copyright 2025 @junka
#ifndef OPS_BUFFER_BINARY_FACTORY_HPP_
#define OPS_BUFFER_BINARY_FACTORY_HPP_

#include "ops/BufferBase.hpp"
#include <string>

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
            // ONNX broadcasting: a 0 dimension means an empty tensor (e.g. the
            // kv_len=0 past_key_values at round0 prefill). Propagate the 0 so
            // the result is empty and the per-element loop is skipped — pairing
            // a 0 with a 1 and taking max(0,1)=1 would instead read element 0
            // of an empty tensor and crash. (Under the -1 sentinel scheme, 0
            // unambiguously means empty for both data and shape-meta tensors;
            // -1 = dynamic never reaches a live tensor's dims_ because the
            // runtime's has_dyn check skips reshape_view on sentinel-bearing
            // recorded shapes.)
            if (dim1 == 0 || dim2 == 0) {
                result[i] = 0;
            } else if (dim1 == 1 || dim2 == 1) {
                result[i] = std::max(dim1, dim2);
            } else if (dim1 == dim2) {
                result[i] = dim1;
            } else {
                std::string sa, sb;
                for (int d : shape1)
                    sa += std::to_string(d) + ",";
                for (int d : shape2)
                    sb += std::to_string(d) + ",";
                throw std::runtime_error(
                    "Shapes are not broadcast-compatible: [" + sa + "] vs [" +
                    sb + "]");
            }
        }
        return result;
    }

  private:
    // Host-compute an int64 elementwise binary op with right-aligned
    // broadcasting. All int64 producers (Shape/NonZero/Equal/Where/Gather/
    // Concat/Range/Slice/Reshape/Transpose/Div/Mul/Add) are CPU ops during
    // the synchronous recording pass, and int64 initializers are
    // CPU-resident, so both inputs are valid via as_tensor<int64_t>().
    // ONNX int64 ops have no half2/uint packing; the data is uploaded to the
    // output SSBO for downstream GPU readers (no GPU op consumes int64 data
    // itself, but the buffer must exist for uniform handling).
    void cpuComputeInt64(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) {
        auto shape_a = inputs[0]->getShape();
        auto shape_b = inputs[1]->getShape();
        auto out_shape = computeBroadcastShape(shape_a, shape_b);
        int total = total_elems(out_shape);

        auto a = core::as_tensor<int64_t>(inputs[0]);
        auto b = core::as_tensor<int64_t>(inputs[1]);
        // int64 inputs may be GPU-resident only (e.g. NonZero's shader-computed
        // output leaves data_ empty). Pull them to the host before (*a)[i].
        if (!a->has_cpu_data()) {
            a->copyToCPU(m_cmdpool_);
        }
        if (!b->has_cpu_data()) {
            b->copyToCPU(m_cmdpool_);
        }

        std::vector<int64_t> out(total);
        for (int i = 0; i < total; ++i) {
            int64_t av = (*a)[broadcast_index(shape_a, out_shape, i)];
            int64_t bv = (*b)[broadcast_index(shape_b, out_shape, i)];
            switch (type_) {
            case OpType::ADD:
                out[i] = av + bv;
                break;
            case OpType::SUB:
                out[i] = av - bv;
                break;
            case OpType::MUL:
                out[i] = av * bv;
                break;
            case OpType::DIV: // ONNX int64 Div is C-style truncation
                out[i] = av / bv;
                break;
            case OpType::POW: { // never reached for int64 in the LLM, keep sane
                int64_t r = 1;
                for (int64_t e = 0; e < bv; ++e) {
                    r *= av;
                }
                out[i] = r;
                break;
            }
            default:
                throw std::runtime_error("int64 binary op not supported");
            }
        }

        auto output = core::as_tensor<int64_t>(outputs[0]);
        output->resize(out_shape);
        output->fillToCPU(out);
        objs_.emplace_back(output->as_storage_buffer(m_dev_, m_cmd_));
        // Explicit src keeps the CPU copy alive for downstream as_tensor<>()
        // readers (copyToGPU would clear data_ otherwise).
        output->copyToGPU(m_cmdpool_, out.data());
    }

    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        // int64 data flows through a synchronous CPU path (see
        // cpuComputeInt64). In the LLM every binary node's inputs are either
        // all-int64 or all-float, so gating on the data input's dtype is safe.
        if (inputs[0]->dtype() == typeid(int64_t)) {
            cpuComputeInt64(inputs, outputs);
            return;
        }

        auto shape_a = inputs[0]->getShape();
        auto shape_b = inputs[1]->getShape();
        auto out_shape = computeBroadcastShape(shape_a, shape_b);
        int total = total_elems(out_shape);

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
