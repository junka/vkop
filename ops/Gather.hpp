// Copyright 2026 @junka
#ifndef OPS_GATHER_HPP_
#define OPS_GATHER_HPP_

#include "core/Tensor.hpp"
#include "ops/BufferBase.hpp"
#include "ops/Operator.hpp"
#include <cstdlib>
#include <numeric>

extern "C" {
extern unsigned char buffer_gather_spv[];
extern unsigned int buffer_gather_spv_len;
extern unsigned char buffer_gather_fp16_spv[];
extern unsigned int buffer_gather_fp16_spv_len;
}
namespace vkop {
namespace ops {

namespace gather {
// Push-constant layout mirrors shaders/buffer/gather.comp (std430). Shapes are
// left-aligned 8-int arrays (fill_dims pads trailing slots with 1). The old
// ivec4 layout capped data rank at 4; the IArr8 fields support up to 8-D —
// needed for the 56 LLM KV-cache gathers on 5-D data [1,2,8,kv,128].
struct GpuGatherParam {
    int inShape[8];
    int indicesShape[8];
    int outShape[8];
    int axis;
    int idims;
    int odims;
    int nindex;
};
} // namespace gather
static_assert(sizeof(gather::GpuGatherParam) <= 128,
              "GpuGatherParam PC overflow");

// SSBO-only op: ONNX Gather along an axis. fp16 picks the fp16 spv variant
// (float16_t elements) via DUAL_FP16_SHADERS.
class Gather : public Operator {
  public:
    explicit Gather(int fp16 = 0)
        : Operator(OpType::GATHER,
                   fp16 ? buffer_gather_fp16_spv : buffer_gather_spv,
                   fp16 ? buffer_gather_fp16_spv_len : buffer_gather_spv_len,
                   {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                   sizeof(gather::GpuGatherParam)) {}

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("axis") != attributes.end()) {
            auto axis = std::stol(attributes.at("axis"));
            param_.axis = axis;
        }
    }

  private:
    static std::vector<int>
    calculateGatherOutputShape(const std::vector<int> &data_shape,
                               const std::vector<int> &indices_shape,
                               int axis) {
        if (data_shape.empty()) {
            return indices_shape;
        }
        int rank = static_cast<int>(data_shape.size());
        int normalized_axis = axis;
        if (normalized_axis < 0) {
            normalized_axis += rank;
        }
        if (normalized_axis < 0 || normalized_axis >= rank) {
            throw std::out_of_range("Axis is out of range for data shape");
        }
        std::vector<int> output_shape;
        output_shape.reserve(rank - 1 + indices_shape.size());
        for (int i = 0; i < rank; ++i) {
            if (i == normalized_axis) {
                output_shape.insert(output_shape.end(), indices_shape.begin(),
                                    indices_shape.end());
            } else {
                output_shape.push_back(data_shape[i]);
            }
        }
        return output_shape;
    }

    // Host-compute an int64-data Gather along `axis`. All 352 int64-data
    // gathers in the LLM have int64 indices produced by Shape/Concat (CPU
    // ops during the synchronous recording pass), so both tensors are valid
    // via as_tensor<int64_t>(). Mirrors gather.comp's indexing: for each
    // output linear index, split it into [data dims, indices dims, data dims]
    // around `axis`, read the gathered index value, and map to the input
    // linear index.
    void cpuComputeInt64(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) {
        auto data_shape = inputs[0]->getShape();
        auto idx_shape = inputs[1]->getShape();
        int rank = static_cast<int>(data_shape.size());
        int axis = param_.axis;
        if (axis < 0) {
            axis += rank;
        }
        auto out_shape =
            calculateGatherOutputShape(data_shape, idx_shape, axis);
        int total = total_elems(out_shape);
        int nindex = static_cast<int>(idx_shape.size());
        int axis_size = data_shape[axis];

        auto data = core::as_tensor<int64_t>(inputs[0]);
        auto indices = core::as_tensor<int64_t>(inputs[1]);
        // Unconditional readback: a cross-round-recycled GPU input may have
        // stale CPU data_ (see SqueezeUnsqueeze/ScatterElements fix).
        data->copyToCPU(m_cmdpool_);
        indices->copyToCPU(m_cmdpool_);
        std::vector<int64_t> out(total);

        // Row-major strides of the data and indices tensors.
        std::vector<int> d_stride(rank, 1);
        for (int i = rank - 2; i >= 0; --i) {
            d_stride[i] = d_stride[i + 1] * data_shape[i + 1];
        }
        std::vector<int> i_stride(nindex, 1);
        for (int i = nindex - 2; i >= 0; --i) {
            i_stride[i] = i_stride[i + 1] * idx_shape[i + 1];
        }
        auto nd_to_linear = [](const std::vector<int> &coord,
                               const std::vector<int> &stride) {
            int idx = 0;
            for (size_t d = 0; d < coord.size(); ++d) {
                idx += coord[d] * stride[d];
            }
            return idx;
        };

        // Out dims and the input dimension each maps to, with the coordinate
        // slot it reads from.
        struct Slot {
            int src_rank; // index into data (0) or indices (1) coordinate
            int src_dim;  // dimension within that tensor
            int dim;      // output dimension
        };
        std::vector<Slot> slots;
        slots.reserve(out_shape.size());
        for (int d = 0; d < axis; ++d) {
            slots.push_back({0, d, d});
        }
        for (int d = 0; d < nindex; ++d) {
            slots.push_back({1, d, axis + d});
        }
        for (int d = axis + 1; d < rank; ++d) {
            slots.push_back({0, d, axis + nindex + (d - axis - 1)});
        }

        // For each output element, decompose its linear index into a full
        // output coordinate, then read the gathered index at the indices
        // sub-coordinate and build the data sub-coordinate.
        std::vector<int> out_stride(out_shape.size());
        if (!out_shape.empty()) {
            out_stride.back() = 1;
            for (int i = static_cast<int>(out_shape.size()) - 2; i >= 0; --i) {
                out_stride[i] = out_stride[i + 1] * out_shape[i + 1];
            }
        }
        std::vector<int> data_coord(rank, 0);
        std::vector<int> idx_coord(nindex, 0);
        std::vector<int> out_coord(out_shape.size(), 0);
        for (int o = 0; o < total; ++o) {
            int r = o;
            for (size_t d = 0; d < out_shape.size(); ++d) {
                out_coord[d] = (r / out_stride[d]) % out_shape[d];
            }
            for (const auto &sl : slots) {
                int v = out_coord[sl.dim];
                if (sl.src_rank == 0) {
                    data_coord[sl.src_dim] = v;
                } else {
                    idx_coord[sl.src_dim] = v;
                }
            }
            int gather_idx = (*indices)[nd_to_linear(idx_coord, i_stride)];
            if (gather_idx < 0) {
                gather_idx += axis_size;
            }
            data_coord[axis] = static_cast<int>(gather_idx);
            out[o] = (*data)[nd_to_linear(data_coord, d_stride)];
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

        auto inshape = inputs[0]->getShape();
        auto indshape = inputs[1]->getShape();
        auto rank = inputs[0]->num_dims();
        if (param_.axis < 0) {
            param_.axis += rank;
        }
        std::vector<int> out_shape =
            calculateGatherOutputShape(inshape, indshape, param_.axis);

        // int64 data flows through a synchronous CPU path (see
        // cpuComputeInt64); all 352 int64 gathers in the LLM are part of the
        // shape/position meta-chain.
        if (inputs[0]->dtype() == typeid(int64_t)) {
            cpuComputeInt64(inputs, outputs);
            return;
        }

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            // Always re-seat the output to the runtime out_shape. The output
            // is pre-created at LoadModel from the converter's recorded shape
            // (e.g. [1,8,1,128] for kv_len=1), but at runtime the data may be
            // smaller (kv_len=0 → [1,8,0,128], empty). Without this re-resize
            // the stale 1024-elem buffer is kept and downstream Concat sees a
            // non-empty shape, producing a half-zero, mis-strided result.
            // resize() is a no-op on element count when the shape already
            // matches (same product → same size_); it only trims/grows the
            // logical view. The backing SSBO is reused (as_storage_buffer
            // returns the existing vkobj_ when present).
            output->resize(out_shape);
            auto output_buffer = output->as_storage_buffer(m_dev_, m_cmd_);
            objs_.emplace_back(output_buffer);
        });

        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto input = core::as_tensor<T>(inputs[0]);
            auto input_buffer = input->as_storage_buffer(m_dev_, m_cmd_);
            objs_.emplace_back(input_buffer);
        });

        dispatch_by_dtype(inputs[1]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto input = core::as_tensor<T>(inputs[1]);
            auto input_buffer = input->as_storage_buffer(m_dev_, m_cmd_);
            objs_.emplace_back(input_buffer);
        });

        param_.idims = rank;
        param_.nindex = inputs[1]->getShape().size();
        // Left-align each shape into an 8-int array (trailing slots = 1) to
        // match gather.comp's IArr8 nd<->linear helpers.
        fill_dims(param_.inShape, inshape);
        fill_dims(param_.indicesShape, indshape);
        fill_dims(param_.outShape, out_shape);
        param_.odims = out_shape.size();
        auto total_size = std::accumulate(out_shape.begin(), out_shape.end(), 1,
                                          std::multiplies<>());
        submit(&param_, UP_DIV(total_size, 256), 1, 1);
    }

    gather::GpuGatherParam param_;
};

} // namespace ops
} // namespace vkop
#endif // OPS_GATHER_HPP_
