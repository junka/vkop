// Copyright 2026 @junka
#ifndef OPS_EXPAND_HPP_
#define OPS_EXPAND_HPP_

#include "core/Tensor.hpp"
#include "ops/BufferBase.hpp"
#include "ops/Operator.hpp"
#include <cmath>
#include <cstdlib>
#include <numeric>

extern "C" {
extern unsigned char buffer_expand_spv[];
extern unsigned int buffer_expand_spv_len;
extern unsigned char buffer_expand_fp16_spv[];
extern unsigned int buffer_expand_fp16_spv_len;
}
namespace vkop {
namespace ops {

namespace expand {
struct GpuExpandParam {
    int rank;
    int fp16;
    int inDims[8];
    int outDims[8];
    int _pad0;
    int _pad1;
};
} // namespace expand

// SSBO-only op: broadcasts input to the given output shape.
class Expand : public Operator {
  public:
    explicit Expand(int fp16 = 0)
        : Operator(OpType::EXPAND,
                   fp16 ? buffer_expand_fp16_spv : buffer_expand_spv,
                   fp16 ? buffer_expand_fp16_spv_len : buffer_expand_spv_len,
                   {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                   sizeof(expand::GpuExpandParam), fp16) {
        param_.fp16 = fp16 ? 1 : 0;
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
            std::vector<int> target_shape(shape_input->num_elements());
            for (int i = 0; i < shape_input->num_elements(); ++i) {
                target_shape[i] = static_cast<int>((*shape_input)[i]);
            }
            // ONNX Expand output shape = right-aligned broadcast of input vs
            // target: dim is the input dim when it is neither 1 nor -1 (a
            // concrete value, including 0=empty), else the target dim. This
            // matters for the NonZero->Transpose->Expand shape-meta chain:
            // an empty source [0,1] expanded to target [1,2048] must yield
            // [0,2048] (empty), NOT [1,2048] — otherwise we read OOB from the
            // 0-element source and feed garbage downstream (Scatter indices).
            size_t maxd = std::max(inshape.size(), target_shape.size());
            out_shape.assign(maxd, 1);
            for (size_t i = 0; i < maxd; ++i) {
                int id =
                    (i < inshape.size()) ? inshape[inshape.size() - 1 - i] : 1;
                int td = (i < target_shape.size())
                             ? target_shape[target_shape.size() - 1 - i]
                             : 1;
                int v = std::max(id, td);
                if (td == 0 || id == 0) {
                    v = 0; // either side concrete-empty -> empty
                } else if (id == -1) {
                    v = td; // input dynamic -> take concrete target
                }
                out_shape[maxd - 1 - i] = v;
            }
            int total = total_elems(out_shape);
            std::vector<int64_t> out(static_cast<size_t>(total));
            auto src = core::as_tensor<int64_t>(inputs[0]);
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

        // ONNX Expand: the shape input (inputs[1]) is the *target* shape, but
        // the real output shape is the element-wise max of the input shape and
        // the target shape (right-aligned broadcasting): an input dim larger
        // than the target dim is kept. E.g. input [1,1,64,1] expanded to target
        // [3,1,1,1] yields output [3,1,64,1]. The host-computed out_shape (from
        // graph shape inference) can be stale/wrong, so recompute it here from
        // the authoritative target buffer + the input shape.
        std::vector<int> target_shape;
        {
            auto sh = core::as_tensor<int64_t>(inputs[1]);
            if (!sh->has_cpu_data())
                sh->copyToCPU(m_cmdpool_);
            target_shape.resize(sh->num_elements());
            for (int i = 0; i < sh->num_elements(); ++i)
                target_shape[i] = static_cast<int>((*sh)[i]);
        }
        size_t maxd = std::max(inshape.size(), target_shape.size());
        out_shape.assign(maxd, 1);
        for (size_t i = 0; i < maxd; ++i) {
            int id = (i < inshape.size()) ? inshape[inshape.size() - 1 - i] : 1;
            int td = (i < target_shape.size())
                         ? target_shape[target_shape.size() - 1 - i]
                         : 1;
            int v = std::max(id, td);
            // target_shape comes from a runtime int64 buffer (inputs[1]) so its
            // 0s are genuine shape values (an empty target dim → empty output).
            // id (the input's own shape) is a live dims_ value: under the -1
            // sentinel scheme, 0 = concrete empty (propagate), -1 = dynamic
            // (shouldn't reach here post-has_dyn, but guard anyway → take the
            // concrete target). Only a real target 0 (td==0) or a
            // concrete-empty input (id==0) propagates an empty dim.
            if (td == 0 || id == 0) {
                v = 0; // either side concrete-empty -> empty
            } else if (id == -1) {
                v = td; // input dynamic -> take concrete target
            }
            out_shape[maxd - 1 - i] = v;
        }
        // Resize the output tensor to the correct broadcasted shape.
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            output->resize(out_shape);
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
        // Fill input + output shapes into the push constant as left-aligned
        // 8-int dim arrays (matches buffer_common.comp's fill_dims convention;
        // the shader uses dims[0..rank-1]). Supports up to rank 8 — the LLM
        // attention key/value Expand broadcasts a 5-D [1,8,1,1,128] ->
        // [1,8,8,1, 128], which the old ivec4 path silently truncated. The
        // shader reads output dims from the push constant — NOT from the
        // (int64, possibly- mismatched) shape buffer — so broadcasting is
        // computed on the host where the dtypes are known.
        param_.rank = static_cast<int>(out_shape.size());
        fill_dims(param_.outDims, out_shape);
        fill_dims_broadcast(param_.inDims, inshape, param_.rank);
        // fp16 packs two elements per uint word; dispatch one thread per word
        // (the shader writes each word once — no read-modify-write race).
        int nthreads = (fp16_ != 0) ? (total_size + 1) / 2 : total_size;
        submit(&param_, UP_DIV(nthreads, 256), 1, 1);
    }

    expand::GpuExpandParam param_;
};

} // namespace ops
} // namespace vkop
#endif // OPS_EXPAND_HPP_
