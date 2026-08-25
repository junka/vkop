// Copyright 2026 @junka
#ifndef OPS_SQUEEZE_UNSQUEEZE_HPP_
#define OPS_SQUEEZE_UNSQUEEZE_HPP_

#include "core/Tensor.hpp"
#include "ops/Operator.hpp"
#include <algorithm>
#include <cstdlib>
#include <set>

namespace vkop {
namespace ops {

// CPU-only view ops: ONNX Squeeze / Unsqueeze. Both are pure shape-metadata
// changes — the element bytes are unchanged, only the logical shape (and thus
// rank) differs. Implemented on the host: pull the input, recompute the
// squeezed/unsqueezed shape, resize the output, and re-upload the same bytes.
//
// Why a runtime op (not folded at conversion): the onnx2vkop FusionOptimizer
// folds Squeeze/Unsqueeze by aliasing the output name to the input and
// propagating the post-view shape to downstream consumers' recorded shapes.
// That fold is only safe when the input shape is fully concrete (a baked
// initializer) — for dynamic inputs (e.g. /Concat_5_output_0 with a symbolic
// kv_len) the recorded shape carries -1 sentinels, the runtime's has_dyn
// guard skips reshape_view, and the downstream op reads the input's live
// (pre-view) shape — wrong rank → wrong broadcast (the rotary Expand
// [1,8,1,128]→[1,8,2,1,128] corruption). Keeping the node + implementing it
// here makes the view real regardless of symbolic dims.
//
// Axes come from input[1] (opset>=13, int64) or the "axes" attribute
// (opset<13). Empty axes: Squeeze removes ALL size-1 dims; Unsqueeze with no
// axes is not valid ONNX (guard: treat as no-op).
class SqueezeUnsqueeze : public Operator {
  public:
    explicit SqueezeUnsqueeze(bool unsqueeze)
        : Operator(unsqueeze ? OpType::UNSQUEEZE : OpType::SQUEEZE, nullptr, 0,
                   {}),
          unsqueeze_(unsqueeze) {}

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("axes") != attributes.end()) {
            auto ax = parse_attr_list<int>(attributes.at("axes"));
            axes_attr_.assign(ax.begin(), ax.end());
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto inshape = inputs[0]->getShape();
        int rank = static_cast<int>(inshape.size());

        // Resolve axes: input[1] (int64) if present, else the attribute.
        std::vector<int> axes;
        if (inputs.size() > 1 && inputs[1]) {
            auto ax = core::as_tensor<int64_t>(inputs[1]);
            if (!ax->has_cpu_data())
                ax->copyToCPU(m_cmdpool_);
            axes.reserve(ax->num_elements());
            for (int i = 0; i < ax->num_elements(); ++i)
                axes.push_back(static_cast<int>((*ax)[i]));
        } else {
            axes = axes_attr_;
        }

        std::vector<int> out_shape;
        if (unsqueeze_) {
            // Insert a size-1 dim at each axis. Per ONNX, axes index into the
            // OUTPUT rank (rank(input)+len(axes)), so negatives normalize
            // against the output rank, NOT the input rank. E.g.
            // Unsqueeze([1,1,20], axes=[-1]) -> output rank 4, axis -1 -> 3 ->
            // [1,1,20,1] (NOT [1,1,1,20], which the old input-rank
            // normalization produced — that misordered the last dim and broke
            // the rotary ScatterND index concat).
            int nd = rank;
            int out_rank = nd + static_cast<int>(axes.size());
            std::set<int> norm;
            for (int a : axes)
                norm.insert(a < 0 ? a + out_rank : a);
            out_shape.reserve(out_rank);
            int ai = 0;
            for (int i = 0; i < out_rank; ++i) {
                if (norm.count(i))
                    out_shape.push_back(1);
                else
                    out_shape.push_back(inshape[ai++]);
            }
        } else {
            // Remove each axis dim (normalize negatives, sort, dedup). Empty
            // axes: remove all size-1 dims.
            std::set<int> norm;
            for (int a : axes)
                norm.insert(a < 0 ? a + rank : a);
            if (norm.empty()) {
                for (int d : inshape)
                    if (d != 1)
                        out_shape.push_back(d);
            } else {
                for (int i = 0; i < rank; ++i)
                    if (!norm.count(i))
                        out_shape.push_back(inshape[i]);
            }
        }

        // View: same bytes, new shape. Pull input to host, resize output,
        // copy the bytes, re-upload. The element count is unchanged
        // (squeeze removes only 1-dims; unsqueeze only adds 1-dims), so the
        // byte copy is exact.
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto input = core::as_tensor<T>(inputs[0]);
            if (!input->has_cpu_data())
                input->copyToCPU(m_cmdpool_);
            auto output = core::as_tensor<T>(outputs[0]);
            output->resize(out_shape);
            // Element count must match (view op); copy the host data straight
            // across. fillToCPU sets data_ to the provided values.
            const std::vector<T> &src = input->data();
            std::vector<T> dst(src.begin(),
                               src.begin() + output->num_elements());
            output->fillToCPU(dst);
            objs_.emplace_back(output->as_storage_buffer(m_dev_, m_cmd_));
            output->copyToGPU(m_cmdpool_, dst.data());
        });
    }

    bool unsqueeze_;
    std::vector<int> axes_attr_;
};

} // namespace ops
} // namespace vkop
#endif // OPS_SQUEEZE_UNSQUEEZE_HPP_
