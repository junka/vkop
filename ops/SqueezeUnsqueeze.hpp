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

// "[1,2]" for the Unsqueeze axis-range error below.
inline std::string axes_str(const std::vector<int> &axes) {
    std::string s = "[";
    for (size_t i = 0; i < axes.size(); ++i)
        s += std::to_string(axes[i]) + (i + 1 == axes.size() ? "" : ",");
    return s + "]";
}

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
    // Single-impl op: always the SSBO path (no image twin).
    bool uses_buffer_backend() const override { return true; }

    explicit SqueezeUnsqueeze(bool unsqueeze)
        : Operator(unsqueeze ? OpType::UNSQUEEZE : OpType::SQUEEZE, {}),
          unsqueeze_(unsqueeze) {}

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("axes") != attributes.end()) {
            auto ax = parse_attr_list<int>(attributes.at("axes"));
            axes_attr_.assign(ax.begin(), ax.end());
        }
    }

    void invalidate_shape_cache() override {
        // Nothing to reset: the axes are read once when they are a
        // read-only leaf and re-read every round otherwise. Kept as an
        // explicit no-op for the virtual contract.
    }
    bool has_host_side_effects() const override { return true; }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        // Host work here: the axes-input readback and the host-only fallback
        // copy below. The GPU-alias path — every LLM instance — reads no bytes
        // and dispatches nothing; it is a pure metadata view.
        //
        // A read-only leaf axes tensor makes this a one-time read (see the
        // branch below); anything else is read every round. Stated here rather
        // than at the early returns so every path has an answer.
        set_did_host_work_last_round(inputs.size() > 1 && inputs[1] &&
                                     !inputs[1]->is_read_only_leaf());
        auto inshape = inputs[0]->getShape();
        int rank = static_cast<int>(inshape.size());

        // Resolve axes: input[1] (int64) if present, else the attribute. The
        // axes input is a tiny int64 tensor that is a read-only leaf in
        // practice (e.g. [-1], [0,2] fixed by the graph).
        std::vector<int> axes;
        if (inputs.size() > 1 && inputs[1]) {
            // Structural fast path: a host-authoritative axes tensor (an
            // initializer, or a graph input the driver filled once) cannot
            // change for its lifetime — the loader only sets that flag for
            // tensors no node writes. So the axes are a CONSTANT of this node,
            // not something to learn: read once, cache, and report no host
            // work forever after.
            //
            // This is the same answer the LEARNING/CONFIRMING/STABLE machine
            // below eventually reaches, but by construction rather than by
            // observation. That distinction matters: the machine only advances
            // when execute() runs, and a segment that is already being
            // re-recorded every round may not reach it (a miss segment
            // re-submits the previous recording instead of re-running its
            // ops), so a predicate that depends on the machine can stay
            // "dynamic" forever. A structural answer cannot deadlock.
            if (inputs[1]->get_host_authoritative()) {
                if (!axes_from_constant_) {
                    auto ax = core::as_tensor<int64_t>(inputs[1]);
                    ax->copyToCPU(m_cmdpool_);  // free: host bytes are current
                    constant_axes_.reserve(ax->num_elements());
                    for (int i = 0; i < ax->num_elements(); ++i)
                        constant_axes_.push_back(static_cast<int>((*ax)[i]));
                    axes_from_constant_ = true;
                }
                axes = constant_axes_;
            } else {
                // GPU-produced axes: read them. No learning — the values are
                // needed to build the output shape either way, and the machine
                // that used to cache them only advanced inside execute(), so it
                // could get stuck (see the constant-axes branch above).
                auto ax = core::as_tensor<int64_t>(inputs[1]);
                // Unconditional readback: a cross-round-recycled GPU input may
                // have stale CPU data_ (see the data-input fix below / the
                // ScatterElements fix — same stale-data_ class).
                ax->copyToCPU(m_cmdpool_);
                axes.reserve(ax->num_elements());
                for (int i = 0; i < ax->num_elements(); ++i)
                    axes.push_back(static_cast<int>((*ax)[i]));
            }
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
            for (int a : axes) {
                // An axis outside [-out_rank, out_rank) means the graph is
                // invalid — in practice the input's rank was lost, because the
                // .vkopbin recorded no shape for this node's input (a shape
                // chain built on an ONNX graph with no value_info). Fail here
                // instead of reading past the end of inshape below.
                if (a < -out_rank || a >= out_rank) {
                    throw std::runtime_error(
                        "Unsqueeze " + get_name() + ": axis " +
                        std::to_string(a) + " out of range for input rank " +
                        std::to_string(rank) + " (axes=" + axes_str(axes) +
                        "); shape chain lost its ranks, re-run ONNX shape "
                        "inference before converting");
                }
                norm.insert(a < 0 ? a + out_rank : a);
            }
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

        // View: same bytes, new shape. The element count is unchanged
        // (squeeze removes only 1-dims; unsqueeze only adds 1-dims), so the
        // byte copy is a pure metadata change.
        //
        // GPU-alias fast path: when the input is GPU-resident (has a
        // VulkanBuffer), the output can SHARE the input's VkBuffer — same
        // bytes, just a different logical shape. This eliminates the ~3.2ms
        // copyToCPU readback that dominated decode (122 Unsqueeze × 3.4ms =
        // 422ms). The output's shape (out_shape, set by resize) drives
        // downstream dispatch; the underlying bytes are identical to the
        // input's.
        //
        // CPU fallback: when the input is host-only (no vkobj_), there is no
        // buffer to alias — pull input to host, resize, copy bytes, re-upload.
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto input = core::as_tensor<T>(inputs[0]);
            auto output = core::as_tensor<T>(outputs[0]);
            output->resize(out_shape);

            if (input->has_gpu_buffer()) {
                // Alias the input's GPU buffer: no readback, no re-upload. The
                // view op preserves element count, so the input's buffer (sized
                // for input->num_elements() == output->num_elements()) is
                // exactly large enough.
                auto src_buff = std::dynamic_pointer_cast<VulkanBuffer>(
                    input->as_storage_buffer(m_dev_, m_cmd_));
                objs_.emplace_back(
                    output->alias_storage_buffer(src_buff, m_cmd_));
                // A byte-preserving view of a read-only leaf is still a leaf:
                // the bytes are the same object, so a consumer may
                // treat the output as a constant too.
                if (input->is_read_only_leaf())
                    output->set_read_only_leaf();
                // Host-shape mode: this is a pure view (element count is
                // unchanged), so propagate the input's authoritative host
                // bytes to the output — downstream host readers then skip
                // their GPU->CPU readback.
                if (host_shape_enabled() && input->get_host_authoritative()) {
                    const std::vector<T> &src = input->data();
                    size_t n =
                        std::min<size_t>(output->num_elements(), src.size());
                    output->fillToCPU(
                        std::vector<T>(src.begin(), src.begin() + n));
                    output->set_host_authoritative();
                }
                return;
            }

            // Host-only input: copy bytes through the CPU. Always read back
            // from the GPU when a buffer exists (defensive; the
            // has_gpu_buffer() gate above already routed GPU inputs to the
            // alias path, so this is the genuinely host-only branch). copyToCPU
            // is a no-op-safe authority check (reads when vkobj_ exists, else
            // just reserveOnCPU).
            input->copyToCPU(m_cmdpool_);
            // Element count must match (view op); copy the host data straight
            // across. fillToCPU sets data_ to the provided values. Guard the
            // count: a mismatched out_shape (dynamic-dim divergence between
            // the live input and the converter-recorded shape) would make
            // output->num_elements() exceed input's actual data size, and the
            // src.begin()+N read / fillToCPU write would overrun (heap
            // corruption that later surfaces as a segfault in unrelated
            // command-buffer code). Clamp to the input's real element count.
            const std::vector<T> &src = input->data();
            size_t n = std::min<size_t>(output->num_elements(), src.size());
            std::vector<T> dst(src.begin(), src.begin() + n);
            output->fillToCPU(dst);
            objs_.emplace_back(output->as_storage_buffer(m_dev_, m_cmd_));
            output->copyToGPUDeferred(m_cmd_);
            if (host_shape_enabled() && inputs[0]->dtype() == typeid(int64_t)) {
                output->set_host_authoritative();
            }
        });
    }

    // Axes read once from a host-authoritative input. The bytes cannot change,
    // so this is a constant of the node — no learning, no per-round readback.
    std::vector<int> constant_axes_;
    bool axes_from_constant_ = false;

    bool unsqueeze_;
    std::vector<int> axes_attr_;

    // Axes are read once when the input is a read-only leaf, and every round
    // otherwise. See execute().
};

} // namespace ops
} // namespace vkop
#endif // OPS_SQUEEZE_UNSQUEEZE_HPP_
