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
extern unsigned char buffer_expand_int64_spv[];
extern unsigned int buffer_expand_int64_spv_len;
}
namespace vkop {
namespace ops {

namespace expand {
// PC layout mirrors shaders/buffer/expand.comp (std430). Shapes are
// left-aligned 8-int arrays (fill_dims pads trailing slots with 1). The `fp16`
// field is the fp16 pack flag on the fp path; on the int64 path (separate
// pipeline, expand_int64.comp) it is REPURPOSED as `total` (output element
// count) — fp16 packing is irrelevant for int64-as-ivec2 data. The int64
// pipeline shares this PC layout/size so only the spv differs.
struct GpuExpandParam {
    int rank;
    int fp16; // fp32/fp16 path: pack flag. int64 path: total (output count).
    int inDims[8];
    int outDims[8];
    int _pad0;
    int _pad1;
};

// Read the ONNX Expand target-shape buffer (inputs[1]) as a vector<int>.
// The shape input is conceptually int64, but callers (tests, older graphs)
// may supply int32 — as_tensor<int64_t> on a Tensor<int> returns null and
// dereferencing it segfaults. Accept any integer dtype by trying the common
// ones; returns an empty vector if none match (caller's maxd() falls back to
// inshape, a safe no-broadcast).
inline std::vector<int>
read_target_shape(const std::shared_ptr<core::ITensor> &t,
                  const std::shared_ptr<VulkanCommandPool> &pool) {
    auto read_as = [&](auto dummy) -> std::vector<int> {
        using U = decltype(dummy);
        auto shaped = core::as_tensor<U>(t);
        if (!shaped)
            return {};
        // Unconditional readback: a cross-round-recycled GPU input may have
        // stale CPU data_ (see SqueezeUnsqueeze/ScatterElements fix).
        shaped->copyToCPU(pool);
        std::vector<int> out(shaped->num_elements());
        for (int i = 0; i < shaped->num_elements(); ++i)
            out[i] = static_cast<int>((*shaped)[i]);
        return out;
    };
    // Try int64 first (ONNX-correct), then int32 (test/legacy). The first
    // non-null cast wins; the others short-circuit.
    auto v = read_as(int64_t{});
    if (!v.empty())
        return v;
    v = read_as(int{});
    if (!v.empty())
        return v;
    return read_as(int32_t{});
}
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

    // Build the int64-data pipeline lazily on first int64 execute. The int64
    // shader (expand_int64.comp) has the same descriptor layout (2× STORAGE:
    // output, input) and PC layout as the fp32 expand, so it builds from the
    // same types_/pc_size_; only the spv differs. Descriptor sets are allocated
    // from the int64 pipeline (sets are pool-specific to a pipeline layout).
    void ensure_int64_pipeline() {
        if (pipeline_int64_)
            return;
        bool use_uab = update_after_bind_ &&
                       m_dev_->is_support_descriptor_update_after_bind();
        pipeline_int64_ = std::make_unique<VulkanPipeline>(
            m_dev_->getLogicalDevice(), types_, pc_size_,
            reinterpret_cast<const uint32_t *>(buffer_expand_int64_spv),
            buffer_expand_int64_spv_len, use_uab, required_subgroup_size_);
        for (auto &ds : m_ds_int64_) {
            ds = pipeline_int64_->allocDescriptorSets();
        }
    }

    // When in int64 mode, bind the int64 pipeline + its descriptor sets
    // instead of the base fp32/fp16 pipeline. The base submit() hardcodes
    // pipeline_/m_ds_; this override redirects them for the int64 dispatch.
    void submit(void *ptr, int width, int height, int layers) override {
        if (!int64_mode_) {
            Operator::submit(ptr, width, height, layers);
            return;
        }
        if (!m_ds_int64_[m_id_]) {
            m_ds_int64_[m_id_] = pipeline_int64_->allocDescriptorSets();
        }
        fillWriteDescriptorSets(m_ds_int64_[m_id_]);
        pipeline_int64_->updateDescriptorSets(writes_);
        m_cmd_->bind(*pipeline_int64_, m_ds_int64_[m_id_]);
        if (ptr) {
            m_cmd_->push_constants(*pipeline_int64_,
                                   static_cast<uint32_t>(pc_size_), ptr);
        }
        m_cmd_->dispatch(width, height, layers);
        if (replay_enabled_) {
            record_fingerprint(ptr, width, height, layers);
        }
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

        // int64 data: GPU broadcast via expand_int64.comp (part of the shape
        // meta-chain). Previously this was a CPU broadcast: read_target_shape
        // (a sync readback) + src->copyToCPU (another readback) + host loop +
        // copyToGPUDeferred. The GPU shader reads the input as ivec2[] and
        // writes the broadcast output as ivec2[] — verbatim 8-byte copies, no
        // 64-bit math needed (Expand is a broadcast indexed copy).
        //
        // The target shape (out dims) is passed via push_constant (outDims) —
        // the runtime resolves it from the target-shape input via the
        // auto-learning cache (decode-stable for most Expands) or a readback
        // (dynamic ones). Either way the CPU knows outDims at dispatch time, so
        // dispatch is exact (UP_DIV(total,256), no over-dispatch).
        if (inputs[0]->dtype() == typeid(int64_t)) {
            // Resolve the target shape from inputs[1] with an auto-learning
            // cache: across decode rounds, most Expand target shapes are
            // round-invariant (only the kv_len-derived ones grow). After two
            // matching readbacks, STABLE skips the readback and reuses the
            // cached target_shape. invalidate_shape_cache() (phase boundary)
            // resets to LEARNING.
            std::vector<int> target_shape;
            bool need_readback = true;
            if (target_cache_state_ == TargetLearnState::STABLE) {
                target_shape = cached_target_;
                need_readback = false;
            }
            if (need_readback) {
                target_shape = expand::read_target_shape(inputs[1], m_cmdpool_);
                if (target_cache_state_ == TargetLearnState::LEARNING) {
                    learned_target_ = target_shape;
                    target_cache_state_ = TargetLearnState::CONFIRMING;
                } else if (target_cache_state_ ==
                           TargetLearnState::CONFIRMING) {
                    if (target_shape == learned_target_) {
                        target_cache_state_ = TargetLearnState::STABLE;
                        cached_target_ = target_shape;
                    } else {
                        target_cache_state_ = TargetLearnState::DYNAMIC;
                    }
                }
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

            ensure_int64_pipeline();
            int64_mode_ = true;

            // Output SSBO (int64 as ivec2 — 8 bytes/elem, same stride).
            auto output = core::as_tensor<int64_t>(outputs[0]);
            output->resize(out_shape);
            objs_.emplace_back(output->as_storage_buffer(m_dev_, m_cmd_));
            // Data input SSBO (nullptr cmd: no barrier — the input was produced
            // by a prior level's shader; the level submit provides ordering).
            auto src = core::as_tensor<int64_t>(inputs[0]);
            objs_.emplace_back(src->as_storage_buffer(m_dev_, nullptr));
            // The base Expand pipeline declares a 3rd STORAGE binding for the
            // target-shape input (used by the fp shader). The int64 shader
            // does NOT read it (outDims come via push_constant), but the
            // descriptor-set layout matches types_ (3× STORAGE), so bind a
            // valid SSBO to keep fillWriteDescriptorSets in bounds. Bind the
            // target tensor's SSBO whatever its dtype (int64 in the LLM; int32
            // in some tests) — dispatch_by_dtype resolves the right as_tensor.
            dispatch_by_dtype(inputs[1]->dtype(), [&](auto dummy) {
                using T = decltype(dummy);
                auto target = core::as_tensor<T>(inputs[1]);
                objs_.emplace_back(target->as_storage_buffer(m_dev_, nullptr));
            });

            param_.rank = static_cast<int>(out_shape.size());
            param_.fp16 = total; // fp16 field repurposed as total on int64 path
            fill_dims(param_.outDims, out_shape);
            fill_dims_broadcast(param_.inDims, inshape, param_.rank);
            submit(&param_, UP_DIV(total, 256), 1, 1);
            int64_mode_ = false;
            // Phase 2: propagate the shape-meta side-channel. When the data
            // input IS shape values (carries shape_ssbo_), the broadcast
            // output is also shape values → tag the output's own SSBO as its
            // shape_ssbo_. rank = out_shape.size() (CPU-known from the
            // broadcast math above, which uses getShape — metadata, not the
            // shape VALUES). Conservative: only propagates along the
            // shape-meta chain.
            if (inputs[0]->has_shape_ssbo()) {
                output->set_shape_ssbo(
                    std::dynamic_pointer_cast<VulkanBuffer>(
                        output->as_storage_buffer(m_dev_, m_cmd_)),
                    static_cast<int>(out_shape.size()));
            }
            return;
        }

        // ONNX Expand: the shape input (inputs[1]) is the *target* shape, but
        // the real output shape is the element-wise max of the input shape and
        // the target shape (right-aligned broadcasting): an input dim larger
        // than the target dim is kept. E.g. input [1,1,64,1] expanded to target
        // [3,1,1,1] yields output [3,1,64,1]. The host-computed out_shape (from
        // graph shape inference) can be stale/wrong, so recompute it here from
        // the authoritative target buffer + the input shape.
        //
        // Auto-learning target-shape cache (mirrors the int64 path above and
        // ReshapeBuffer's cache): most fp Expand target shapes are round-
        // invariant across decode (only kv_len-derived ones grow). After two
        // matching readbacks, STABLE skips copyToCPU (the sync readback) and
        // reuses the cached target_shape. invalidate_shape_cache() resets to
        // LEARNING at the prefill→decode boundary. This is the dominant Expand
        // cost (59 readbacks/round, ~57ms onexec).
        std::vector<int> target_shape;
        bool need_readback = true;
        if (target_cache_state_ == TargetLearnState::STABLE) {
            target_shape = cached_target_;
            need_readback = false;
        }
        if (need_readback) {
            target_shape = expand::read_target_shape(inputs[1], m_cmdpool_);
            if (target_cache_state_ == TargetLearnState::LEARNING) {
                learned_target_ = target_shape;
                target_cache_state_ = TargetLearnState::CONFIRMING;
            } else if (target_cache_state_ == TargetLearnState::CONFIRMING) {
                if (target_shape == learned_target_) {
                    target_cache_state_ = TargetLearnState::STABLE;
                    cached_target_ = target_shape;
                } else {
                    target_cache_state_ = TargetLearnState::DYNAMIC;
                }
            }
            // DYNAMIC: stay DYNAMIC (readback every round, no caching).
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

  public:
    // Phase boundary (prefill->decode): reset the target-shape learning state
    // so the new phase re-learns stability. A target shape stable in prefill
    // may differ in decode (seq collapses), so STABLE can't carry across.
    void invalidate_shape_cache() override {
        target_cache_state_ = TargetLearnState::LEARNING;
    }

  private:
    // --- target-shape (inputs[1]) value cache (runtime auto-learning) ---
    // Across decode rounds, most Expand target shapes are round-invariant
    // (only the kv_len-derived ones grow). Learn stability by comparing two
    // consecutive readbacks, then STABLE skips copyToCPU. See execute() for
    // the state machine.
    enum class TargetLearnState { LEARNING, CONFIRMING, STABLE, DYNAMIC };
    TargetLearnState target_cache_state_ = TargetLearnState::LEARNING;
    std::vector<int> learned_target_; // round-0 target (compared vs round-1)
    std::vector<int> cached_target_;  // STABLE target reused on round 2+

    expand::GpuExpandParam param_{};
    std::unique_ptr<VulkanPipeline> pipeline_int64_;
    VkDescriptorSet m_ds_int64_[vkop::kInflight] = {nullptr};
    bool int64_mode_ = false;
};

} // namespace ops
} // namespace vkop
#endif // OPS_EXPAND_HPP_
