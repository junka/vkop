// Copyright 2025 @junka
#ifndef OPS_RESHAPE_HPP_
#define OPS_RESHAPE_HPP_

#include "ops/BufferBase.hpp"
#include "ops/PimplFacade.hpp"
#include <cstdio>
#include <cstdlib>
#include <numeric>

#include "core/Tensor.hpp"
#include "ops/Operator.hpp"
extern "C" {
extern unsigned char image_reshape_spv[];
extern unsigned int image_reshape_spv_len;
extern unsigned char buffer_reshape_spv[];
extern unsigned int buffer_reshape_spv_len;
extern unsigned char buffer_reshape_fp16_spv[];
extern unsigned int buffer_reshape_fp16_spv_len;
}
namespace vkop {
namespace ops {

namespace reshape {
struct GpuReshapeParam {
    ivec4 inImgSize;
    ivec4 outImgSize;
    ivec4 inShape;
    ivec4 outShape;
};

} // namespace reshape

class ReshapeImage : public Operator {
  public:
    explicit ReshapeImage()
        : Operator(OpType::RESHAPE, image_reshape_spv, image_reshape_spv_len,
                   {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER},
                   sizeof(reshape::GpuReshapeParam)) {}
    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("allowzero") != attributes.end()) {
            allowzero_ = std::stol(attributes.at("allowzero"));
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto inshape = inputs[0]->getShape();
        auto shape = core::as_tensor<int64_t>(inputs[1]);
        assert(shape->num_dims() == 1);
        int n = shape->num_elements();

        std::vector<int> dim(n);
        for (int i = 0; i < n; i++) {
            dim[i] = static_cast<int>((*shape)[i]);
        }
        auto total = std::accumulate(inshape.begin(), inshape.end(), 1,
                                     std::multiplies<>());
        for (int i = 0; i < n; i++) {
            if (!allowzero_ && dim[i] == 0) {
                dim[i] = inshape[i];
            }
        }
        for (int i = 0; i < n; i++) {
            if (dim[i] != 0 && dim[i] != -1)
                total = total / dim[i];
        }
        for (int i = 0; i < n; i++) {
            if (dim[i] == -1)
                dim[i] = total;
        }

        bool noop = false;
        if (inshape.size() == dim.size()) {
            noop = true;
            for (size_t i = 0; i < inshape.size(); i++) {
                if (inshape[i] != dim[i]) {
                    noop = false;
                    break;
                }
            }
        }
        if (inshape.size() == 4 && dim.size() == 3) {
            if (inshape[0] == 1 && inshape[1] == dim[0] &&
                inshape[2] == dim[1] && inshape[3] == dim[2]) {
                noop = true;
            }
        } else if (inshape.size() == 3 && dim.size() == 4) {
            if (dim[0] == 1 && dim[1] == inshape[0] && dim[2] == inshape[1] &&
                dim[3] == inshape[2]) {
                noop = true;
            }
        } else if (inshape.size() == 4 && dim.size() == 2) {
            if (inshape[0] == 1 && inshape[1] == 1 && inshape[2] == dim[0] &&
                inshape[3] == dim[1]) {
                noop = true;
            }
        } else if (inshape.size() == 2 && dim.size() == 4) {
            if (dim[0] == 1 && dim[1] == 1 && inshape[0] == dim[2] &&
                inshape[1] == dim[3]) {
                noop = true;
            }
        }

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->num_elements() != total_elems(dim)) {
                output->resize(dim);
            }
            if (dim.size() <= 2) {
                auto output_buff = output->as_storage_buffer(m_dev_);
                objs_.emplace_back(output_buff);
            } else {
                auto output_image = output->as_output_image(m_dev_, m_cmd_);
                objs_.emplace_back(output_image);
            }
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto input = core::as_tensor<T>(inputs[0]);
            if (inputs[0]->num_dims() <= 2) {
                auto input_buff = input->as_storage_buffer(m_dev_);
                objs_.emplace_back(input_buff);
            } else {
                auto input_image = input->as_input_image(m_dev_, m_cmd_);
                objs_.emplace_back(input_image);
            }
        });

        if (noop) {
            // copy directly, could be optimized by preprocess/compiler
            if (inshape.size() < 3) {
                auto output_buff =
                    std::dynamic_pointer_cast<VulkanBuffer>(objs_[0]);
                auto input_buff =
                    std::dynamic_pointer_cast<VulkanBuffer>(objs_[1]);
                input_buff->copyBufferToStageBuffer(m_cmd_->get(),
                                                    output_buff->getBuffer(), 0,
                                                    output_buff->getSize(), 0);
            } else {
                auto output_image =
                    std::dynamic_pointer_cast<VulkanImage>(objs_[0]);
                auto input_image =
                    std::dynamic_pointer_cast<VulkanImage>(objs_[1]);
                input_image->transferReadBarrier(m_cmd_->get());
                output_image->copyImageToImage(m_cmd_->get(), input_image,
                                               {0, 0, 0}, 0);
            }
            return;
        }

        if (dim.size() <= 2) {
            auto output_buff =
                std::dynamic_pointer_cast<VulkanBuffer>(objs_[0]);
            auto input_image = std::dynamic_pointer_cast<VulkanImage>(objs_[1]);
            input_image->copyImageToBuffer(m_cmd_->get(),
                                           output_buff->getBuffer(), 0);
            return;
        }

        auto out_gpu_shape = outputs[0]->getGPUShape();
        auto in_gpu_shape = inputs[0]->getGPUShape();
        reshape::GpuReshapeParam param;
        param.inImgSize[0] = in_gpu_shape[0];
        param.inImgSize[1] = in_gpu_shape[1];
        param.inImgSize[2] = in_gpu_shape[2];
        param.inImgSize[3] = 1;
        param.outImgSize[0] = out_gpu_shape[0];
        param.outImgSize[1] = out_gpu_shape[1];
        param.outImgSize[2] = out_gpu_shape[2];
        param.outImgSize[3] = 1;
        if (inshape.size() == 4) {
            param.inShape[0] = inshape[0];
            param.inShape[1] = inshape[1];
            param.inShape[2] = inshape[2];
            param.inShape[3] = inshape[3];
        } else if (inshape.size() == 3) {
            param.inShape[0] = 1;
            param.inShape[1] = inshape[0];
            param.inShape[2] = inshape[1];
            param.inShape[3] = inshape[2];
        }
        if (n == 4) {
            param.outShape[0] = dim[0];
            param.outShape[1] = dim[1];
            param.outShape[2] = dim[2];
            param.outShape[3] = dim[3];
        } else if (n == 3) {
            param.outShape[0] = 1;
            param.outShape[1] = dim[0];
            param.outShape[2] = dim[1];
            param.outShape[3] = dim[2];
        }
        submit(&param, UP_DIV(out_gpu_shape[0], 16),
               UP_DIV(out_gpu_shape[1], 16), out_gpu_shape[2]);
    }

    int allowzero_ = 0;
};

// Buffer (SSBO, compact row-major) reshape. Data is flat row-major so a
// reshape is a 1:1 copy of `total` contiguous scalars; the shape change is
// metadata-only. fp16 packs two elements per uint word (one thread/word).
class ReshapeBuffer : public BufferFactory {
  public:
    explicit ReshapeBuffer(int fp16)
        : BufferFactory(OpType::RESHAPE,
                        fp16 ? buffer_reshape_fp16_spv : buffer_reshape_spv,
                        fp16 ? buffer_reshape_fp16_spv_len
                             : buffer_reshape_spv_len,
                        {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                        sizeof(ReshapePC), fp16) {}

    // Receive the converter's per-input value_dynamic annotation (option C).
    // Used as a HINT only — the runtime auto-learning cache below is the
    // authority. value_dynamic=true means "likely changes per round" so we
    // skip the 2-round learning warmup and readback every round; false means
    // "likely stable" so we still learn to confirm. Either way, the learned
    // state (STABLE) is what actually skips readback.
    void set_input_value_dynamic(const std::vector<bool> &vd) override {
        shape_value_dynamic_ = (vd.size() > 1) ? vd[1] : true;
    }
    void invalidate_shape_cache() override {
        // Phase boundary (prefill->decode): reset learning state so the new
        // phase re-learns stability. The dims that were stable in prefill may
        // differ in decode (seq collapses), so we can't carry STABLE across.
        cache_state_ = LearnState::LEARNING;
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto in_shape = inputs[0]->getShape();
        auto shape = core::as_tensor<int64_t>(inputs[1]);

        // Resolve the output dim[] from the int64 shape input (inputs[1]).
        //
        // Runtime auto-learning cache (replaces converter-annotation-only
        // caching): across decode rounds, most Reshape shape inputs are
        // round-invariant (125/181 — only 56 kv_len-derived ones grow). Rather
        // than trust the converter's value_dynamic hint, we LEARN by comparing
        // two consecutive rounds' readback results:
        //   LEARNING  (round 0): readback, store as learned_dim_.
        //   CONFIRMING(round 1): readback, compare to learned_dim_.
        //               match  -> STABLE: skip readback on round 2+ (reuse).
        //               differ -> DYNAMIC: readback every round.
        //   STABLE    (round 2+): reuse cached dim[], no copyToCPU stall.
        //   DYNAMIC   (round 2+): readback every round (value changes).
        // invalidate_shape_cache() (prefill->decode boundary) resets to
        // LEARNING so the new phase re-learns. The 3.2ms submit+wait per
        // Reshape is the dominant decode cost (181 × 3.2ms = 586ms); STABLE
        // reshapes skip it entirely.
        std::vector<int> dim;
        bool need_readback = true;
        if (cache_state_ == LearnState::STABLE) {
            // Reuse the previously-read dim[] — no GPU->CPU sync this round.
            dim = cached_dim_;
            need_readback = false;
        }
        if (need_readback) {
            // The shape input may be GPU-resident only (produced by the int64
            // Concat/Gather GPU shader in a prior level). Read it back so
            // data_ is populated for the (*shape)[i] access below — same
            // pattern as Unsqueeze/Slice/Cast/Expand. No-op (reserveOnCPU) if
            // host-only.
            shape->copyToCPU(m_cmdpool_);
            int n = shape->num_elements();
            dim.resize(n);
            for (int i = 0; i < n; ++i) {
                dim[i] = static_cast<int>((*shape)[i]);
            }
            // Advance the learning state machine.
            if (cache_state_ == LearnState::LEARNING) {
                learned_dim_ = dim;
                cache_state_ = LearnState::CONFIRMING;
            } else if (cache_state_ == LearnState::CONFIRMING) {
                if (dim == learned_dim_) {
                    cache_state_ = LearnState::STABLE;
                    cached_dim_ = dim; // stable value to reuse next round
                } else {
                    cache_state_ = LearnState::DYNAMIC;
                }
            }
            // DYNAMIC: stay DYNAMIC (readback every round, no caching).
        }
        int n = static_cast<int>(dim.size());
        int total = total_elems(in_shape);
        // resolve a 0 dim by copying from the input, and -1 from the remainder
        for (int i = 0; i < n; ++i) {
            if (dim[i] == 0 && i < static_cast<int>(in_shape.size())) {
                dim[i] = in_shape[i];
            }
        }
        int known = total;
        for (int i = 0; i < n; ++i) {
            if (dim[i] != 0 && dim[i] != -1) {
                known /= dim[i];
            }
        }
        for (int i = 0; i < n; ++i) {
            if (dim[i] == -1) {
                dim[i] = known;
            }
        }

        // int64 data: a reshape is a metadata-only change — the element bytes
        // are identical, only the logical shape differs. This is part of the
        // int64 shape meta-chain.
        //
        // GPU-alias fast path (Phase 2): when the input is GPU-resident, the
        // output SHARES the input's VkBuffer — same bytes, just a different
        // shape. No readback, no re-upload. The view op preserves element count
        // (total from inputs[0] == output total), so the input's buffer is
        // exactly large enough. `dim` (the new shape) is still resolved above
        // — via the auto-learning cache for stable reshapes (no readback) or a
        // shape-input readback for dynamic ones — because downstream needs the
        // CPU-known dims_ for getShape()/dispatch. But the DATA no longer
        // crosses GPU->CPU.
        //
        // CPU fallback: when the input is host-only, copy bytes through CPU.
        if (inputs[0]->dtype() == typeid(int64_t)) {
            auto src = core::as_tensor<int64_t>(inputs[0]);
            auto output = core::as_tensor<int64_t>(outputs[0]);
            output->resize(dim);

            if (src->has_gpu_buffer()) {
                // Alias the input's GPU buffer: no data readback, no re-upload.
                auto src_buff = std::dynamic_pointer_cast<VulkanBuffer>(
                    src->as_storage_buffer(m_dev_, m_cmd_));
                objs_.emplace_back(
                    output->alias_storage_buffer(src_buff, m_cmd_));
                return;
            }

            // Host-only input: copy bytes through the CPU.
            std::vector<int64_t> out(static_cast<size_t>(total));
            // Unconditional readback: a cross-round-recycled GPU input may
            // have stale CPU data_ (see SqueezeUnsqueeze/ScatterElements fix).
            src->copyToCPU(m_cmdpool_);
            for (int i = 0; i < total; ++i) {
                out[static_cast<size_t>(i)] = (*src)[i];
            }
            output->fillToCPU(out);
            // as_storage_buffer creates vkobj_ (STORAGE|TRANSFER_DST); the
            // deferred upload records vkCmdUpdateBuffer into the level cmd
            // buffer (no submit+wait stall) instead of copyToGPU's sync flush.
            objs_.emplace_back(output->as_storage_buffer(m_dev_, m_cmd_));
            output->copyToGPUDeferred(m_cmd_);
            return;
        }

        // int8/bool data (e.g. the LLM's image_pad_mask): a reshape is a
        // metadata-only byte copy. The buffer reshape shader reads uint words
        // (fp32/fp16), which would misread 1-byte int8 elements as packed
        // words — so on the CPU fallback we copy byte-by-byte. But when the
        // input is GPU-resident, alias its buffer (same bytes, new shape) — no
        // readback needed. The mask is tiny either way.
        if (inputs[0]->dtype() == typeid(int8_t)) {
            auto src = core::as_tensor<int8_t>(inputs[0]);
            auto output = core::as_tensor<int8_t>(outputs[0]);
            output->resize(dim);

            if (src->has_gpu_buffer()) {
                auto src_buff = std::dynamic_pointer_cast<VulkanBuffer>(
                    src->as_storage_buffer(m_dev_, m_cmd_));
                objs_.emplace_back(
                    output->alias_storage_buffer(src_buff, m_cmd_));
                return;
            }

            std::vector<int8_t> out(static_cast<size_t>(total));
            src->copyToCPU(m_cmdpool_);
            for (int i = 0; i < total; ++i) {
                out[static_cast<size_t>(i)] = (*src)[i];
            }
            output->fillToCPU(out);
            objs_.emplace_back(output->as_storage_buffer(m_dev_, m_cmd_));
            output->copyToGPUDeferred(m_cmd_);
            return;
        }

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->num_elements() != total_elems(dim)) {
                output->resize(dim);
            }
            bind_ssbo<T>(outputs[0], /*is_output=*/true);
        });
        // The shape input (inputs[1]) is int64_t and lives on the CPU; the
        // buffer shader only reads from inputs[0], so the shape tensor does
        // not need an SSBO binding (the host already consumed it above).
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            bind_ssbo<T>(inputs[0], /*is_output=*/false);
        });

        ReshapePC pc{};
        pc.rank_in = static_cast<int>(in_shape.size());
        fill_dims(pc.inDims, in_shape);
        pc.rank_out = static_cast<int>(dim.size());
        fill_dims(pc.outDims, dim);
        pc.total = total;
        int nthreads = (fp16_ != 0) ? (total + 1) / 2 : total;
        submit(&pc, UP_DIV(nthreads, 256), 1, 1);
    }

    // --- shape-input value cache (runtime auto-learning) ---
    // Most Reshape shape inputs are round-invariant across decode rounds
    // (125/181; only 56 kv_len-derived grow). We learn stability by comparing
    // two consecutive readbacks, then STABLE reshapes skip copyToCPU (the
    // ~3.2ms submit+wait that dominates decode). See execute() for the state
    // machine. invalidate_shape_cache() resets to LEARNING at phase boundaries.
    enum class LearnState { LEARNING, CONFIRMING, STABLE, DYNAMIC };
    LearnState cache_state_ = LearnState::LEARNING;
    bool shape_value_dynamic_ = true; // converter hint (unused by the learner)
    std::vector<int> learned_dim_; // round-0 dim[] (compared against round-1)
    std::vector<int> cached_dim_;  // STABLE dim[] reused on round 2+
};

// PIMPL façade: buffer SSBO impl when backend_buffer is set, else image.
class Reshape : public PimplFacade {
  public:
    Reshape(int fp16, bool backend_buffer) : PimplFacade(OpType::RESHAPE) {
        impl_ = backend_buffer ? std::unique_ptr<Operator>(
                                     std::make_unique<ReshapeBuffer>(fp16))
                               : std::make_unique<ReshapeImage>();
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_RESHAPE_HPP_
