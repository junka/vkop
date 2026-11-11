// Copyright 2025 @junka
#ifndef OPS_TRANSPOSE_HPP_
#define OPS_TRANSPOSE_HPP_

#include "core/Tensor.hpp"
#include "ops/BufferBase.hpp"
#include "ops/Operator.hpp"
#include "ops/PimplFacade.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
namespace vkop {
namespace ops {

namespace transpose {
// True when `perm` merely reorders unit-size dims, so the permuted tensor's
// flat (row-major) memory is byte-identical to the source's — the Transpose is
// a pure VIEW on the bytes.
//
// The shader emits output flat index o from input flat index
//     in(o) = sum_i c_i * in_stride[perm[i]]
// (it assigns in_coord[perm[i]] = out_coord[i] and reads the input linearly),
// while o itself is sum_i c_i * out_stride[i]. A 1:1 flat copy is therefore
// exact iff out_stride[i] == in_stride[perm[i]] for every i whose source dim
// size[perm[i]] is > 1 (a size-1 dim has c_i == 0, so its stride is moot).
//
// This is not a heuristic: it is the exact condition under which the naive
// gather degenerates to an identity, and it is the common case in decode —
// [1,heads,1,head_dim] under perm [0,2,1,3], and [1,1,head_dim] under [0,2,1].
inline bool flat_identity_perm(const std::vector<int> &shape,
                               const std::vector<int> &perm) {
    int rank = static_cast<int>(shape.size());
    if (static_cast<int>(perm.size()) != rank)
        return false;
    std::vector<int> in_stride(rank, 1), out_stride(rank, 1);
    for (int d = rank - 2; d >= 0; --d) {
        in_stride[d] = in_stride[d + 1] * shape[d + 1];
    }
    for (int d = rank - 2; d >= 0; --d) {
        out_stride[d] = out_stride[d + 1] * shape[perm[d + 1]];
    }
    for (int i = 0; i < rank; ++i) {
        if (shape[perm[i]] == 1) {
            continue;
        }
        if (out_stride[i] != in_stride[perm[i]]) {
            return false;
        }
    }
    return true;
}

struct GpuTransposeParam {
    ivec4 inShape;
    ivec4 outShape;
    ivec4 perms;
    ivec4 reverse_perms;
};

} // namespace transpose

class TransposeImage : public Operator {
  public:
    explicit TransposeImage()
        : Operator(OpType::TRANSPOSE,
                   {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER},
                   sizeof(transpose::GpuTransposeParam)) {}
    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("perm") != attributes.end()) {
            perm_ = parse_attr_list<int>(attributes.at("perm"));
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {

        auto inshape = inputs[0]->getShape();
        std::vector<int> outshape(4);
        for (size_t i = 0; i < 4; ++i) {
            outshape[i] = inshape[perm_[i]];
        }

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->num_elements() != total_elems(outshape)) {
                output->resize(outshape);
            }
            auto output_image = output->as_output_image(m_dev_, m_cmd_);
            objs_.emplace_back(output_image);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto input = core::as_tensor<T>(inputs[0]);
            auto input_image = input->as_input_image(m_dev_, m_cmd_);
            objs_.emplace_back(input_image);
        });

        std::vector<int> reverse_perms(4);
        for (int i = 0; i < 4; ++i) {
            reverse_perms[perm_[i]] = i;
        }

        transpose::GpuTransposeParam param;
        for (int i = 0; i < 4; i++) {
            param.inShape[i] = inshape[i];
            param.outShape[i] = outshape[i];
            param.perms[i] = perm_[i];
            param.reverse_perms[i] = reverse_perms[i];
        }
        submit(&param, UP_DIV(outshape[3], 16),
               UP_DIV(outshape[2] * outshape[0], 16), UP_DIV(outshape[1], 4));
    }

    std::vector<int> perm_ = {3, 2, 1, 0};
};
// Transpose buffer op. Arbitrary perm up to 8-D. fp16 path uses the
// buffer_transpose_fp16 shader (word-per-thread, packed-half reads).
class TransposeBuffer : public BufferFactory {
  public:
    explicit TransposeBuffer(int fp16)
        : BufferFactory(OpType::TRANSPOSE,
                        {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                        sizeof(TransposePC), fp16) {}

    bool supports_bda_variant() const override { return true; }
    size_t bda_pc_size() const override { return sizeof(TransposeBdaPC); }

    // int64-data pipeline (buffer/transpose_int64). Same 2× STORAGE layout and
    // the same 100-byte TransposePC as the base module, so it reuses types_
    // and pc_size_; only the SPIR-V differs.
    void ensure_int64_pipeline() {
        if (pipeline_int64_) return;
        bool use_uab = update_after_bind_ &&
                       m_dev_->is_support_descriptor_update_after_bind();
        auto i64 = resolve_spv("buffer_transpose_int64");
        pipeline_int64_ = std::make_unique<VulkanPipeline>(
            m_dev_->getLogicalDevice(), types_, sizeof(TransposePC),
            i64.first, i64.second, use_uab, required_subgroup_size_);
        for (auto &ds : m_ds_int64_) {
            ds = pipeline_int64_->allocDescriptorSets();
        }
        build_bda_table_module("buffer_transpose_int64_bda",
                               sizeof(TransposePC), 2, bda_i64_);
    }
    void submit(void *ptr, int width, int height, int layers) override {
        if (!int64_mode_) {
            Operator::submit(ptr, width, height, layers);
            return;
        }
        if (submit_bda_table_module(bda_i64_, ptr, width, height, layers)) {
            return;
        }
        if (!m_ds_int64_[m_id_]) {
            m_ds_int64_[m_id_] = pipeline_int64_->allocDescriptorSets();
        }
        fillWriteDescriptorSets(m_ds_int64_[m_id_]);
        pipeline_int64_->updateDescriptorSets(writes_);
        m_cmd_->bind(*pipeline_int64_, m_ds_int64_[m_id_]);
        if (ptr) {
            m_cmd_->push_constants(
                *pipeline_int64_,
                static_cast<uint32_t>(sizeof(TransposePC)), ptr);
        }
        m_cmd_->dispatch(width, height, layers);
        if (replay_enabled_) {
            record_fingerprint(ptr, width, height, layers);
        }
    }
    std::unique_ptr<VulkanPipeline> pipeline_int64_;
    std::array<VkDescriptorSet, vkop::kInflight> m_ds_int64_{};
    BdaTableModule bda_i64_;
    bool int64_mode_ = false;

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("perm") != attributes.end()) {
            perm_ = parse_attr_list<int>(attributes.at("perm"));
        }
    }
    bool has_host_side_effects() const override { return true; }

    // The output SHAPE is recomputed every round from the input's, and
    // execute() applies it with output->resize(outshape). A replay skips that,
    // so the output would keep the previous round's dims while the recorded
    // dispatch writes the new ones — measured on the Qwen3-VL shape-meta chain:
    // /mrope_static/Transpose goes (3,1,7) -> (3,1,1) at prefill->decode, and
    // without this report its level looked static and replayed, leaving the
    // downstream Where/Expand reading a stale rank.
    //
    // This is orthogonal to has_host_side_effects(): the fp16/int8 paths do
    // their work on the GPU, but the SHAPE they publish is host metadata.
    bool recording_changes_per_round() const override {
        return transpose_shape_changed_;
    }
    // Kept conservative for the type question: the fp16 paths do no host data
    // work, so a segment may replay them once the shape is stable.
    bool did_host_work_last_round() const override {
        return host_work_last_round_;
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        // Every path is now a GPU dispatch — the fp16/fp32 gather, the
        // flat-identity alias, and (since buffer/transpose_int64.comp) the
        // int64 move. No host work, so a segment may replay. The one thing a
        // replay must still watch is the output SHAPE, reported separately by
        // recording_changes_per_round().
        set_did_host_work_last_round(false);
        auto inshape = inputs[0]->getShape();
        int rank = static_cast<int>(inshape.size());
        std::vector<int> outshape(rank);
        for (int i = 0; i < rank; ++i) {
            outshape[i] = inshape[perm_[i]];
        }
        // Does this round publish a different output shape than the previous
        // one? That is what makes a recorded dispatch unsafe to reuse (see the
        // declaration above); it does not mean the op did host DATA work.
        transpose_shape_changed_ =
            outshape != last_outshape_ || rank != (int)last_outshape_.size();
        last_outshape_ = outshape;

        // Fail closed on a <= 1-byte payload: no byte build in this mover (an
        // fp8/int8/bool transpose would mis-gather 4-per-word). int64 takes the
        // CPU permute below.
        core::require_word_movable_elem(inputs[0]->elem_kind(), "Transpose");

        // int64 transpose: a GPU ivec2 move (buffer/transpose_int64.comp).
        //
        // This used to be a host permute — read the int64 tensor back, permute
        // it in a C++ loop, re-upload. That made the level unreplayable: a
        // replay skips execute(), so the permute stopped and the output kept
        // the previous round's bytes. The kernel needs no 64-bit arithmetic
        // (it moves two 32-bit words per element), so unlike the int64 Cast it
        // does not require the shaderInt64 feature.
        if (inputs[0]->dtype() == typeid(int64_t)) {
            int total = total_elems(outshape);
            auto output = core::as_tensor<int64_t>(outputs[0]);
            if (output->num_elements() != total) {
                output->resize(outshape);
            }
            objs_.emplace_back(output->as_storage_buffer(m_dev_, m_cmd_));
            auto src = core::as_tensor<int64_t>(inputs[0]);
            // nullptr cmd: the producer's level already ordered this buffer;
            // the same convention the other int64 twins use for inputs.
            objs_.emplace_back(src->as_storage_buffer(m_dev_, nullptr));

            TransposePC pc{};
            pc.rank = rank;
            for (int i = 0; i < 8 && i < rank; ++i) {
                pc.inDims[i] = inshape[i];
                pc.outDims[i] = outshape[i];
                pc.perm[i] = perm_[i];
            }
            ensure_int64_pipeline();
            int64_mode_ = true;
            // An empty tensor (this instance is (1,0) in decode) has nothing to
            // move; UP_DIV(0,256) is 0, which vkCmdDispatch rejects.
            if (total > 0) {
                submit(&pc, UP_DIV(total, 256), 1, 1);
                auto out_buf =
                    std::dynamic_pointer_cast<VulkanBuffer>(objs_[0]);
                if (out_buf) {
                    out_buf->shaderWriteBarrier(m_cmd_->get(),
                                                (VkDeviceSize)total * 8);
                }
            }
            return;
        }

        // Flat-identity perm (see flat_identity_perm): the permuted bytes are
        // identical to the source's, so this Transpose is a pure VIEW — alias
        // the input's VkBuffer exactly like Reshape does (ops/Reshape.hpp:413),
        // skipping the gather dispatch entirely. 24 dispatches/decode round on
        // the Qwen2.5 attention output ([1,heads,1,head_dim], perm [0,2,1,3]),
        // plus the rotary [1,1,head_dim] instance during decode. In prefill the
        // seq dim is > 1 and the perm is no longer an identity, so those levels
        // keep the real dispatch — this only fires where the bytes genuinely do
        // not move.
        if (transpose::flat_identity_perm(inshape, perm_)) {
            bool aliased = false;
            dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
                using T = decltype(dummy);
                auto src = core::as_tensor<T>(inputs[0]);
                if (!src->has_gpu_buffer()) {
                    return;
                }
                auto output = core::as_tensor<T>(outputs[0]);
                if (output->num_elements() != total_elems(outshape)) {
                    output->resize(outshape);
                }
                auto src_buff = std::dynamic_pointer_cast<VulkanBuffer>(
                    src->as_storage_buffer(m_dev_, m_cmd_));
                objs_.emplace_back(
                    output->alias_storage_buffer(src_buff, m_cmd_));
                aliased = true;
            });
            if (aliased) {
                // Pure view this round: report it so Runtime does not stamp this
                // node as the buffer's writer (see
                // Operator::did_alias_output_last_round).
                set_did_alias_output_last_round(true);
                return;
            }
        }
        // Real gather dispatch below: this node WRITES its output.
        set_did_alias_output_last_round(false);

        std::shared_ptr<VulkanBuffer> out_buf;
        std::shared_ptr<VulkanBuffer> in_buf;
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->num_elements() != total_elems(outshape)) {
                output->resize(outshape);
            }
            out_buf = bind_ssbo<T>(outputs[0], /*is_output=*/true);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            in_buf = bind_ssbo<T>(inputs[0], /*is_output=*/false);
        });

        TransposePC pc{};
        pc.rank = rank;
        fill_dims(pc.inDims, inshape);
        fill_dims(pc.outDims, outshape);
        for (int i = 0; i < 8; ++i) {
            pc.perm[i] = (i < rank) ? perm_[i] : i;
        }
        // fp16 packs two elements per uint word; dispatch one thread per output
        // word (the fp16 shader writes each word once — no RMW race).
        int total = total_elems(outshape);
        int nthreads = (fp16_ != 0) ? (total + 1) / 2 : total;
        int total_groups = UP_DIV(nthreads, 256);
        if (bda_ready() && out_buf && in_buf && out_buf->hasDeviceAddress() &&
            in_buf->hasDeviceAddress()) {
            const VkDeviceAddress out_addr = out_buf->getDeviceAddress();
            const VkDeviceAddress in_addr = in_buf->getDeviceAddress();
            if ((out_addr % 4) == 0 && (in_addr % 4) == 0) {
                TransposeBdaPC bpc{};
                std::memcpy(&bpc, &pc, offsetof(TransposeBdaPC, out_addr));
                bpc.out_addr = out_addr;
                bpc.in_addr = in_addr;
                submit_bda(&bpc, sizeof(bpc), total_groups, 1, 1);
                return;
            }
        }
        if (!m_ds_[m_id_]) {
            m_ds_[m_id_] = pipeline_->allocDescriptorSets();
        }
        submit_per_ds(m_ds_[m_id_], &pc, total_groups, 1, 1);
    }

    std::vector<int> perm_ = {3, 2, 1, 0};
    // The previous round's output shape. recording_changes_per_round() reports
    // whether it moved, which is what makes a recorded dispatch reusable.
    std::vector<int> last_outshape_;
    bool transpose_shape_changed_ = true;
    void invalidate_shape_cache() override {
        last_outshape_.clear();
        transpose_shape_changed_ = true;
    }
};

// PIMPL façade: buffer SSBO impl when backend_buffer is set, else image.
class Transpose : public PimplFacade {
  public:
    Transpose(int fp16, bool backend_buffer) : PimplFacade(OpType::TRANSPOSE) {
        impl_ = backend_buffer ? std::unique_ptr<Operator>(
                                     std::make_unique<TransposeBuffer>(fp16))
                               : std::make_unique<TransposeImage>();
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_TRANSPOSE_HPP_
