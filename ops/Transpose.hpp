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
extern "C" {
extern unsigned char image_transpose_spv[];
extern unsigned int image_transpose_spv_len;
extern unsigned char buffer_transpose_spv[];
extern unsigned int buffer_transpose_spv_len;
extern unsigned char buffer_transpose_fp16_spv[];
extern unsigned int buffer_transpose_fp16_spv_len;
}
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
        : Operator(OpType::TRANSPOSE, image_transpose_spv,
                   image_transpose_spv_len,
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
// buffer_transpose_fp16_spv shader (word-per-thread, packed-half reads).
class TransposeBuffer : public BufferFactory {
  public:
    explicit TransposeBuffer(int fp16)
        : BufferFactory(OpType::TRANSPOSE,
                        fp16 ? buffer_transpose_fp16_spv : buffer_transpose_spv,
                        fp16 ? buffer_transpose_fp16_spv_len
                             : buffer_transpose_spv_len,
                        {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                        sizeof(TransposePC), fp16) {}

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
        int rank = static_cast<int>(inshape.size());
        std::vector<int> outshape(rank);
        for (int i = 0; i < rank; ++i) {
            outshape[i] = inshape[perm_[i]];
        }

        // Fail closed on a <= 1-byte payload: no byte build in this mover (an
        // fp8/int8/bool transpose would mis-gather 4-per-word). int64 takes the
        // CPU permute below.
        core::require_word_movable_elem(inputs[0]->elem_kind(), "Transpose");

        // int64 transpose: CPU permute (the single instance feeds NonZero,
        // part of the shape meta-chain).
        if (inputs[0]->dtype() == typeid(int64_t)) {
            int total = total_elems(outshape);
            std::vector<int64_t> out(static_cast<size_t>(total));
            auto src = core::as_tensor<int64_t>(inputs[0]);
            // The input may be GPU-resident only (e.g. NonZero's output is
            // computed by a shader, leaving data_ empty) OR a cross-round-
            // recycled GPU input with stale CPU data_ (see SqueezeUnsqueeze/
            // ScatterElements fix). Pull it back to the host either way.
            src->copyToCPU(m_cmdpool_);
            std::vector<int> in_stride(rank, 1);
            for (int d = rank - 2; d >= 0; --d) {
                in_stride[d] = in_stride[d + 1] * inshape[d + 1];
            }
            std::vector<int> out_stride(rank, 1);
            for (int d = rank - 2; d >= 0; --d) {
                out_stride[d] = out_stride[d + 1] * outshape[d + 1];
            }
            for (int o = 0; o < total; ++o) {
                int r = o;
                int in_lin = 0;
                for (int d = 0; d < rank; ++d) {
                    int coord = (r / out_stride[d]) % outshape[d];
                    in_lin += coord * in_stride[perm_[d]];
                }
                out[static_cast<size_t>(o)] = (*src)[in_lin];
            }
            auto output = core::as_tensor<int64_t>(outputs[0]);
            output->resize(outshape);
            output->fillToCPU(out);
            objs_.emplace_back(output->as_storage_buffer(m_dev_, m_cmd_));
            // Deferred upload keeps data_ populated (no sync submit+wait) for
            // downstream host readers. Host-shape mode: also mark the bytes
            // authoritative so those readers skip their GPU->CPU readback.
            output->copyToGPUDeferred(m_cmd_);
            if (host_shape_enabled()) {
                output->set_host_authoritative();
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
                return;
            }
        }

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->num_elements() != total_elems(outshape)) {
                output->resize(outshape);
            }
            bind_ssbo<T>(outputs[0], /*is_output=*/true);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            bind_ssbo<T>(inputs[0], /*is_output=*/false);
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
        if (!m_ds_[m_id_]) {
            m_ds_[m_id_] = pipeline_->allocDescriptorSets();
        }
        submit_per_ds(m_ds_[m_id_], &pc, total_groups, 1, 1);
    }

    std::vector<int> perm_ = {3, 2, 1, 0};
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
