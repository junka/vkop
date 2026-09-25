// Copyright 2025 @junka
#ifndef OPS_ROTARY_EMBEDDING_HPP_
#define OPS_ROTARY_EMBEDDING_HPP_

#include "ops/PimplFacade.hpp"
#include <memory>

#include "Operator.hpp"
#include "ops/BufferBase.hpp"
extern "C" {
extern unsigned char buffer_rotary_spv[];
extern unsigned int buffer_rotary_spv_len;
extern unsigned char buffer_rotary_fp16_spv[];
extern unsigned int buffer_rotary_fp16_spv_len;
}
namespace vkop {
namespace ops {

// Push constant for shaders/buffer/rotary.comp. 5 ints (20B), padded to 16.
// X is [B, num_heads, seq, head_dim]; output is the same shape. cos/sin are
// pre-broadcast to [B, 1, S, head_dim] (axis-1 unsqueeze from /rotary_emb/*)
// and index without the heads axis: cs_idx = (b*seq + s)*head_dim + d.
// When input_untransposed is set, X is actually laid out as
// [B, seq, num_heads, head_dim] (a Transpose(perm=[0,2,1,3]) feeding RE was
// folded away); the shader remaps the read index. num_heads/seq are still
// passed in the output-layout order (num_heads = shape[-3], seq = shape[-2]
// of the UNTRANSPOSED input, which the execute() swaps before filling PC).
struct alignas(16) RotaryPC {
    int total;              // Y element count = B * num_heads * seq * head_dim
    int head_dim;           // full head dim (rotary_embedding_dim)
    int num_heads;          // heads dim of X
    int seq;                // sequence len of X
    int input_untransposed; // 1 = X is [B, seq, num_heads, head_dim] (folded
                            // Transpose)
    int seq_major; // 1 = X is [seq, num_heads, head_dim] (rank 3, vision
                   // encoder: the seq axis leads the head axis)
};
static_assert(sizeof(RotaryPC) <= 128, "RotaryPC PC overflow");

// Buffer-backend (SSBO) RotaryEmbedding. Half-split (non-interleaved), full
// head_dim rotation. The shader runs ONE THREAD PER OUTPUT WORD (two packed
// elements) so the fp16 build avoids the half2 read-modify-write race
// ([[expand-fp16-race]]): each output word is owned by a single thread that
// computes both halves and writes the whole word once. Dispatch is therefore
// words = ceil(total/2) threads, not total.
//
// Bindings: out=0, X=1, cos=2, sin=3.
class RotaryEmbeddingBuffer : public BufferFactory {
  public:
    explicit RotaryEmbeddingBuffer(int fp16)
        : BufferFactory(OpType::ROTARY_EMBEDDING,
                        fp16 ? buffer_rotary_fp16_spv : buffer_rotary_spv,
                        fp16 ? buffer_rotary_fp16_spv_len
                             : buffer_rotary_spv_len,
                        std::vector<VkDescriptorType>{
                            DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                            DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                        sizeof(RotaryPC), fp16),
          fp16_(fp16) {
        update_after_bind_ = true;
    }

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("input_untransposed") != attributes.end()) {
            input_untransposed_ =
                std::stol(attributes.at("input_untransposed")) != 0;
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        // X is [B, num_heads, seq, head_dim] by default. When
        // input_untransposed_ is set, a Transpose(perm=[0,2,1,3]) feeding X
        // was folded away, so X is actually laid out [B, seq, num_heads,
        // head_dim] — shape[-2] is num_heads and shape[-3] is seq (swapped
        // relative to the default). The output is still [B, num_heads, seq,
        // head_dim]; the shader remaps the read index accordingly.
        std::vector<int> xshape = inputs[0]->getShape();
        int head_dim = xshape.back();
        int seq = xshape[xshape.size() - 2];
        int num_heads = xshape.size() >= 3 ? xshape[xshape.size() - 3] : 1;
        if (input_untransposed_) {
            // X is [B, seq, num_heads, head_dim]: swap the two middle axes.
            std::swap(seq, num_heads);
        }
        // Leading dims (batch etc.) collapse into `total - heads*seq*head_dim`.
        int total = total_elems(xshape);
        // Rank-3 [seq, heads, head_dim] (vision encoder: batch axis squeezed
        // away and seq leading). cos/sin carry `seq` rows, so the row count
        // they cover identifies the layout: xshape[-2] is heads, not seq.
        bool seq_major =
            (xshape.size() == 3 && !input_untransposed_ &&
             total_elems(inputs[1]->getShape()) / head_dim == xshape[0]);
        if (seq_major)
            std::swap(seq, num_heads);
        // Output layout is [B, num_heads, seq, head_dim] (the transposed layout
        // downstream MatMul Q@K^T / Concat K-cache expect). When X is
        // untransposed ([B, seq, num_heads, head_dim]), the OUTPUT shape is the
        // swapped input shape — NOT xshape. Resizing to xshape would set the
        // output dims_ metadata to the untransposed layout and corrupt every
        // downstream op that reads getShape() for dispatch/broadcast math.
        std::vector<int> out_shape = xshape;
        if (input_untransposed_ && out_shape.size() >= 3) {
            std::swap(out_shape[out_shape.size() - 2],
                      out_shape[out_shape.size() - 3]);
        }

        // Output has the same shape as X; resize if the runtime created it
        // with a placeholder (e.g. dynamic -1 -> 1 at load).
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto type_tag) {
            using T = decltype(type_tag);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->num_elements() != total) {
                output->resize(out_shape);
            }
            bind_ssbo<T>(outputs[0], /*is_output=*/true);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto type_tag) {
            using T = decltype(type_tag);
            bind_ssbo<T>(inputs[0], /*is_output=*/false); // X
        });
        dispatch_by_dtype(inputs[1]->dtype(), [&](auto type_tag) {
            using T = decltype(type_tag);
            bind_ssbo<T>(inputs[1], /*is_output=*/false); // cos
        });
        dispatch_by_dtype(inputs[2]->dtype(), [&](auto type_tag) {
            using T = decltype(type_tag);
            bind_ssbo<T>(inputs[2], /*is_output=*/false); // sin
        });

        RotaryPC pc{};
        pc.total = total;
        pc.head_dim = head_dim;
        pc.num_heads = num_heads;
        pc.seq = seq;
        pc.input_untransposed = input_untransposed_ ? 1 : 0;
        pc.seq_major = seq_major ? 1 : 0;

        // Dispatch: fp16 packs 2 elements/word and the shader runs one thread
        // per output word (race-free whole-word write, [[expand-fp16-race]]);
        // fp32 is 1 element/word so one thread per element.
        if (fp16_) {
            int words = (total + 1) / 2;
            submit(&pc, UP_DIV(words, 256), 1, 1);
        } else {
            submit(&pc, UP_DIV(total, 256), 1, 1);
        }
    }

  private:
    int fp16_;
    bool input_untransposed_ =
        false; // X laid out as [B, seq, num_heads, head_dim]
};

// PIMPL façade. Buffer-only (per the runtime-op authorization); the image
// backend is not implemented.
class RotaryEmbedding : public PimplFacade {
  public:
    RotaryEmbedding(int fp16, bool backend_buffer)
        : PimplFacade(OpType::ROTARY_EMBEDDING) {
        if (!backend_buffer) {
            throw std::runtime_error(
                "RotaryEmbedding is buffer-backend only (no image impl).");
        }
        impl_ = std::unique_ptr<Operator>(
            std::make_unique<RotaryEmbeddingBuffer>(fp16));
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_ROTARY_EMBEDDING_HPP_
