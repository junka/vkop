// Copyright 2025 @junka
#ifndef OPS_ROTARY_EMBEDDING_HPP_
#define OPS_ROTARY_EMBEDDING_HPP_

#include "ops/PimplFacade.hpp"
#include <memory>

#include "Operator.hpp"
#include "ops/BufferBase.hpp"
namespace vkop {
namespace ops {

// Push constant for shaders/buffer/rotary.comp（字段顺序必须与 shader 里的
// push_constant 完全一致，尾部可多不可少）。
// X is [B, num_heads, seq, head_dim]; output is the same shape. cos/sin are
// pre-broadcast to [B, 1, S, rotary_dim]（rotary_dim == head_dim 时即整维旋转）
// 且按 (b*seq + s)*rotary_dim + d 索引，不含 heads 轴。
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
    // cos/sin 每行的列数 == RoPE 实际作用的维数。等于 head_dim
    // 时整维旋转（Qwen、 DiT），小于 head_dim 时是 partial rotary（Phi-4:
    // head_dim 128 只转前 96 维， 后 32 维直通）。0 = 按 head_dim
    // 处理，兼容不填这个字段的旧图。
    int rotary_dim;
    // 1 = cos/sin 是 fp32（X 允许是 fp16）。视觉塔的导出把整个 RoPE 放在 fp32
    // 域里，激活却是 fp16；fp16 build 的 shader 据此按 fp32 步长读 cos/sin，
    // 于是两个 build 内部的 fp32 运算逐位一致 —— 这正是把包裹 RE 的
    // Cast(f16->f32)/Cast(f32->f16) 对折掉仍然逐位相同的前提。
    // 0 = cos/sin 与 X 同 dtype（LLM 的 MRoPE 是 fp16）。
    int cs_fp32;
    // >= 0: X 是视觉塔 qkv 的 [seq, C, num_heads, head_dim] 张量，本 op 只读第
    // comp 个分量，Y 仍是 rank-3 [seq, num_heads, head_dim]。converter 的
    // fuse_qkv_components 把「Transpose([N,C,H,D]->[C,N,H,D]) + Split(axis=0)
    // + Squeeze」三个 op 折成这条寻址（它们只为让 Split 落在 axis 0 而存在，
    // 见 [[zero-copy-range-views]]）。head_dim 在两种布局里都是最内维，所以
    // 只有读点的 stride 变了。 -1 = 关闭（默认）。
    int comp;
    // comp >= 0 时 X 相邻 seq 行的距离（元素）= C*num_heads*head_dim。
    int in_seq_stride;
};
static_assert(sizeof(RotaryPC) <= 128, "RotaryPC PC overflow");

// Buffer-backend (SSBO) RotaryEmbedding. Half-split (non-interleaved); 整维旋转
// (rotary_dim == head_dim) 或 partial rotary（cos/sin
// 行宽更窄时，超出的一段直通）。The shader runs ONE THREAD PER OUTPUT WORD (two
// packed
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
        // input_component: read one component out of a [seq, C, heads, dim]
        // tensor instead of a whole [seq, heads, dim] one (see RotaryPC::comp).
        // Set by the converter's fuse_qkv_components; absent on every other
        // graph, which leaves the addressing exactly as it was.
        if (attributes.find("input_component") != attributes.end()) {
            input_component_ =
                static_cast<int>(std::stol(attributes.at("input_component")));
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
        // Component read: X is the vision tower's rank-4 qkv tensor
        // [seq, C, heads, dim] and only component `input_component_` is
        // rotated. Y is still the rank-3 [seq, heads, dim] the Split+Squeeze
        // chain used to hand over, so heads/seq come from the other two axes
        // and C is the component count. Checked rather than assumed: reading
        // the wrong axis here would rotate the wrong tensor silently.
        int comp_count = 1;
        if (input_component_ >= 0) {
            if (xshape.size() != 4) {
                throw std::runtime_error(
                    "RotaryEmbedding: input_component needs a rank-4 "
                    "[seq, C, heads, dim] input");
            }
            seq = xshape[0];
            comp_count = xshape[1];
            num_heads = xshape[2];
            if (comp_count <= 0 || input_component_ >= comp_count) {
                throw std::runtime_error(
                    "RotaryEmbedding: input_component " +
                    std::to_string(input_component_) +
                    " is outside C=" + std::to_string(comp_count));
            }
            if (seq <= 0 || num_heads <= 0) {
                throw std::runtime_error("RotaryEmbedding: input_component "
                                         "needs resolved seq/heads");
            }
        } else if (input_untransposed_) {
            // X is [B, seq, num_heads, head_dim]: swap the two middle axes.
            std::swap(seq, num_heads);
        }
        // cos/sin 的行宽就是 RoPE 真正作用的维数：Phi-4 这类 partial rotary
        // 模型里 它只有 head_dim 的一部分（128 维只转前 96
        // 维），内核据此决定哪一段参与 旋转、哪一段直通。取不到（旧图/缺 shape
        // 信息）时按整维处理，行为与原来 一致。
        std::vector<int> cshape = inputs[1]->getShape();
        int rotary_dim = cshape.empty() ? head_dim : cshape.back();
        if (rotary_dim <= 0 || rotary_dim > head_dim)
            rotary_dim = head_dim;

        // Leading dims (batch etc.) collapse into `total - heads*seq*head_dim`.
        // A component read outputs the rank-3 [seq, heads, dim], not X's
        // [seq, C, heads, dim], so its total is the product of those three.
        int total = (input_component_ >= 0) ? seq * num_heads * head_dim
                                            : total_elems(xshape);
        // Rank-3 [seq, heads, head_dim] (vision encoder: batch axis squeezed
        // away and seq leading). cos/sin carry `seq` rows, so the row count
        // they cover identifies the layout: xshape[-2] is heads, not seq.
        bool seq_major =
            (input_component_ >= 0 ||
             (xshape.size() == 3 && !input_untransposed_ &&
              total_elems(inputs[1]->getShape()) / rotary_dim == xshape[0]));
        if (seq_major && input_component_ < 0)
            std::swap(seq, num_heads);
        // Output layout is [B, num_heads, seq, head_dim] (the transposed layout
        // downstream MatMul Q@K^T / Concat K-cache expect). When X is
        // untransposed ([B, seq, num_heads, head_dim]), the OUTPUT shape is the
        // swapped input shape — NOT xshape. Resizing to xshape would set the
        // output dims_ metadata to the untransposed layout and corrupt every
        // downstream op that reads getShape() for dispatch/broadcast math.
        // A component read likewise outputs [seq, heads, dim], not X's rank-4.
        std::vector<int> out_shape = xshape;
        if (input_component_ >= 0) {
            out_shape = {seq, num_heads, head_dim};
        } else if (input_untransposed_ && out_shape.size() >= 3) {
            std::swap(out_shape[out_shape.size() - 2],
                      out_shape[out_shape.size() - 3]);
        }

        // Output has the same shape as X; resize if the runtime created it
        // with a placeholder (e.g. dynamic -1 -> 1 at load). A component read
        // is the one case where the output layout is NOT X's, and its element
        // count alone cannot tell the two apart ([1,S,H,D] and [S,H,D] hold
        // the same number), so the shape has to be compared too — the same
        // guard Attention's seq_major resize uses.
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto type_tag) {
            using T = decltype(type_tag);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->num_elements() != total ||
                (input_component_ >= 0 && output->getShape() != out_shape)) {
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
        pc.rotary_dim = rotary_dim;
        // cos/sin are read at their own dtype's stride, which is independent of
        // X's. The vision tower folds away the fp16->fp32 Cast that used to
        // feed this op, leaving fp16 X next to the exporter's fp32 cos/sin.
        pc.cs_fp32 = (inputs[1]->dtype() == typeid(uint16_t)) ? 0 : 1;
        pc.comp = input_component_;
        pc.in_seq_stride = comp_count * num_heads * head_dim;

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
    // Read component `input_component_` of a [seq, C, heads, dim] input;
    // -1 (default) reads the whole [seq, heads, dim] tensor. Set by the
    // converter's fuse_qkv_components.
    int input_component_ = -1;
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
        } // namespace ops
        impl_ = std::unique_ptr<Operator>(
            std::make_unique<RotaryEmbeddingBuffer>(fp16));
    } // namespace vkop
};

} // namespace ops
} // namespace vkop
#endif // OPS_ROTARY_EMBEDDING_HPP_
