// Copyright 2026 @junka
#ifndef OPS_FUSED_ELEMWISE_HPP_
#define OPS_FUSED_ELEMWISE_HPP_

#include "core/Tensor.hpp"
#include "ops/BufferBase.hpp"
#include "ops/Operator.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <vector>

extern "C" {
extern unsigned char buffer_fused_elemwise_spv[];
extern unsigned int buffer_fused_elemwise_spv_len;
extern unsigned char buffer_fused_elemwise_fp16_spv[];
extern unsigned int buffer_fused_elemwise_fp16_spv_len;
}

namespace vkop {
namespace ops {

// Fused elementwise op: executes a chain of N elemwise operations in ONE
// GPU dispatch (one thread per output element) instead of N separate
// dispatches. This is the kernel-fusion engine — collapses chains like the
// 9-op SwiGLU block (Add Sqrt Div Mul Cast Mul Sigmoid Mul Mul) to amortize
// the ~0.3ms Intel ARL iGPU kernel-launch overhead that dominates decode
// (3270 fragmented dispatches/round). See
// plan-kernel-fusion-dispatch-reduction.md.
//
// The op program (op codes + operand encodings + per-input broadcast shapes
// + scalar constants) is encoded as an int32 SSBO ("program buffer") bound
// at binding MAX_INPUTS+1, built once from converter-supplied attributes.
// The push constant carries only sizes (total/nInputs/nOps/nScalars) — all
// shape/program detail lives in the program SSBO to stay within the 128B PC
// limit for long chains.
//
// Op-code set (mirrors shaders/buffer/fused_elemwise.comp):
//   ADD, SUB, MUL, DIV, POW (binary); SQRT, SIGMOID, NEG, EXP, TANH, CAST
//   (unary). Operand encoding: 0..MAX_REGS-1 = register; 100..100+nScalars-1
//   = scalar constant. Inputs are preloaded into reg[0..nInputs-1].
//
// The converter (model/pypi/onnx2vkop/optimizer.py fuse_elemwise_chain)
// emits a FUSED_ELEMWISE node whose attributes encode the chain. This op
// parses them into the program buffer at setAttribute time.
inline constexpr int kFusedMaxInputs = 8;
inline constexpr int kFusedMaxRegs = 16;
inline constexpr int kFusedMaxOps = 32;

// Op codes — must match shaders/buffer/fused_elemwise.comp.
enum class FusedOp {
    ADD = 1,
    SUB = 2,
    MUL = 3,
    DIV = 4,
    POW = 5,
    SQRT = 6,
    SIGMOID = 7,
    NEG = 8,
    EXP = 9,
    TANH = 10,
    CAST = 11,
    WHERE = 12,
};

struct alignas(16) FusedElemwisePC {
    int total;    // output element count
    int nInputs;  // number of bound input SSBOs (≤ kFusedMaxInputs)
    int nOps;     // number of ops in the program
    int nScalars; // number of scalar constants in the program
    int _pad0;
    int _pad1;
    int _pad2;
    int _pad3;
};
static_assert(sizeof(FusedElemwisePC) <= 128, "FusedElemwisePC PC overflow");

class FusedElemwise : public BufferFactory {
  public:
    explicit FusedElemwise(int fp16 = 0)
        : BufferFactory(
              OpType::FUSED_ELEMWISE,
              fp16 ? buffer_fused_elemwise_fp16_spv : buffer_fused_elemwise_spv,
              fp16 ? buffer_fused_elemwise_fp16_spv_len
                   : buffer_fused_elemwise_spv_len,
              // Bindings: 0 = output, 1..8 = inputs (kFusedMaxInputs), 9 =
              // prog. All STORAGE (the program buffer is read-only but
              // SSBO-typed in the shader; UPDATE_AFTER_BIND lets unused input
              // slots stay unbound when nInputs < kFusedMaxInputs).
              []() {
                  std::vector<VkDescriptorType> t(
                      1 + kFusedMaxInputs + 1,
                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
                  return t;
              }(),
              sizeof(FusedElemwisePC), fp16) {}

    // Attributes (from the converter):
    //   "ops"        flat int list, 4 per op: {op, dst, a, b}
    //   "input_shapes" flat int list, rank*nInputs dims (left-aligned, padded
    //                to `rank` per input; rank is inferred from the longest)
    //   "scalars"    flat float list (encoded as their int bits in the program)
    //   "rank"       common rank for nd<->linear
    //   "out_shape"  output shape (also written as input slot 0's dims when
    //                input 0 is the full-shape tensor; if input 0 is scalar,
    //                the host still places out_shape in the dims block slot 0)
    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.count("ops")) {
            ops_ = parse_attr_list<int>(attributes.at("ops"));
        }
        if (attributes.count("input_shapes")) {
            input_shapes_ = parse_attr_list<int>(attributes.at("input_shapes"));
        }
        if (attributes.count("scalars")) {
            scalar_floats_ = parse_attr_list<float>(attributes.at("scalars"));
        }
        if (attributes.count("rank")) {
            rank_ = std::stoi(attributes.at("rank"));
        }
        if (attributes.count("out_shape")) {
            out_shape_ = parse_attr_list<int>(attributes.at("out_shape"));
        }
    }

  private:
    std::vector<int> ops_;          // 4 ints/op
    std::vector<int> input_shapes_; // rank*nInputs, left-aligned per input
    std::vector<float> scalar_floats_;
    std::vector<int> out_shape_;
    int rank_ = 0;
    // The program buffer is (re)built on first execute (needs m_dev_/m_cmd_).
    std::shared_ptr<VulkanBuffer> prog_buf_;
    std::vector<int32_t> prog_data_;
    bool prog_built_ = false;

    // Build the program SSBO contents. Layout (mirrors the shader):
    //   [0] nOps  [1] nInputs  [2] nScalars  [3] rank
    //   [4 .. 4+rank*nInputs) input dims (per input, left-aligned, padded to
    //   rank; slot 0 = output shape)
    //   [...] ops: 4 ints/op
    //   [...] scalars (float bits as uint)
    //
    // Shapes are derived from the LIVE input/output tensors at execute time
    // (the converter-time shapes carry -1 for dynamic dims like kv_len, which
    // grows each decode round). The program (ops + scalars) is static; only
    // the dims block is rebuilt each round.
    void
    build_program(int nInputs,
                  const std::vector<std::shared_ptr<core::ITensor>> &inputs,
                  const std::shared_ptr<core::ITensor> &output) {
        int nOps = static_cast<int>(ops_.size()) / 4;
        int nScalars = static_cast<int>(scalar_floats_.size());

        // Resolve the output shape. The output tensor was resized by execute()
        // to the broadcast-max shape (or the converter out_shape_ if concrete).
        // For the dims block, slot 0 = output shape. Rank = max input rank.
        std::vector<std::vector<int>> in_shapes;
        in_shapes.reserve(nInputs);
        int rank = 0;
        for (int k = 0; k < nInputs; ++k) {
            auto s = inputs[k]->getShape();
            in_shapes.push_back(s);
            rank = std::max(rank, static_cast<int>(s.size()));
        }
        std::vector<int> out_shp = output->getShape();
        if (out_shp.empty())
            out_shp = out_shape_; // fallback to converter
        rank = std::max(rank, static_cast<int>(out_shp.size()));
        if (rank == 0)
            rank = 1;
        rank = std::min(rank, 8); // IArr8 cap

        prog_data_.clear();
        prog_data_.push_back(nOps);
        prog_data_.push_back(nInputs);
        prog_data_.push_back(nScalars);
        prog_data_.push_back(rank);

        // Slot 0 = output shape (left-aligned, padded to rank with 1).
        auto push_left_aligned = [&](const std::vector<int> &s) {
            for (int i = 0; i < rank; ++i) {
                prog_data_.push_back(
                    (i < static_cast<int>(s.size()) && s[i] > 0) ? s[i] : 1);
            }
        };
        push_left_aligned(out_shp);
        // Slots 1..nInputs-1 = each input's shape, ONNX right-aligned (pad
        // LEADING with 1 so a lower-rank input broadcasts on its trailing
        // axes — matches the shader's broadcast_index which treats any
        // dim==1 as a broadcast axis).
        for (int k = 0; k < nInputs; ++k) {
            const auto &s = in_shapes[k];
            // Right-align: leading 1s, then the shape in trailing slots.
            std::vector<int> padded(rank, 1);
            int r =
                static_cast<int>(std::min(s.size(), static_cast<size_t>(rank)));
            for (int i = 0; i < r; ++i) {
                padded[rank - r + i] = (s[i] > 0) ? s[i] : 1;
            }
            // The shader reads input[k]'s dims at dimsBase + k*rank. Slot 0
            // above is the output; input 0's dims are also written here (the
            // shader reads outDims from input slot 0, so input 0 must carry
            // the OUTPUT shape). For k==0 we already pushed out_shp; for a
            // pointwise chain input 0 == output shape, so this is consistent.
            // For k>=1 push the (right-aligned) input shape.
            if (k == 0)
                continue; // already pushed as out_shp
            for (int i = 0; i < rank; ++i)
                prog_data_.push_back(padded[i]);
        }
        // Ops.
        for (int v : ops_)
            prog_data_.push_back(v);
        // Scalars (float bits as uint32).
        for (float s : scalar_floats_) {
            uint32_t bits;
            std::memcpy(&bits, &s, sizeof(bits));
            prog_data_.push_back(static_cast<int32_t>(bits));
        }
    }

    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        // Resolve dtype from the output (the chain's terminal dtype). All
        // fused inputs share the output dtype (dtype-crossing casts are NOT
        // fused — the converter skips chains containing them).
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);

            int nInputs = static_cast<int>(inputs.size());
            nInputs = std::min(nInputs, kFusedMaxInputs);

            // Compute the broadcast-max output shape from the LIVE input
            // tensors (the converter-time shapes carry -1 for dynamic dims
            // like kv_len, which grows each decode round). ONNX right-aligned
            // broadcast: align to the max rank, element-wise max (1
            // broadcasts).
            int rank = 0;
            for (int k = 0; k < nInputs; ++k) {
                rank = std::max(rank,
                                static_cast<int>(inputs[k]->getShape().size()));
            }
            rank = std::min(rank, 8); // IArr8 cap
            std::vector<int> out_shp(rank, 1);
            for (int k = 0; k < nInputs; ++k) {
                auto s = inputs[k]->getShape();
                int r = static_cast<int>(
                    std::min(s.size(), static_cast<size_t>(rank)));
                for (int i = 0; i < r; ++i) {
                    int dim = s[r - 1 - i] > 0 ? s[r - 1 - i] : 1;
                    int idx = rank - 1 - i;
                    out_shp[idx] = std::max(out_shp[idx], dim);
                }
            }
            if (out_shp.empty())
                out_shp = out_shape_;
            int total = total_elems(out_shp);
            if (output->num_elements() != total) {
                output->resize(out_shp);
            }

            auto out_buf = bind_ssbo<T>(outputs[0], /*is_output=*/true);
            for (int k = 0; k < nInputs; ++k) {
                bind_ssbo<T>(inputs[k], /*is_output=*/false);
            }
            // The program SSBO lives at binding MAX_INPUTS+1 (=9). Descriptor
            // binding is positional (objs_[i] -> binding i), so the unused
            // input slots nInputs..MAX_INPUTS-1 must be filled with a valid
            // SSBO placeholder (the output buffer is fine — the shader's
            // input-preload loop is gated by nInputs and never reads them).
            for (int k = nInputs; k < kFusedMaxInputs; ++k) {
                objs_.emplace_back(out_buf);
            }

            // Build the program SSBO each round (dims change with kv_len; the
            // op program is static). The program is tiny (a few hundred bytes)
            // so vkCmdUpdateBuffer handles it inline.
            build_program(nInputs, inputs, outputs[0]);
            if (!prog_built_) {
                prog_buf_ = std::make_shared<VulkanBuffer>(
                    m_dev_,
                    static_cast<VkDeviceSize>(prog_data_.size() *
                                              sizeof(int32_t)),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                prog_built_ = true;
            }
            // Upload (data may have grown if the dims changed across rounds).
            prog_buf_->updateBuffer(m_cmd_->get(), prog_data_.data(),
                                    prog_data_.size() * sizeof(int32_t));
            objs_.emplace_back(prog_buf_);

            FusedElemwisePC pc{};
            pc.total = total;
            pc.nInputs = nInputs;
            pc.nOps = static_cast<int>(ops_.size()) / 4;
            pc.nScalars = static_cast<int>(scalar_floats_.size());
            // fp16: one thread per uint word (two packed half elements).
            int nthreads = total;
            submit(&pc, UP_DIV(nthreads, 256), 1, 1);
        });
    }
};

} // namespace ops
} // namespace vkop

#endif // OPS_FUSED_ELEMWISE_HPP_
