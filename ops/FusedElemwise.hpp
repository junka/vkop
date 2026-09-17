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
    //   rank)
    //   [...] ops: 4 ints/op
    //   [...] scalars (float bits as uint)
    // Input dims slot 0 = output shape (the shader reads outDims from there).
    void build_program(int nInputs) {
        int nOps = static_cast<int>(ops_.size()) / 4;
        int nScalars = static_cast<int>(scalar_floats_.size());
        prog_data_.clear();
        prog_data_.push_back(nOps);
        prog_data_.push_back(nInputs);
        prog_data_.push_back(nScalars);
        prog_data_.push_back(rank_);

        // Per-input dims. The converter left-aligns each input's shape and
        // pads to `rank`; slot 0 is the output shape.
        int dimsCount = rank_ * nInputs;
        // Ensure input_shapes_ has exactly dimsCount entries (pad with 1).
        while (static_cast<int>(input_shapes_.size()) < dimsCount) {
            input_shapes_.push_back(1);
        }
        // Slot 0 = output shape (left-aligned, padded to rank).
        for (int i = 0; i < rank_; ++i) {
            prog_data_.push_back(
                i < static_cast<int>(out_shape_.size()) ? out_shape_[i] : 1);
        }
        // Slots 1..nInputs-1 = the converter-supplied input shapes.
        for (int k = 1; k < nInputs; ++k) {
            for (int i = 0; i < rank_; ++i) {
                int idx = k * rank_ + i;
                prog_data_.push_back(
                    idx < static_cast<int>(input_shapes_.size())
                        ? input_shapes_[idx]
                        : 1);
            }
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
            int total = total_elems(out_shape_.empty() ? output->getShape()
                                                       : out_shape_);
            if (output->num_elements() != total) {
                output->resize(out_shape_.empty() ? output->getShape()
                                                  : out_shape_);
            }
            auto out_buf = bind_ssbo<T>(outputs[0], /*is_output=*/true);

            int nInputs = static_cast<int>(inputs.size());
            nInputs = std::min(nInputs, kFusedMaxInputs);
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

            // Build + upload the program buffer (once; reused across rounds
            // since the chain structure is static). The program is tiny (a
            // few hundred bytes) so vkCmdUpdateBuffer handles it inline.
            if (!prog_built_) {
                build_program(nInputs);
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
            // Upload the program (cheap; data may change if the converter
            // re-ran, but structure is static — still re-upload to be safe
            // since the buffer may have been recycled).
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
