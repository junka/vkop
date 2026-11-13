// Copyright 2025 @junka
#ifndef OPS_CUMSUM_HPP_
#define OPS_CUMSUM_HPP_

#include "Operator.hpp"
#include "ops/BufferBase.hpp"
#include "ops/PimplFacade.hpp"
#include <cstring>

namespace vkop {
namespace ops {

// Cumulative sum along one axis (ONNX CumSum).
//
// Buffer-only: the image backend has no strided-prefix primitive, and the
// only consumer (Qwen3.5's gated delta rule) is a dense-SSBO graph. The op
// therefore ignores `backend_buffer` and always builds the buffer leaf, the
// same choice Attention and EmbeddingForward make.
//
// Push constant: the resolved axis and its strides. The workgroup count is
// outer*axis_stride (one per line); each workgroup walks its line serially
// because a prefix sum carries a running total.
//
// fp32 and fp16. Three shader builds share one source (see
// docs/fp16-storage-layouts.md):
//   <key>          fp32
//   <key>_fp16     float16_t storage -- native half loads/stores
//   <key>_fp16_packed  two halves per uint word, unpack/pack per access
// The kernel runs ONE thread per line, which is what makes both fp16 layouts
// safe: with the packed one, two lanes of a single word would race.
class CumsumBuffer : public BufferFactory {
  public:
    explicit CumsumBuffer(int fp16)
        : BufferFactory(OpType::CUMSUM,
                        {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                        sizeof(CumsumPC), fp16) {}

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("axis") != attributes.end()) {
            axis_ = std::stol(attributes.at("axis"));
        }
        // ONNX CumSum also carries `exclusive` / `reverse` as INPUTS, not
        // attributes; the converter folds those into attributes when they are
        // constant, which every real graph has.
        if (attributes.find("exclusive") != attributes.end()) {
            exclusive_ = std::stol(attributes.at("exclusive")) != 0;
        }
        if (attributes.find("reverse") != attributes.end()) {
            reverse_ = std::stol(attributes.at("reverse")) != 0;
        }
    }

  private:
    bool supports_bda_variant() const override { return true; }
    size_t bda_pc_size() const override { return sizeof(CumsumBdaPC); }

    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        if (inputs.empty() || !inputs[0]) {
            throw std::runtime_error("CumSum: no input bound");
        }
        if (outputs.empty() || !outputs[0]) {
            throw std::runtime_error("CumSum: no output bound");
        }
    
        const auto shape = inputs[0]->getShape();
        const int rank = static_cast<int>(shape.size());
        if (rank == 0 || rank > 8) {
            throw std::runtime_error("CumSum: rank must be 1..8");
        }
        int axis = axis_;
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) {
            throw std::runtime_error("CumSum: axis out of range");
        }
        const int64_t total = total_elems(shape);
        if (total <= 0) {
            throw std::runtime_error("CumSum: empty input");
        }

        int outer = 1;
        for (int i = 0; i < axis; i++) outer *= shape[i];
        int axis_len = shape[axis];
        int axis_stride = 1;
        for (int i = axis + 1; i < rank; i++) axis_stride *= shape[i];

        // The shader build is chosen from the tensor dtype (the runtime's
        // op_shader_fp16), so this only has to bind the storage the way the
        // matching build indexes it: one element per uint for fp32, a real
        // half array for native fp16, or the packed word pair.
        std::shared_ptr<VulkanBuffer> out_buf, in_buf;
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto tag) {
            using T = decltype(tag);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->num_elements() != total) {
                output->resize(shape);
            } else if (output->getShape() != shape) {
                output->reshape_view(shape);
            }
            out_buf = bind_ssbo<T>(outputs[0], /*is_output=*/true);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto tag) {
            using T = decltype(tag);
            in_buf = bind_ssbo<T>(inputs[0], /*is_output=*/false);
        });

        CumsumPC pc{};
        pc.axis = axis;
        pc.axis_len = axis_len;
        pc.axis_stride = axis_stride;
        pc.outer = outer;
        pc.exclusive = exclusive_ ? 1 : 0;
        pc.reverse = reverse_ ? 1 : 0;

        // One thread per PAIR of adjacent lines: in fp16 the two elements at
        // lanes s and s+1 of a slice share a 32-bit word, so a thread that owns
        // both keeps its writes word-disjoint from every other thread's.
        const int lane_pairs = (axis_stride + 1) / 2;
        int groups = UP_DIV(outer * lane_pairs, 256);

        if (bda_ready() && in_buf && out_buf && in_buf->hasDeviceAddress() &&
            out_buf->hasDeviceAddress()) {
            const VkDeviceAddress in_addr = in_buf->getDeviceAddress();
            const VkDeviceAddress out_addr = out_buf->getDeviceAddress();
            if ((in_addr % 4) == 0 && (out_addr % 4) == 0) {
                CumsumBdaPC bpc{};
                std::memcpy(&bpc, &pc, offsetof(CumsumBdaPC, out_addr));
                bpc.out_addr = out_addr;
                bpc.in_addr = in_addr;
                submit_bda(&bpc, sizeof(bpc), groups, 1, 1);
                return;
            }
        }
        submit(&pc, groups, 1, 1);
    }

    int axis_ = 0;
    bool exclusive_ = false;
    bool reverse_ = false;
};

// PIMPL façade. CumSum has no image backend, so both branches construct the
// buffer leaf; `backend_buffer` is accepted for the factory signature only.
class Cumsum : public PimplFacade {
  public:
    Cumsum(int fp16, bool /*backend_buffer*/) : PimplFacade(OpType::CUMSUM) {
        impl_ = std::make_unique<CumsumBuffer>(fp16);
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_CUMSUM_HPP_
