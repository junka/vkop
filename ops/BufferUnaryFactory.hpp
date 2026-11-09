// Copyright 2025 @junka
#ifndef OPS_BUFFER_UNARY_FACTORY_HPP_
#define OPS_BUFFER_UNARY_FACTORY_HPP_

#include "ops/BufferBase.hpp"
#include <cstring>

// Shared base for elementwise unary buffer (SSBO) ops. Mirrors the image
// path's UnaryFactory: each per-op leaf (ReluBuffer, SigmoidBuffer, ...)
// derives from this and just names its own OpType; the shared execute() below
// binds the input/output SSBOs, fills the UnaryElemPC push constant (shape +
// element count), and dispatches one thread per element (fp32) or per uint
// word (fp16 packed half2).
//
// fp16 note: where a per-op shader ships a -DFP16 build, shader_key_for picks
// it off the fp16 flag (the leaf ctor passes `fp16` through and the base
// resolves the matching key). The fp16 path packs two elements per uint word,
// so dispatch is one thread per word (two elements) — no two threads target
// the same word.

namespace vkop {
namespace ops {

class BufferUnaryFactory : public BufferFactory {
  public:
    BufferUnaryFactory(OpType type, int fp16)
        : BufferFactory(type,
                        {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                        sizeof(UnaryElemPC), fp16) {}

  private:
    // Every unary leaf shares one push-constant layout and dispatch shape, so
    // the BDA plumbing lives here once. The archive decides whether a given
    // leaf actually has a <key>_bda kernel; the descriptor pipeline remains
    // the per-dispatch fallback (see try_build_bda_pipeline).
    bool supports_bda_variant() const override { return true; }
    size_t bda_pc_size() const override { return sizeof(UnaryBdaPC); }

    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto shape = inputs[0]->getShape();
        int total = total_elems(shape);

        std::shared_ptr<VulkanBuffer> out_buf;
        std::shared_ptr<VulkanBuffer> in_buf;
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->num_elements() != total_elems(shape)) {
                output->resize(shape);
            }
            out_buf = bind_ssbo<T>(outputs[0], /*is_output=*/true);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            in_buf = bind_ssbo<T>(inputs[0], /*is_output=*/false);
        });

        UnaryElemPC pc{};
        pc.rank = static_cast<int>(shape.size());
        fill_dims(pc.outDims, shape);
        fill_dims(pc.in0Dims, shape);
        pc.total = total;
        // fp16 packs two elements per uint word; dispatch one thread per
        // word (two elements) so no two threads target the same word.
        int nthreads = (fp16_ != 0) ? (total + 1) / 2 : total;
        int groups = UP_DIV(nthreads, 256);

        // BDA variant: same math, addresses instead of bindings. It is chosen
        // per dispatch, not per op: a host-imported tensor or a view whose
        // offset cannot satisfy buffer_reference_align=4 takes the descriptor
        // pipeline that is still alive.
        if (bda_ready() && in_buf && out_buf && in_buf->hasDeviceAddress() &&
            out_buf->hasDeviceAddress()) {
            const VkDeviceAddress in_addr = in_buf->getDeviceAddress();
            const VkDeviceAddress out_addr = out_buf->getDeviceAddress();
            if ((in_addr % 4) == 0 && (out_addr % 4) == 0) {
                UnaryBdaPC bpc{};
                std::memcpy(&bpc, &pc, offsetof(UnaryBdaPC, in_addr));
                bpc.in_addr = in_addr;
                bpc.out_addr = out_addr;
                submit_bda(&bpc, sizeof(bpc), groups, 1, 1);
                return;
            }
        }
        submit(&pc, groups, 1, 1);
    }
};

} // namespace ops
} // namespace vkop

#endif // OPS_BUFFER_UNARY_FACTORY_HPP_
