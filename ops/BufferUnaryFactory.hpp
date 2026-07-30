// Copyright 2025 @junka
#ifndef OPS_BUFFER_UNARY_FACTORY_HPP_
#define OPS_BUFFER_UNARY_FACTORY_HPP_

#include "ops/BufferBase.hpp"

// Shared base for elementwise unary buffer (SSBO) ops. Mirrors the image
// path's UnaryFactory: each per-op leaf (ReluBuffer, SigmoidBuffer, ...)
// derives from this and just passes its own `buffer_<op>_spv` symbol; the
// shared execute() below binds the input/output SSBOs, fills the
// UnaryElemPC push constant (shape + element count), and dispatches one
// thread per element (fp32) or per uint word (fp16 packed half2).
//
// fp16 note: where a per-op shader ships a -DFP16 build, the leaf ctor
// passes that spv when fp16 is set; otherwise fp32. The fp16 path packs
// two elements per uint word, so dispatch is one thread per word (two
// elements) — no two threads target the same word.

namespace vkop {
namespace ops {

class BufferUnaryFactory : public BufferFactory {
  public:
    BufferUnaryFactory(OpType type, uint8_t *spv, uint32_t spv_len, int fp16)
        : BufferFactory(type, spv, spv_len,
                        {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                        sizeof(UnaryElemPC), fp16) {}

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto shape = inputs[0]->getShape();
        int total = total_elems(shape);

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->size() == 0) {
                output->resize(shape);
            }
            bind_ssbo<T>(outputs[0], /*is_output=*/true);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            bind_ssbo<T>(inputs[0], /*is_output=*/false);
        });

        UnaryElemPC pc{};
        pc.rank = static_cast<int>(shape.size());
        fill_dims(pc.outDims, shape);
        fill_dims(pc.in0Dims, shape);
        pc.total = total;
        // fp16 packs two elements per uint word; dispatch one thread per
        // word (two elements) so no two threads target the same word.
        int nthreads = (fp16_ != 0) ? (total + 1) / 2 : total;
        submit(&pc, UP_DIV(nthreads, 256), 1, 1);
    }
};

} // namespace ops
} // namespace vkop

#endif // OPS_BUFFER_UNARY_FACTORY_HPP_
