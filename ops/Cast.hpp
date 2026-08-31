// Copyright 2026 @junka
#ifndef OPS_CAST_HPP_
#define OPS_CAST_HPP_

#include "ops/BufferBase.hpp"

extern "C" {
extern unsigned char buffer_cast_spv[];
extern unsigned int buffer_cast_spv_len;
}

namespace vkop {
namespace ops {

namespace cast {
struct alignas(16) CastPC {
    int mode;  // 0 = fp32->fp16, 1 = fp16->fp32
    int total; // input element count
    int _pad0;
    int _pad1;
};
} // namespace cast

// SSBO-only op: ONNX Cast (fp32 <-> fp16). The model has 227 Cast nodes
// converting between float32 (to=1) and float16 (to=10); there are no
// int64 casts. Every input is GPU-produced (fp16 hidden states, fp32 rotary
// products), so this must be a GPU shader op — reading back a GPU-produced
// input during execute() would see stale data (the runtime records all
// command buffers before submitting any).
class Cast : public BufferFactory {
  public:
    explicit Cast()
        : BufferFactory(OpType::CAST, buffer_cast_spv, buffer_cast_spv_len,
                        {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                        sizeof(cast::CastPC)) {}

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("to") != attributes.end()) {
            to_ = std::stol(attributes.at("to"));
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto shape = inputs[0]->getShape();
        int total = total_elems(shape);

        // int64 -> fp32 cast (the rotary position-id Cast). int64 tensors are
        // CPU-resident during recording (shape-meta domain), so cast on the
        // host: read the int64 input, write fp32 values, resize + re-upload.
        // The buffer_cast shader only handles fp32<->fp16, not int64.
        if (inputs[0]->dtype() == typeid(int64_t) &&
            outputs[0]->dtype() == typeid(float)) {
            auto src = core::as_tensor<int64_t>(inputs[0]);
            // Unconditional readback: a cross-round-recycled GPU input may
            // have stale CPU data_ (see SqueezeUnsqueeze/ScatterElements fix).
            src->copyToCPU(m_cmdpool_);
            int src_avail = src->num_elements();
            std::vector<float> out(total);
            for (int i = 0; i < total && i < src_avail; ++i) {
                out[i] = static_cast<float>((*src)[i]);
            }
            auto output = core::as_tensor<float>(outputs[0]);
            output->resize(shape);
            output->fillToCPU(out);
            objs_.emplace_back(output->as_storage_buffer(m_dev_, m_cmd_));
            output->copyToGPU(m_cmdpool_, out.data());
            return;
        }

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->num_elements() != total_elems(shape)) {
                output->resize(shape);
            }
            bind_ssbo<T>(outputs[0], /*is_output=*/true);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            bind_ssbo<T>(inputs[0], /*is_output=*/false);
        });

        cast::CastPC pc{};
        // mode 0 (f32->f16): each thread packs two fp32 words into one half2
        // output word. mode 1 (f16->f32): each thread unpacks one half2 input
        // word into two fp32 output words. Either way one thread per word.
        pc.mode = (inputs[0]->dtype() == typeid(uint16_t)) ? 1 : 0;
        pc.total = total;
        submit(&pc, UP_DIV((total + 1) / 2, 256), 1, 1);
    }

    int to_ = 0;
};

} // namespace ops
} // namespace vkop
#endif // OPS_CAST_HPP_
