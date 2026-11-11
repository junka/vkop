// Copyright 2026 @junka
#ifndef OPS_CAST_HPP_
#define OPS_CAST_HPP_

#include "ops/BufferBase.hpp"
#include <cstdio>
#include <cstdlib>

namespace vkop {
namespace ops {

namespace cast {
struct alignas(16) CastPC {
    int mode;  // 0 = fp32->fp16, 1 = fp16->fp32
    int total; // input element count
    int _pad0;
    int _pad1;
};

// Device-address variant (buffer/cast.comp built with -DBDA). The single
// module handles both conversion directions, so there is no _fp16 key.
struct alignas(16) CastBdaPC {
    CastPC base;
    VkDeviceAddress out_addr;
    VkDeviceAddress in_addr;
};
static_assert(offsetof(CastBdaPC, out_addr) == 16, "CastBdaPC drift");
static_assert(offsetof(CastBdaPC, in_addr) == 24, "CastBdaPC drift");
static_assert(sizeof(CastBdaPC) <= 128, "CastBdaPC PC overflow");

// int64 -> fp32 (buffer/cast_int64.comp). Same 16-byte scalar prefix as
// CastPC; the two buffers are out (float) then in (int64 as ivec2). Needs the
// shaderInt64 device feature — this is the only int64 kernel that does 64-bit
// ARITHMETIC rather than moving the bytes.
struct alignas(16) CastInt64PC {
    int total;
    int _pad0;
    int _pad1;
    int _pad2;
};
static_assert(sizeof(CastInt64PC) == 16, "CastInt64PC drift");
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
        : BufferFactory(OpType::CAST,
                        {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                        sizeof(cast::CastPC)) {}

    // The int64->fp32 Cast used to read position-id data back to the host every
    // round and convert it in a C++ loop, with a "+1 prediction" shortcut
    // (LINEAR_GROW) that guessed next round's value instead of reading it.
    // Both are gone: with shaderInt64 enabled the conversion runs in
    // buffer/cast_int64.comp, so it is a plain dispatch and the level replays.
    // The CPU fallback below remains for a device without the feature, and it
    // no longer predicts anything — a prediction cannot survive a replay.

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("to") != attributes.end()) {
            to_ = std::stol(attributes.at("to"));
        }
    }
    bool has_host_side_effects() const override { return true; }
    // The only host work left on this path is the CPU fallback's readback, and
    // that path is taken only when the device lacks shaderInt64 (or for the
    // fp32<->fp16 dispatches, which do none). Reported per instance so a
    // replayable GPU round does not look dynamic.
    bool did_host_work_last_round() const override {
        return host_work_observed_ ? host_work_last_round_ : false;
    }

  private:
    bool supports_bda_variant() const override { return true; }
    size_t bda_pc_size() const override { return sizeof(cast::CastBdaPC); }
    bool bda_variant_ignores_fp16() const override { return true; }

    // int64-data pipeline (buffer/cast_int64). Built lazily on the first int64
    // execute; same 2× STORAGE layout as the base module, so it reuses types_.
    void ensure_int64_pipeline() {
        if (pipeline_int64_) return;
        bool use_uab = update_after_bind_ &&
                       m_dev_->is_support_descriptor_update_after_bind();
        auto i64 = resolve_spv("buffer_cast_int64");
        pipeline_int64_ = std::make_unique<VulkanPipeline>(
            m_dev_->getLogicalDevice(), types_, sizeof(cast::CastInt64PC),
            i64.first, i64.second, use_uab, required_subgroup_size_);
        for (auto &ds : m_ds_int64_) {
            ds = pipeline_int64_->allocDescriptorSets();
        }
        build_bda_table_module("buffer_cast_int64_bda",
                               sizeof(cast::CastInt64PC), 2, bda_i64_);
    }
    // Route the dispatch through the int64 pipeline while int64_mode_ is set.
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
                static_cast<uint32_t>(sizeof(cast::CastInt64PC)), ptr);
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

    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        // Host work here is the int64->fp32 CPU fallback's readback, taken only
        // on a device without shaderInt64; every other path (including the
        // shaderInt64 int64 cast) is a pure dispatch. The instance answer is
        // refined further down by the branch that actually runs.
        set_did_host_work_last_round(false);
        auto shape = inputs[0]->getShape();
        int total = total_elems(shape);

        // `to_` (ONNX's TensorProto id: 1 = fp32, 10 = fp16) and the output
        // tensor's dtype must agree. The runtime sizes the output tensor from
        // this same attribute and the mode selection below reads the tensor
        // dtypes, so a disagreement means one of the two is stale — and the
        // shader would then reinterpret the buffer at the wrong element
        // stride. A third dtype (int64 indices, int8, bool) has no kernel at
        // all: it must not fall through to a float word copy.
        const int to_dtype = outputs[0]->dtype() == typeid(uint16_t) ? 10
                             : outputs[0]->dtype() == typeid(float)  ? 1
                                                                     : 0;
        if (to_ != 0 && to_ != to_dtype) {
            throw std::runtime_error(
                std::string("Cast: the graph says to=") + std::to_string(to_) +
                " but the output tensor is " +
                core::elem_name(outputs[0]->elem_kind()) +
                " (node's input is " + core::elem_name(inputs[0]->elem_kind()) +
                "); only FLOAT and FLOAT16 are supported");
        }

        // int64 -> fp32 cast (the rotary position-id Cast).
        //
        // Two implementations, and the choice is a DEVICE capability:
        //  - shaderInt64 enabled: the shader recombines each ivec2 into a real
        //    64-bit value and converts. Pure dispatch, no host work, so the
        //    level replays like any other.
        //  - otherwise: the CPU path below (readback, convert, re-upload).
        //    It is real host work every round, and there is no way to make a
        //    replay reproduce it — the +1 "prediction" this branch used to
        //    carry was exactly that mistake: it skipped the readback, so a
        //    replay froze the value at the last recorded round (measured: the
        //    Cast output stuck at round 3 while position_ids kept advancing).
        if (inputs[0]->dtype() == typeid(int64_t) &&
            outputs[0]->dtype() == typeid(float)) {
            if (m_dev_ && m_dev_->is_support_shader_int64()) {
                ensure_int64_pipeline();
                int64_mode_ = true;
                cast::CastInt64PC pc{};
                pc.total = total;
                // out (fp32) then in (int64 as ivec2) — same binding order as
                // every other int64 twin.
                dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
                    using T = decltype(dummy);
                    auto output = core::as_tensor<T>(outputs[0]);
                    if (output->num_elements() != total_elems(shape)) {
                        output->resize(shape);
                    }
                    objs_.emplace_back(
                        output->as_storage_buffer(m_dev_, m_cmd_));
                });
                auto src = core::as_tensor<int64_t>(inputs[0]);
                objs_.emplace_back(src->as_storage_buffer(m_dev_, nullptr));
                submit(&pc, UP_DIV(total, 256), 1, 1);
                // The shader writes the output; flush it so the next level's
                // reader sees the values (same pattern as Equal's int64 path).
                auto out_buf =
                    std::dynamic_pointer_cast<VulkanBuffer>(objs_[0]);
                if (out_buf) {
                    out_buf->shaderWriteBarrier(m_cmd_->get(),
                                                (VkDeviceSize)total_elems(shape) *
                                                    sizeof(float));
                }
                return;
            }
            // CPU fallback: read back, convert on the host, re-upload.
            // No +1 prediction here — it cannot survive a replay.
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
            output->copyToGPUDeferred(m_cmd_);
            return;
        }

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

        cast::CastPC pc{};
        // mode 0 (f32->f16): each thread packs two fp32 words into one half2
        // output word. mode 1 (f16->f32): each thread unpacks one half2 input
        // word into two fp32 output words. Either way one thread per word.
        // mode 2 (same dtype both sides — an identity Cast like torch's
        // .to(fp16) on an already-fp16 tensor): plain half2 word copy;
        // picking 0/1 here reinterprets the buffer at the wrong element
        // stride and writes 2x the output bytes.
        //
        // Anything else is a conversion no kernel implements: the old code
        // treated "not fp16" as fp32 and would have run the f32->f16 packing
        // over, say, an int8 buffer.
        const auto &idt = inputs[0]->dtype();
        if (idt != typeid(float) && idt != typeid(uint16_t)) {
            throw std::runtime_error(
                std::string("Cast: no kernel converts from ") +
                core::elem_name(inputs[0]->elem_kind()) +
                " (only fp32<->fp16, plus the int64->fp32 host path)");
        }
        if (idt == outputs[0]->dtype()) {
            pc.mode = 2;
        } else {
            pc.mode = (idt == typeid(uint16_t)) ? 1 : 0;
        }
        pc.total = total;
        const int groups = UP_DIV((total + 1) / 2, 256);
        if (bda_ready() && out_buf && in_buf && out_buf->hasDeviceAddress() &&
            in_buf->hasDeviceAddress()) {
            const VkDeviceAddress out_addr = out_buf->getDeviceAddress();
            const VkDeviceAddress in_addr = in_buf->getDeviceAddress();
            if ((out_addr % 4) == 0 && (in_addr % 4) == 0) {
                cast::CastBdaPC bpc{};
                bpc.base = pc;
                bpc.out_addr = out_addr;
                bpc.in_addr = in_addr;
                submit_bda(&bpc, sizeof(bpc), groups, 1, 1);
                return;
            }
        }
        submit(&pc, groups, 1, 1);
    }

    int to_ = 0;
};

} // namespace ops
} // namespace vkop
#endif // OPS_CAST_HPP_
