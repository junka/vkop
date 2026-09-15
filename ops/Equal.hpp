// Copyright 2026 @junka
#ifndef OPS_EQUAL_HPP_
#define OPS_EQUAL_HPP_

#include "ops/BufferBinaryFactory.hpp"
#include <numeric>

extern "C" {
extern unsigned char buffer_equal_spv[];
extern unsigned int buffer_equal_spv_len;
extern unsigned char buffer_equal_int64_spv[];
extern unsigned int buffer_equal_int64_spv_len;
}

namespace vkop {
namespace ops {

// SSBO-only op: C[i] = (A[i] == B[i]) ? 1 : 0. Supports broadcasting.
// Output is uint (1=true, 0=false), matching Where's condition buffer.
class Equal : public BufferBinaryFactory {
  public:
    explicit Equal()
        : BufferBinaryFactory(OpType::EQUAL, buffer_equal_spv,
                              buffer_equal_spv_len, /*fp16=*/0) {}

    // Build the int64-data pipeline lazily on first int64 execute. The int64
    // shader (equal_int64.comp) has the same descriptor layout (3× STORAGE:
    // output, A, B) and PC layout (BinaryElemPC) as the fp32 equal, so it
    // builds from the inherited types_/pc_size_; only the spv differs.
    void ensure_int64_pipeline() {
        if (pipeline_int64_)
            return;
        bool use_uab = update_after_bind_ &&
                       m_dev_->is_support_descriptor_update_after_bind();
        pipeline_int64_ = std::make_unique<VulkanPipeline>(
            m_dev_->getLogicalDevice(), types_, pc_size_,
            reinterpret_cast<const uint32_t *>(buffer_equal_int64_spv),
            buffer_equal_int64_spv_len, use_uab, required_subgroup_size_);
        for (auto &ds : m_ds_int64_) {
            ds = pipeline_int64_->allocDescriptorSets();
        }
    }

    // When in int64 mode, bind the int64 pipeline + its descriptor sets
    // instead of the base fp32 pipeline.
    void submit(void *ptr, int width, int height, int layers) override {
        if (!int64_mode_) {
            Operator::submit(ptr, width, height, layers);
            return;
        }
        if (!m_ds_int64_[m_id_]) {
            m_ds_int64_[m_id_] = pipeline_int64_->allocDescriptorSets();
        }
        fillWriteDescriptorSets(m_ds_int64_[m_id_]);
        pipeline_int64_->updateDescriptorSets(writes_);
        m_cmd_->bind(*pipeline_int64_, m_ds_int64_[m_id_]);
        if (ptr) {
            m_cmd_->push_constants(*pipeline_int64_,
                                   static_cast<uint32_t>(pc_size_), ptr);
        }
        m_cmd_->dispatch(width, height, layers);
        if (replay_enabled_) {
            record_fingerprint(ptr, width, height, layers);
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto shape_a = inputs[0]->getShape();
        auto shape_b = inputs[1]->getShape();
        auto out_shape = computeBroadcastShape(shape_a, shape_b);
        int total = total_elems(out_shape);

        // int64 comparison runs on the GPU (equal_int64.comp). The runtime
        // allocates the output as int64 when the inputs are int64. The shader
        // reads int64 A/B as ivec2[] and writes int64 1/0 (ivec2(eq,0)) as the
        // condition for downstream Where (which reads `cond != 0`). The output
        // stays GPU-resident — Where's int64 GPU path reads it as an SSBO, so
        // the Equal->Where chain never crosses GPU->CPU. (When Where is NOT yet
        // GPU-ified, its copyToCPU(cond) re-reads the GPU buffer
        // authoritatively — correct, just a sync readback that this
        // optimization aims to remove by also GPU-ifying Where.)
        if (inputs[0]->dtype() == typeid(int64_t)) {
            ensure_int64_pipeline();
            int64_mode_ = true;

            // Output SSBO (int64 as ivec2 — 8 bytes/elem, same stride).
            auto output = core::as_tensor<int64_t>(outputs[0]);
            output->resize(out_shape);
            objs_.emplace_back(output->as_storage_buffer(m_dev_, m_cmd_));
            // A/B input SSBOs (nullptr cmd: no barrier — produced by a prior
            // level's shader; the level submit provides ordering).
            auto a = core::as_tensor<int64_t>(inputs[0]);
            objs_.emplace_back(a->as_storage_buffer(m_dev_, nullptr));
            auto b = core::as_tensor<int64_t>(inputs[1]);
            objs_.emplace_back(b->as_storage_buffer(m_dev_, nullptr));

            BinaryElemPC pc{};
            pc.rank = static_cast<int>(out_shape.size());
            fill_dims(pc.outDims, out_shape);
            fill_dims_broadcast(pc.in0Dims, shape_a, pc.rank);
            fill_dims_broadcast(pc.in1Dims, shape_b, pc.rank);
            pc.activation = 0;
            pc.broadcast =
                (shape_a == out_shape && shape_b == out_shape) ? 0 : 1;
            pc.total = total;
            submit(&pc, UP_DIV(total, 256), 1, 1);
            int64_mode_ = false;
            return;
        }

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->num_elements() != total_elems(out_shape)) {
                output->resize(out_shape);
            }
            bind_ssbo<T>(outputs[0], true);
        });
        bind_ssbo<float>(inputs[0], false);
        bind_ssbo<float>(inputs[1], false);

        BinaryElemPC pc{};
        pc.rank = static_cast<int>(out_shape.size());
        fill_dims(pc.outDims, out_shape);
        fill_dims_broadcast(pc.in0Dims, shape_a, pc.rank);
        fill_dims_broadcast(pc.in1Dims, shape_b, pc.rank);
        pc.activation = 0;
        pc.broadcast = (shape_a == out_shape && shape_b == out_shape) ? 0 : 1;
        pc.total = total;
        submit(&pc, UP_DIV(total, 256), 1, 1);
    }

    std::unique_ptr<VulkanPipeline> pipeline_int64_;
    VkDescriptorSet m_ds_int64_[vkop::kInflight] = {nullptr};
    bool int64_mode_ = false;
};

} // namespace ops
} // namespace vkop
#endif // OPS_EQUAL_HPP_
