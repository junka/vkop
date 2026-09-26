// Copyright 2026 @junka
#ifndef OPS_SHAPE_HPP_
#define OPS_SHAPE_HPP_

#include "core/Tensor.hpp"
#include "ops/Operator.hpp"
#include <cstdlib>

// CPU-only op: ONNX Shape. Outputs a 1-D int64 tensor holding the input's
// dims. All 358 Shape nodes in llm.vkopbin read float/fp16 GPU-produced
// data but only need getShape() (host metadata), so no shader is required —
// the output is filled on the host and uploaded to the GPU SSBO.
//
// Perf: this is the #1 decode bottleneck (804ms/round, 46% of the 1.75s —
// measured via VKOP_OPPROF). The cost was the synchronous copyToGPU upload
// (cmd.submit()+cmd.wait() per call, ~2.66ms × 302 calls) of a 32-64 byte
// payload. We now upload NON-synchronously via vkCmdUpdateBuffer recorded
// into the level's command buffer (m_cmd_) — no staging pool, no submit+wait.
// The copy executes when the level submits, batched with all other ops.
// Data is GPU-resident (safe for GPU consumers like the int64 Gather shader)
// but costs ~0 to upload.
namespace vkop {
namespace ops {

class Shape : public Operator {
  public:
    explicit Shape() : Operator(OpType::SHAPE, nullptr, 0, {}) {}

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto shape = inputs[0]->getShape();
        std::vector<int64_t> dims(shape.begin(), shape.end());

        auto output = core::as_tensor<int64_t>(outputs[0]);
        output->resize(std::vector<int>{static_cast<int>(shape.size())});
        output->fillToCPU(dims);
        // Create/reuse the SSBO, then upload non-synchronously into the level
        // command buffer (no submit+wait stall). The explicit src in fillToCPU
        // keeps data_ populated for copyToGPUDeferred and for downstream
        // as_tensor<int64_t>() CPU readers.
        objs_.emplace_back(output->as_storage_buffer(m_dev_, m_cmd_));
        output->copyToGPUDeferred(m_cmd_);
        // Phase 2: the Shape output IS the shape-value SSBO for the dynamic
        // shape-meta chain. Tag it so downstream GPU-driven consumers (Phase 3
        // Binary/MatMul) can read broadcast/dispatch dims from here instead of
        // readback. rank = number of dims (CPU-known from getShape above).
        output->set_shape_ssbo(std::dynamic_pointer_cast<VulkanBuffer>(
                                   output->as_storage_buffer(m_dev_, m_cmd_)),
                               static_cast<int>(shape.size()));
        // Host-shape mode: the dims were filled on the host above and data_ is
        // recomputed every round, so it stays authoritative — downstream
        // host-side shape consumers (Gather/Concat/Reshape) read it with no
        // GPU readback.
        if (host_shape_enabled()) {
            output->set_host_authoritative();
        }
    }

    // Shape is CPU-only (no pipeline/spv, no submit()). Its output changes
    // every round (kv_len grows -> getShape() differs), so it must NOT be
    // replay-cached — the record-once-replay state machine (keyed on submit()
    // fingerprints) would see an empty fingerprint, mark it CACHED, and skip
    // execute() on later rounds, leaving the output stale. Refuse replay.
    void enable_replay(bool /*v*/) override { replay_enabled_ = false; }
};

} // namespace ops
} // namespace vkop
#endif // OPS_SHAPE_HPP_
