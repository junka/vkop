// Copyright 2026 @junka
#ifndef OPS_SHAPE_HPP_
#define OPS_SHAPE_HPP_

#include "core/Tensor.hpp"
#include "ops/Operator.hpp"
#include <cstdlib>

// CPU-only op: ONNX Shape. Outputs a 1-D int64 tensor holding the input's
// dims. All 358 Shape nodes in llm.vkopbin read float/fp16 GPU-produced
// data but only need getShape() (host metadata), so no shader is required —
// the output is filled on the host and uploaded to the GPU SSBO for any
// downstream int64 consumers (all of which read it via as_tensor<int64_t>()
// on the CPU copy).
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
        objs_.emplace_back(output->as_storage_buffer(m_dev_, m_cmd_));
        // Explicit src keeps the CPU copy alive for downstream as_tensor<>()
        // readers; without it copyToGPU would clear data_ after upload.
        output->copyToGPU(m_cmdpool_, dims.data());
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_SHAPE_HPP_
