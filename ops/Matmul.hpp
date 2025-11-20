// Copyright 2025 @junka
#ifndef OPS_MATMUL_HPP_
#define OPS_MATMUL_HPP_

#include "Operator.hpp"

#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanCommandBuffer.hpp"
#include "vulkan/VulkanCommandPool.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanImage.hpp"
#include "vulkan/VulkanPipeline.hpp"
#include "vulkan/VulkanQueryPool.hpp"

extern unsigned char matmul_spv[];
extern unsigned int matmul_spv_len;

namespace vkop {
namespace ops {

namespace matmul {
struct alignas(16) GpuMatMulParam {
    int M;
    int N;
    int K;
};
} // namespace matmul
class MatMul : public Operator {
  public:
    MatMul() : Operator(OpType::MATMUL){};

    template <typename T>
    void prepare(std::vector<std::shared_ptr<core::ITensor>> inputs,
                 std::vector<std::shared_ptr<core::ITensor>> outputs) {
        auto inputa = core::as_tensor<T>(inputs[0]);
        auto inputb = core::as_tensor<T>(inputs[1]);
        auto output = core::as_tensor<T>(outputs[0]);
        int m = inputa->getShape()[0];
        int n = inputb->getShape()[1];
        if (output->size() == 0) {
            output->resize(std::vector<int>{m, n});
        }

        auto inputa_buffer = inputa->as_storage_buffer(m_dev_);
        auto inputb_buffer = inputb->as_storage_buffer(m_dev_);
        auto output_buffer = output->as_storage_buffer(m_dev_);

        types_ = {output_buffer->getDescriptorType(),
                  inputa_buffer->getDescriptorType(),
                  inputb_buffer->getDescriptorType()};
        objs_ = {output_buffer, inputa_buffer, inputb_buffer};
    }

    void
    apply(const std::vector<std::shared_ptr<core::ITensor>> &inputs,
          const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        if (inputs[0]->dtype() == typeid(float)) {
            prepare<float>(inputs, outputs);
        } else if (inputs[0]->dtype() == typeid(uint16_t)) {
            prepare<uint16_t>(inputs, outputs);
        } else {
            LOG_ERROR("Unsupported data type");
        }
    }

    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        if (inputs[0]->dtype() == typeid(float)) {
            auto inputa = core::as_tensor<float>(inputs[0]);
            auto inputb = core::as_tensor<float>(inputs[1]);
            auto output = core::as_tensor<float>(outputs[0]);
            int m = inputa->getShape()[0];
            int n = inputb->getShape()[1];

            inputa->copyToGPU(m_dev_, m_cmdpool_);
            inputb->copyToGPU(m_dev_, m_cmdpool_);

            matmul::GpuMatMulParam para;
            para.M = m;
            para.N = n;
            para.K = inputa->getShape()[1];

            submit(&para, sizeof(matmul::GpuMatMulParam), matmul_spv,
                   matmul_spv_len, UP_DIV(n, 16), UP_DIV(m, 16));
        }
    }

  private:
};

} // namespace ops
} // namespace vkop
#endif // OPS_MATMUL_HPP_
