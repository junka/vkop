// Copyright 2025 @junka
#ifndef OPS_GEMM_HPP_
#define OPS_GEMM_HPP_

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

extern unsigned char gemm_spv[];
extern unsigned int gemm_spv_len;

namespace vkop {
namespace ops {

namespace gemm {
struct alignas(16) GpuGemmParam {
    int M;
    int N;
    int K;
    int has_bias;
    int transA;
    int transB;
    float alpha;
    float beta;
};
} // namespace gemm

class Gemm : public Operator {
  public:
    Gemm() : Operator(OpType::GEMM) {}

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("alpha") != attributes.end()) {
            auto alpha = std::stof(attributes.at("alpha"));
            alpha_ = alpha;
        }
        if (attributes.find("beta") != attributes.end()) {
            auto beta = std::stof(attributes.at("beta"));
            beta_ = beta;
        }
        if (attributes.find("transA") != attributes.end()) {
            transA_ = std::stoi(attributes.at("transA"));
        }
        if (attributes.find("transB") != attributes.end()) {
            transB_ = std::stoi(attributes.at("transB"));
        }
    }

    template <typename T>
    void prepare(std::vector<std::shared_ptr<core::ITensor>> inputs,
                 std::vector<std::shared_ptr<core::ITensor>> outputs) {
        auto inputa = core::as_tensor<T>(inputs[0]);
        auto inputb = core::as_tensor<T>(inputs[1]);
        auto output = core::as_tensor<T>(outputs[0]);
        auto inputc =
            inputs.size() > 2 ? core::as_tensor<T>(inputs[2]) : nullptr;
        int m = inputa->getShape()[0];
        int n = inputb->getShape()[1];
        if (transA_) {
            m = inputa->getShape()[1];
        }
        if (transB_) {
            n = inputb->getShape()[0];
        }
        if (output->size() == 0) {
            output->resize(std::vector<int>{m, n});
        }

        auto inputa_buffer = inputa->as_storage_buffer(m_dev_);
        auto inputb_buffer = inputb->as_storage_buffer(m_dev_);
        auto inputc_buffer =
            inputc ? inputc->as_storage_buffer(m_dev_)
                   : std::make_shared<VulkanBuffer>(
                         m_dev_, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        auto output_buffer = output->as_storage_buffer(m_dev_);

        types_ = {output_buffer->getDescriptorType(),
                  inputa_buffer->getDescriptorType(),
                  inputb_buffer->getDescriptorType(),
                  inputc_buffer->getDescriptorType()};
        objs_ = {output_buffer, inputa_buffer, inputb_buffer, inputc_buffer};
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
            auto inputc =
                inputs.size() > 2 ? core::as_tensor<float>(inputs[2]) : nullptr;
            auto output = core::as_tensor<float>(outputs[0]);
            int m = inputa->getShape()[0];
            int n = inputb->getShape()[1];
            int k = inputa->getShape()[1];
            if (transA_) {
                m = inputa->getShape()[1];
                k = inputa->getShape()[0];
            }
            if (transB_) {
                n = inputb->getShape()[0];
                k = inputb->getShape()[1];
            }

            inputa->copyToGPU(m_dev_, m_cmdpool_);
            inputb->copyToGPU(m_dev_, m_cmdpool_);
            if (inputc) {
                inputc->copyToGPU(m_dev_, m_cmdpool_);
            }

            gemm::GpuGemmParam para;
            para.M = m;
            para.N = n;
            para.K = k;
            para.alpha = alpha_;
            para.beta = beta_;
            para.transA = transA_;
            para.transB = transB_;
            para.has_bias = (inputs.size() > 2 ? 1 : 0);

            submit(&para, sizeof(gemm::GpuGemmParam), gemm_spv, gemm_spv_len,
                   UP_DIV(n, 16), UP_DIV(m, 16));
        }
    }

  private:
    float alpha_ = 1.0F;
    float beta_ = 1.0F;
    int transA_ = 0;
    int transB_ = 0;
};

} // namespace ops
} // namespace vkop
#endif // OPS_GEMM_HPP_
