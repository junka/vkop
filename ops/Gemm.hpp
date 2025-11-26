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
    Gemm()
        : Operator(OpType::GEMM, gemm_spv, gemm_spv_len,
                   sizeof(gemm::GpuGemmParam)) {
        n_imgs_ = 0;
        types_ = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
        objs_.reserve(types_.size());
    }

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

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        int m = inputs[0]->getShape()[0];
        int n = inputs[1]->getShape()[1];
        int k = inputs[0]->getShape()[1];
        if (transA_) {
            m = inputs[0]->getShape()[1];
            k = inputs[0]->getShape()[0];
        }
        if (transB_) {
            n = inputs[1]->getShape()[0];
            k = inputs[1]->getShape()[1];
        }

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto t) {
            using T = decltype(t);
            auto outputptr = core::as_tensor<T>(outputs[0]);
            if (outputptr->size() == 0) {
                outputptr->resize(std::vector<int>{m, n});
            }
            auto output_buffer = outputptr->as_storage_buffer(m_dev_);
            objs_.emplace_back(output_buffer);
        });
        for (const auto &input : inputs) {
            dispatch_by_dtype(input->dtype(), [&](auto t) {
                using T = decltype(t);
                auto inputptr = core::as_tensor<T>(input);
                auto input_buffer = inputptr->as_storage_buffer(m_dev_);
                inputptr->copyToGPU(m_dev_, m_cmdpool_);
                objs_.emplace_back(input_buffer);
            });
        }
        if (inputs.size() <= 2) {
            objs_.emplace_back(dummyBuffer_);
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

        submit(&para, UP_DIV(n, 16), UP_DIV(m, 16));
    }
    void
    set_runtime_device(std::shared_ptr<VulkanDevice> &dev,
                       std::shared_ptr<VulkanCommandPool> &cmdpool,
                       std::shared_ptr<VulkanCommandBuffer> &cmd) override {
        Operator::set_runtime_device(dev, cmdpool, cmd);

        dummyBuffer_ =
            std::make_shared<VulkanBuffer>(m_dev_, 4,
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                               VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }

  private:
    float alpha_ = 1.0F;
    float beta_ = 1.0F;
    int transA_ = 0;
    int transB_ = 0;
    std::shared_ptr<VulkanBuffer> dummyBuffer_;
};

} // namespace ops
} // namespace vkop
#endif // OPS_GEMM_HPP_
