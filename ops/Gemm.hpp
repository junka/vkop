// Copyright 2025 @junka
#ifndef OPS_GEMM_HPP_
#define OPS_GEMM_HPP_

#include "Operator.hpp"
#include "ops/Conv2d.hpp"

extern "C" {
extern unsigned char gemm_spv[];
extern unsigned int gemm_spv_len;
};

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
    int fp16a;
    int fp16b;
    int fp16c;
    int activation;
};
} // namespace gemm

class Gemm : public Operator {
  public:
    Gemm()
        : Operator(OpType::GEMM, gemm_spv, gemm_spv_len,
                   {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                    DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                   sizeof(gemm::GpuGemmParam)) {}

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
            transA_ = std::stol(attributes.at("transA"));
        }
        if (attributes.find("transB") != attributes.end()) {
            transB_ = std::stol(attributes.at("transB"));
        }

        if (attributes.find("activation") != attributes.end()) {
            std::string activation = attributes.at("activation");
            if (activation == "Relu") {
                activation_ = conv2d::ActivationMode::RELU;
            } else if (activation == "Sigmoid") {
                activation_ = conv2d::ActivationMode::SIGMOID;
            } else if (activation == "Tanh") {
                activation_ = conv2d::ActivationMode::TANH;
            } else if (activation == "HardSwish") {
                activation_ = conv2d::ActivationMode::HARDSWISH;
            } else if (activation == "Mish") {
                activation_ = conv2d::ActivationMode::MISH;
            } else if (activation == "Relu6") {
                activation_ = conv2d::ActivationMode::RELU6;
            } else if (activation == "Swish") {
                activation_ = conv2d::ActivationMode::SWISH;
            } else {
                throw std::invalid_argument("Unsupported activation: " +
                                            activation);
            }
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
                objs_.emplace_back(input_buffer);
            });
        }
        if (inputs.size() <= 3) {
            objs_.emplace_back(dummy_buffer_);
        }
        gemm::GpuGemmParam para;
        para.M = m;
        para.N = n;
        para.K = k;
        para.alpha = alpha_;
        para.beta = beta_;
        para.transA = transA_;
        para.transB = transB_;
        if (inputs[0]->dtype() == typeid(uint16_t)) {
            para.fp16a = 1;
        } else if (inputs[0]->dtype() == typeid(uint8_t)) {
            para.fp16a = 2;
        } else {
            para.fp16a = 0;
        }
        if (inputs[1]->dtype() == typeid(uint16_t)) {
            para.fp16b = 1;
        } else if (inputs[1]->dtype() == typeid(uint8_t)) {
            para.fp16b = 2;
        } else {
            para.fp16b = 0;
        }
        para.activation = static_cast<int>(activation_);

        if (inputs.size() > 2) {
            para.has_bias = 1;
            if (inputs[2]->dtype() == typeid(uint16_t)) {
                para.fp16c = 1;
            } else if (inputs[2]->dtype() == typeid(uint8_t)) {
                para.fp16c = 2;
            } else {
                para.fp16c = 0;
            }
        }

        submit(&para, UP_DIV(n, 16), UP_DIV(m, 16), 1);
    }
    void set_runtime_device(
        const std::shared_ptr<VulkanDevice> &dev,
        const std::shared_ptr<VulkanCommandPool> &cmdpool) override {
        Operator::set_runtime_device(dev, cmdpool);
    }

    float alpha_ = 1.0F;
    float beta_ = 1.0F;
    int transA_ = 0;
    int transB_ = 0;
    conv2d::ActivationMode activation_ = conv2d::ActivationMode::NONE;
};

} // namespace ops
} // namespace vkop
#endif // OPS_GEMM_HPP_
