// Copyright 2025 @junka
#ifndef OPS_SOFTMAX_HPP_
#define OPS_SOFTMAX_HPP_

#include "UnaryFactory.hpp"

extern unsigned char softmax_spv[];
extern unsigned int softmax_spv_len;

namespace vkop {
namespace ops {
namespace softmax {

using ivec4 = int[4];
using ivec2 = int[2];

struct GpuSoftMaxParam {
    ivec4 outImgSize;
    ivec4 outShape;
    int axis; // 0: N, 1: C, 2: H, 3: W
};

} // namespace softmax

class Softmax : public Operator {
  public:
    Softmax() : Operator(OpType::SOFTMAX) {}

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("axis") != attributes.end()) {
            auto axis = std::stoi(attributes.at("axis"));
            axis_ = axis;
        } else if (attributes.find("dim") != attributes.end()) {
            auto axis = std::stoi(attributes.at("dim"));
            axis_ = axis;
        }
    }

    template <typename T>
    void prepare(std::vector<std::shared_ptr<core::ITensor>> inputs,
                 std::vector<std::shared_ptr<core::ITensor>> outputs) {
        auto input = core::as_tensor<T>(inputs[0]);
        auto output = core::as_tensor<T>(outputs[0]);

        auto input_shape = input->getShape();

        if (input_shape.size() != 4) {
            throw std::invalid_argument("Input must have 4 dimensions.");
        }

        if (output->size() == 0) {
            output->resize(input_shape);
        }

        auto input_image = input->as_input_image(m_dev_, m_cmdpool_);
        auto output_image = output->as_output_image(m_dev_, m_cmdpool_);

        paramBuffer_ = std::make_shared<VulkanBuffer>(
            m_dev_, sizeof(softmax::GpuSoftMaxParam),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        types_ = {output_image->getDescriptorType(),
                  input_image->getDescriptorType(),
                  paramBuffer_->getDescriptorType()};
        objs_ = {output_image, input_image, paramBuffer_};
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
            auto input = core::as_tensor<float>(inputs[0]);
            auto output = core::as_tensor<float>(outputs[0]);
            auto input_shape = input->getShape();

            int batch = input_shape[0];
            int depth = input_shape[1];
            int out_height = input_shape[2];
            int out_width = input_shape[3];

            int realwidth = out_width * UP_DIV(depth, 4);
            int realheight = out_height * batch;

            auto *para = static_cast<softmax::GpuSoftMaxParam *>(
                paramBuffer_->getMappedMemory());
            // vkimage params
            para->outImgSize[0] = realwidth;
            para->outImgSize[1] = realheight;
            para->outImgSize[2] = 1;
            para->outImgSize[3] = 0;
            para->outShape[0] = batch;
            para->outShape[1] = out_height;
            para->outShape[2] = out_width;
            para->outShape[3] = depth;
            para->axis = axis_;
            paramBuffer_->unmapMemory();

            if (axis_ == 0) {
                submit(softmax_spv, softmax_spv_len, out_width,
                       out_height * UP_DIV(depth, 4));
            } else if (axis_ == 1) {
                submit(softmax_spv, softmax_spv_len, out_width,
                       out_height * batch);
            } else if (axis_ == 2) {
                submit(softmax_spv, softmax_spv_len, out_width,
                       UP_DIV(depth, 4) * batch);
            } else if (axis_ == 3) {
                submit(softmax_spv, softmax_spv_len, out_height,
                       UP_DIV(depth, 4) * batch);
            }
        }
    }

  private:
    int axis_;

    std::shared_ptr<VulkanBuffer> paramBuffer_;
};

} // namespace ops
} // namespace vkop
#endif // OPS_SOFTMAX_HPP_
