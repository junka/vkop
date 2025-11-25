// Copyright 2025 @junka
#ifndef OPS_SOFTMAX_HPP_
#define OPS_SOFTMAX_HPP_

#include "Operator.hpp"

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
    Softmax()
        : Operator(OpType::SOFTMAX, softmax_spv, softmax_spv_len,
                   sizeof(softmax::GpuSoftMaxParam)) {
        types_ = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER};
    }

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

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {

        auto input_shape = inputs[0]->getShape();
        if (input_shape.size() != 4) {
            throw std::invalid_argument("Input must have 4 dimensions.");
        }
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->size() == 0) {
                output->resize(inputs[0]->getShape());
            }
            auto output_image = output->as_output_image(m_dev_, m_cmd_);
            // types_.emplace_back(output_image->getDescriptorType());
            objs_.emplace_back(output_image);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto input = core::as_tensor<T>(inputs[0]);
            auto input_image = input->as_input_image(m_dev_, m_cmd_);

            // types_.emplace_back(input_image->getDescriptorType());
            objs_.emplace_back(input_image);
        });

        int batch = input_shape[0];
        int depth = input_shape[1];
        int out_height = input_shape[2];
        int out_width = input_shape[3];

        int realwidth = out_width * UP_DIV(depth, 4);
        int realheight = out_height * batch;

        softmax::GpuSoftMaxParam para;
        ;
        // vkimage params
        para.outImgSize[0] = realwidth;
        para.outImgSize[1] = realheight;
        para.outImgSize[2] = 1;
        para.outImgSize[3] = 0;
        para.outShape[0] = batch;
        para.outShape[1] = out_height;
        para.outShape[2] = out_width;
        para.outShape[3] = depth;
        para.axis = axis_;

        if (axis_ == 0) {
            submit(&para, out_width, out_height * UP_DIV(depth, 4));
        } else if (axis_ == 1) {
            submit(&para, out_width, out_height * batch);
        } else if (axis_ == 2) {
            submit(&para, out_width, UP_DIV(depth, 4) * batch);
        } else if (axis_ == 3) {
            submit(&para, out_height, UP_DIV(depth, 4) * batch);
        }
    }

  private:
    int axis_;
};

} // namespace ops
} // namespace vkop
#endif // OPS_SOFTMAX_HPP_
