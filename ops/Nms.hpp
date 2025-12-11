// Copyright 2025 @junka
#ifndef OPS_NMS_HPP_
#define OPS_NMS_HPP_

#include "UnaryFactory.hpp"

extern unsigned char nms_spv[];
extern unsigned int nms_spv_len;

namespace vkop {
namespace ops {
namespace nms {

using ivec4 = int[4];
using ivec2 = int[2];

struct alignas(16) GpuNMSParam {
    ivec4 inputSize;
    ivec4 outputSize;
    ivec2 pad;
    ivec2 kernelSize;
    ivec2 stride;
};

} // namespace nms

class Nms : public Operator {
  public:
    Nms()
        : Operator(OpType::MAXPOOL2D, nms_spv, nms_spv_len,
                   sizeof(nms::GpuNMSParam)) {
        n_imgs_ = 2;
        types_ = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER};
        objs_.reserve(types_.size());
    }

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("center_point_box") != attributes.end()) {
            std::string dila_str = attributes.at("center_point_box ");
            center_point_box_ = std::stol(dila_str);
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto input_shape = inputs[0]->getShape();

        int batch = input_shape[0];
        int depth = input_shape[1];

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->size() == 0) {
                output->resize(batch, depth, out_height, out_width);
            }
            auto output_image = output->as_output_image(m_dev_, m_cmd_);
            objs_.emplace_back(output_image);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto input = core::as_tensor<T>(inputs[0]);
            auto input_image = input->as_input_image(m_dev_, m_cmd_);

            objs_.emplace_back(input_image);
        });
        int realheight = out_height * batch;

        nms::GpuMaxpoolParam para;

        para.inputSize[0] = input_shape[3];
        para.inputSize[1] = input_shape[2];
        para.inputSize[2] = depth;
        para.inputSize[3] = batch;
        para.outputSize[0] = out_width;
        para.outputSize[1] = out_height;
        para.outputSize[2] = depth;
        para.outputSize[3] = batch;

        para.pad[0] = pads_[0];
        para.pad[1] = pads_[1];
        para.kernelSize[0] = kernel_shape_[0];
        para.kernelSize[1] = kernel_shape_[1];
        para.stride[0] = strides_[0];
        para.stride[1] = strides_[1];

        submit(&para, UP_DIV(out_width, 16), UP_DIV(realheight, 16),
               UP_DIV(depth, 4));
    }

  private:
    int center_point_box_ = 0;
};

} // namespace ops
} // namespace vkop
#endif // OPS_NMS_HPP_
