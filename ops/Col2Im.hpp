// Copyright 2025 @junka
#ifndef OPS_COL2IM_HPP_
#define OPS_COL2IM_HPP_

#include "Operator.hpp"

extern unsigned char col2im_spv[];
extern unsigned int col2im_spv_len;

namespace vkop {
namespace ops {
namespace col2im {

using ivec4 = int[4];
using ivec2 = int[2];

struct alignas(16) GpuCol2ImParam {
    ivec4 outImgSize;
    ivec4 outShape;
    int groupSize;
    int totalGroups;
};

} // namespace col2im

class Col2im : public Operator {
  public:
    Col2im()
        : Operator(OpType::COL2IM, col2im_spv, col2im_spv_len,
                   sizeof(col2im::GpuCol2ImParam)) {
        types_ = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER};
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {

        std::vector<int> input_shape = inputs[0]->getShape();

        if (input_shape.size() != 4) {
            throw std::invalid_argument("Input must have 4 dimensions.");
        }
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto t) {
            using T = decltype(t);
            auto outputptr = core::as_tensor<T>(outputs[0]);
            if (outputptr->size() == 0) {
                outputptr->resize(input_shape);
            }
            auto output_image = outputptr->as_output_image(m_dev_, m_cmd_);
            // types_.emplace_back(output_image->getDescriptorType());
            objs_.emplace_back(output_image);
        });

        for (const auto &input : inputs) {
            dispatch_by_dtype(input->dtype(), [&](auto t) {
                using T = decltype(t);
                auto inputptr = core::as_tensor<T>(input);
                auto input_image = inputptr->as_input_image(m_dev_, m_cmd_);
                // types_.emplace_back(input_image->getDescriptorType());
                objs_.emplace_back(input_image);
            });
        }

        int batch = input_shape[0];
        int depth = input_shape[1];
        int out_height = input_shape[2];
        int out_width = input_shape[3];

        int realwidth = out_width * UP_DIV(depth, 4);
        int realheight = out_height * batch;

        col2im::GpuCol2ImParam para;
        // vkimage params
        para.outImgSize[0] = realwidth;
        para.outImgSize[1] = realheight;
        para.outImgSize[2] = 1;
        para.outImgSize[3] = 0;
        para.outShape[0] = batch;
        para.outShape[1] = out_height;
        para.outShape[2] = out_width;
        para.outShape[3] = depth;

        submit(&para, out_width, out_height);
    }

  private:
};

} // namespace ops
} // namespace vkop
#endif // OPS_COL2IM_HPP_
