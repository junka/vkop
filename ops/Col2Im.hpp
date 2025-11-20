// Copyright 2025 @junka
#ifndef OPS_COL2IM_HPP_
#define OPS_COL2IM_HPP_

#include "UnaryFactory.hpp"

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
    Col2im() : Operator(OpType::COL2IM) {}

    template <typename T>
    void prepare(std::vector<std::shared_ptr<core::ITensor>> inputs,
                 std::vector<std::shared_ptr<core::ITensor>> outputs) {
        auto input = core::as_tensor<T>(inputs[0]);
        auto output = core::as_tensor<T>(outputs[0]);

        auto input_shape = input->getShape();

        if (output->size() == 0) {
            output->resize(input_shape);
        }

        auto input_image = input->as_input_image(m_dev_, m_cmdpool_);
        auto output_image = output->as_output_image(m_dev_, m_cmdpool_);

        types_ = {output_image->getDescriptorType(),
                  input_image->getDescriptorType()};
        objs_ = {output_image, input_image};
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

            if (input_shape.size() != 4) {
                throw std::invalid_argument("Input must have 4 dimensions.");
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

            submit(&para, sizeof(col2im::GpuCol2ImParam), col2im_spv,
                   col2im_spv_len, out_width, out_height);
        } else {
            LOG_ERROR("Unsupported data type");
        }
    }

  private:
};

} // namespace ops
} // namespace vkop
#endif // OPS_COL2IM_HPP_
