// Copyright 2025 @junka
#ifndef OPS_RESHAPE_HPP_
#define OPS_RESHAPE_HPP_

#include "core/Tensor.hpp"
#include "ops/Operator.hpp"

extern unsigned char reshape_spv[];
extern unsigned int reshape_spv_len;
namespace vkop {
namespace ops {

namespace reshape {
using ivec4 = int[4];
using ivec2 = int[2];
struct GpuReshapeParam {
    ivec4 inImgSize;
    ivec4 outImgSize;
    ivec4 inShape;
    ivec4 outShape;
};

} // namespace reshape

class Reshape : public Operator {
  public:
    explicit Reshape()
        : Operator(OpType::RESHAPE, reshape_spv, reshape_spv_len,
                   sizeof(reshape::GpuReshapeParam)) {
        n_imgs_ = 2;
        types_ = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER};
        objs_.reserve(2);
    }

  private:
    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("allowzero") != attributes.end()) {
            allowzero_ = std::stoi(attributes.at("allowzero"));
        }
    }

    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {

        auto shape = core::as_tensor<int64_t>(inputs[1]);
        assert(shape->num_dims() == 1);
        int n = shape->num_elements();
        assert(n >= 3);
        std::vector<int> dim(4);
        dim[3] = (*shape)[n - 1];
        dim[2] = (*shape)[n - 2];
        dim[1] = (*shape)[n - 3];
        if (n == 3) {
            dim[0] = 1;
        } else if (n == 4) {
            dim[0] = (*shape)[0];
        }
        for (int i = 0; i < 4; i++) {
            if (!allowzero_) {
                dim[i] = inputs[0]->getShape()[i];
            }
        }
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->size() == 0) {
                output->resize(dim);
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

        auto inshape = inputs[0]->getShape();
        auto outshape = outputs[0]->getShape();
        reshape::GpuReshapeParam param;
        param.inImgSize[0] = inshape[3];
        param.inImgSize[1] = inshape[2] * inshape[0];
        param.inImgSize[2] = UP_DIV(inshape[1], 4);
        param.inImgSize[3] = 1;
        param.outImgSize[0] = outshape[3];
        param.outImgSize[1] = outshape[2] * outshape[0];
        param.outImgSize[2] = UP_DIV(outshape[1], 4);
        param.outImgSize[3] = 1;
        for (int i = 0; i < 4; i++) {
            param.inShape[i] = inshape[i];
            param.outShape[i] = dim[i];
        }
        submit(&param, UP_DIV(dim[3], 16), UP_DIV(dim[2] * dim[0], 16),
               UP_DIV(dim[1], 4));
    }

    int allowzero_ = 0;
};

} // namespace ops
} // namespace vkop
#endif // OPS_RESHAPE_HPP_
