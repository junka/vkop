// Copyright 2025 @junka
#ifndef OPS_BATCHNORM2D_HPP_
#define OPS_BATCHNORM2D_HPP_

#include "Operator.hpp"

#include <memory>

extern unsigned char batchnorm_spv[];
extern unsigned int batchnorm_spv_len;

namespace vkop {
namespace ops {
namespace batchnorm {

using ivec4 = int[4];

// torch.nn.functional.batch_norm(input, running_mean, running_var, weight=None,
//                                bias=None, training=False, momentum=0.1,
//                                eps=1e-05)
struct alignas(16) GpuBatchNormParam {
    ivec4 outShape;
    float eps;      // default 1e-5
    float momentum; // default 0.1
};
} // namespace batchnorm

class BatchNorm : public Operator {
  public:
    BatchNorm()
        : Operator(OpType::BATCHNORM, batchnorm_spv, batchnorm_spv_len,
                   sizeof(batchnorm::GpuBatchNormParam)) {
        n_imgs_ = 2;
        types_ = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
        objs_.reserve(types_.size());
    }
    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        attributes.find("training") != attributes.end()
            ? training_ = (attributes.at("align_corners") == "1" ||
                           attributes.at("align_corners") == "true")
            : training_ = false;
        if (attributes.find("eps") != attributes.end()) {
            eps_ = std::stof(attributes.at("eps"));
        }
        if (attributes.find("momentum") != attributes.end()) {
            momentum_ = std::stof(attributes.at("momentum"));
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto input_shape = inputs[0]->getShape();

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto t) {
            using T = decltype(t);
            auto outputptr = core::as_tensor<T>(outputs[0]);
            if (outputptr->size() == 0) {
                outputptr->resize(input_shape);
            }
            auto output_image = outputptr->as_output_image(m_dev_, m_cmd_);
            objs_.emplace_back(output_image);
        });

        dispatch_by_dtype(inputs[0]->dtype(), [&](auto t) {
            using T = decltype(t);
            auto inputptr = core::as_tensor<T>(inputs[0]);
            auto input_image = inputptr->as_input_image(m_dev_, m_cmd_);
            objs_.emplace_back(input_image);
        });

        dispatch_by_dtype(inputs[1]->dtype(), [&](auto t) {
            using T = decltype(t);
            auto para = core::as_tensor<T>(inputs[1]);
            auto para_buffer = para->as_storage_buffer(m_dev_);
            objs_.emplace_back(para_buffer);
        });

        int batch = input_shape[0];
        int depth = input_shape[1];
        int out_height = input_shape[2];
        int out_width = input_shape[3];

        int realheight = out_height * batch;

        batchnorm::GpuBatchNormParam para;
        para.eps = eps_;
        para.momentum = momentum_;
        para.outShape[0] = batch;
        para.outShape[1] = depth;
        para.outShape[2] = out_height;
        para.outShape[3] = out_width;

        submit(&para, UP_DIV(out_width, 16), UP_DIV(realheight, 16),
               UP_DIV(depth, 4));
    }

  private:
    bool training_ = false;
    float momentum_ = 0.1;
    float eps_ = 1e-5;

    std::shared_ptr<VulkanBuffer> tensorBuffer_;
};

} // namespace ops
} // namespace vkop
#endif // OPS_BATCHNORM2D_HPP_
