// Copyright 2025 @junka
#ifndef OPS_CONCAT_HPP_
#define OPS_CONCAT_HPP_

#include "ops/Operator.hpp"

extern unsigned char concat_spv[];
extern unsigned int concat_spv_len;

namespace vkop {
namespace ops {

namespace concat {

using ivec4 = int[4];
struct ConcatParam {
    ivec4 inShape;
    ivec4 outShape;
    ivec4 offset;
    int axis;
};

} // namespace concat

class Concat : public Operator {
  public:
    explicit Concat()
        : Operator(OpType::CONCAT, concat_spv, concat_spv_len,
                   sizeof(concat::ConcatParam)) {
        n_imgs_ = 2;
        types_ = {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        };
        objs_.reserve(types_.size());
    }

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("axis") != attributes.end()) {
            axis_ = std::stol(attributes.at("axis"));
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        int rank = inputs[0]->num_dims();
        if (axis_ < 0) {
            axis_ = rank + axis_;
        }
        assert(rank >= 3);
        std::vector<int> out_shape = inputs[0]->getShape();
        for (size_t i = 1; i < inputs.size(); i++) {
            auto shape = inputs[i]->getShape();
            out_shape[axis_] += shape[axis_];
        }
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->size() == 0) {
                output->resize(out_shape);
            }
            auto output_image = output->as_output_image(m_dev_, m_cmd_);
            objs_.emplace_back(output_image);
        });
        auto output_image = std::dynamic_pointer_cast<VulkanImage>(objs_[0]);
        int offset = 0;
        for (const auto &in : inputs) {
            dispatch_by_dtype(in->dtype(), [&](auto dummy) {
                using T = decltype(dummy);
                auto input = core::as_tensor<T>(in);
                auto input_image = input->as_input_image(m_dev_, m_cmd_);
                auto inGPUshape = input->getGPUShape();
                if (axis_ + 4 - rank == 1 && (offset % 4 == 0)) {
                    // assume they can be divided by 4
                    input_image->transferReadBarrier(m_cmd_->get());
                    output_image->copyImageToImage(m_cmd_->get(), input_image,
                                                   {0, 0, 0}, offset / 4);
                    offset += in->get_channel();
                } else if (axis_ + 4 - rank == 0) {
                    input_image->transferReadBarrier(m_cmd_->get());
                    output_image->copyImageToImage(m_cmd_->get(), input_image,
                                                   {0, offset, 0}, 0);
                    offset += inGPUshape[1];
                } else if (axis_ + 4 - rank == 3) {
                    input_image->transferReadBarrier(m_cmd_->get());
                    output_image->copyImageToImage(m_cmd_->get(), input_image,
                                                   {offset, 0, 0}, 0);
                    offset += inGPUshape[0];
                } else if (axis_ + 4 - rank == 2) {
                    if (objs_.size() == 2) {
                        objs_.pop_back();
                    }
                    objs_.emplace_back(input_image);
                    concat::ConcatParam para = {};
                    auto input_shape = in->getShape();
                    if (in->num_dims() == 4) {
                        for (size_t j = 0; j < 4; j++) {
                            para.inShape[j] = input_shape[j];
                            para.outShape[j] = out_shape[j];
                        }
                    } else if (in->num_dims() == 3) {
                        para.inShape[0] = 1;
                        para.inShape[1] = input_shape[0];
                        para.inShape[2] = input_shape[1];
                        para.inShape[3] = input_shape[2];
                        para.outShape[0] = 1;
                        para.outShape[1] = out_shape[0];
                        para.outShape[2] = out_shape[1];
                        para.outShape[3] = out_shape[2];
                    }
                    para.offset[0] = 0;
                    para.offset[1] = 0;
                    para.offset[2] = offset;
                    para.offset[3] = 0;
                    para.axis = 2;
                    offset += in->get_height();
                    submit(&para, UP_DIV(inGPUshape[0], 16),
                           UP_DIV(inGPUshape[1], 16), inGPUshape[2]);
                } else if (axis_ + 4 - rank == 1) {
                    if (objs_.size() == 2) {
                        objs_.pop_back();
                    }
                    objs_.emplace_back(input_image);
                    auto input_shape = in->getShape();
                    concat::ConcatParam para = {};
                    if (in->num_dims() == 4) {
                        for (size_t j = 0; j < 4; j++) {
                            para.inShape[j] = input_shape[j];
                            para.outShape[j] = out_shape[j];
                        }
                    } else if (in->num_dims() == 3) {
                        para.inShape[0] = 1;
                        para.inShape[1] = input_shape[0];
                        para.inShape[2] = input_shape[1];
                        para.inShape[3] = input_shape[2];
                        para.outShape[0] = 1;
                        para.outShape[1] = out_shape[0];
                        para.outShape[2] = out_shape[1];
                        para.outShape[3] = out_shape[2];
                    }
                    para.offset[0] = 0;
                    para.offset[1] = offset;
                    para.offset[2] = 0;
                    para.offset[3] = 0;
                    para.axis = 1;
                    offset += in->get_channel();
                    submit(&para, UP_DIV(inGPUshape[0], 16),
                           UP_DIV(inGPUshape[1], 16), inGPUshape[2]);
                }
            });
        }
    }

  private:
    int axis_ = 1;
};

} // namespace ops
} // namespace vkop
#endif // OPS_CONCAT_HPP_
