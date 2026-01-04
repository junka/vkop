// Copyright 2025 @junka
#ifndef OPS_SOFTMAX_HPP_
#define OPS_SOFTMAX_HPP_

#include "Operator.hpp"

extern unsigned char softmax_spv[];
extern unsigned int softmax_spv_len;
extern unsigned char softmax2_spv[];
extern unsigned int softmax2_spv_len;

namespace vkop {
namespace ops {
namespace softmax {

struct GpuSoftMaxParam {
    ivec4 outShape;
    int axis; // 0: N, 1: C, 2: H, 3: W
};

} // namespace softmax

class Softmax : public Operator {
  public:
    Softmax(const Softmax &) = delete;
    Softmax &operator=(const Softmax &) = delete;
    Softmax(Softmax &&) = delete;
    Softmax &operator=(Softmax &&) = delete;
    explicit Softmax(bool use_ssbo = false)
        : Operator(OpType::SOFTMAX, use_ssbo ? softmax2_spv : softmax_spv,
                   use_ssbo ? softmax2_spv_len : softmax_spv_len,
                   sizeof(softmax::GpuSoftMaxParam)) {
        n_imgs_ = use_ssbo ? 0 : 2;
        if (use_ssbo) {
            types_ = {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE};
        } else {
            types_ = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER};
        }
        objs_.reserve(types_.size());
    }

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("axis") != attributes.end()) {
            axis_ = std::stol(attributes.at("axis"));
        } else if (attributes.find("dim") != attributes.end()) {
            axis_ = std::stol(attributes.at("dim"));
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {

        auto input_shape = inputs[0]->getShape();
        int rank = inputs[0]->num_dims();

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->size() == 0) {
                output->resize(inputs[0]->getShape());
            }
            if (types_[0] == DESCRIPTOR_TYPE_STORAGE) {
                auto output_buffer = output->as_storage_buffer(m_dev_);
                objs_.emplace_back(output_buffer);
            } else {
                auto output_image = output->as_output_image(m_dev_, m_cmd_);
                objs_.emplace_back(output_image);
            }
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto input = core::as_tensor<T>(inputs[0]);
            if (types_[1] == DESCRIPTOR_TYPE_STORAGE) {
                auto input_buffer = input->as_storage_buffer(m_dev_);
                objs_.emplace_back(input_buffer);
            } else {
                auto input_image = input->as_input_image(m_dev_, m_cmd_);
                objs_.emplace_back(input_image);
            }
        });

        int batch = input_shape[0];
        int depth = input_shape[1];
        int out_height = input_shape[2];
        int out_width = input_shape[3];

        int realheight = out_height * batch;

        softmax::GpuSoftMaxParam para;
        // vkimage params
        para.outShape[0] = batch;
        para.outShape[1] = depth;
        para.outShape[2] = out_height;
        para.outShape[3] = out_width;
        if (axis_ < 0) {
            axis_ = rank + axis_;
        }
        para.axis = axis_;

        if (types_[0] == DESCRIPTOR_TYPE_STORAGE) {
            if (rank == 1) {
                para.outShape[0] = 1;
                para.outShape[1] = batch;
                para.axis = 1;
            }
            if (para.axis == 1) {
                submit(&para, para.outShape[0], 1, 1);
            } else {
                submit(&para, para.outShape[1], 1, 1);
            }
        } else {
            if (axis_ == 0) {
                submit(&para, UP_DIV(out_width, 16), UP_DIV(out_height, 16),
                       UP_DIV(depth, 4));
            } else if (axis_ == 1) {
                submit(&para, UP_DIV(out_width, 16), UP_DIV(realheight, 16),
                       batch);
            } else if (axis_ == 2) {
                submit(&para, UP_DIV(out_width, 16), UP_DIV(batch, 16),
                       UP_DIV(depth, 4));
            } else if (axis_ == 3) {
                submit(&para, UP_DIV(out_height, 16), UP_DIV(batch, 16),
                       UP_DIV(depth, 4));
            }
        }
    }

    int axis_;
};

} // namespace ops
} // namespace vkop
#endif // OPS_SOFTMAX_HPP_
