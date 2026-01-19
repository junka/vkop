// Copyright 2025 @junka
#ifndef OPS_AVERAGEPOOL_HPP_
#define OPS_AVERAGEPOOL_HPP_

#include "ops/Operator.hpp"

extern "C" {
extern unsigned char averagepool_spv[];
extern unsigned int averagepool_spv_len;
};

namespace vkop {
namespace ops {
namespace averagepool {
struct alignas(16) GpuAveragePoolParam {
    ivec4 inShape; // NCHW
    ivec4 outShape;
    ivec4 pads;
    ivec2 kernel_shape;
    ivec2 strides;
    int count_include_pad;
};
} // namespace averagepool

class AveragePool : public Operator {
  public:
    AveragePool()
        : Operator(OpType::AVERAGEPOOL, averagepool_spv, averagepool_spv_len,
                   {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER},
                   sizeof(averagepool::GpuAveragePoolParam)) {}

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("ceil_mode") != attributes.end()) {
            ceil_mode_ = std::stol(attributes.at("ceil_mode"));
        }
        if (attributes.find("count_include_pad") != attributes.end()) {
            count_include_pad_ = std::stol(attributes.at("count_include_pad"));
        }
        if (attributes.find("dilations") != attributes.end()) {
            std::string dila_str = attributes.at("dilations");
            if (dila_str.find(',') != std::string::npos) {
                dilations_ = parse_attr_list<int>(dila_str);
            } else {
                int d = std::stol(dila_str);
                dilations_ = {d, d};
            }
        }

        if (attributes.find("kernel_shape") != attributes.end()) {
            std::string kernel_str = attributes.at("kernel_shape");
            if (kernel_str.find(',') != std::string::npos) {
                kernel_shape_ = parse_attr_list<int>(kernel_str);
            } else {
                int k = std::stol(kernel_str);
                kernel_shape_ = {k, k};
            }
        } else if (attributes.find("kernel_size") != attributes.end()) {
            std::string kernel_str = attributes.at("kernel_size");
            if (kernel_str.find(',') != std::string::npos) {
                kernel_shape_ = parse_attr_list<int>(kernel_str);
            } else {
                int k = std::stol(kernel_str);
                kernel_shape_ = {k, k};
            }
        }

        if (attributes.find("pads") != attributes.end()) {
            std::string pad_str = attributes.at("pads");
            if (pad_str.find(',') != std::string::npos) {
                pads_ = parse_attr_list<int>(pad_str);
            } else {
                int p = std::stol(pad_str);
                pads_ = {p, p};
            }
        }

        if (attributes.find("strides") != attributes.end()) {
            std::string stride_str = attributes.at("strides");
            if (stride_str.find(',') != std::string::npos) {
                strides_ = parse_attr_list<int>(stride_str);
            } else {
                int s = std::stol(stride_str);
                strides_ = {s, s};
            }
        }
        if (attributes.find("auto_pad") != attributes.end()) {
            std::string auto_pad = attributes.at("auto_pad");
            if (auto_pad == "VALID") {
                auto_pad_ = 1;
            } else if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
                auto_pad_ = 2;
            } else if (auto_pad == "NOTSET") {
                // do nothing
            } else {
                throw std::invalid_argument("Unsupported auto_pad: " +
                                            auto_pad);
            }
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto input_shape = inputs[0]->getShape();

        int rank = inputs[0]->num_dims();
        auto outshape = input_shape;

        if (auto_pad_ != 0) {
            if (auto_pad_ == 1) {
                for (int i = 2; i < rank; i++) {
                    outshape[i] = (input_shape[i] + kernel_shape_[i - 2]) /
                                      strides_[i - 2] +
                                  1;
                }
            } else {
                for (int i = 2; i < rank; i++) {
                    outshape[i] = (input_shape[i] + strides_[i - 2] - 1) /
                                  strides_[i - 2];
                }
            }
        } else {
            for (int i = 2; i < rank; i++) {
                auto padding = pads_[i - 2] + pads_[i];
                if (ceil_mode_ == 0) {
                    outshape[i] =
                        (input_shape[i] + padding - kernel_shape_[i - 2]) /
                            strides_[i - 2] +
                        1;
                } else {
                    outshape[i] = (input_shape[i] + padding -
                                   kernel_shape_[i - 2] + strides_[i - 2] - 1) /
                                      strides_[i - 2] +
                                  1;
                }
            }
        }
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->size() == 0) {
                output->resize(outshape);
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

        averagepool::GpuAveragePoolParam para;
        for (int i = 0; i < rank; i++) {
            para.inShape[i] = input_shape[i];
            para.outShape[i] = outshape[i];
            para.pads[i] = pads_[i];
        }
        for (int i = 0; i < 2; i++) {
            para.kernel_shape[i] = kernel_shape_[i];
            para.strides[i] = strides_[i];
        }
        para.count_include_pad = count_include_pad_;
        auto out_gpu_shape = outputs[0]->getGPUShape();

        submit(&para, UP_DIV(out_gpu_shape[0], 16),
               UP_DIV(out_gpu_shape[1], 16), out_gpu_shape[2]);
    }

    std::vector<int> kernel_shape_;
    std::vector<int> strides_ = {1, 1};
    std::vector<int> pads_ = {0, 0, 0, 0};
    std::vector<int> dilations_ = {1, 1};
    int ceil_mode_ = 0;
    int count_include_pad_ = 0;
    int auto_pad_ = 0; // notset
};

} // namespace ops
} // namespace vkop
#endif // OPS_AVERAGEPOOL_HPP_
