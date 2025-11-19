// Copyright 2025 @junka
#ifndef OPS_MAXPOOL2D_HPP_
#define OPS_MAXPOOL2D_HPP_

#include "UnaryFactory.hpp"

extern unsigned char maxpool2d_spv[];
extern unsigned int maxpool2d_spv_len;

namespace vkop {
namespace ops {
namespace maxpool2d {

using ivec4 = int[4];
using ivec2 = int[2];

struct GpuMaxpoolParam {
    ivec4 inputSize;
    ivec4 outputSize;
    ivec2 pad;
    ivec2 kernelSize;
    ivec2 stride;
};

} // namespace maxpool2d

class Maxpool2d : public Operator {
  public:
    Maxpool2d() : Operator(OpType::MAXPOOL2D) {}

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("auto_pad ") != attributes.end()) {
            std::string auto_pad = attributes.at("auto_pad");
            if (auto_pad == "VALID") {
            } else if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
                // SAME would let out_h = ceil(in_h/stride_h)
                // so padding_h = ((out_h-1)*stride_h + (kernel_h-1)*dilations_h
                // + 1 - in_h)/2 here we just set padding to kernel_size/2, and
                // only support stride=1,dilation=1 case
                if (strides_[0] != 1 || strides_[1] != 1 ||
                    dilations_[0] != 1 || dilations_[1] != 1) {
                    throw std::invalid_argument("Only support stride=1 and "
                                                "dilation=1 for SAME auto_pad");
                }
                pads_ = {(kernel_shape_[0] - 1) * strides_[0] / 2,
                         (kernel_shape_[1] - 1) * strides_[1] / 2};
            } else if (auto_pad == "NOTSET") {
                // do nothing
                pads_ = {0, 0};
            } else {
                throw std::invalid_argument("Unsupported auto_pad: " +
                                            auto_pad);
            }
        }

        if (attributes.find("dilations") != attributes.end()) {
            std::string dila_str = attributes.at("dilations");
            if (dila_str.find(',') != std::string::npos) {
                dilations_ = parse_attr_list(dila_str);
            } else {
                int d = std::stoi(dila_str);
                dilations_ = {d, d};
            }
        } else {
            dilations_ = {1, 1};
        }

        if (attributes.find("ceil_mode") != attributes.end()) {
            ceil_mode_ = std::stoi(attributes.at("ceil_mode"));
        } else {
            ceil_mode_ = 1;
        }

        if (attributes.find("kernel_shape") != attributes.end()) {
            std::string kernel_str = attributes.at("kernel_shape");
            if (kernel_str.find(',') != std::string::npos) {
                kernel_shape_ = parse_attr_list(kernel_str);
            } else {
                int k = std::stoi(kernel_str);
                kernel_shape_ = {k, k};
            }
        } else {
            // should be inferred from weight shape
            kernel_shape_ = {0, 0};
        }

        if (attributes.find("pads") != attributes.end()) {
            std::string pad_str = attributes.at("pads");
            if (pad_str.find(',') != std::string::npos) {
                pads_ = parse_attr_list(pad_str);
            } else {
                int p = std::stoi(pad_str);
                pads_ = {p, p};
            }
        } else {
            pads_ = {0, 0};
        }
        if (attributes.find("storage_order") != attributes.end()) {
            storage_order_ = std::stoi(attributes.at("storage_order"));
        } else {
            storage_order_ = 1;
        }
        if (attributes.find("strides") != attributes.end()) {
            std::string stride_str = attributes.at("strides");
            if (stride_str.find(',') != std::string::npos) {
                strides_ = parse_attr_list(stride_str);
            } else {
                int s = std::stoi(stride_str);
                strides_ = {s, s};
            }
        } else {
            // default stride value should be kernel size
            strides_ = kernel_shape_;
        }
    }
    template <typename T>
    void prepare(std::vector<std::shared_ptr<core::ITensor>> inputs,
                 std::vector<std::shared_ptr<core::ITensor>> outputs) {
        auto input = core::as_tensor<T>(inputs[0]);
        auto output = core::as_tensor<T>(outputs[0]);

        auto input_shape = input->getShape();
        int batch = input_shape[0];
        int depth = input_shape[1];

        int out_height;
        int out_width;
        if (ceil_mode_) {
            out_height =
                (input_shape[2] - dilations_[0] * (kernel_shape_[0] - 1) +
                 2 * pads_[0] + strides_[0] - 1) /
                    strides_[0] +
                1;
            out_width =
                (input_shape[3] - dilations_[1] * (kernel_shape_[1] - 1) +
                 2 * pads_[1] + strides_[1] - 1) /
                    strides_[1] +
                1;
        } else {
            out_height =
                (input_shape[2] - dilations_[0] * (kernel_shape_[0] - 1) +
                 2 * pads_[0] - 1) /
                    strides_[0] +
                1;
            out_width =
                (input_shape[3] - dilations_[1] * (kernel_shape_[1] - 1) +
                 2 * pads_[1] - 1) /
                    strides_[1] +
                1;
        }

        if (output->size() == 0) {
            output->resize(batch, depth, out_height, out_width);
        }

        auto input_image = input->as_input_image(m_dev_, m_cmdpool_);
        auto output_image = output->as_output_image(m_dev_, m_cmdpool_);

        paramBuffer_ = std::make_shared<VulkanBuffer>(
            m_dev_, sizeof(maxpool2d::GpuMaxpoolParam),
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

            if (input_shape.size() != 4) {
                throw std::invalid_argument("Input must have 4 dimensions.");
            }
            int batch = input_shape[0];
            int depth = input_shape[1];

            int out_height;
            int out_width;
            if (ceil_mode_) {
                out_height =
                    (input_shape[2] - dilations_[0] * (kernel_shape_[0] - 1) +
                     2 * pads_[0] + strides_[0] - 1) /
                        strides_[0] +
                    1;
                out_width =
                    (input_shape[3] - dilations_[1] * (kernel_shape_[1] - 1) +
                     2 * pads_[1] + strides_[1] - 1) /
                        strides_[1] +
                    1;
            } else {
                out_height =
                    (input_shape[2] - dilations_[0] * (kernel_shape_[0] - 1) +
                     2 * pads_[0] - 1) /
                        strides_[0] +
                    1;
                out_width =
                    (input_shape[3] - dilations_[1] * (kernel_shape_[1] - 1) +
                     2 * pads_[1] - 1) /
                        strides_[1] +
                    1;
            }

            int realwidth = out_width * UP_DIV(depth, 4);
            int realheight = out_height * batch;

            auto *para = static_cast<maxpool2d::GpuMaxpoolParam *>(
                paramBuffer_->getMappedMemory());
            // vkimage params
            para->inputSize[0] = input_shape[3];
            para->inputSize[1] = input_shape[2];
            para->inputSize[2] = UP_DIV(depth, 4);
            para->inputSize[3] = batch;
            para->outputSize[0] = out_width;
            para->outputSize[1] = out_height;
            para->outputSize[2] = UP_DIV(depth, 4);
            para->outputSize[3] = batch;

            para->pad[0] = pads_[0];
            para->pad[1] = pads_[1];
            para->kernelSize[0] = kernel_shape_[0];
            para->kernelSize[1] = kernel_shape_[1];
            para->stride[0] = strides_[0];
            para->stride[1] = strides_[1];

            paramBuffer_->unmapMemory();
            submit(maxpool2d_spv, maxpool2d_spv_len, UP_DIV(realwidth, 16),
                   UP_DIV(realheight, 16));
        }
    }

  private:
    std::vector<int> kernel_shape_;
    std::vector<int> strides_;
    std::vector<int> pads_;
    std::vector<int> dilations_;

    int storage_order_;
    int ceil_mode_;

    std::shared_ptr<VulkanBuffer> paramBuffer_;
};

} // namespace ops
} // namespace vkop
#endif // OPS_MAXPOOL2D_HPP_
