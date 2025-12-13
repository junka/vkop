// Copyright 2025 @junka
#ifndef OPS_OCONV2D_HPP_
#define OPS_OCONV2D_HPP_

#include <memory>
#include <string>
#include <unordered_map>

#include "Operator.hpp"

extern unsigned char conv2d_spv[];
extern unsigned int conv2d_spv_len;
namespace vkop {
namespace ops {

namespace conv2d {

enum class PaddingMode { ZEROS, REFLECT, REPLICATE, CIRCULAR };
enum class ActivationMode {
    NONE,
    RELU,
    SIGMOID,
    TANH,
    HARDSWISH,
    MISH,
    RELU6,
    GATE_SIGMOID,
};

using ivec4 = int[4];
using ivec2 = int[2];

struct alignas(16) GPUConv2dParam {
    ivec4 inputSize;
    ivec4 outputSize;
    ivec4 kernel_shape;
    ivec4 imageSize;
    ivec2 stride;
    ivec2 padding;
    ivec2 dilation;

    int groups;
    int bias;
    int transpose;
    int pack;
    int activation;

    int fuse_bn;
    float eps;
};

} // namespace conv2d

class Conv2d : public Operator {
  public:
    Conv2d()
        : Operator(OpType::CONV2D, conv2d_spv, conv2d_spv_len,
                   sizeof(conv2d::GPUConv2dParam)) {
        n_imgs_ = 3;
        types_ = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
        objs_.reserve(types_.size());
        activation_ = conv2d::ActivationMode::NONE;
    }

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("dilations") != attributes.end()) {
            std::string dila_str = attributes.at("dilations");
            if (dila_str.find(',') != std::string::npos) {
                dilations_ = parse_attr_list(dila_str);
            } else {
                int d = std::stol(dila_str);
                dilations_ = {d, d};
            }
        }

        if (attributes.find("group") != attributes.end()) {
            groups_ = std::stol(attributes.at("group"));
        }

        if (attributes.find("kernel_shape_") != attributes.end()) {
            std::string kernel_str = attributes.at("kernel_shape_");
            if (kernel_str.find(',') != std::string::npos) {
                kernel_shape_ = parse_attr_list(kernel_str);
            } else {
                int k = std::stol(kernel_str);
                kernel_shape_ = {k, k};
            }
        }

        if (attributes.find("pads") != attributes.end()) {
            std::string pad_str = attributes.at("pads");
            if (pad_str.find(',') != std::string::npos) {
                pads_ = parse_attr_list(pad_str);
            } else {
                int p = std::stol(pad_str);
                pads_ = {p, p};
            }
        }

        if (attributes.find("strides") != attributes.end()) {
            std::string stride_str = attributes.at("strides");
            if (stride_str.find(',') != std::string::npos) {
                strides_ = parse_attr_list(stride_str);
            } else {
                int s = std::stol(stride_str);
                strides_ = {s, s};
            }
        }
        if (attributes.find("auto_pad") != attributes.end()) {
            std::string auto_pad = attributes.at("auto_pad");
            if (auto_pad == "VALID") {
                pads_ = {0, 0};
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
                pads_ = {kernel_shape_[0] / 2, kernel_shape_[1] / 2};
            } else if (auto_pad == "NOTSET") {
                // do nothing
            } else {
                throw std::invalid_argument("Unsupported auto_pad: " +
                                            auto_pad);
            }
        }
        if (attributes.find("fused_bn") != attributes.end()) {
            fuse_bn_ = std::stol(attributes.at("fused_bn"));
        }
        if (attributes.find("eps") != attributes.end()) {
            eps_ = std::stof(attributes.at("eps"));
        }

        if (attributes.find("activation") != attributes.end()) {
            std::string activation = attributes.at("activation");
            if (activation == "Relu") {
                activation_ = conv2d::ActivationMode::RELU;
            } else if (activation == "Sigmoid") {
                activation_ = conv2d::ActivationMode::SIGMOID;
            } else if (activation == "Tanh") {
                activation_ = conv2d::ActivationMode::TANH;
            } else if (activation == "HardSwish") {
                activation_ = conv2d::ActivationMode::HARDSWISH;
            } else if (activation == "Mish") {
                activation_ = conv2d::ActivationMode::MISH;
            } else if (activation == "Relu6") {
                activation_ = conv2d::ActivationMode::RELU6;
            } else if (activation == "GateSigmoid") {
                activation_ = conv2d::ActivationMode::GATE_SIGMOID;
            } else {
                throw std::invalid_argument("Unsupported activation: " +
                                            activation);
            }
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        std::vector<int> input_shape = inputs[0]->getShape();
        std::vector<int> weight_shape = inputs[1]->getShape();

        int batch = input_shape[0];
        int depth = input_shape[1];
        int in_height = input_shape[2];
        int in_width = input_shape[3];
        int out_batch = batch;
        int out_depth = weight_shape[0];
        int kernel_h =
            kernel_shape_[0] == 0 ? weight_shape[2] : kernel_shape_[0];
        int kernel_w =
            kernel_shape_[1] == 0 ? weight_shape[3] : kernel_shape_[1];
        int out_height =
            (in_height + 2 * pads_[0] - dilations_[0] * (kernel_h - 1) - 1) /
                strides_[0] +
            1;
        int out_width =
            (in_width + 2 * pads_[1] - dilations_[1] * (kernel_w - 1) - 1) /
                strides_[1] +
            1;
        auto process_output = [&](auto type_tag) {
            using T = decltype(type_tag);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->size() == 0) {
                output->resize(out_batch, out_depth, out_height, out_width);
            }

            auto output_image = output->as_output_image(m_dev_, m_cmd_);
            objs_.emplace_back(output_image);
        };
        dispatch_by_dtype(outputs[0]->dtype(), process_output);

        auto process_input = [&](auto type_tag) {
            using T = decltype(type_tag);
            auto input = core::as_tensor<T>(inputs[0]);
            auto input_image = input->as_input_image(m_dev_, m_cmd_);
            objs_.emplace_back(input_image);
        };

        dispatch_by_dtype(inputs[0]->dtype(), process_input);

        auto process_weight_and_bias = [&](auto type_tag) {
            using T = decltype(type_tag);
            auto weight = core::as_tensor<T>(inputs[1]);
            auto bias =
                (inputs.size() > 2) ? core::as_tensor<T>(inputs[2]) : nullptr;

            auto weight_image = weight->as_input_image(m_dev_, m_cmd_);
            auto bias_buffer =
                bias ? bias->as_storage_buffer(m_dev_) : dummyBuffer_;
            objs_.emplace_back(weight_image);
            objs_.emplace_back(bias_buffer);

            if (bias)
                bias->copyToGPU(m_cmdpool_);
        };
        dispatch_by_dtype(inputs[1]->dtype(), process_weight_and_bias);

        if (inputs.size() > 2) {
            dispatch_by_dtype(inputs[2]->dtype(), [&](auto type_tag) {
                using T = decltype(type_tag);
                auto para = core::as_tensor<T>(inputs[2]);
                auto para_buffer = para->as_storage_buffer(m_dev_);
                objs_.emplace_back(para_buffer);
            });
        } else {
            auto para_buffer = dummyBuffer_;
            objs_.emplace_back(para_buffer);
        }

        auto out_gpu_shape = outputs[0]->getGPUShape();
        conv2d::GPUConv2dParam para;
        // vkimage params
        para.inputSize[0] = in_width;
        para.inputSize[1] = in_height;
        para.inputSize[2] = depth;
        para.inputSize[3] = batch;
        para.outputSize[0] = out_width;
        para.outputSize[1] = out_height;
        para.outputSize[2] = out_depth;
        para.outputSize[3] = out_batch;
        para.kernel_shape[0] = weight_shape[3];
        para.kernel_shape[1] = weight_shape[2];
        para.kernel_shape[2] = weight_shape[1];
        para.kernel_shape[3] = weight_shape[0];
        para.imageSize[0] = out_gpu_shape[0];
        para.imageSize[1] = out_gpu_shape[1];
        para.imageSize[2] = out_gpu_shape[2];
        para.imageSize[3] = 1;
        para.stride[0] = strides_[0];
        para.stride[1] = strides_[1];
        para.padding[0] = pads_[0];
        para.padding[1] = pads_[1];
        para.dilation[0] = dilations_[0];
        para.dilation[1] = dilations_[1];

        para.groups = groups_;
        para.bias = ((inputs.size() > 2)) ? 1 : 0;
        para.transpose = inputs[1]->get_transpose() ? 1 : 0;
        para.pack = inputs[1]->get_pack() ? 1 : 0;
        para.activation = static_cast<int>(activation_);

        para.fuse_bn = fuse_bn_;
        para.eps = eps_;

        submit(&para, UP_DIV(out_gpu_shape[0], 16),
               UP_DIV(out_gpu_shape[1], 16), out_gpu_shape[2]);
    }
    void set_runtime_device(
        const std::shared_ptr<VulkanDevice> &dev,
        const std::shared_ptr<VulkanCommandPool> &cmdpool) override {
        Operator::set_runtime_device(dev, cmdpool);
        dummyBuffer_ =
            std::make_shared<VulkanBuffer>(m_dev_, 4,
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                               VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }

  private:
    std::vector<int> kernel_shape_ = {0, 0};
    std::vector<int> strides_ = {1, 1};
    std::vector<int> pads_ = {0, 0};
    std::vector<int> dilations_ = {1, 1};
    int groups_ = 1;
    float eps_ = 1e-5;
    int fuse_bn_ = 0;
    conv2d::ActivationMode activation_ = conv2d::ActivationMode::NONE;
    std::shared_ptr<VulkanBuffer> dummyBuffer_;
}; // namespace ops

} // namespace ops
} // namespace vkop
#endif // OPS_OCONV2D_HPP_