// Copyright 2025 @junka
#ifndef OPS_OCONV2D_HPP_
#define OPS_OCONV2D_HPP_

#include <string>
#include <unordered_map>
#include <utility>

#include "Operator.hpp"

#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanCommandBuffer.hpp"
#include "vulkan/VulkanImage.hpp"
#include "vulkan/VulkanPipeline.hpp"
#include "vulkan/VulkanQueryPool.hpp"

extern unsigned char conv2d_spv[];
extern unsigned int conv2d_spv_len;
namespace vkop {
namespace ops {

namespace conv2d {

enum class PaddingMode { ZEROS, REFLECT, REPLICATE, CIRCULAR };
enum class ActivationMode { NONE, RELU, SIGMOID, TANH, HARDSWISH, MISH };

using ivec4 = int[4];
using ivec2 = int[2];

struct GPUConv2dParam {
    ivec4 inputSize;
    ivec4 outputSize;
    ivec4 kernel_shape;

    ivec2 stride;
    ivec2 padding;
    ivec2 dilation;

    int groups;
    int bias;
    int activation;
};

} // namespace conv2d

class Conv2d : public Operator {
  public:
    Conv2d() : Operator(OpType::CONV2D) {}

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
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

        if (attributes.find("group") != attributes.end()) {
            groups_ = std::stoi(attributes.at("group"));
        } else {
            groups_ = 1;
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

        if (attributes.find("strides") != attributes.end()) {
            std::string stride_str = attributes.at("strides");
            if (stride_str.find(',') != std::string::npos) {
                strides_ = parse_attr_list(stride_str);
            } else {
                int s = std::stoi(stride_str);
                strides_ = {s, s};
            }
        } else {
            strides_ = {1, 1};
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
            } else {
                throw std::invalid_argument("Unsupported activation: " +
                                            activation);
            }
        } else {
            activation_ = conv2d::ActivationMode::NONE;
        }
    }

    template <typename T>
    void prepare(std::vector<std::shared_ptr<core::ITensor>> inputs,
                 std::vector<std::shared_ptr<core::ITensor>> outputs) {
        auto input = core::as_tensor<T>(inputs[0]);
        auto output = core::as_tensor<T>(outputs[0]);
        auto weight = core::as_tensor<T>(inputs[1]);
        auto bias =
            (inputs.size() > 2) ? core::as_tensor<float>(inputs[2]) : nullptr;

        auto input_shape = input->getShape();
        auto weight_shape = weight->getShape();

        int batch = input_shape[0];
        int in_height = input_shape[2];
        int in_width = input_shape[3];
        int out_batch = batch;
        int out_depth = weight_shape[0];
        int out_height = (in_height + 2 * pads_[0] -
                          dilations_[0] * (weight_shape[2] - 1) - 1) /
                             strides_[0] +
                         1;
        int out_width = (in_width + 2 * pads_[1] -
                         dilations_[1] * (weight_shape[3] - 1) - 1) /
                            strides_[1] +
                        1;
        if (output->size() == 0) {
            output->resize(out_batch, out_depth, out_height, out_width);
        }

        auto input_image = input->as_input_image(m_dev_, m_cmdpool_);
        auto output_image = output->as_output_image(m_dev_, m_cmdpool_);

        auto weight_image = weight->as_input_image(m_dev_, m_cmdpool_);

        auto bias_buffer =
            bias ? bias->as_storage_buffer(m_dev_)
                 : std::make_shared<VulkanBuffer>(
                       m_dev_, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        paramBuffer_ = std::make_shared<VulkanBuffer>(
            m_dev_, sizeof(conv2d::GPUConv2dParam),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        types_ = {output_image->getDescriptorType(),
                  input_image->getDescriptorType(),
                  weight_image->getDescriptorType(),
                  bias ? bias_buffer->getDescriptorType()
                       : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                  paramBuffer_->getDescriptorType()};
        objs_ = {output_image, input_image, weight_image, bias_buffer,
                 paramBuffer_};
    }

    void
    apply(const std::vector<std::shared_ptr<core::ITensor>> &inputs,
          const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        if (inputs[0]->dtype() == typeid(float)) {
            prepare<float>(inputs, outputs);
        } else {
            LOG_ERROR("unsupported");
        }
    }

    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {

        if (inputs[0]->dtype() == typeid(float)) {
            auto input = core::as_tensor<float>(inputs[0]);
            auto output = core::as_tensor<float>(outputs[0]);
            auto weight = core::as_tensor<float>(inputs[1]);
            auto bias = (inputs.size() > 2) ? core::as_tensor<float>(inputs[2])
                                            : nullptr;

            auto input_shape = input->getShape();
            auto weight_shape = weight->getShape();

            int batch = input_shape[0];
            int depth = input_shape[1];
            int in_height = input_shape[2];
            int in_width = input_shape[3];

            int out_batch = batch;
            int out_depth = weight_shape[0];
            int out_height = (in_height + 2 * pads_[0] -
                              dilations_[0] * (weight_shape[2] - 1) - 1) /
                                 strides_[0] +
                             1;
            int out_width = (in_width + 2 * pads_[1] -
                             dilations_[1] * (weight_shape[3] - 1) - 1) /
                                strides_[1] +
                            1;
            int realwidth = out_width * UP_DIV(out_depth, 4);
            int realheight = out_height * batch;

            auto *para = static_cast<conv2d::GPUConv2dParam *>(
                paramBuffer_->getMappedMemory());
            // vkimage params
            para->inputSize[0] = in_width;
            para->inputSize[1] = in_height;
            para->inputSize[2] = depth;
            para->inputSize[3] = batch;
            para->outputSize[0] = out_width;
            para->outputSize[1] = out_height;
            para->outputSize[2] = out_depth;
            para->outputSize[3] = out_batch;
            // original params

            para->kernel_shape[0] = weight_shape[3];
            para->kernel_shape[1] = weight_shape[2];
            para->kernel_shape[2] = weight_shape[1];
            para->kernel_shape[3] = weight_shape[0];
            para->stride[0] = strides_[0];
            para->stride[1] = strides_[1];
            para->padding[0] = pads_[0];
            para->padding[1] = pads_[1];
            para->dilation[0] = dilations_[0];
            para->dilation[1] = dilations_[1];

            para->groups = groups_;
            para->bias = bias ? 1 : 0;
            para->activation = static_cast<int>(activation_);
            paramBuffer_->unmapMemory();

            if (bias)
                bias->copyToGPU(m_dev_, m_cmdpool_);

            submit(conv2d_spv, conv2d_spv_len, UP_DIV(realwidth, 16),
                   UP_DIV(realheight, 16));
        } else {
            LOG_ERROR("Unsupported data type");
        }
    }

  private:
    std::vector<int> kernel_shape_;
    std::vector<int> strides_;
    std::vector<int> pads_;
    std::vector<int> dilations_;
    int groups_;
    conv2d::ActivationMode activation_;

    // conv2d::PaddingMode padding_mode_;

    std::shared_ptr<VulkanBuffer> paramBuffer_;
    std::shared_ptr<VulkanBuffer> biasBuffer_;
};

} // namespace ops
} // namespace vkop
#endif // OPS_OCONV2D_HPP_