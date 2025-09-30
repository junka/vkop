// Copyright 2025 @junka
#ifndef OPS_OCONV2D_HPP_
#define OPS_OCONV2D_HPP_

#include <string>
#include <unordered_map>
#include <utility>

#include "Operator.hpp"

#include "core/Tensor.hpp"
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

using ivec4 = int[4];
using ivec2 = int[2];

struct GPUConv2dParam {

    // ivec4 outImgSize;
    ivec4 inputSize;
    ivec4 outputSize;
    // ivec4 offset;  //batchOffset, hOffset, outputHeight, other

    int in_channels;
    int out_channels;

    ivec2 kernel_shape;
    ivec2 stride;
    ivec2 padding;
    ivec2 dilation;

    int groups;
    bool bias;
    int padding_mode;
};

} // namespace conv2d

class Conv2d : public Operator {
  public:
    Conv2d() = default;

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("auto_pad ") != attributes.end()) {
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
    }

    template <typename T>
    void prepare(std::vector<std::shared_ptr<core::Tensor<T>>> inputs,
                 std::vector<std::shared_ptr<core::Tensor<T>>> outputs) {
        auto input = inputs[0];
        auto weight = inputs[1];

        auto output = outputs[0];
        auto input_shape = input->getTensorShape();
        auto weight_shape = weight->getTensorShape();

        int out_channel = weight_shape[0];

        VkDevice device = m_dev_->getLogicalDevice();
        int exflags = 0;
        if (m_dev_->is_support_host_image_copy()) {
#ifdef VK_EXT_host_image_copy
            exflags |= VK_IMAGE_USAGE_HOST_TRANSFER_BIT;
#endif
        }

        outputImage_ = output->make_vkimg(
            m_dev_, VK_IMAGE_USAGE_STORAGE_BIT |
                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | exflags);

        inputImage_ = input->make_vkimg(
            m_dev_, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT | exflags);

        weightImage_ = weight->make_vkimg(
            m_dev_, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT | exflags);

        biasImage_ = std::make_shared<VulkanImage>(
            m_dev_, VkExtent3D{static_cast<uint32_t>(out_channel), 1, 1},
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | exflags,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        paramBuffer_ = std::make_shared<VulkanBuffer>(
            m_dev_, sizeof(conv2d::GPUConv2dParam),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

#ifdef VK_EXT_host_image_copy
        if (m_dev_->is_support_host_image_copy()) {
            if (m_dev_->checkHostImageCopyDstLayoutSupport(
                    VK_IMAGE_LAYOUT_GENERAL)) {
                outputImage_->hostImaggeTransition(VK_IMAGE_LAYOUT_GENERAL);
            } else {
                outputImage_->hostImaggeTransition(
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            }
            inputImage_->hostImaggeTransition(
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            weightImage_->hostImaggeTransition(
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            biasImage_->hostImaggeTransition(
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        } else
#endif
        {
            VulkanCommandBuffer cmd(device, m_cmdpool_->getCommandPool());
            cmd.begin();
            outputImage_->writeBarrier(cmd.get());
            inputImage_->readBarrier(cmd.get());
            weightImage_->readBarrier(cmd.get());
            biasImage_->readBarrier(cmd.get());
            cmd.end();
            cmd.submit(m_dev_->getComputeQueue());
        }
    }

    template <typename T>
    void apply(std::vector<std::shared_ptr<core::Tensor<T>>> inputs,
               std::vector<std::shared_ptr<core::Tensor<T>>> outputs) {
        auto input = inputs[0];
        auto weight = inputs[1];
        std::shared_ptr<core::Tensor<T>> bias;
        if (inputs.size() > 2) {
            bias = inputs[2];
        }
        auto output = outputs[0];

        auto input_shape = input->getTensorShape();
        auto weight_shape = weight->getTensorShape();

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
        int realwidth = out_width * UP_DIV(depth, 4);
        int realheight = out_height * batch;

        if (output->size() == 0) {
            output->resize(out_batch, out_depth, out_height, out_width);
        }
        prepare(inputs, outputs);

        auto *para = static_cast<conv2d::GPUConv2dParam *>(
            paramBuffer_->getMappedMemory());
        // vkimage params
        // para->outImgSize[0] = realwidth;
        // para->outImgSize[1] = realheight;
        // para->outImgSize[2] = 1;
        // para->outImgSize[3] = 0;
        para->inputSize[0] = in_width;
        para->inputSize[1] = in_height;
        para->inputSize[2] = UP_DIV(depth, 4);
        para->inputSize[3] = batch;
        para->outputSize[0] = out_width;
        para->outputSize[1] = out_height;
        para->outputSize[2] = UP_DIV(out_depth, 4);
        para->outputSize[3] = out_batch;
        // original params
        para->in_channels = depth;
        para->out_channels = out_depth;
        para->kernel_shape[0] = kernel_shape_[0];
        para->kernel_shape[1] = kernel_shape_[1];
        para->stride[0] = strides_[0];
        para->stride[1] = strides_[1];
        para->padding[0] = pads_[0];
        para->padding[1] = pads_[1];
        para->dilation[0] = dilations_[0];
        para->dilation[1] = dilations_[1];

        para->groups = groups_;
        para->bias = (inputs.size() > 2);
        para->padding_mode = static_cast<int>(padding_mode_);
        paramBuffer_->unmapMemory();

        VkDevice device = m_dev_->getLogicalDevice();

        auto input_rgba = input->convertTensorToRGBA();

        // for depthwise conv, always has same kernel num with input channels.
        // each conv kernel channles do work for one input channel,
        // usually the kernel shape is (oc, ic, kh, kw)
        // but for depthwise kernel shape is (ic, 1, kh, kw)
        // which would has a follow-up pointwise action with shape (oc, ic, 1,
        // 1)

        // Split the weight into depthwise and pointwise components
        // auto *depthwise_weight = new core::Tensor<T>(
        //     weight_shape[1], 1, weight_shape[2], weight_shape[3]);
        // auto *pointwise_weight =
        //     new core::Tensor<T>(weight_shape[0], weight_shape[1], 1, 1);

        // auto *weight_ptr = weight->data();
        // auto *depthwise_ptr = depthwise_weight->data();
        // auto *pointwise_ptr = pointwise_weight->data();
        // Extract depthwise weights (ic, 1, kh, kw)
        // for (int ic = 0; ic < weight_shape[1]; ++ic) {
        //     for (int kh = 0; kh < weight_shape[2]; ++kh) {
        //         for (int kw = 0; kw < weight_shape[3]; ++kw) {
        //             depthwise_ptr[ic * weight_shape[2] * weight_shape[3] +
        //                           kh * weight_shape[3] + kw] =
        //                 weight_ptr[ic * weight_shape[2] * weight_shape[3] +
        //                            kh * weight_shape[3] + kw];
        //         }
        //     }
        // }

        // Extract pointwise weights (oc, ic, 1, 1)
        // for (int oc = 0; oc < weight_shape[0]; ++oc) {
        //     for (int ic = 0; ic < weight_shape[1]; ++ic) {
        //         pointwise_ptr[oc * weight_shape[1] + ic] =
        //             weight_ptr[oc * weight_shape[1] * weight_shape[2] *
        //                            weight_shape[3] +
        //                        ic];
        //     }
        // }

        auto weight_rgba = weight->convertTensorToRGBA();
        // Convert depthwise and pointwise weights to RGBA format
        // auto depthwise_rgba = depthwise_weight->convertTensorToRGBA();
        // auto pointwise_rgba = weightImage_->convertNCHWToRGBA(
        //     pointwise_weight.data(), {weight_shape[0], weight_shape[1], 1,
        //     1});

        // bias as 1D image data
#ifdef VK_EXT_host_image_copy
        if (m_dev_->is_support_host_image_copy()) {
            inputImage_->hostImageCopyToDevice(input_rgba.data());
            weightImage_->hostImageCopyToDevice(weight_rgba.data());
            biasImage_->hostImageCopyToDevice(bias->data());
        } else
#endif
        {
            VulkanCommandBuffer cmdstg(device, m_cmdpool_->getCommandPool());
            cmdstg.begin();
            inputImage_->stagingBufferCopyToImage(cmdstg.get(),
                                                  input_rgba.data());
            weightImage_->stagingBufferCopyToImage(cmdstg.get(),
                                                   weight_rgba.data());
            biasImage_->stagingBufferCopyToImage(cmdstg.get(), bias->data());
            cmdstg.end();
            cmdstg.submit(m_dev_->getComputeQueue());
        }

        VulkanCommandBuffer cmd(device, m_cmdpool_->getCommandPool());
        cmd.begin();
        inputImage_->readBarrier(cmd.get());
        weightImage_->readBarrier(cmd.get());
        biasImage_->readBarrier(cmd.get());
        cmd.end();
        cmd.submit(m_dev_->getComputeQueue());

        submit(conv2d_spv, conv2d_spv_len, out_width, out_height);

        std::vector<T> tmp(realheight * realwidth * 4);
        T *ptr = tmp.data();
#ifdef VK_EXT_host_image_copy
        if (m_dev_->is_support_host_image_copy()) {
            outputImage_->hostImageCopyToHost(ptr);
        } else
#endif
        {
            VulkanCommandBuffer cmd(device, m_cmdpool_->getCommandPool());
            cmd.begin();
            VulkanCommandBuffer cmdstg1(device, m_cmdpool_->getCommandPool());
            cmdstg1.begin();
            outputImage_->stagingBufferCopyToHost(cmdstg1.get());
            cmdstg1.end();
            cmdstg1.submit(m_dev_->getComputeQueue());
            outputImage_->readStaingBuffer(ptr);
        }

        output->convertRGBAToTensor(ptr);
    }

    void execute(
        std::vector<std::shared_ptr<core::Tensor<uint16_t>>> inputs,
        std::vector<std::shared_ptr<core::Tensor<uint16_t>>> outputs) override;
    void
    execute(std::vector<std::shared_ptr<core::Tensor<float>>> inputs,
            std::vector<std::shared_ptr<core::Tensor<float>>> outputs) override;
    void
    execute(std::vector<std::shared_ptr<core::Tensor<int>>> inputs,
            std::vector<std::shared_ptr<core::Tensor<int>>> outputs) override;

  private:
    std::vector<int> kernel_shape_;
    std::vector<int> strides_;
    std::vector<int> pads_;
    std::vector<int> dilations_;
    int groups_;

    conv2d::PaddingMode padding_mode_;

    std::shared_ptr<VulkanImage> outputImage_;
    std::shared_ptr<VulkanImage> inputImage_;
    std::shared_ptr<VulkanImage> weightImage_;
    std::shared_ptr<VulkanBuffer> paramBuffer_;
    std::shared_ptr<VulkanImage> biasImage_;

    void submit(const unsigned char *spv, unsigned int spv_len, int out_width,
                int out_height) override;
};

} // namespace ops
} // namespace vkop
#endif // OPS_OCONV2D_HPP_