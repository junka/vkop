// Copyright 2025 @junka
#ifndef OPS_OCONV2D_HPP_
#define OPS_OCONV2D_HPP_

#include <string>
#include <utility>

#include "Operator.hpp"

#include "Tensor.hpp"
#include "VulkanBuffer.hpp"
#include "VulkanCommandBuffer.hpp"
#include "VulkanCommandPool.hpp"
#include "VulkanDevice.hpp"
#include "VulkanImage.hpp"
#include "VulkanPipeline.hpp"
#include "VulkanQueryPool.hpp"

extern unsigned char conv2d_spv[];
extern unsigned int conv2d_spv_len;
namespace vkop {
namespace ops {

namespace conv2d {

enum class PaddingMode { ZEROS, REFLECT, REPLICATE, CIRCULAR };

using ivec4 = int[4];
using ivec2 = int[2];

struct GPUConv2dParam {

    ivec4 outImgSize;
    ivec4 inputSize;
    // ivec4 offset;  //batchOffset, hOffset, outputHeight, other

    int in_channels;
    int out_channels;

    ivec2 kernel_size;
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

    explicit Conv2d(
        int in_channels, int out_channels, int kernel_size, int stride,
        int padding, int dilation = 1, int groups = 1, bool bias = true,
        conv2d::PaddingMode padding_mode = conv2d::PaddingMode::ZEROS)
        : in_channels_(in_channels), out_channels_(out_channels),
          kernel_size_({kernel_size, kernel_size}), stride_({stride, stride}),
          padding_({padding, padding}), dilation_({dilation, dilation}),
          groups_(groups), bias_(bias), padding_mode_(padding_mode) {}

    explicit Conv2d(
        int in_channels, int out_channels, int kernel_h, int kernel_w,
        int stride_h, int stride_w, int padding_h, int padding_w,
        int dilation_h = 1, int dilation_w = 1, int groups = 1,
        bool bias = true,
        conv2d::PaddingMode padding_mode = conv2d::PaddingMode::ZEROS)
        : in_channels_(in_channels), out_channels_(out_channels),
          kernel_size_({kernel_h, kernel_w}), stride_({stride_h, stride_w}),
          padding_({padding_h, padding_w}), dilation_({dilation_h, dilation_w}),
          groups_(groups), bias_(bias), padding_mode_(padding_mode) {}

    explicit Conv2d(
        int in_channels, int out_channels,
        std::pair<int, int> kernel_size = {1, 1},
        std::pair<int, int> stride = {1, 1},
        std::pair<int, int> padding = {0, 0},
        std::pair<int, int> dilation = {1, 1}, int groups = 1, bool bias = true,
        conv2d::PaddingMode padding_mode = conv2d::PaddingMode::ZEROS)
        : in_channels_(in_channels), out_channels_(out_channels),
          kernel_size_(std::move(kernel_size)), stride_(std::move(stride)),
          padding_(std::move(padding)), dilation_(std::move(dilation)),
          groups_(groups), bias_(bias), padding_mode_(padding_mode) {}

    void setAttribute(
        int in_channels, int out_channels,
        std::pair<int, int> kernel_size = {1, 1},
        std::pair<int, int> stride = {1, 1},
        std::pair<int, int> padding = {0, 0},
        std::pair<int, int> dilation = {1, 1}, int groups = 1, bool bias = true,
        conv2d::PaddingMode padding_mode = conv2d::PaddingMode::ZEROS) {
        in_channels_ = in_channels;
        out_channels_ = out_channels;
        kernel_size_ = std::move(kernel_size);
        stride_ = std::move(stride);
        padding_ = std::move(padding);
        dilation_ = std::move(dilation);
        groups_ = groups;
        bias_ = bias;
        padding_mode_ = padding_mode;
    }

    template <typename T>
    void prepare(std::vector<std::shared_ptr<core::Tensor<T>>> inputs,
                 std::vector<std::shared_ptr<core::Tensor<T>>> outputs) {
        auto input = inputs[0];
        auto weight = inputs[1];
        // core::Tensor<T> *bias = inputs[2];

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

        outputImage_ =
            output->make_vkimg(m_phydev_, m_dev_,
                               VK_IMAGE_USAGE_STORAGE_BIT |
                                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT | exflags);

        inputImage_ = input->make_vkimg(
            m_phydev_, m_dev_,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | exflags);

        weightImage_ = weight->make_vkimg(
            m_phydev_, m_dev_,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | exflags);

        biasImage_ = std::make_shared<VulkanImage>(
            m_phydev_, m_dev_->getComputeQueueFamilyIndex(), device,
            VkExtent3D{static_cast<uint32_t>(out_channel), 1, 1},
            VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | exflags,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        paramBuffer_ = std::make_shared<VulkanBuffer>(
            m_phydev_, m_dev_->getComputeQueueFamilyIndex(), device,
            sizeof(conv2d::GPUConv2dParam), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

#ifdef VK_EXT_host_image_copy
        if (m_dev->is_support_host_image_copy()) {
            if (m_dev->checkHostImageCopyDstLayoutSupport(
                    VK_IMAGE_LAYOUT_GENERAL)) {
                outputImage->hostImaggeTransition(VK_IMAGE_LAYOUT_GENERAL);
            } else {
                outputImage->hostImaggeTransition(
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            }
            inputImage->hostImaggeTransition(
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            weightImage->hostImaggeTransition(
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            biasImage->hostImaggeTransition(
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
        auto bias = inputs[2];
        auto output = outputs[0];

        auto input_shape = input->getTensorShape();
        auto weight_shape = weight->getTensorShape();

        int batch = input_shape[0];
        int depth = input_shape[1];
        int in_height = input_shape[2];
        int in_width = input_shape[3];

        int out_batch = batch;
        int out_depth = weight_shape[0];
        int out_height = (in_height + 2 * padding_.first -
                          dilation_.first * (weight_shape[2] - 1) - 1) /
                             stride_.first +
                         1;
        int out_width = (in_width + 2 * padding_.second -
                         dilation_.second * (weight_shape[3] - 1) - 1) /
                            stride_.second +
                        1;
        int realwidth = out_width * UP_DIV(depth, 4);
        int realheight = out_height * batch;

        if (output->size() == 0) {
            output->resize(out_batch, out_depth, out_height, out_width);
        }
        prepare(inputs, outputs);

        auto *para = paramBuffer_->getMappedMemory<conv2d::GPUConv2dParam>();
        // vkimage params
        para->outImgSize[0] = realwidth;
        para->outImgSize[1] = realheight;
        para->outImgSize[2] = 1;
        para->outImgSize[3] = 0;
        // original params
        para->in_channels = in_channels_;
        para->out_channels = out_channels_;
        para->kernel_size[0] = kernel_size_.first;
        para->kernel_size[1] = kernel_size_.second;
        para->stride[0] = stride_.first;
        para->stride[1] = stride_.second;
        para->padding[0] = padding_.first;
        para->padding[1] = padding_.second;
        para->dilation[0] = dilation_.first;
        para->dilation[1] = dilation_.second;

        para->groups = groups_;
        para->bias = bias_;
        para->padding_mode = static_cast<int>(padding_mode_);
        // para->dilation = dilation_;
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
        auto *depthwise_weight = new core::Tensor<T>(
            weight_shape[1], 1, weight_shape[2], weight_shape[3]);
        auto *pointwise_weight =
            new core::Tensor<T>(weight_shape[0], weight_shape[1], 1, 1);
        auto *weight_ptr = weight->data();
        auto *depthwise_ptr = depthwise_weight->data();
        auto *pointwise_ptr = pointwise_weight->data();
        // Extract depthwise weights (ic, 1, kh, kw)
        for (int ic = 0; ic < weight_shape[1]; ++ic) {
            for (int kh = 0; kh < weight_shape[2]; ++kh) {
                for (int kw = 0; kw < weight_shape[3]; ++kw) {
                    depthwise_ptr[ic * weight_shape[2] * weight_shape[3] +
                                  kh * weight_shape[3] + kw] =
                        weight_ptr[ic * weight_shape[2] * weight_shape[3] +
                                   kh * weight_shape[3] + kw];
                }
            }
        }

        // Extract pointwise weights (oc, ic, 1, 1)
        for (int oc = 0; oc < weight_shape[0]; ++oc) {
            for (int ic = 0; ic < weight_shape[1]; ++ic) {
                pointwise_ptr[oc * weight_shape[1] + ic] =
                    weight_ptr[oc * weight_shape[1] * weight_shape[2] *
                                   weight_shape[3] +
                               ic];
            }
        }

        // Convert depthwise and pointwise weights to RGBA format
        auto depthwise_rgba = depthwise_weight->convertTensorToRGBA();
        // auto pointwise_rgba = weightImage_->convertNCHWToRGBA(
        //     pointwise_weight.data(), {weight_shape[0], weight_shape[1], 1,
        //     1});

        // bias as 1D image data
#ifdef VK_EXT_host_image_copy
        if (m_dev->is_support_host_image_copy()) {
            inputImage->hostImageCopyToDevice(inputRGBA.data());
            weightImage_->hostImageCopyToDevice(depthwise_rgba.data());
            biasImage_->hostImageCopyToDevice(bias.data());
        } else
#endif
        {
            VulkanCommandBuffer cmdstg(device, m_cmdpool_->getCommandPool());
            cmdstg.begin();
            inputImage_->stagingBufferCopyToImage(cmdstg.get(),
                                                  input_rgba.data());
            weightImage_->stagingBufferCopyToImage(cmdstg.get(),
                                                   depthwise_rgba.data());
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
        if (m_dev->is_support_host_image_copy()) {
            outputImage->hostImageCopyToHost(ptr);
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

    void
    execute(std::vector<std::shared_ptr<core::Tensor<float>>> inputs,
            std::vector<std::shared_ptr<core::Tensor<float>>> outputs) override;
    void
    execute(std::vector<std::shared_ptr<core::Tensor<int>>> inputs,
            std::vector<std::shared_ptr<core::Tensor<int>>> outputs) override;

  private:
    int in_channels_;
    int out_channels_;
    std::pair<int, int> kernel_size_;
    std::pair<int, int> stride_;
    std::pair<int, int> padding_;
    std::pair<int, int> dilation_;
    int groups_;
    bool bias_;
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