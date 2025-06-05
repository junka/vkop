// Copyright 2025 @junka
#ifndef OPS_OCONV2D_HPP_
#define OPS_OCONV2D_HPP_

#include "Operator.hpp"

#include "Tensor.hpp"
#include "VulkanBuffer.hpp"
#include "VulkanCommandBuffer.hpp"
#include "VulkanCommandPool.hpp"
#include "VulkanDevice.hpp"
#include "VulkanImage.hpp"
#include "VulkanPipeline.hpp"
#include "VulkanQueryPool.hpp"

namespace vkop {
namespace ops {

class Add : public Operator {
  public:
    Add() = default;

    template <typename T>
    void prepare(std::vector<std::shared_ptr<core::Tensor<T>>> inputs,
                 std::vector<std::shared_ptr<core::Tensor<T>>> outputs) {
        auto input_a = inputs[0];
        auto input_b = inputs[1];
        auto output = outputs[0];

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

        inputAImage_ = input_a->make_vkimg(
            m_phydev_, m_dev_,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | exflags);
        inputBImage_ = input_b->make_vkimg(
            m_phydev_, m_dev_,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | exflags);
#ifdef VK_EXT_host_image_copy
        if (m_dev->is_support_host_image_copy()) {
            if (m_dev->checkHostImageCopyDstLayoutSupport(
                    VK_IMAGE_LAYOUT_GENERAL)) {
                outputImage->hostImaggeTransition(VK_IMAGE_LAYOUT_GENERAL);
            } else {
                outputImage->hostImaggeTransition(
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            }
            inputAImage->hostImaggeTransition(
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            inputBImage->hostImaggeTransition(
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        } else
#endif
        {
            VulkanCommandBuffer cmd(device, m_cmdpool_->getCommandPool());
            cmd.begin();
            outputImage_->writeBarrier(cmd.get());
            inputAImage_->readBarrier(cmd.get());
            inputBImage_->readBarrier(cmd.get());
            cmd.end();
            cmd.submit(m_dev_->getComputeQueue());
        }
    }
    template <typename T>
    void apply(std::vector<std::shared_ptr<core::Tensor<T>>> inputs,
               std::vector<std::shared_ptr<core::Tensor<T>>> outputs) {
        auto input_a = inputs[0];
        auto input_b = inputs[1];
        auto output = outputs[0];

        auto input_shape = input_a->getTensorShape();
        int batch = input_shape[0];
        int depth = input_shape[1];
        int out_height = input_shape[2];
        int out_width = input_shape[3];

        int realwidth = out_width * UP_DIV(depth, 4);
        int realheight = out_height * batch;
        if (output->size() == 0) {
            output->resize(input_a->getTensorShape());
        }
        prepare(inputs, outputs);

        VkDevice device = m_dev_->getLogicalDevice();

        auto inputa_rgba = input_a->convertTensorToRGBA();
        auto inputb_rgba = input_b->convertTensorToRGBA();
#ifdef VK_EXT_host_image_copy
        if (m_dev->is_support_host_image_copy()) {
            inputAImage_->hostImageCopyToDevice(inputa_rgba.data());
            inputBImage_->hostImageCopyToDevice(inputb_rgba.data());
        } else
#endif
        {
            VulkanCommandBuffer cmdstg(device, m_cmdpool_->getCommandPool());
            cmdstg.begin();
            inputAImage_->stagingBufferCopyToImage(cmdstg.get(),
                                                   inputa_rgba.data());
            inputBImage_->stagingBufferCopyToImage(cmdstg.get(),
                                                   inputb_rgba.data());
            cmdstg.end();
            cmdstg.submit(m_dev_->getComputeQueue());
        }
        VulkanCommandBuffer cmd(device, m_cmdpool_->getCommandPool());
        cmd.begin();
        inputAImage_->readBarrier(cmd.get());
        inputBImage_->readBarrier(cmd.get());
        cmd.end();
        cmd.submit(m_dev_->getComputeQueue());

        submit(out_width, out_height);

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
    std::shared_ptr<VulkanImage> outputImage_;
    std::shared_ptr<VulkanImage> inputAImage_;
    std::shared_ptr<VulkanImage> inputBImage_;

    void submit(int out_width, int out_height) override;
};

} // namespace ops
} // namespace vkop
#endif