// Copyright 2025 @junka
#ifndef OPS_SOFTMAX_HPP_
#define OPS_SOFTMAX_HPP_

#include "UnaryFactory.hpp"

extern unsigned char softmax_spv[];
extern unsigned int softmax_spv_len;

namespace vkop {
namespace ops {
namespace softmax {

using ivec4 = int[4];
using ivec2 = int[2];

struct GpuSoftMaxParam {
    ivec4 outImgSize;
    ivec4 outShape;
    int groupSize;
    int totalGroups;
    int axis; // 0: N, 1: C, 2: H, 3: W
};

} // namespace softmax

class Softmax : public Operator {
  public:
    Softmax() = default;

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("axis ") != attributes.end()) {
            auto axis = std::stoi(attributes.at("axis"));
            axis_ = axis;
        }
    }

    template <typename T>
    void prepare(std::vector<std::shared_ptr<core::Tensor<T>>> inputs,
                 std::vector<std::shared_ptr<core::Tensor<T>>> outputs) {
        auto input = inputs[0];
        auto output = outputs[0];

        auto input_shape = input->getTensorShape();

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

        paramBuffer_ = std::make_shared<VulkanBuffer>(
            m_dev_, sizeof(softmax::GpuSoftMaxParam),
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
        } else
#endif
        {
            VulkanCommandBuffer cmd(device, m_cmdpool_->getCommandPool());
            cmd.begin();
            outputImage_->writeBarrier(cmd.get());
            inputImage_->readBarrier(cmd.get());
            cmd.end();
            cmd.submit(m_dev_->getComputeQueue());
        }
    }

    template <typename T>
    void apply(std::vector<std::shared_ptr<core::Tensor<T>>> inputs,
               std::vector<std::shared_ptr<core::Tensor<T>>> outputs) {
        auto input = inputs[0];
        auto output = outputs[0];
        auto input_shape = input->getTensorShape();

        if (input_shape.size() != 4) {
            throw std::invalid_argument("Input must have 4 dimensions.");
        }
        int batch = input_shape[0];
        int depth = input_shape[1];
        int out_height = input_shape[2];
        int out_width = input_shape[3];

        int realwidth = out_width * UP_DIV(depth, 4);
        int realheight = out_height * batch;

        if (output->size() == 0) {
            output->resize(batch, depth, out_height, out_width);
        }
        prepare(inputs, outputs);

        auto *para = static_cast<softmax::GpuSoftMaxParam *>(
            paramBuffer_->getMappedMemory());
        // vkimage params
        para->outImgSize[0] = realwidth;
        para->outImgSize[1] = realheight;
        para->outImgSize[2] = 1;
        para->outImgSize[3] = 0;
        para->outShape[0] = batch;
        para->outShape[1] = out_height;
        para->outShape[2] = out_width;
        para->outShape[3] = depth;
        int total_groups = 1;
        for (int i = 0; i < 4; ++i) {
            if (i != axis_) {
                total_groups *= input_shape[i];
            }
        }
        para->groupSize = input_shape[axis_];
        para->totalGroups = total_groups;
        para->axis = axis_;
        paramBuffer_->unmapMemory();

        VkDevice device = m_dev_->getLogicalDevice();

        // auto input_rgba = inputImage_->convertNCHWToRGBA(input);
        auto input_rgba = input->convertTensorToRGBA();
#ifdef VK_EXT_host_image_copy
        if (m_dev_->is_support_host_image_copy()) {
            inputImage_->hostImageCopyToDevice(input_rgba.data());
        } else
#endif
        {
            VulkanCommandBuffer cmdstg(device, m_cmdpool_->getCommandPool());
            cmdstg.begin();
            inputImage_->stagingBufferCopyToImage(cmdstg.get(),
                                                  input_rgba.data());
            cmdstg.end();
            cmdstg.submit(m_dev_->getComputeQueue());
        }

        VulkanCommandBuffer cmd(device, m_cmdpool_->getCommandPool());
        cmd.begin();
        inputImage_->readBarrier(cmd.get());
        cmd.end();
        cmd.submit(m_dev_->getComputeQueue());

        submit(softmax_spv, softmax_spv_len, out_width, out_height);

        std::vector<T> tmp(realheight * realwidth * 4);
        T *ptr = tmp.data();
#ifdef VK_EXT_host_image_copy
        if (m_dev_->is_support_host_image_copy()) {
            outputImage_->hostImageCopyToHost(ptr);
        } else
#endif
        {
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

    void execute(
        std::vector<std::shared_ptr<core::Tensor<uint16_t>>> inputs,
        std::vector<std::shared_ptr<core::Tensor<uint16_t>>> outputs) override;

  private:
    int axis_;

    std::shared_ptr<VulkanImage> outputImage_;
    std::shared_ptr<VulkanImage> inputImage_;

    std::shared_ptr<VulkanBuffer> paramBuffer_;

    void submit(const unsigned char *spv, unsigned int spv_len, int out_width,
                int out_height) override;
};

} // namespace ops
} // namespace vkop
#endif // OPS_SOFTMAX_HPP_
