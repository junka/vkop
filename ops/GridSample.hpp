// Copyright 2025 @junka
#ifndef OPS_GRIDSAMPLE_HPP_
#define OPS_GRIDSAMPLE_HPP_

#include <unistd.h>
#include <vector>

#include "Operator.hpp"

#include "core/Tensor.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanCommandBuffer.hpp"
#include "vulkan/VulkanImage.hpp"
#include "vulkan/VulkanPipeline.hpp"
#include "vulkan/VulkanQueryPool.hpp"

#include "include/logger.hpp"

extern unsigned char grid_sample_spv[];
extern unsigned int grid_sample_spv_len;

namespace vkop {
namespace ops {

namespace gridsample {
enum class InterpolationMode { BILINEAR, NEAREST };

enum class PaddingMode { ZEROS, BORDER, REFLECTION };

using ivec4 = int[4];
using ivec2 = int[2];

struct GpuGridSampleParam {
    ivec4 outImgSize;
    ivec2 inShape;
    ivec2 outShape;
    bool align_corners;
    int padding_mode;
    int interpolation_mode;
};
} // namespace gridsample

class GridSample : public Operator {
  public:
    GridSample() = default;

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        attributes.find("align_corners") != attributes.end()
            ? align_corners_ = (attributes.at("align_corners") == "1" ||
                                attributes.at("align_corners") == "true")
            : align_corners_ = false;
        if (attributes.find("interpolation_mode") != attributes.end()) {
            std::string mode = attributes.at("interpolation_mode");
            if (mode == "linear" || mode == "bilinear") {
                interpolation_mode_ = gridsample::InterpolationMode::BILINEAR;
            } else if (mode == "nearest") {
                interpolation_mode_ = gridsample::InterpolationMode::NEAREST;
            } else {
                LOG_ERROR("Unsupported interpolation_mode: " + mode);
                throw std::invalid_argument("Unsupported interpolation_mode: " +
                                            mode);
            }
        } else {
            interpolation_mode_ = gridsample::InterpolationMode::BILINEAR;
        }
        if (attributes.find("padding_mode") != attributes.end()) {
            std::string mode = attributes.at("padding_mode");
            if (mode == "zeros") {
                padding_mode_ = gridsample::PaddingMode::ZEROS;
            } else if (mode == "border") {
                padding_mode_ = gridsample::PaddingMode::BORDER;
            } else if (mode == "reflection") {
                padding_mode_ = gridsample::PaddingMode::REFLECTION;
            } else {
                LOG_ERROR("Unsupported padding_mode: " + mode);
                throw std::invalid_argument("Unsupported padding_mode: " +
                                            mode);
            }
        } else {
            padding_mode_ = gridsample::PaddingMode::ZEROS;
        }
    }

    template <typename T>
    void prepare(std::vector<std::shared_ptr<core::Tensor<T>>> inputs,
                 std::vector<std::shared_ptr<core::Tensor<T>>> outputs) {
        auto input = inputs[0];
        auto grid = inputs[1];
        auto output = outputs[0];

        auto input_shape = input->getTensorShape();
        auto grid_shape = grid->getTensorShape();

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

        gridImage_ = grid->make_vkimg(
            m_dev_, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT | exflags);

        paramBuffer_ = std::make_shared<VulkanBuffer>(
            m_dev_, sizeof(gridsample::GpuGridSampleParam),
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
            gridImage_->hostImaggeTransition(
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        } else
#endif
        {
            VulkanCommandBuffer cmd(device, m_cmdpool_->getCommandPool());
            cmd.begin();
            outputImage_->writeBarrier(cmd.get());
            inputImage_->readBarrier(cmd.get());
            gridImage_->readBarrier(cmd.get());
            cmd.end();
            cmd.submit(m_dev_->getComputeQueue());
        }
    }

    template <typename T>
    void apply(std::vector<std::shared_ptr<core::Tensor<T>>> inputs,
               std::vector<std::shared_ptr<core::Tensor<T>>> outputs) {
        auto input = inputs[0];
        auto grid = inputs[1];
        auto output = outputs[0];
        auto input_shape = input->getTensorShape();
        auto grid_shape = grid->getTensorShape();

        if (input_shape.size() != 4 || grid_shape.size() != 4) {
            throw std::invalid_argument(
                "Input and grid must have 4 dimensions.");
        }
        int batch = input_shape[0];
        int depth = input_shape[1];
        int in_height = input_shape[2];
        int in_width = input_shape[3];
        int out_height = grid_shape[1];
        int out_width = grid_shape[2];

        int realwidth = out_width * UP_DIV(depth, 4);
        int realheight = out_height * batch;

        if (output->size() == 0) {
            output->resize(batch, depth, out_height, out_width);
        }
        prepare(inputs, outputs);

        auto *para = static_cast<gridsample::GpuGridSampleParam *>(
            paramBuffer_->getMappedMemory());
        // vkimage params
        para->outImgSize[0] = realwidth;
        para->outImgSize[1] = realheight;
        para->outImgSize[2] = 1;
        para->outImgSize[3] = 0;
        // original params
        para->inShape[0] = in_width;
        para->inShape[1] = in_height;
        para->outShape[0] = out_width;
        para->outShape[1] = out_height;
        para->align_corners = align_corners_;
        para->padding_mode = static_cast<int>(padding_mode_);
        para->interpolation_mode = static_cast<int>(interpolation_mode_);
        paramBuffer_->unmapMemory();

        VkDevice device = m_dev_->getLogicalDevice();

        // auto input_rgba = inputImage_->convertNCHWToRGBA(input);
        auto input_rgba = input->convertTensorToRGBA();
        // auto grid_rgba = gridImage_->convertNCHWToRGBA(grid);
        auto grid_rgba = grid->convertTensorToRGBA();
#ifdef VK_EXT_host_image_copy
        if (m_dev_->is_support_host_image_copy()) {
            inputImage_->hostImageCopyToDevice(input_rgba.data());
            gridImage_->hostImageCopyToDevice(grid_rgba.data());
        } else
#endif
        {
            VulkanCommandBuffer cmdstg(device, m_cmdpool_->getCommandPool());
            cmdstg.begin();
            inputImage_->stagingBufferCopyToImage(cmdstg.get(),
                                                  input_rgba.data());
            gridImage_->stagingBufferCopyToImage(cmdstg.get(),
                                                 grid_rgba.data());
            cmdstg.end();
            cmdstg.submit(m_dev_->getComputeQueue());
        }

        VulkanCommandBuffer cmd(device, m_cmdpool_->getCommandPool());
        cmd.begin();
        inputImage_->readBarrier(cmd.get());
        gridImage_->readBarrier(cmd.get());
        cmd.end();
        cmd.submit(m_dev_->getComputeQueue());

        submit(grid_sample_spv, grid_sample_spv_len, out_width, out_height);

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
    gridsample::InterpolationMode interpolation_mode_;
    gridsample::PaddingMode padding_mode_;
    bool align_corners_;

    std::shared_ptr<VulkanImage> outputImage_;
    std::shared_ptr<VulkanImage> inputImage_;
    std::shared_ptr<VulkanImage> gridImage_;
    std::shared_ptr<VulkanBuffer> paramBuffer_;

    void submit(const unsigned char *spv, unsigned int spv_len, int out_width,
                int out_height) override;
};

} // namespace ops
} // namespace vkop

#endif // OPS_GRIDSAMPLE_HPP_
