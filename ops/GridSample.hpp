// Copyright 2025 @junka
#ifndef OPS_GRIDSAMPLE_HPP_
#define OPS_GRIDSAMPLE_HPP_

#include <unistd.h>
#include <vector>

#include "Operator.hpp"
#include "vulkan/vulkan.hpp"

#include "VulkanBuffer.hpp"
#include "VulkanCommandBuffer.hpp"
#include "VulkanCommandPool.hpp"
#include "VulkanDevice.hpp"
#include "VulkanImage.hpp"
#include "VulkanPipeline.hpp"
#include "VulkanQueryPool.hpp"

#include "logger.hpp"

namespace vkop {
namespace ops {

enum class InterpolationMode { BILINEAR, NEAREST };

enum class PaddingMode { ZEROS, BORDER, REFLECTION };

#define UP_DIV(x, y) (((x) + (y)-1) / (y))

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

class GridSample : public Operator {
  public:
    explicit GridSample(
        InterpolationMode interp_mode = InterpolationMode::BILINEAR,
        PaddingMode pad_mode = PaddingMode::ZEROS, bool align_corners = false)
        : interpolation_mode_(interp_mode), padding_mode_(pad_mode),
          align_corners_(align_corners) {}

    template <typename T>
    std::vector<T> apply(const std::vector<T> &input,
                         const std::vector<T> &grid,
                         const std::vector<int> &input_shape,
                         const std::vector<int> &grid_shape) {
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

        prepare(input_shape, grid_shape);

        auto *para = reinterpret_cast<GpuGridSampleParam *>(
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

        const T *input_ptr = input.data();
        const T *grid_ptr = grid.data();
        auto input_rgba =
            inputImage_->convertNCHWToRGBA(input_ptr, input_shape);
        auto grid_rgba = gridImage_->convertNCHWToRGBA(grid_ptr, grid_shape);
#ifdef VK_EXT_host_image_copy
        if (m_dev->is_support_host_image_copy()) {
            inputImage->hostImageCopyToDevice(inputRGBA.data());
            gridImage->hostImageCopyToDevice(gridRGBA.data());
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

        submit(out_width, out_height);

        std::vector<T> tmp(realheight * realwidth * 4);
        T *ptr = tmp.data();
#ifdef VK_EXT_host_image_copy
        if (m_dev->is_support_host_image_copy()) {
            outputImage->hostImageCopyToHost(ptr);
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

        std::vector<T> output = outputImage_->convertRGBAToNCHW<T>(
            ptr, {batch, depth, out_height, out_width});
        return output;
    }

    void set_runtime_device(VkPhysicalDevice phydev,
                            std::shared_ptr<VulkanDevice> dev,
                            VulkanCommandPool *cmdpool) {
        m_phydev_ = phydev;
        m_dev_ = std::move(dev);
        m_cmdpool_ = cmdpool;
    }

  private:
    InterpolationMode interpolation_mode_;
    PaddingMode padding_mode_;
    bool align_corners_;

    std::shared_ptr<VulkanImage> outputImage_;
    std::shared_ptr<VulkanImage> inputImage_;
    std::shared_ptr<VulkanImage> gridImage_;
    std::shared_ptr<VulkanBuffer> paramBuffer_;

    VkPhysicalDevice m_phydev_;
    std::shared_ptr<VulkanDevice> m_dev_;
    VulkanCommandPool *m_cmdpool_;

    void prepare(const std::vector<int> &input_shape,
                 const std::vector<int> &grid_shape);
    void submit(int out_width, int out_height);
};

} // namespace ops
} // namespace vkop

#endif // OPS_GRIDSAMPLE_HPP_
