#ifndef GRIDSAMPLE_HPP
#define GRIDSAMPLE_HPP

#include <vector>

#include "Operator.hpp"
#include "vulkan/vulkan.hpp"

#include "VulkanDevice.hpp"
#include "VulkanBuffer.hpp"
#include "VulkanImage.hpp"
#include "VulkanPipeline.hpp"
#include "VulkanCommandPool.hpp"
#include "VulkanCommandBuffer.hpp"
#include "VulkanQueryPool.hpp"


namespace vkop {
namespace ops {

enum class InterpolationMode {
    Bilinear,
    Nearest
};

enum class PaddingMode {
    Zeros,
    Border,
    Reflection
};

#define UP_DIV(x, y) (((x) + (y) - 1) / (y))

typedef int ivec4[4];
typedef int ivec2[2];

struct GpuGridSampleParam {
    ivec4 outImgSize;
    ivec2 inShape;
    ivec2 outShape;
};

class GridSample: public Operator {
public:
    GridSample(InterpolationMode interp_mode = InterpolationMode::Bilinear,
               PaddingMode pad_mode = PaddingMode::Zeros,
               bool align_corners = false)
        : interpolation_mode(interp_mode),
          padding_mode(pad_mode),
          align_corners(align_corners) {}

    template <typename T>
    std::vector<T> apply(const std::vector<T>& input,
                        const std::vector<T>& grid,
                        const std::vector<int>& input_shape,
                        const std::vector<int>& grid_shape)
    {
        if (input_shape.size() != 4 || grid_shape.size() != 4) {
            throw std::invalid_argument("Input and grid must have 4 dimensions.");
        }
        int batch = input_shape[0];
        int depth = input_shape[1];
        int inHeight = input_shape[2];
        int inWidth = input_shape[3];
        int outHeight = grid_shape[1];
        int outWidth = grid_shape[2];

        prepare(input_shape, grid_shape);

        struct GpuGridSampleParam *para = reinterpret_cast<GpuGridSampleParam*>(paramBuffer->getMappedMemory());
        // vkimage params
        para->outImgSize[0] = UP_DIV(depth, 4) * outWidth;
        para->outImgSize[1] = outHeight * batch;
        para->outImgSize[2] = 1;
        para->outImgSize[3] = 0;
        // original params
        para->inShape[0] = inWidth;
        para->inShape[1] = inHeight;
        para->outShape[0] = outWidth;
        para->outShape[1] = outHeight;
        paramBuffer->unmapMemory();

        VkDevice device = m_dev->getLogicalDevice();

        const T* inputPtr = input.data();
        const T* gridPtr = grid.data();
        auto inputRGBA = inputImage->convertNCHWToRGBA(inputPtr, input_shape);
        auto gridRGBA = gridImage->convertNCHWToRGBA(gridPtr, grid_shape);
#ifdef VK_EXT_host_image_copy
        if (m_dev->is_support_host_image_copy()) {
            inputImage->hostImageCopyToDevice(inputRGBA.data());
            gridImage->hostImageCopyToDevice(gridRGBA.data());
        } else
#endif
        {
            VulkanCommandBuffer cmdstg(device, m_cmdpool->getCommandPool());
            cmdstg.begin();
            inputImage->stagingBufferCopyToImage(cmdstg.get(), inputRGBA.data());
            gridImage->stagingBufferCopyToImage(cmdstg.get(), gridRGBA.data());
            cmdstg.end();
            cmdstg.submit(m_dev->getComputeQueue());
        }

        VulkanCommandBuffer cmd(device, m_cmdpool->getCommandPool());
        cmd.begin();
        inputImage->readBarrier(cmd.get());
        gridImage->readBarrier(cmd.get());
        cmd.end();
        cmd.submit(m_dev->getComputeQueue());

        submit(outWidth, outHeight);

        int realwidth = outWidth * UP_DIV(depth, 4);
        int realheight = outHeight * batch;
        std::vector<T> tmp(realheight * realwidth * 4);
        T *ptr = tmp.data();
#ifdef VK_EXT_host_image_copy
        if (m_dev->is_support_host_image_copy()) {
            outputImage->hostImageCopyToHost(ptr);
        } else
#endif
        {
            VulkanCommandBuffer cmdstg1(device, m_cmdpool->getCommandPool());
            cmdstg1.begin();
            outputImage->stagingBufferCopyToHost(cmdstg1.get(), ptr);
            cmdstg1.end();
            cmdstg1.submit(m_dev->getComputeQueue());
        }
        std::vector<T> output = outputImage->convertRGBAToNCHW<T>(ptr, {batch, depth, outHeight, outWidth});
        return output;
    }


    void set_runtime_device(VkPhysicalDevice phydev, std::shared_ptr<VulkanDevice> dev, VulkanCommandPool *cmdpool) {
        m_phydev = phydev;
        m_dev = dev;
        m_cmdpool = cmdpool;
    };

private:
    InterpolationMode interpolation_mode;
    PaddingMode padding_mode;
    bool align_corners;

    std::shared_ptr<VulkanImage> outputImage;
    std::shared_ptr<VulkanImage> inputImage;
    std::shared_ptr<VulkanImage> gridImage;
    std::shared_ptr<VulkanBuffer> paramBuffer;


    VkPhysicalDevice m_phydev;
    std::shared_ptr<VulkanDevice> m_dev;
    VulkanCommandPool *m_cmdpool;

    void prepare(const std::vector<int>& input_shape, const std::vector<int>& grid_shape);
    void submit(int outWidth, int outHeight);
};

} // namespace ops
} // namespace vkop

#endif // GRIDSAMPLE_HPP