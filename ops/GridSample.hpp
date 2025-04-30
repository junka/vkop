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

        const T* inputPtr = input.data();
        const T* gridPtr = grid.data();
        inputImage->convertNCHWToRGBA(inputPtr, input_shape);
        gridImage->convertNCHWToRGBA(gridPtr, grid_shape);

        VkDevice device = m_dev->getLogicalDevice();
        VulkanCommandBuffer cmd(device, m_cmdpool->getCommandPool());
        cmd.begin();
        inputImage->transitionImageLayout(cmd.get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT);
        gridImage->transitionImageLayout(cmd.get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT);
        cmd.end();
        cmd.submit(m_dev->getComputeQueue());

        submit(outWidth, outHeight);

        // outputImage->transitionImageLayout(cmd.get(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT);

        std::vector<T> output = outputImage->convertRGBAToNCHW<T>({batch, depth, outHeight, outWidth});
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