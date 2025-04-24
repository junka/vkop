#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <chrono>
#include <cmath>
#include <sys/types.h>
#include "src/VulkanBuffer.hpp"
#include "src/VulkanDevice.hpp"
#include "src/VulkanImage.hpp"
#include "src/VulkanInstance.hpp"
#include "src/VulkanPipeline.hpp"
#include "src/VulkanCommandPool.hpp"
#include "src/VulkanCommandBuffer.hpp"

#include "grid_sample.h"
#include "src/VulkanQueryPool.hpp"

using namespace vkop;

typedef int ivec4[4];
typedef int ivec2[2];
struct GpuGridSampleParam {
    ivec4 outImgSize;
    ivec2 inShape;
    ivec2 outShape;
};


// 反归一化处理坐标
/*
* x_norm 范围在 [-1.0, 1.0], 那么首先 +1 后范围在 [0, 2]
* 那么乘以 range  后范围在 [0, 2*range - 1]
* 再除以 2 后范围在 [0, range - 0.5]
*/
static float getPosition(float x_norm, int range, bool alignCorners) {
    (void)alignCorners;
    // float a = alignCorners ? 1.0f : 0.0f;
    // float b = alignCorners ? 0.0f : 1.0f;
    return (((1.0f + x_norm) * (range) - 1.0f) * 0.5f);
}

// padding zero
// 获取坐标对应的值, 如果超过范围, 补零
static float sample(int y, int x, const float *buffer, int height, int width) {
    if (y < 0 || y >= height || x < 0 || x >= width) {
        return 0.0f;
    }

    return buffer[y * width + x];
}

// 双线性插值算法,
/*
* 首先计算出上下左右四个点的坐标, 对浮点取整,floor下值,ceil上值
* 然后sample取值, 计算出四个点的值
* 计算出权重,
*/
static float interpolate(float h, float w, const float *buffer, int height, int width) {
    // mode == GridSampleMode_BILINEAR
    int w0_h = ::floor(h);
    int w0_w = ::floor(w);
    int w1_h = ::ceil(h);
    int w1_w = ::ceil(w);

    // 左下角
    float i00 = sample(w0_h, w0_w, buffer, height, width);
    // 右下角
    float i01 = sample(w0_h, w1_w, buffer, height, width);
    // 左上角
    float i10 = sample(w1_h, w0_w, buffer, height, width);
    // 右上角
    float i11 = sample(w1_h, w1_w, buffer, height, width);

    // 权重, 左边界归一化
    float fx2 = w - w0_w;
    // 右边界归一化
    float fx1 = 1.0f - fx2;
    // 上边界归一化
    float fy2 = h - w0_h;
    // 下边界归一化
    float fy1 = 1.0f - fy2;

    // 插值. 水平方向
    float i0 = ((i00) * fx1 + (i01) * fx2);
    float i1 = ((i10) * fx1 + (i11) * fx2);

    // 插值, 竖直方向
    return ((i0 * fy1) + (i1 * fy2));
}

static void reference_grid_sample(const float *inputPtr, const float *gridPtr, std::vector<float> &output,
                                  int batch, int inHeight, int inWidth, int outHeight, int outWidth, int depth,
                                  bool alignCorners) {
    output.resize(batch * outHeight * outWidth * depth);

    float *outputPtr = output.data();

    // 按照 NCHW 的顺序, HW 以output 为目标,
    // grid 的hw 和output是一致的
    // input不参与循环hw, 在每个NC的循环中, 直接整个图以HW尺寸输入,保证grid操作单个channel
    for (auto b = 0; b < batch; ++b) {
        const float *_inputPtr = inputPtr + b * inHeight * inWidth * depth;
        const float *_gridPtr = gridPtr + b * outHeight * outWidth * 2;
        float *_outputPtr = outputPtr + b * outHeight * outWidth * depth;

        for (auto c = 0; c < depth; ++c) {
            auto __inputPtr = _inputPtr + c * inHeight * inWidth;
            auto __outputPtr = _outputPtr + c * outHeight * outWidth;

            for (auto h = 0; h < outHeight; ++h) {
                auto __gridPtr = _gridPtr + h * outWidth * 2;
                auto ___outputPtr = __outputPtr + h * outWidth;

                for (auto w = 0; w < outWidth; ++w) {
                    // 首先反归一化得到坐标
                    auto x = getPosition(__gridPtr[2 * w + 0], inWidth, alignCorners);
                    auto y = getPosition(__gridPtr[2 * w + 1], inHeight, alignCorners);
                    // 然后插值,得到的值输出
                    ___outputPtr[w] = interpolate(y, x, __inputPtr, inHeight, inWidth);
                }
            }
        }
    }
}

struct test_params {

    int batch;
    int depth;
    int inHeight;
    int inWidth ;
    int outHeight;
    int outWidth;

    std::vector<float> originInputData;
    std::vector<float> originGridData;
    std::vector<float> expectedOutput;

    void initTestdata(void)
    {
        std::vector<int> t = {
            // 1, 3, 5, 10, 5, 10
            1, 4, 6, 16, 6, 17
        };
        batch = t[0];
        depth = t[1];
        inHeight = t[2];
        inWidth = t[3];
        outHeight = t[4];
        outWidth = t[5];

        auto inputSize = batch * depth * inHeight * inWidth;
        auto gridSize = batch * outHeight * outWidth * 2;
        auto outputSize = batch * outHeight * outWidth * depth;

        originInputData.resize(inputSize);
        originGridData.resize(gridSize); //for 2d, last dim is x,y
        expectedOutput.resize(outputSize);

        float *inputPtr = originInputData.data();
        float *gridPtr = originGridData.data();

        std::random_device rd{};
        std::mt19937 gen{rd()};
        gen.seed(1024);
        std::normal_distribution<> inputDist{0.0f, 1.0};
        std::normal_distribution<> gridDist{0.0f, 3.0f / outWidth};
        for (int i = 0; i < inputSize; i++) {
            inputPtr[i] = inputDist(gen);
        }
        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < outHeight; h++) {
                for (int w = 0; w < outWidth; w++) {
                    float offsetH = gridDist(gen);
                    float offsetW = gridDist(gen);
                    gridPtr[b * outHeight * outWidth * 2 + h * outWidth * 2 + w * 2 + 0] = (2.0f * w / (outWidth-1) - 1.0f + offsetW);
                    gridPtr[b * outHeight * outWidth * 2 + h * outWidth * 2 + w * 2 + 1] = (2.0f * h / (outHeight-1) - 1.0f + offsetH);
                }
            }
        }
        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        reference_grid_sample(inputPtr, gridPtr, expectedOutput, batch, inHeight, inWidth, outHeight, outWidth, depth, false);
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        std::cout << "reference grid sample time: " << duration.count() << " s" << std::endl;
    }
};

#define UP_DIV(x, y) (((x) + (y) - 1) / (y))
void setImageData(VkInstance ins, float *data, std::vector<int> nchw, std::shared_ptr<VulkanImage> image)
{
    int batch = nchw[0];//1
    int depth = nchw[1];//6
    int height = nchw[2];//17
    int width = nchw[3];//2

    int stride_w = 1;
    int stride_h = width;
    int stride_c = width * height;
    int stride_n = width * height * depth;
    int realdepth = UP_DIV(depth, 4);
    int realwidth = width * UP_DIV(depth, 4);

    // since format is VK_FORMAT_R32G32B32A32_SFLOAT
    float *ptr = (float *)malloc(batch * height * realdepth * realwidth * 4 * sizeof(float));

    uint32_t rowPitch = realwidth * 4 * sizeof(float);
    float *dst = reinterpret_cast<float *>(ptr);
    for (int b = 0; b < batch; b++) {
        float* batchstart = reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(ptr) + b * height * rowPitch);
        for (int c = 0; c < realdepth; c++) {
            dst = reinterpret_cast<float *>(batchstart) + c * 4 * width;
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int offset = b * stride_n + 4 * c * stride_c + h * stride_h + w * stride_w;

                    float r = data[offset];
                    float g = (4 * c + 1 < depth) ? data[stride_c + offset] : 0.0f;
                    float b = (4 * c + 2 < depth) ? data[2 * stride_c + offset] : 0.0f;
                    float a = (4 * c + 3 < depth) ? data[3 * stride_c + offset] : 0.0f;

                    // Write RGBA values to the Vulkan image memory
                    dst[w * 4 + 0] = r;
                    dst[w * 4 + 1] = g;
                    dst[w * 4 + 2] = b;
                    dst[w * 4 + 3] = a;
                }
                // Move to the next row in the Vulkan image memory
                dst = reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(dst) + rowPitch);
            }
        }
    }
    image->hostImageCopyFrom(ins, ptr);
    free(ptr);
}


bool verifyImageData(VkInstance ins, float *data, std::vector<int> nchw, std::shared_ptr<VulkanImage> image) {
    int batch = nchw[0];
    int depth = nchw[1];
    int height = nchw[2];
    int width = nchw[3];

    int stride_w = 1;
    int stride_h = width;
    int stride_c = width * height;
    int stride_n = width * height * depth;

    bool ret = true;
    int realdepth = UP_DIV(depth, 4);
    int realwidth = width * UP_DIV(depth, 4);
    
    float *ptr = (float *)malloc(batch * height * realdepth * realwidth * 4 * sizeof(float));
    image->hostImageCopyTo(ins, ptr);

    uint32_t rowPitch = realwidth * 4 * sizeof(float);
    // since format is VK_FORMAT_R32G32B32A32_SFLOAT
    float *dst = reinterpret_cast<float *>(ptr);
    for (int b = 0; b < batch; b++) {
        float* batchstart = reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(ptr) + b * height * rowPitch);
        for (int c = 0; c < realdepth; c++) {
            dst = reinterpret_cast<float *>(batchstart) + c * width * 4;
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int offset = b * stride_n + 4 * c * stride_c + h * stride_h + w * stride_w;
                    float r = data[offset];
                    float g = (4 * c + 1 < depth) ? data[stride_c + offset] : 0.0f;
                    float b = (4 * c + 2 < depth) ? data[stride_c * 2 + offset] : 0.0f;
                    float a = (4 * c + 3 < depth) ? data[stride_c * 3 + offset] : 0.0f;
                    if (std::fabs(dst[w * 4 + 0] - r) > 0.01||
                        std::fabs(dst[w * 4 + 1] - g) > 0.01||
                        std::fabs(dst[w * 4 + 2] - b) > 0.01||
                        std::fabs(dst[w * 4 + 3] - a) > 0.01) {
                        std::cout << "data mismatch at " << offset << ", result " << dst[w * 4 + 0] << " expect " << r << std::endl;
                        std::cout << "data mismatch at " << offset + stride_c << ", result " << dst[w * 4 + 1] << " expect " << g << std::endl;
                        // std::cout << "data mismatch at " << i + width * height * 2 << ", result " << dst[w * 4 + 2] << " expect " << b << std::endl;
                        // std::cout << "data mismatch at " << i << ", result " << dst[w * 4 + 3] << " expect " << a << std::endl;
                        ret = false;
                        // break;
                    } else {
                        // std::cout << "data match at " << i << ", result " << dst[w * 4 + 0] << " expect " << r << std::endl;
                    }
                }
                dst = reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(dst) + rowPitch);
            }
        }
    }
    free(ptr);
    return ret;
}

int main() {
    struct test_params test_params;
    test_params.initTestdata();
    try {
        VulkanInstance instance;

        auto phydevs = instance.getPhysicalDevices();
        for (auto pdev : phydevs) {
            auto dev = std::make_shared<VulkanDevice>(pdev);
            if (dev->getDeviceName().find("llvmpipe")!= std::string::npos) {
                continue;
            }
            VkDevice device = dev->getLogicalDevice();

            VulkanCommandPool cmdpool(device, dev->getComputeQueueFamilyIndex());

            auto img1 = std::make_shared<VulkanImage>(pdev, dev->getComputeQueueFamilyIndex(), device, VkExtent3D {
                (uint32_t)test_params.outWidth * UP_DIV(test_params.depth, 4),
                (uint32_t)test_params.outHeight,
                1
            }, VK_FORMAT_R32G32B32A32_SFLOAT,
                VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                VK_IMAGE_TYPE_2D);

            auto img2 = std::make_shared<VulkanImage>(pdev, dev->getComputeQueueFamilyIndex(), device, VkExtent3D{
                (uint32_t)test_params.inWidth * UP_DIV(test_params.depth, 4),
                (uint32_t)test_params.inHeight,
                1
            }, VK_FORMAT_R32G32B32A32_SFLOAT,
                VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                VK_IMAGE_TYPE_2D);

            auto img3 = std::make_shared<VulkanImage>(pdev, dev->getComputeQueueFamilyIndex(), device, VkExtent3D{
                2 * UP_DIV((uint32_t)test_params.outHeight, 4),
                (uint32_t)test_params.outWidth,
                1
            }, VK_FORMAT_R32G32B32A32_SFLOAT,
                VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                VK_IMAGE_TYPE_2D);//VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL

            auto b = std::make_shared<VulkanBuffer>(pdev, dev->getComputeQueueFamilyIndex(), device, sizeof(GpuGridSampleParam),
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

            // cmd.begin();
            // img1->transitionImageLayout(cmd.get(), VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT);
            // img2->transitionImageLayout(cmd.get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT);
            // img3->transitionImageLayout(cmd.get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT);
            // cmd.end();
            if (dev->checkHostImageCopyDstLayoutSupport(VK_IMAGE_LAYOUT_GENERAL)) {
                img1->hostImaggeTransition(instance.getInstance(), VK_IMAGE_LAYOUT_GENERAL);
            } else {
                img1->hostImaggeTransition(instance.getInstance(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            }
            img2->hostImaggeTransition(instance.getInstance(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            img3->hostImaggeTransition(instance.getInstance(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            
            // Ensure shared pointers are retained before cmd.submit
            std::vector<std::shared_ptr<VulkanResource>> objs = {
                img1,
                img2,
                img3,
                b
            };

            float *inputPtr = test_params.originInputData.data();
            float *gridPtr = test_params.originGridData.data();
            setImageData(instance.getInstance(), inputPtr, {test_params.batch, test_params.depth, test_params.inHeight, test_params.inWidth}, img2);
            setImageData(instance.getInstance(), gridPtr, {test_params.batch, test_params.outHeight, test_params.outWidth, 2}, img3);

            // can not do transition with host image copy since the limit to dst layout?
            VulkanCommandBuffer cmd(device, cmdpool.getCommandPool());
            cmd.begin();
            img2->transitionImageLayout(cmd.get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT);
            img3->transitionImageLayout(cmd.get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT);
            cmd.end();
            cmd.submit(dev->getComputeQueue());

            struct GpuGridSampleParam *para = reinterpret_cast<GpuGridSampleParam*>(b->getMappedMemory());
            para->outImgSize[0] = test_params.outWidth;
            para->outImgSize[1] = test_params.outHeight;
            para->outImgSize[2] = 1;
            para->outImgSize[3] = 0;
            para->inShape[0] = test_params.inWidth;
            para->inShape[1] = test_params.inHeight;
            para->outShape[0] = test_params.outWidth;
            para->outShape[1] = test_params.outHeight;
            b->unmapMemory();

            std::vector<VkDescriptorType> types = {
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
            };
            VulkanPipeline pipeline(device, types, objs, reinterpret_cast<const uint32_t *>(grid_sample_spv), grid_sample_spv_len);

            VulkanCommandBuffer cmd2(device, cmdpool.getCommandPool());
            VulkanQueryPool queryPool(device, 2, VK_QUERY_TYPE_TIMESTAMP);
            cmd2.begin();
            cmd2.bind(pipeline);
            queryPool.begin(cmd2.get());
            cmd2.dispatch(test_params.outWidth, test_params.outHeight, 1);
            queryPool.end(cmd2.get());
            cmd2.end();
            cmd2.submit(dev->getComputeQueue());
            auto r = queryPool.getResults();
            double ts = double(r[1]-r[0])* double(1e-9) * dev->getTimestampPeriod();
            std::cout << "Time: " << ts  << " s" << std::endl;

            bool ret = verifyImageData(instance.getInstance(), test_params.expectedOutput.data(), {test_params.batch, test_params.depth, test_params.outHeight, test_params.outWidth}, img1);
            // bool ret = img1.verifyImageData(test_params.expectedOutput.data(), {test_params.batch, test_params.depth, test_params.outHeight, test_params.outWidth});
            if (ret) {
                std::cout << "Test Passed" << std::endl;
            } else {
                std::cout << "Test Failed" << std::endl;
            }

        }
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}