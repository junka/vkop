
#include <cstdint>
#include <memory>
#include <random>
#include <chrono>
#include <cmath>
#include <sys/types.h>
#include "VulkanDevice.hpp"
#include "VulkanInstance.hpp"
#include "Renderdoc.hpp"

#include "logger.hpp"

#include "GridSample.hpp"

using namespace vkop;


// 反归一化处理坐标
/*
* x_norm 范围在 [-1.0, 1.0], 那么首先 +1 后范围在 [0, 2]
* 那么乘以 range  后范围在 [0, 2*range - 1]
* 再除以 2 后范围在 [0, range - 0.5]
*/
template <typename T>
static T getPosition(T x_norm, int range, bool alignCorners) {
    T a = alignCorners ? 1.0f : 0.0f;
    T b = alignCorners ? 0.0f : 1.0f;
    return (T)(((1.0f + x_norm) * (range - a) - b) * 0.5f);
}

// padding zero
// 获取坐标对应的值, 如果超过范围, 补零
template <typename T>
static T sample(int y, int x, const T *buffer, int height, int width) {
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
template <typename T>
static T interpolate(T h, T w, const T *buffer, int height, int width) {
    // mode == GridSampleMode_BILINEAR
    int w0_h = ::floor(h);
    int w0_w = ::floor(w);
    int w1_h = ::ceil(h);
    int w1_w = ::ceil(w);

    // 左下角
    T i00 = sample(w0_h, w0_w, buffer, height, width);
    // 右下角
    T i01 = sample(w0_h, w1_w, buffer, height, width);
    // 左上角
    T i10 = sample(w1_h, w0_w, buffer, height, width);
    // 右上角
    T i11 = sample(w1_h, w1_w, buffer, height, width);

    // 权重, 左边界归一化
    T fx2 = w - w0_w;
    // 右边界归一化
    T fx1 = 1.0f - fx2;
    // 上边界归一化
    T fy2 = h - w0_h;
    // 下边界归一化
    T fy1 = 1.0f - fy2;

    // 插值. 水平方向
    T i0 = ((i00) * fx1 + (i01) * fx2);
    T i1 = ((i10) * fx1 + (i11) * fx2);

    // 插值, 竖直方向
    return ((i0 * fy1) + (i1 * fy2));
}

template <typename T>
static void reference_grid_sample(const T *inputPtr, const T *gridPtr, std::vector<T> &output,
                        int batch, int inHeight, int inWidth, int outHeight, int outWidth, int depth,
                        bool alignCorners) {
    output.resize(batch * outHeight * outWidth * depth);

    T *outputPtr = output.data();

    // 按照 NCHW 的顺序, HW 以output 为目标,
    // grid 的hw 和output是一致的
    // input不参与循环hw, 在每个NC的循环中, 直接整个图以HW尺寸输入,保证grid操作单个channel
    for (auto b = 0; b < batch; ++b) {
        const T *_inputPtr = inputPtr + b * inHeight * inWidth * depth;
        const T *_gridPtr = gridPtr + b * outHeight * outWidth * 2;
        T *_outputPtr = outputPtr + b * outHeight * outWidth * depth;

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

class GridSampleTest {
public:
    int batch;
    int depth;
    int inHeight;
    int inWidth ;
    int outHeight;
    int outWidth;

    std::vector<float> originInputData;
    std::vector<float> originGridData;
    std::vector<float> expectedOutput;

    GridSampleTest() {
        initTestdata();
    }
private:
    void initTestdata(void)
    {
        std::vector<int> t = {
            1, 3, 4, 4, 4, 4
            // 2, 5, 4, 4, 4, 4
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
        reference_grid_sample<float>(inputPtr, gridPtr, expectedOutput, batch, inHeight, inWidth, outHeight, outWidth, depth, false);
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        LOG_INFO("reference grid sample time: %f s", duration.count());
    }
};




int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", true);
    GridSampleTest tp;
    // auto rdoc = Renderdoc(VulkanInstance::getVulkanInstance().getInstance());
    try {
        auto phydevs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
        for (auto pdev : phydevs) {
            auto dev = std::make_shared<VulkanDevice>(pdev);
            if (dev->getDeviceName().find("llvmpipe")!= std::string::npos) {
                continue;
            }

            VkDevice device = dev->getLogicalDevice();
            VulkanCommandPool cmdpool(device, dev->getComputeQueueFamilyIndex());

            ops::GridSample gs;
            gs.set_runtime_device(pdev, dev, &cmdpool);
            // Ensure shared pointers are retained before cmd.submit
            auto ret = gs.apply<float>(tp.originInputData, tp.originGridData,
                {tp.batch, tp.depth, tp.inHeight, tp.inWidth},
                {tp.batch, tp.outHeight, tp.outWidth, 2});
            for (uint32_t i = 0; i < ret.size(); i++) {
                if (std::fabs(ret.data()[i] - tp.expectedOutput.data()[i]) > 0.01) {
                    LOG_ERROR("Test Fail at (%d): %f, %f", i, ret.data()[i], tp.expectedOutput.data()[i]);
                    return -1;
                }
            }
            
            LOG_INFO("Test Passed");

        }
    } catch (const std::exception &e) {
        LOG_ERROR("%s\n", e.what());
        return EXIT_FAILURE;
    }
    // rdoc.EndRenderDocCapture(VulkanInstance::getVulkanInstance().getInstance());

    return EXIT_SUCCESS;
}