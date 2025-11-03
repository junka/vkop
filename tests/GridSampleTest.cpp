
#include <cstdint>
#include <memory>
#include <random>
#include <chrono>
#include <cmath>
#include <sys/types.h>

#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "setup.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;

namespace {
// 反归一化处理坐标
/*
* x_norm 范围在 [-1.0, 1.0], 那么首先 +1 后范围在 [0, 2]
* 那么乘以 range  后范围在 [0, 2*range - 1]
* 再除以 2 后范围在 [0, range - 0.5]
*/
template <typename T>
T getPosition(T x_norm, int range, bool alignCorners) {
    T a = alignCorners ? 1.0F : 0.0F;
    T b = alignCorners ? 0.0F : 1.0F;
    return static_cast<T>(((1.0F + x_norm) * (range - a) - b) * 0.5F);
}

// padding zero
// 获取坐标对应的值, 如果超过范围, 补零
template <typename T>
T sample(int y, int x, const T *buffer, int height, int width) {
    if (y < 0 || y >= height || x < 0 || x >= width) {
        return 0.0F;
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
T interpolate(T h, T w, const T *buffer, int height, int width) {
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
    T fx1 = 1.0F - fx2;
    // 上边界归一化
    T fy2 = h - w0_h;
    // 下边界归一化
    T fy1 = 1.0F - fy2;

    // 插值. 水平方向
    T i0 = ((i00) * fx1 + (i01) * fx2);
    T i1 = ((i10) * fx1 + (i11) * fx2);

    // 插值, 竖直方向
    return ((i0 * fy1) + (i1 * fy2));
}

template <typename T>
void reference_grid_sample(const T *inputPtr, const T *gridPtr, std::vector<T> &output,
                        int batch, int inHeight, int inWidth, int outHeight, int outWidth, int depth,
                        bool alignCorners) {
    output.resize(batch * outHeight * outWidth * depth);

    T *output_ptr = output.data();

    // 按照 NCHW 的顺序, HW 以output 为目标,
    // grid 的hw 和output是一致的
    // input不参与循环hw, 在每个NC的循环中, 直接整个图以HW尺寸输入,保证grid操作单个channel
    for (auto b = 0; b < batch; ++b) {
        const T *b_input_ptr = inputPtr + b * inHeight * inWidth * depth;
        const T *b_grid_ptr = gridPtr + b * outHeight * outWidth * 2;
        T *b_output_ptr = output_ptr + b * outHeight * outWidth * depth;

        for (auto c = 0; c < depth; ++c) {
            auto *c_input_ptr = b_input_ptr + c * inHeight * inWidth;
            auto *c_output_ptr = b_output_ptr + c * outHeight * outWidth;

            for (auto h = 0; h < outHeight; ++h) {
                auto * h_grid_ptr = b_grid_ptr + h * outWidth * 2;
                auto * h_output_ptr = c_output_ptr + h * outWidth;

                for (auto w = 0; w < outWidth; ++w) {
                    // 首先反归一化得到坐标
                    auto x = getPosition(h_grid_ptr[2 * w + 0], inWidth, alignCorners);
                    auto y = getPosition(h_grid_ptr[2 * w + 1], inHeight, alignCorners);
                    // 然后插值,得到的值输出
                    h_output_ptr[w] = interpolate(y, x, c_input_ptr, inHeight, inWidth);
                }
            }
        }
    }
}

class GridSampleTest: public TestCase {
public:
    std::shared_ptr<Tensor<float>> input;
    std::shared_ptr<Tensor<float>> grid;
    std::vector<float> expectedOutput;

    GridSampleTest(): TestCase("GridSample") {
        initTestdata();
    }
private:
    void initTestdata()
    {
        std::vector<int> t = {
            // 1, 3, 4, 4, 4, 4
            2, 5, 4, 4, 4, 4
        };
        auto batch = t[0];
        auto depth = t[1];
        auto in_height = t[2];
        auto in_width = t[3];
        auto out_height = t[4];
        auto out_width = t[5];

        input = std::make_shared<Tensor<float>>(batch, depth, in_height, in_width);
        grid = std::make_shared<Tensor<float>>(batch, out_height, out_width, 2);

        auto *input_ptr = input->data();
        auto *grid_ptr = grid->data();

        std::random_device rd{};
        std::mt19937 gen{rd()};
        gen.seed(1024);
        std::normal_distribution<> input_dist{0.0F, 1.0F};
        std::normal_distribution<> grid_dist{0.0F, 3.0F / out_width};
        for (int i = 0; i < input->num_elements(); i++) {
            input_ptr[i] = input_dist(gen);
        }
        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < out_height; h++) {
                for (int w = 0; w < out_width; w++) {
                    float offset_h = grid_dist(gen);
                    float offset_w = grid_dist(gen);
                    grid_ptr[b * out_height * out_width * 2 + h * out_width * 2 + w * 2 + 0] = (2.0F * w / (out_width - 1) - 1.0F + offset_w);
                    grid_ptr[b * out_height * out_width * 2 + h * out_width * 2 + w * 2 + 1] = (2.0F * h / (out_height - 1) - 1.0F + offset_h);
                }
            }
        }
        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        reference_grid_sample<float>(input_ptr, grid_ptr, expectedOutput, batch, in_height, in_width, out_height, out_width, depth, false);
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        LOG_INFO("reference grid sample time: %f s", duration.count());
    }
};

}


int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", false);
    GridSampleTest gst;
    if (!gst.run_test({gst.input, gst.grid}, gst.expectedOutput)) {
        return -1;
    }
    return EXIT_SUCCESS;
}
