#include <memory>
#include <random>
#include <chrono>
#include <cmath>
#include <utility>

#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "setup.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;

namespace {
/*
* x_norm ranges [-1.0, 1.0], make it +1 to [0, 2]
* then it rangs [0, 2*range - 1] after multiply range 
* div 2 to [0, range - 0.5]
*/
template <typename T>
T getPosition(T x_norm, int range, bool alignCorners) {
    T a = alignCorners ? 1.0F : 0.0F;
    T b = alignCorners ? 0.0F : 1.0F;
    return static_cast<T>(((1.0F + x_norm) * (range - a) - b) * 0.5F);
}

// padding zero
template <typename T>
T sample(int y, int x, const std::shared_ptr<Tensor<T>> &input, int offset, int height, int width) {
    if (y < 0 || y >= height || x < 0 || x >= width) {
        return 0.0F;
    }

    return (*input)[offset + (y * width) + x];
}

template <typename T>
T interpolate(T h, T w, const std::shared_ptr<Tensor<T>> &buffer, int offset, int height, int width) {
    // mode == GridSampleMode_BILINEAR
    int w0_h = ::floor(h);
    int w0_w = ::floor(w);
    int w1_h = ::ceil(h);
    int w1_w = ::ceil(w);

    // left down
    T i00 = sample(w0_h, w0_w, buffer, offset, height, width);
    // right down
    T i01 = sample(w0_h, w1_w, buffer, offset, height, width);
    // left top
    T i10 = sample(w1_h, w0_w, buffer, offset, height, width);
    // right top
    T i11 = sample(w1_h, w1_w, buffer, offset, height, width);

    T fx2 = w - w0_w;
    T fx1 = 1.0F - fx2;
    T fy2 = h - w0_h;
    T fy1 = 1.0F - fy2;

    T i0 = (((i00) * fx1) + ((i01) * fx2));
    T i1 = (((i10) * fx1) + ((i11) * fx2));

    return ((i0 * fy1) + (i1 * fy2));
}

template <typename T>
void reference_grid_sample(const std::shared_ptr<Tensor<T>> &input, const std::shared_ptr<Tensor<T>> &grid, std::shared_ptr<Tensor<T>> &output,
                        int batch, int inHeight, int inWidth, int outHeight, int outWidth, int depth,
                        bool alignCorners) {
    for (auto b = 0; b < batch; ++b) {
        int b_input_offset = b * inHeight * inWidth * depth;
        int b_grid_offset = b * outHeight * outWidth * 2;
        int b_output_offset = b * outHeight * outWidth * depth;

        for (auto c = 0; c < depth; ++c) {
            auto c_input_offset = b_input_offset + (c * inHeight * inWidth);
            auto c_output_offset = b_output_offset + (c * outHeight * outWidth);

            for (auto h = 0; h < outHeight; ++h) {
                auto h_grid_offset = b_grid_offset + (h * outWidth * 2);
                auto h_output_offset = c_output_offset + (h * outWidth);

                for (auto w = 0; w < outWidth; ++w) {
                    auto x = getPosition((*grid)[h_grid_offset + (2 * w) + 0], inWidth, alignCorners);
                    auto y = getPosition((*grid)[h_grid_offset + (2 * w) + 1], inHeight, alignCorners);
                    (*output)[h_output_offset+w] = interpolate(y, x, input, c_input_offset, inHeight, inWidth);
                }
            }
        }
    }
}

class GridSampleTest: public TestCase {
public:
    std::shared_ptr<Tensor<float>> input;
    std::shared_ptr<Tensor<float>> grid;
    std::shared_ptr<Tensor<float>> output;

    GridSampleTest(std::vector<int> input_shape, std::vector<int> output_shape): 
        TestCase("GridSample"), input_shape_(std::move(input_shape)), output_shape_(std::move(output_shape)) {
        initTestdata();
    }
private:
    std::vector<int> input_shape_;
    std::vector<int> output_shape_;
    void initTestdata()
    {
        int batch = input_shape_[0];
        int depth = input_shape_[1];
        int in_height = input_shape_[2];
        int in_width = input_shape_[3];
        int out_height = output_shape_[2];
        int out_width = output_shape_[3];
        input = std::make_shared<Tensor<float>>(batch, depth, in_height, in_width);
        grid = std::make_shared<Tensor<float>>(batch, out_height, out_width, 2);
        output = std::make_shared<Tensor<float>>(batch, depth, out_height, out_width);
        input->reserveOnCPU();
        grid->reserveOnCPU();
        output->reserveOnCPU();

        std::random_device rd{};
        std::mt19937 gen{rd()};
        gen.seed(1024);
        std::normal_distribution<> input_dist{0.0F, 1.0F};
        std::normal_distribution<> grid_dist{0.0F, 3.0F / out_width};
        for (int i = 0; i < input->num_elements(); i++) {
            (*input)[i] = input_dist(gen);
        }
        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < out_height; h++) {
                for (int w = 0; w < out_width; w++) {
                    float offset_h = grid_dist(gen);
                    float offset_w = grid_dist(gen);
                    (*grid)[(b * out_height * out_width * 2) + (h * out_width * 2) + (w * 2) + 0] = (2.0F * w / (out_width - 1) - 1.0F + offset_w);
                    (*grid)[(b * out_height * out_width * 2) + (h * out_width * 2) + (w * 2) + 1] = (2.0F * h / (out_height - 1) - 1.0F + offset_h);
                }
            }
        }
        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        reference_grid_sample<float>(input, grid, output, batch, in_height, in_width, out_height, out_width, depth, false);
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        LOG_INFO("reference grid sample time: %f s", duration.count());
    }
};

}

TEST(GridSampleTest, GridSampleComprehensiveTest) {
    
    std::vector<std::tuple<std::vector<int>, std::vector<int>>> test_cases = {
        {{1, 3, 4, 4}, {1, 3, 4, 4}},
        {{2, 5, 4, 4}, {2, 5, 4, 4}}
    };
    for (const auto& t : test_cases) { 
        auto [input_shape, output_shape] = t;
        GridSampleTest gst(input_shape, output_shape);
        EXPECT_TRUE(gst.run_test<float>({gst.input, gst.grid}, {gst.output}));
    }
}
