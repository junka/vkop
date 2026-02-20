#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "setup.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;

namespace {
#ifdef USE_CPP_REF
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
#endif

template <typename T>
class GridSampleTest: public TestCase<T> {
public:
    std::shared_ptr<Tensor<T>> input;
    std::shared_ptr<Tensor<T>> grid;
    std::shared_ptr<Tensor<T>> output;

    GridSampleTest(std::vector<int> input_shape, std::vector<int> output_shape): 
        TestCase<T>("GridSample"), input_shape_(std::move(input_shape)), output_shape_(std::move(output_shape)) {
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
        input = std::make_shared<Tensor<T>>(batch, depth, in_height, in_width);
        grid = std::make_shared<Tensor<T>>(batch, out_height, out_width, 2);
        output = std::make_shared<Tensor<T>>(batch, depth, out_height, out_width);
        torch::manual_seed(42);
        auto torch_input = torch::randn({batch, depth, in_height, in_width}, this->getTorchConf());
        auto torch_grid = torch::randn({batch, out_height, out_width, 2}, this->getTorchConf());
        auto torch_output = torch::grid_sampler_2d(torch_input, torch_grid, 0, 0, false);
        this->fillTensorFromTorch(input, torch_input);
        this->fillTensorFromTorch(grid, torch_grid);
        this->fillTensorFromTorch(output, torch_output);
#if 0
        reference_grid_sample<T>(input, grid, output, batch, in_height, in_width, out_height, out_width, depth, false);
#endif
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

        LOG_INFO("Testing gridsample fp32");
        GridSampleTest<float> gst(input_shape, output_shape);
        EXPECT_TRUE(gst.run_test({gst.input, gst.grid}, {gst.output}));

        LOG_INFO("Testing gridsample fp16");
        GridSampleTest<uint16_t> gst1(input_shape, output_shape);
        EXPECT_TRUE(gst1.run_test({gst1.input, gst1.grid}, {gst1.output}));
    }
}
