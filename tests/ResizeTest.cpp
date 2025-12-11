#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <cassert>
#include <chrono>

#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "setup.hpp"
#include "ops/Resize.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;
using vkop::ops::Resize;

#define USE_CPP_REFER 1

namespace {
#if USE_CPP_REFER

template<typename T>
inline T clamp(T v, T low, T high) {
    return std::max(low, std::min(v, high));
}

// Catmull-Rom cubic interpolation kernel
float cubic_kernel(float x, float a = -0.75F) {
    x = std::abs(x);
    if (x <= 1.0F) {
        return (a + 2.0F) * x * x * x - (a + 3.0F) * x * x + 1.0F;
    }
    if (x < 2.0F) {
        return a * x * x * x - 5.0F * a * x * x + 8.0F * a * x - 4.0F * a;
    }
    return 0.0F;
}

template<typename T>
// CPU resize for 4D NCHW float tensor
std::vector<T> reference_resize(
    const std::shared_ptr<Tensor<T>>& input,
    int batch, int channels, int in_h, int in_w,
    int out_h, int out_w,
    const std::string& mode = "bilinear",
    bool align_corners = false,
    float cubic_coeff_a = -0.75F
) {
    std::vector<T> output(batch * channels * out_h * out_w);

    // Precompute projection maps (from output coords to input coords)
    auto project = [&](int out_idx, int in_size, int out_size) -> float {
        if (out_size == 1) {
            return align_corners ? 0.0f : (static_cast<float>(in_size - 1) / 2.0f);
        }
        if (align_corners) {
            return static_cast<float>(out_idx) * (in_size - 1) / (out_size - 1);
        }  
        // PyTorch default: half-pixel centers
        return (static_cast<float>(out_idx) + 0.5f) * in_size / out_size - 0.5f;
       
    };
    auto get_pixel = [&](int n, int c, int y, int x) -> float {
        // if (y < 0 || y >= in_h || x < 0 || x >= in_w) return 0.0f;
        y = clamp(y, 0, in_h - 1);
        x = clamp(x, 0, in_w - 1);
        return (*input)[((n * channels + c) * in_h + y) * in_w + x];
    };
    for (int n = 0; n < batch; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int oy = 0; oy < out_h; ++oy) {
                for (int ox = 0; ox < out_w; ++ox) {
                    float fx = project(ox, in_w, out_w);
                    float fy = project(oy, in_h, out_h);

                    float val = 0.0F;

                    if (mode == "nearest") {
                        // Modern PyTorch (>=1.6) default: left-top of each block for integer scale
                        int ix = static_cast<int>(static_cast<float>(ox) * static_cast<float>(in_w) / static_cast<float>(out_w));
                        int iy = static_cast<int>(static_cast<float>(oy) * static_cast<float>(in_h) / static_cast<float>(out_h));
                        ix = clamp(ix, 0, in_w - 1);
                        iy = clamp(iy, 0, in_h - 1);
                        val = get_pixel(n, c, iy, ix);
                    }
                    else if (mode == "bilinear") {
                        fx = clamp(fx, 0.0F, static_cast<float>(in_w - 1));
                        fy = clamp(fy, 0.0F, static_cast<float>(in_h - 1));

                        int x0 = static_cast<int>(std::floor(fx));
                        int y0 = static_cast<int>(std::floor(fy));
                        int x1 = std::min(x0 + 1, in_w - 1);
                        int y1 = std::min(y0 + 1, in_h - 1);

                        float dx = fx - x0;
                        float dy = fy - y0;

                        float top = get_pixel(n, c, y0, x0) * (1 - dx) + get_pixel(n, c, y0, x1) * dx;
                        float bottom = get_pixel(n, c, y1, x0) * (1 - dx) + get_pixel(n, c, y1, x1) * dx;
                        val = top * (1 - dy) + bottom * dy;
                    } else if (mode == "bicubic") {
                        // Bicubic: sample 4x4 neighborhood
                        int x = static_cast<int>(std::floor(fx));
                        int y = static_cast<int>(std::floor(fy));
                        float dx = fx - x;
                        float dy = fy - y;

                        val = 0.0F;
                        for (int i = -1; i <= 2; ++i) {
                            for (int j = -1; j <= 2; ++j) {
                                float coeff = cubic_kernel(dy - i, cubic_coeff_a) * cubic_kernel(dx - j, cubic_coeff_a);
                                val += get_pixel(n, c, y + i, x + j) * coeff;
                            }
                        }
                    } else {
                        throw std::invalid_argument("Unsupported mode: " + mode);
                    }

                    output[((n * channels + c) * out_h + oy) * out_w + ox] = val;
                }
            }
        }
    }

    return output;
}
#endif
} // namespace

template<typename T>
class ResizeTest: public TestCase {
public:
    std::vector<int> input_shape_ = {1, 3, 4, 8};
    std::vector<int> resize_ = {1, 3, 6, 4};
    bool align_corners_ = false;
    std::string mode_ = "bilinear";
    bool antialias_ = false;
    float cubic_coeff_a_ = -0.75; // pytorch fixed value

    std::string buildSizeString(const std::vector<int>& sizes) {
        std::string result = "[";
        int rank = sizes.size();
        for (size_t i = rank-2; i < sizes.size(); ++i) {
            result += std::to_string(sizes[i]);
            if (i < sizes.size() - 1) {
                result += ", ";
            }
        }
        result += "]";
        return result;
    }
    std::unordered_map<std::string, std::string> attributes = {
        {"size", buildSizeString(resize_)},
        // 用于上采样的算法: 'nearest'| 'linear'| 'bilinear'| 'bicubic'| 'trilinear'| 'area'| 'nearest-exact'
        // 对于2D 操作，仅考虑 nearest， bilinear, bicubic, area, nearest-exact
        {"mode", mode_},
        // align_corners works only for linear, bilinear, bicubic, trilinear
        // {"coordinate_transformation_mode", "half_pixel"},
        {"align_corners", align_corners_ ? "True": "False"},
        // antialias works for 'bilinear', 'bicubic'
        {"antialias", antialias_ ? "True" : "False"},
        // {"cubic_coeff_a", std::to_string(cubic_coeff_a_)},
    };

    std::shared_ptr<Tensor<T>> input_data_;
    std::shared_ptr<Tensor<T>> output_data_;

    ResizeTest(): TestCase("Resize") {
        initTestData();
    }


private:
    void initTestData() {
        std::vector<std::vector<int>> shapes;
        shapes.emplace_back(input_shape_);

        std::tuple<std::vector<std::vector<float>>, std::vector<int>> k = TestCase::execute_torch_operator("interpolate", shapes, attributes);
        std::vector<std::vector<float>> torch_tensors = std::get<0>(k);
        auto torch_output = torch_tensors[0];
        auto torch_input = torch_tensors[1];
        std::vector<int> output_shape = std::get<1>(k);

        printf("torch output size: [%d, %d, %d, %d]\n", output_shape[0], output_shape[1], output_shape[2], output_shape[3]);

        printf("\n===Input==============\n");
        for (int i = 0; i < input_shape_[0]; i++) {
            printf("[\n");
            for (int j = 0; j < input_shape_[1]; j++) {
                printf("[\n");
                for (int k = 0; k < input_shape_[2]; k++) {
                    printf("[");
                    for (int l = 0; l < input_shape_[3]; l++) {
                        int idx = i * input_shape_[1] * input_shape_[2] * input_shape_[3] +
                                j * input_shape_[2] * input_shape_[3] +
                                k * input_shape_[3] + l;
                        printf("%.4f, ", torch_input[idx]);
                    }
                    printf("]\n");
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n===Output==============\n");
        for (int i = 0; i < output_shape[0]; i++) {
            printf("[\n");
            for (int j = 0; j < output_shape[1]; j++) {
                printf("[\n");
                for (int k = 0; k < output_shape[2]; k++) {
                    printf("[");
                    for (int l = 0; l < output_shape[3]; l++) {
                        int idx = i * output_shape[1] * output_shape[2] * output_shape[3] +
                                j * output_shape[2] * output_shape[3] +
                                k * output_shape[3] + l;
                        printf("%.4f, ", torch_output[idx]);
                    }
                    printf("]\n");
                }
                printf("\n");
            }
            printf("\n");
        }
        input_data_ = std::make_shared<Tensor<float>>(input_shape_);
        input_data_->fillToCPU(torch_input);
        output_data_ = std::make_shared<Tensor<float>>(output_shape);
        output_data_->fillToCPU(torch_output);
#if USE_CPP_REFER
        auto ret = reference_resize(input_data_, input_shape_[0], input_shape_[1], input_shape_[2], input_shape_[3],
                    resize_[0], resize_[1], mode_, align_corners_, cubic_coeff_a_);
        printf("\n===Reference Output==============\n");
        for (int i = 0; i < output_shape[0]; i++) {
            for (int j = 0; j < output_shape[1]; j++) {
                for (int k = 0; k < output_shape[2]; k++) {
                    printf("[");
                    for (int l = 0; l < output_shape[3]; l++) {
                        int idx = i * output_shape[1] * output_shape[2] * output_shape[3] +
                                j * output_shape[2] * output_shape[3] +
                                k * output_shape[3] + l;
                        printf("%.4f, ", ret[idx]);
                        if (fabs(ret[idx] - torch_output[idx]) > 0.0001F) {
                            printf("  <-- MISMATCH ");
                        }
                    }
                    printf("]\n");
                }
                printf("\n");
            }
            printf("\n");
        }
#endif
    }
};


int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", true);
    ResizeTest<float> rt;

    if (!rt.run_test<float>({rt.input_data_, nullptr, nullptr, nullptr}, {rt.output_data_},
        [&rt](std::unique_ptr<vkop::ops::Operator> &op) {
        auto *resize_op = dynamic_cast<Resize *>(op.get());
        if (!resize_op) {
            LOG_ERROR("Failed to cast operator to Resize");
            return;
        }
        resize_op->setAttribute(rt.attributes);

    })) {
        return -1;
    }
    
    return 0;
}