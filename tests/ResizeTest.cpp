#include <memory>
#include <string>
#include <vector>
#include <cassert>

#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "setup.hpp"
#include "ops/Resize.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;
using vkop::ops::Resize;


template<typename T>
class ResizeTest: public TestCase {
public:
    std::vector<int> input_shape_ = {1, 2, 4, 4};
    std::vector<int> resize_ = {1, 2, 2, 2};
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
        const auto& torch_output = torch_tensors[0];
        const auto& torch_input = torch_tensors[1];
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
                        int idx = (i * input_shape_[1] * input_shape_[2] * input_shape_[3]) +
                                (j * input_shape_[2] * input_shape_[3]) +
                                (k * input_shape_[3]) + l;
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
                        int idx = (i * output_shape[1] * output_shape[2] * output_shape[3]) +
                                (j * output_shape[2] * output_shape[3]) +
                                (k * output_shape[3]) + l;
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