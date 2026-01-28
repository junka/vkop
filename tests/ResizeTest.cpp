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
        torch::manual_seed(42);
        auto torch_input = torch::randn({input_shape_[0], input_shape_[1], input_shape_[2], input_shape_[3]});

        std::vector<int64_t> sizes;
        bool has_size = false;
        if (attributes.count("size") > 0) {
            std::string size_str = attributes.at("size");
            if (size_str[0] == '[' && size_str[size_str.length()-1] == ']') {
                std::string content = size_str.substr(1, size_str.length() - 2);
                std::stringstream ss(content);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    sizes.push_back(std::stoll(item));
                }
            } else {
                sizes.push_back(std::stoll(size_str));
            }
            has_size = !sizes.empty();
        }

        std::vector<double> scale_factors;
        bool has_scale_factor = false;
        if (attributes.count("scale_factor") > 0) {
            std::string scale_str = attributes.at("scale_factor");
            if (scale_str[0] == '[' && scale_str[scale_str.length()-1] == ']') {
                std::string content = scale_str.substr(1, scale_str.length() - 2);
                std::stringstream ss(content);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    scale_factors.push_back(std::stod(item));
                }
            } else {
                scale_factors.push_back(std::stod(scale_str));
            }
            has_scale_factor = !scale_factors.empty();
        }

        bool align_corners = attributes.count("align_corners") ? (attributes.at("align_corners") == "True" || attributes.at("align_corners") == "true") : false;
        std::string mode = attributes.count("mode") ? attributes.at("mode") : "nearest";

        std::optional<std::vector<int64_t>> opt_size = std::nullopt;
        if (has_size) {
            opt_size = sizes;
        }

        std::optional<std::vector<double>> opt_scale_factor = std::nullopt;
        if (has_scale_factor) {
            opt_scale_factor = scale_factors;
        }

        auto options = torch::nn::functional::InterpolateFuncOptions()
            .size(opt_size)
            .scale_factor(opt_scale_factor);

        if (mode == "bilinear") {
            options = options.mode(torch::kBilinear).align_corners(align_corners);
        } else if (mode == "bicubic") {
            options = options.mode(torch::kBicubic).align_corners(align_corners);
        } else if (mode == "trilinear") {
            options = options.mode(torch::kTrilinear).align_corners(align_corners);
        } else if (mode == "linear") {
            options = options.mode(torch::kLinear).align_corners(align_corners);
        } else {
            options = options.mode(torch::kNearest);
        }

        auto torch_output = torch::nn::functional::interpolate(torch_input, options);

        std::vector<int> output_shape = {};
        output_shape.reserve(torch_output.dim());
        for (int i = 0; i < torch_output.dim(); i++) {
            output_shape.push_back(torch_output.size(i));
        }

        printf("torch output size: [%d, %d, %d, %d]\n", output_shape[0], output_shape[1], output_shape[2], output_shape[3]);
        printf("\n===Input==============\n");
        std::cout << torch_input << std::endl;
        printf("\n===Output==============\n");
        std::cout << torch_output << std::endl;

        input_data_ = std::make_shared<Tensor<float>>(input_shape_);
        auto input_cpu = torch_input.cpu().contiguous();
        std::vector<float> input_vector;
        input_vector.reserve(input_cpu.numel());
        auto input_accessor = input_cpu.accessor<float, 4>();
        for (int i = 0; i < input_shape_[0]; i++) {
            for (int j = 0; j < input_shape_[1]; j++) {
                for (int k = 0; k < input_shape_[2]; k++) {
                    for (int l = 0; l < input_shape_[3]; l++) {
                        input_vector.push_back(input_accessor[i][j][k][l]);
                    }
                }
            }
        }
        input_data_->fillToCPU(input_vector);

        output_data_ = std::make_shared<Tensor<float>>(output_shape);
        auto output_cpu = torch_output.cpu().contiguous();
        std::vector<float> output_vector;
        output_vector.reserve(output_cpu.numel());
        auto output_accessor = output_cpu.accessor<float, 4>();
        for (int i = 0; i < output_shape[0]; i++) {
            for (int j = 0; j < output_shape[1]; j++) {
                for (int k = 0; k < output_shape[2]; k++) {
                    for (int l = 0; l < output_shape[3]; l++) {
                        output_vector.push_back(output_accessor[i][j][k][l]);
                    }
                }
            }
        }
        output_data_->fillToCPU(output_vector);
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