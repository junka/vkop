#include <memory>
#include <string>
#include <utility>
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
    float cubic_coeff_a_ = -0.75;

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
    std::unordered_map<std::string, std::string> attributes;

    std::shared_ptr<Tensor<T>> input_data_;
    std::shared_ptr<Tensor<T>> output_data_;

    ResizeTest(const std::vector<int>& input_shape, const std::vector<int>& resize, 
               bool align_corners, std::string mode, bool antialias, float cubic_coeff_a = -0.75)
        : TestCase("Resize"), input_shape_(input_shape), resize_(resize), align_corners_(align_corners), 
          mode_(std::move(mode)), antialias_(antialias), cubic_coeff_a_(cubic_coeff_a) {
        
        attributes = {
            {"size", buildSizeString(resize_)},
            {"mode", mode_},
            {"align_corners", align_corners_ ? "True": "False"},
            {"antialias", antialias_ ? "True" : "False"},
        };
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

        printf("\n===Input==============\n");
        std::cout << torch_input << std::endl;
        printf("\n===Output==============\n");
        std::cout << torch_output << std::endl;

        input_data_ = std::make_shared<Tensor<float>>(input_shape_);
        fillTensorFromTorch(input_data_, torch_input);

        output_data_ = std::make_shared<Tensor<float>>(output_shape);
        fillTensorFromTorch(output_data_, torch_output);
    }
};


int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", false);
    vkop::tests::TestEnv::initialize();

    std::vector<std::tuple<std::vector<int>, std::vector<int>, bool, std::string, bool>> test_cases = {
        {{1, 2, 4, 4}, {1, 2, 2, 2}, false, "bilinear", false},
        {{1, 3, 8, 8}, {1, 3, 4, 4}, false, "nearest", false},
        {{1, 4, 5, 5}, {1, 4, 10, 10}, false, "nearest", false},
        // {{1, 2, 3, 3}, {1, 2, 6, 6}, true, "bilinear", true},
        // {{2, 1, 6, 6}, {2, 1, 12, 12}, true, "bilinear", false},
    };
    for (const auto& test_case : test_cases) {
        auto [input_shape, resize, align_corners, mode, antialias] = test_case;
        LOG_INFO("Running test case: input=%d,%d,%d,%d, resize=%d,%d,%d,%d, mode=%s, align_corners=%s, antialias=%s",
               input_shape[0], input_shape[1], input_shape[2], input_shape[3],
               resize[0], resize[1], resize[2], resize[3], mode.c_str(), 
               align_corners ? "true" : "false", antialias ? "true" : "false");
        ResizeTest<float> rt(input_shape, resize, align_corners, mode, antialias);

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
    }
    vkop::tests::TestEnv::cleanup();
    return 0;
}