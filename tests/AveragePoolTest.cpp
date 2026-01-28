#include <vector>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "ops/AveragePool.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;
using vkop::ops::AveragePool;

namespace {
class AveragePoolTest : public TestCase {
public:
    std::shared_ptr<Tensor<float>> input;
    std::shared_ptr<Tensor<float>> output;
    std::unordered_map<std::string, std::string> attributes = {
        {"auto_pad", "NOTSET"},
        {"pads", "[0,0,0,0]"},
        {"strides", "[2,4]"},
        {"kernel_shape", "[4,8]"}
    };

    AveragePoolTest():TestCase("AveragePool") {
        initTestdata();
    }
private:
    void initTestdata()
    {
        std::vector<int> input_shape = {
            1, 3, 8, 8
        };
        torch::manual_seed(42);
        auto torch_input = torch::randn({input_shape[0], input_shape[1], input_shape[2], input_shape[3]});

        std::vector<int64_t> kernel_sizes;
        std::string kernel_shape_str = attributes.count("kernel_shape") ? attributes.at("kernel_shape") : "[2,2]";
        if (kernel_shape_str[0] == '[' && kernel_shape_str.back() == ']') {
            std::string content = kernel_shape_str.substr(1, kernel_shape_str.length() - 2);
            std::stringstream ss(content);
            std::string item;
            while (std::getline(ss, item, ',')) {
                kernel_sizes.push_back(std::stoll(item));
            }
        } else {
            int64_t val = std::stoll(kernel_shape_str);
            kernel_sizes = {val, val};
        }

        std::vector<int64_t> strides;
        std::string strides_str = attributes.count("strides") ? attributes.at("strides") : "[1,1]";
        if (strides_str[0] == '[' && strides_str.back() == ']') {
            std::string content = strides_str.substr(1, strides_str.length() - 2);
            std::stringstream ss(content);
            std::string item;
            while (std::getline(ss, item, ',')) {
                strides.push_back(std::stoll(item));
            }
        } else {
            int64_t val = std::stoll(strides_str);
            strides = {val, val};
        }

        std::vector<int64_t> paddings;
        std::string pads_str = attributes.count("pads") ? attributes.at("pads") : "[0,0]";
        if (pads_str[0] == '[' && pads_str.back() == ']') {
            std::string content = pads_str.substr(1, pads_str.length() - 2);
            std::stringstream ss(content);
            std::string item;
            while (std::getline(ss, item, ',')) {
                paddings.push_back(std::stoll(item));
            }
        } else {
            int64_t val = std::stoll(pads_str);
            paddings = {val, val};
        }
        if (paddings.size() > 2) {
            paddings = {paddings[0],paddings[1]};
        }

        bool ceil_mode = attributes.count("ceil_mode") ? (std::stoi(attributes.at("ceil_mode")) != 0) : false;
        bool count_include_pad = attributes.count("count_include_pad") ? (std::stoi(attributes.at("count_include_pad")) != 0) : true;

        auto torch_output = torch::avg_pool2d(torch_input,
            torch::ArrayRef<int64_t>(kernel_sizes),
            torch::ArrayRef<int64_t>(strides),
            torch::ArrayRef<int64_t>(paddings),
            ceil_mode,
            count_include_pad
        );

        std::vector<int> output_shape = {};
        output_shape.reserve(torch_output.dim());
        for (int i = 0; i < torch_output.dim(); i++) {
            output_shape.push_back(torch_output.size(i));
        }

        input = std::make_shared<Tensor<float>>(input_shape);
        auto input_cpu = torch_input.cpu().contiguous();
        std::vector<float> input_vector;
        input_vector.reserve(input_cpu.numel());
        auto input_accessor = input_cpu.accessor<float, 4>();
        for (int i = 0; i < input_shape[0]; i++) {
            for (int j = 0; j < input_shape[1]; j++) {
                for (int k = 0; k < input_shape[2]; k++) {
                    for (int l = 0; l < input_shape[3]; l++) {
                        input_vector.push_back(input_accessor[i][j][k][l]);
                    }
                }
            }
        }
        input->fillFP32ToCPU(input_vector);

        output = std::make_shared<Tensor<float>>(output_shape);
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
        output->fillFP32ToCPU(output_vector);

        printf("\n===Input==============\n");
        std::cout << torch_input << std::endl;

        printf("\n===Output==============\n");
        printf("[ %d, %d, %d, %d ]\n", output_shape[0], output_shape[1], output_shape[2], output_shape[3]);
        std::cout << torch_output << std::endl;
    }
};
}

int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", false);

    AveragePoolTest aptest;
    if (!aptest.run_test<float>({aptest.input}, {aptest.output},
        [&aptest](std::unique_ptr<vkop::ops::Operator> &op) {
        auto *ap_op = dynamic_cast<AveragePool *>(op.get());
        if (!ap_op) {
            LOG_ERROR("Failed to cast operator to Conv2d");
            return;
        }
        ap_op->setAttribute(aptest.attributes);
    })) {
        return -1;
    }

    return 0;
}