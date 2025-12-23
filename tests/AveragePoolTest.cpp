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
        std::vector<std::vector<int>> shapes;
        shapes.push_back(input_shape);
        std::tuple<std::vector<std::vector<float>>, std::vector<int>> k = TestCase::execute_torch_operator("avg_pool2d", shapes, attributes);
        std::vector<std::vector<float>> torch_tensors = std::get<0>(k);
        auto& torch_output = torch_tensors[0];
        auto& torch_input = torch_tensors[1];
        std::vector<int> output_shape = std::get<1>(k);
        input = std::make_shared<Tensor<float>>(input_shape);
        input->fillFP32ToCPU(torch_input);
        output = std::make_shared<Tensor<float>>(output_shape);
        output->fillFP32ToCPU(torch_output);

        printf("\n===Input==============\n");
        for (int i = 0; i < input_shape[0]; i++) {
            printf("[\n");
            for (int j = 0; j < input_shape[1]; j++) {
                printf("[\n");
                for (int k = 0; k < input_shape[2]; k++) {
                    printf("[");
                    for (int l = 0; l < input_shape[3]; l++) {
                        int idx = (i * input_shape[1] * input_shape[2] * input_shape[3]) +
                                (j * input_shape[2] * input_shape[3]) +
                                (k * input_shape[3]) + l;
                        printf("%.4f, ", torch_input[idx]);
                    }
                    printf("],\n");
                }
                printf("],\n");
            }
            printf("]\n");
        }

        printf("\n===Output==============\n");
        printf("[ %d, %d, %d, %d ]\n", output_shape[0], output_shape[1], output_shape[2], output_shape[3]);
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