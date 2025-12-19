#include <vector>
#include <random>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "ops/Transpose.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;
using vkop::ops::Transpose;

namespace {

    std::vector<int> transpose_shape(const std::vector<int>& shape, const std::vector<int>& perm) {
        std::vector<int> new_shape(perm.size());
        for (size_t i = 0; i < perm.size(); ++i) {
            new_shape[i] = shape[perm[i]];
        }
        return new_shape;
    }

    template <typename T>
    void transpose(const std::shared_ptr<Tensor<T>>& input, std::shared_ptr<Tensor<T>>& output, const std::vector<int>& perm) {
        const auto& in_shape = input->getShape();
        auto out_shape = transpose_shape(in_shape, perm);

        int in_C = in_shape[1];
        int in_H = in_shape[2];
        int in_W = in_shape[3];
        int out_C = out_shape[1];
        int out_H = out_shape[2];
        int out_W = out_shape[3];
        for (int n = 0; n < in_shape[0]; ++n) {
            for (int c = 0; c < in_shape[1]; ++c) {
                for (int h = 0; h < in_shape[2]; ++h) {
                    for (int w = 0; w < in_shape[3]; ++w) {
                        int new_n = perm[0] == 0 ? n : (perm[0] == 1 ? c : (perm[0] == 2 ? h : w));
                        int new_c = perm[1] == 0 ? n : (perm[1] == 1 ? c : (perm[1] == 2 ? h : w));
                        int new_h = perm[2] == 0 ? n : (perm[2] == 1 ? c : (perm[2] == 2 ? h : w));
                        int new_w = perm[3] == 0 ? n : (perm[3] == 1 ? c : (perm[3] == 2 ? h : w));
                        (*output)[(new_n * out_C * out_H * out_W) + (new_c * out_H * out_W) + (new_h * out_W) + new_w] =
                            input->at((n * in_C * in_H * in_W) + (c * in_H * in_W) + (h * in_W) + w);
                    }
                }
            }
        }
    }

class TransposeTest : public TestCase {
public:
    std::vector<int> input_shape_ = {
        2, 1, 2, 3
    };
    std::vector<int> perm_ = {
        0, 2, 3, 1
    };
    const std::unordered_map<std::string, std::string> param = {
        {"perm", "[0,2,3,1]"}
    };
    std::shared_ptr<Tensor<float>> input;
    std::shared_ptr<Tensor<float>> output;

    TransposeTest():TestCase("Transpose") {
        initTestdata();
    }
private:
    void initTestdata()
    {
        input = std::make_shared<Tensor<float>>(input_shape_);
        input->reserveOnCPU();
        auto out_shape = transpose_shape(input_shape_, perm_);
        output = std::make_shared<Tensor<float>>(out_shape);
        output->reserveOnCPU();

        std::random_device rd{};
        std::mt19937 gen{rd()};
        gen.seed(1024);
        std::normal_distribution<> input_dist{-1.0F, 1.0F};
        for (int i = 0; i < input->num_elements(); i++) {
            auto a = input_dist(gen);
            (*input)[i] = a;
        }
        printf("Input:\n");
        for (int i = 0; i < input_shape_[0]; i++) {
            printf("[\n");
            for (int j = 0; j < input_shape_[1]; j++) {
                printf("[\n");
                for (int k = 0; k < input_shape_[2]; k++) {
                    printf("[");
                    for (int l = 0; l < input_shape_[3]; l++) {
                        int idx = (i * input_shape_[1] * input_shape_[2] * input_shape_[3]) + (j * input_shape_[2] * input_shape_[3]) +
                            (k * input_shape_[3]) + l;
                        printf("%.4f, ", (*input)[idx]);
                    }
                    printf("]\n");
                }
                printf("]\n");
            }
            printf("]\n");
        }
        transpose(input, output, perm_);
        const auto& in_shape = input->getShape();

        printf("Output:\n");
        for (int n = 0; n < out_shape[0]; n++) {
            printf("[\n");
            for (int c = 0; c < out_shape[1]; c++) {
                printf("[\n");
                for (int h = 0; h < out_shape[2]; h++) {
                    printf("[");
                    for (int w = 0; w < out_shape[3]; w++) {
                        int idx = (n * out_shape[1] * out_shape[2] * out_shape[3]) +
                                  (c * out_shape[2] * out_shape[3]) + (h * out_shape[3]) + w;
                        printf("%f ", (*output)[idx]);
                    }
                    printf("]\n");
                }
                printf("]\n");
            }
            printf("]\n");
        }
    }
};
}


int main()
{
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", false);

    TransposeTest trans_test;
    if (!trans_test.run_test<float>({trans_test.input}, {trans_test.output}, [&trans_test](std::unique_ptr<vkop::ops::Operator> &op) {
            auto *tran_op = dynamic_cast<Transpose *>(op.get());
            if (!tran_op) {
                LOG_ERROR("Failed to cast operator to Transpose");
                return;
            }
            tran_op->setAttribute(trans_test.param);
        })) {
        return -1;
    }
    return 0;
}