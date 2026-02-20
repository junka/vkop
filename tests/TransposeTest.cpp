#include <vector>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "ops/Transpose.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;
using vkop::ops::Transpose;

namespace {

    
#ifdef USE_CPP_REF
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
#endif
template <typename T>
class TransposeTest : public TestCase<T> {
public:
    std::vector<int> input_shape_;
    std::vector<int> perm_;
    std::unordered_map<std::string, std::string> attributes;
    std::shared_ptr<Tensor<T>> input;
    std::shared_ptr<Tensor<T>> output;

    TransposeTest(std::vector<int>& input_shape, std::vector<int> & permute):TestCase<T>("Transpose"), input_shape_(input_shape), perm_(permute) {
        if (!perm_.empty() && perm_.size() > 0) {
            std::string str = "[";
            for (int i = 0; i < static_cast<int>(perm_.size()); ++i) {
                str += std::to_string(i);
                if (i < static_cast<int>(perm_.size()) - 1) {
                    str += ",";
                }
            }
            str += "]";
            attributes = {
                {"perm", str}
            };
        }
        
        initTestdata();
    }
private:
    void initTestdata()
    {
        input = std::make_shared<Tensor<T>>(input_shape_);

        std::vector<int64_t> inshape(input_shape_.begin(), input_shape_.end());
        auto torch_input = torch::randn(inshape, this->getTorchConf());

        int ndim = input->num_dims();
        std::vector<int64_t> perm(ndim);
        if (perm_.empty() || perm_.size() == 0) {
            for (int i = 0; i < ndim; ++i) {
                perm[i] = ndim - 1 - i;
            }
        } else {
            for (int i = 0; i < ndim; ++i) {
                perm[i] = perm_[i];
            }
        }
        auto torch_output = torch::permute(torch_input, perm);
        auto oshape = torch_output.sizes();
        std::vector<int> out_shape(oshape.begin(), oshape.end());;
        output = std::make_shared<Tensor<T>>(out_shape);

        this->fillTensorFromTorch(input, torch_input);
        this->fillTensorFromTorch(output, torch_output);

        printf("Input:\n");
        input->print_tensor();
#if 0
        transpose(input, output, perm_);
#endif
        const auto& in_shape = input->getShape();

        printf("Output:\n");
        output->print_tensor();
    }
};
}

TEST(TransposeTest, TransposeComprehensiveTest) {

    std::vector<std::tuple<std::vector<int>, std::vector<int>>> test_cases = {
        {{2, 1, 2, 3}, {0, 2, 3, 1}},
        {{2, 3, 4, 5}, {}},
        // {{2, 1, 3}, {}},
    };
    for (const auto& test_case : test_cases) {
        auto [input_shape, perm] = test_case;

        LOG_INFO("Transpose FP32");
        TransposeTest<float> trans_test(input_shape, perm);
        EXPECT_TRUE(trans_test.run_test({trans_test.input}, {trans_test.output}, [&trans_test](std::unique_ptr<vkop::ops::Operator> &op) {
            auto *tran_op = dynamic_cast<Transpose *>(op.get());
            if (!tran_op) {
                LOG_ERROR("Failed to cast operator to Transpose");
                return;
            }
            tran_op->setAttribute(trans_test.attributes);
        }));

        LOG_INFO("Transpose FP16");
        TransposeTest<uint16_t> trans_test1(input_shape, perm);
        EXPECT_TRUE(trans_test1.run_test({trans_test1.input}, {trans_test1.output}, [&trans_test1](std::unique_ptr<vkop::ops::Operator> &op) {
            auto *tran_op = dynamic_cast<Transpose *>(op.get());
            if (!tran_op) {
                LOG_ERROR("Failed to cast operator to Transpose");
                return;
            }
            tran_op->setAttribute(trans_test1.attributes);
        }));
    }

}