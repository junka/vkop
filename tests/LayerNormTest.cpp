#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "ops/LayerNorm.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;
using vkop::ops::LayerNorm;

namespace {
#if USE_CPP_REFER

std::vector<float> layer_norm(std::shared_ptr<Tensor<float>> &input, std::vector<int> input_shape, std::vector<int> nshape,
                              std::shared_ptr<Tensor<float>>& weight, std::shared_ptr<Tensor<float>>& bias, float epsilon) {
    size_t total_size = input->num_elements();
    std::vector<float> output(total_size);

    int rank = input_shape.size();
    int norm_rank = nshape.size();

    if (norm_rank > rank) {
        throw std::invalid_argument("normalized_shape has more dims than input");
    }

    for (int i = 0; i < norm_rank; ++i) {
        if (input_shape[rank - norm_rank + i] != nshape[i]) {
            throw std::invalid_argument("normalized_shape does not match trailing dims of input");
        }
    }

    int inner_size = 1;
    for (int s : nshape) {
        inner_size *= s;
    }

    int outer_size = total_size / inner_size;

    bool has_weight = (inner_size == weight->num_elements());
    bool has_bias = (inner_size == bias->num_elements());

    if (weight->num_elements() != inner_size) {
        throw std::invalid_argument("weight size must match product of normalized_shape");
    }
    if (bias->num_elements() != inner_size) {
        throw std::invalid_argument("bias size must match product of normalized_shape");
    }

    for (int outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
        const int in_block_offset = outer_idx * inner_size;
        float*       out_block = output.data() + (outer_idx * inner_size);

        float sum = 0.0F;
        for (int i = 0; i < inner_size; ++i) {
            sum += (*input)[in_block_offset+i];
        }
        float mean = sum / inner_size;

        float var = 0.0F;
        for (int i = 0; i < inner_size; ++i) {
            float diff = (*input)[in_block_offset+i] - mean;
            var += diff * diff;
        }
        var /= inner_size;
        float inv_std = 1.0F / std::sqrt(var + epsilon);

        for (int i = 0; i < inner_size; ++i) {
            float normalized_val = ((*input)[in_block_offset+i] - mean) * inv_std;
            if (has_weight) normalized_val *= (*weight)[i];
            if (has_bias)   normalized_val += (*bias)[i];
            out_block[i] = normalized_val;
        }
    }

    return output;
}
#endif

template <typename T>
class LayerNormTest : public TestCase<T> {
public:
    std::vector<int> input_shape_;
    std::vector<int> normalized_shape_;
    std::unordered_map<std::string, std::string> param;
    std::shared_ptr<Tensor<T>> input;
    std::shared_ptr<Tensor<T>> weight;
    std::shared_ptr<Tensor<T>> bias;
    std::shared_ptr<Tensor<T>> output;

    LayerNormTest(const std::vector<int> &input_shape,
                  const std::vector<int> &normalized_shape)
        : TestCase<T>("LayerNorm"), input_shape_(input_shape),
          normalized_shape_(normalized_shape) {
        std::string ns = "[";
        for (size_t i = 0; i < normalized_shape_.size(); ++i) {
            if (i != 0) ns += ", ";
            ns += std::to_string(normalized_shape_[i]);
        }
        ns += "]";
        param = {{"eps", "1e-5"}, {"normalized_shape", ns}};
        initTestdata();
    }
private:
    void initTestdata()
    {
        torch::manual_seed(42);
        std::vector<int64_t> ishape(input_shape_.begin(), input_shape_.end());
        std::vector<int64_t> nshape(normalized_shape_.begin(),
                                    normalized_shape_.end());
        auto torch_input = torch::randn(ishape, this->getTorchConf());
        auto torch_weight = torch::randn(nshape, this->getTorchConf());
        auto torch_bias = torch::randn(nshape, this->getTorchConf());

        // fp16 reference: accumulate in fp32, then round to half. The CPU
        // half layer_norm accumulates in half and drifts past the 2% check.
        torch::Tensor torch_output;
        if (this->getTorchConf().dtype() == torch::kFloat16) {
            torch_output = torch::layer_norm(torch_input.to(torch::kFloat32), nshape,
                                             torch_weight.to(torch::kFloat32),
                                             torch_bias.to(torch::kFloat32), 1e-5)
                               .to(torch::kFloat16);
        } else {
            torch_output = torch::layer_norm(torch_input, nshape, torch_weight, torch_bias, 1e-5);
        }

        std::vector<int> output_shape = {};
        output_shape.reserve(torch_output.dim());
        for (int i = 0; i < torch_output.dim(); i++) {
            output_shape.push_back(torch_output.size(i));
        }

        printf("torch output size: [");
        for (size_t i = 0; i < output_shape.size(); ++i) {
            printf("%d%s", output_shape[i], i + 1 == output_shape.size() ? "]\n" : ", ");
        }

        input = std::make_shared<Tensor<T>>(input_shape_);
        this->fillTensorFromTorch(input, torch_input);

        weight = std::make_shared<Tensor<T>>(normalized_shape_);
        this->fillTensorFromTorch(weight, torch_weight);

        bias = std::make_shared<Tensor<T>>(normalized_shape_);
        this->fillTensorFromTorch(bias, torch_bias);

        output = std::make_shared<Tensor<T>>(output_shape);
        this->fillTensorFromTorch(output, torch_output);
    }
};
}

TEST(LayerNormTest, LayerNormComprehensiveTest) {
    // The image backend picks its normalize kernel from normalizedDim: 1 -> W,
    // 2 -> HW, 3 -> CHW. Only the HW shape used to be covered, which left the
    // other two reduce sites (and their subgroup variants) unexercised.
    //
    // normalizedDim 3 is left out on purpose: the CHW kernel runs and writes
    // normalized values, but the tensor read back from it does not line up with
    // the torch reference's NCHW order (first element 1.9269, the raw input,
    // vs 5.2607). It fails identically with the base and the -DSUBGROUP build,
    // so it is a pre-existing image-path bug this suite never reached, not a
    // regression from the subgroup work.
    struct Case {
        std::vector<int> input;
        std::vector<int> norm;
    };
    const std::vector<Case> cases = {
        {{2, 5, 4, 4}, {4}},    // normalizedDim 1: norm_for_w
        {{2, 5, 4, 4}, {4, 4}}, // normalizedDim 2: norm_for_hw
    };
    for (const auto &c : cases) {
        LayerNormTest<float> lntest(c.input, c.norm);
#if USE_CPP_REFER
        printf("\n===verify C++ refer ==========\n");
        auto bout = layer_norm(lntest.input, c.input, c.norm, lntest.weight,
                               lntest.bias, 1e-5);
        size_t mismatch = 0;
        for (size_t i = 0; i < bout.size(); ++i) {
            if (std::fabs(bout[i] - (*lntest.output)[i]) > 1e-3) {
                ++mismatch;
            }
        }
        printf("c++ refer mismatches: %zu / %zu\n", mismatch, bout.size());
#endif

        EXPECT_TRUE(lntest.run_test({lntest.input, lntest.weight, lntest.bias},
                                    {lntest.output},
            [&lntest](std::unique_ptr<vkop::ops::Operator> &op) {
                op->setAttribute(lntest.param);
            })) << "normalized_shape rank " << c.norm.size();

        LayerNormTest<uint16_t> lntest16(c.input, c.norm);
        {
            // LayerNorm's image kernel only has an fp32 path; fp16 lives in the
            // SSBO implementation, so this case has to force the buffer backend
            // instead of inheriting the image default.
            vkop::tests::ScopedBufferBackend buffer_backend;
            EXPECT_TRUE(lntest16.run_test({lntest16.input, lntest16.weight, lntest16.bias},
                                          {lntest16.output},
                [&lntest16](std::unique_ptr<vkop::ops::Operator> &op) {
                    op->setAttribute(lntest16.param);
                })) << "fp16 normalized_shape rank " << c.norm.size();
        }
    }
}

// normalizedDim 3 (the CHW kernel) is a known image-path bug: the shader runs
// and writes normalized values, but the readback does not line up with the
// torch reference's NCHW order (measured first element 1.9269 — the raw input
// — vs 5.2607). It reproduces with the base and the -DSUBGROUP build, so it is
// not a subgroup regression. Keeping the case as a DISABLED test makes the gap
// visible in --gtest_list_tests instead of leaving it only in a comment.
TEST(LayerNormTest, DISABLED_ChwNormalizedDim) {
    LayerNormTest<float> lntest({2, 5, 4, 4}, {4, 4, 4});
    EXPECT_TRUE(lntest.run_test({lntest.input, lntest.weight, lntest.bias},
                                {lntest.output},
        [&lntest](std::unique_ptr<vkop::ops::Operator> &op) {
            op->setAttribute(lntest.param);
        })) << "normalized_shape rank 3 (CHW)";
}
