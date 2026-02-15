#include <vector>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "ops/Topk.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;
using vkop::ops::Topk;

namespace {
#if USE_CPP_REFER
std::vector<std::vector<std::pair<int, float>>> get_top_k_predictions(const std::vector<std::vector<float>>& probs, int k) {
    std::vector<std::vector<std::pair<int, float>>> ret;
    int rows = probs.size();
    int cols = probs[0].size();
    for (int i = 0; i < rows; ++i) {
        std::vector<float> softmax_probs = probs[i];

        std::vector<std::pair<int, float>> indexed_probs;
        indexed_probs.reserve(softmax_probs.size());
        for (size_t j = 0; j < softmax_probs.size(); ++j) {
            indexed_probs.emplace_back((i * cols) + j, softmax_probs[j]);
        }

        std::sort(indexed_probs.begin(), indexed_probs.end(),
                [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                    return a.second > b.second;
                });

        if (indexed_probs.size() > static_cast<size_t>(k)) {
            indexed_probs.resize(k);
        }
        ret.emplace_back(indexed_probs);
    }
    return ret;
}
#endif

template<typename T>
class TopkTest : public TestCase {
public:
    std::shared_ptr<Tensor<T>> input;
    std::shared_ptr<Tensor<int>> indexs;
    std::shared_ptr<Tensor<T>> output;
    std::vector<int> t;
    int k = 7;
    int axis = -1;
    std::unordered_map<std::string, std::string> attrs;

    TopkTest(const std::vector<int> &shape, const int k, const int axis):TestCase("TopK"),t (shape), k(k), axis(axis) {
        attrs = {
            {"k", std::to_string(k)},
            {"axis", std::to_string(axis)},
            {"largest", "1"},
            {"sorted", "1"}
        };
        initTestdata();
    }
private:
    void initTestdata()
    {
        auto out_shape = t;
        input = std::make_shared<Tensor<T>>(t);
        axis = axis < 0 ? input->num_dims() + axis : axis;
        out_shape[axis] = k;
        indexs = std::make_shared<Tensor<int>>(out_shape);
        output = std::make_shared<Tensor<T>>(out_shape);

        torch::TensorOptions conf;
        if constexpr (std::is_same_v<T, float>) {
            conf = torch::TensorOptions().dtype(torch::kFloat32);
        } else if constexpr (std::is_same_v<T, uint16_t>) {
            conf = torch::TensorOptions().dtype(torch::kFloat16);
        }
        torch::manual_seed(56);
        auto torch_input = torch::randn({t[0], t[1]}, conf);
        auto torch_topk = torch::topk(torch_input, k, axis, true, true);
        auto [value, indices64] = torch_topk;
        auto indices32 = indices64.to(torch::kInt);

        printf("\n===input =======\n");
        std::cout << torch_input << std::endl;
        printf("\n===topk value =======\n");
        std::cout << value << std::endl;
        printf("\n===topk index =======\n");
        std::cout << indices32 << std::endl;

        fillTensorFromTorch(input, torch_input);
        fillTensorFromTorch(output, value);
        fillTensorFromTorch(indexs, indices32);
    }
};
}

TEST(TopkTest, TopkComprehensiveTest) {

    const std::vector<std::tuple<std::vector<int>, int, int>> test_cases = {
        {{1, 200}, 7, -1},
        {{1, 400}, 8, -1},
        {{1, 600}, 7, -1},
        {{3, 2000}, 7, -1},
        {{1, 2000}, 6, -1},
    };

    for (const auto &test_case : test_cases) {
        auto [t, k, axis] = test_case;
        
        LOG_INFO("Testing TopK with shape: [%d, %d], k: %d, axis: %d",
                 t[0], t[1], k, axis);
        LOG_INFO("Testing fp32");
        TopkTest<float> toptest(t, k, axis);
        EXPECT_TRUE (toptest.run_test<float>({toptest.input}, {toptest.output, toptest.indexs},[&toptest](std::unique_ptr<vkop::ops::Operator> &op) {
            auto *topk_op = dynamic_cast<Topk *>(op.get());
            if (!topk_op) {
                LOG_ERROR("Failed to cast operator to Topk");
                return;
            }
            topk_op->setAttribute(toptest.attrs);
        }));

        LOG_INFO("Testing fp16");
        TopkTest<uint16_t> toptest1(t, k, axis);
        EXPECT_TRUE(toptest1.run_test<uint16_t>({toptest1.input}, {toptest1.output, toptest1.indexs},[&toptest1](std::unique_ptr<vkop::ops::Operator> &op) {
            auto *topk_op = dynamic_cast<Topk *>(op.get());
            if (!topk_op) {
                LOG_ERROR("Failed to cast operator to Topk");
                return;
            }
            topk_op->setAttribute(toptest1.attrs);
        }));
    }
}