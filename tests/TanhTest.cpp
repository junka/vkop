#include <vector>

#include "UnaryTest.hpp"
#include "include/logger.hpp"

using vkop::tests::UnaryTest;

// Tanh 测试：element-wise tanh，走 UnaryFactory 路径（与 Erf/Sin/Cos/Neg 一致）。
// 参考输出用 torch::tanh。fp32 + fp16，多 shape。

namespace {

template <typename T>
class TanhTest : public vkop::tests::UnaryTest<T> {
public:
    explicit TanhTest(const std::vector<int> &shape)
        : vkop::tests::UnaryTest<T>("Tanh", shape) {
        // tanh 输入范围控制在 [-3,3]：randn 可能过大使 fp16 tanh 近 ±1 全饱和，
        // 不利于检测实现正确性。用 3*tanh(randn/3) 缩放，保持梯度可分辨。
        auto scaled = 3.0 * torch::tanh(this->torch_input / 3.0);
        this->torch_input = scaled;
        this->fillTensorFromTorch(this->input, this->torch_input);
        auto torch_output = torch::tanh(this->torch_input);
        this->fillTensorFromTorch(this->output, torch_output);
    }
};
} // namespace

TEST(TanhTest, TanhComprehensiveTest) {
    std::vector<std::tuple<std::vector<int>>> test_cases = {
        {{1, 3, 64, 64}},
        {{1, 8, 16, 16}},
        {{3, 64, 64}},        // 3D（无 batch 维）
        {{1, 1, 224, 224}},   // 单 channel 大图，贴近 visual block mlp 维度
    };
    for (const auto &test_case : test_cases) {
        auto [shape] = test_case;
        TanhTest<float> tanhtest(shape);
        EXPECT_TRUE(tanhtest.run_test());

        TanhTest<uint16_t> tanhtest16(shape);
        EXPECT_TRUE(tanhtest16.run_test());
    }
}
