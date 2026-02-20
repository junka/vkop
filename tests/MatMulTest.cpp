#include <vector>
#include <cmath>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;

#ifdef USE_CPP_REF
static void reference_matmul(const std::shared_ptr<Tensor<float>> &inputa, const std::shared_ptr<Tensor<float>> &inputb, std::shared_ptr<Tensor<float>> &output) {
    int M = inputa->get_height(); // A: [..., M, K]
    int K = inputa->get_width();
    int N = inputb->get_width();
    auto batch = inputa->get_batch();
    auto chan = inputa->get_channel();
    printf("M: %d, K: %d, N: %d\n", M, K, N);

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < chan; c++) {
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    float sum = 0.0F;
                    size_t idxc = (b * chan * M * N) + (c * M * N) + (i * N) + j;
                    for (int k = 0; k < K; k++) {
                        // M * k, K * N
                        size_t idxa = (b * chan * M * K) + (c * M * K) + (i * K) + k;
                        size_t idxb = (b * chan * K * N) + (c * K * N) + (k * N) + j;
                        sum += (*inputa)[idxa] * (*inputb)[idxb];
                    }
                    (*output)[idxc] = sum;
                }
            }
        }
    }
}
#endif
namespace {

template<typename T>
class MatMulTest : public TestCase<T> {
public:
    std::vector<int> t1;
    std::vector<int> t2;
    std::shared_ptr<Tensor<T>> inputa;
    std::shared_ptr<Tensor<T>> inputb;
    std::shared_ptr<Tensor<T>> output;

    MatMulTest(const std::vector<int> &t1, const std::vector<int> &t2) : TestCase<T>("MatMul"), t1(t1), t2(t2) {
        initTestdata();
    }

private:
    void initTestdata() {
        inputa = std::make_shared<Tensor<T>>(t1);
        inputb = std::make_shared<Tensor<T>>(t2);
        size_t rank = t1.size();
        std::vector<int> to = {3, t1[rank-2], t2[rank-1]};
        int kk = t1[rank-1];
        output = std::make_shared<Tensor<T>>(to);
        std::vector<int64_t> t1shape(t1.begin(), t1.end());
        std::vector<int64_t> t2shape(t2.begin(), t2.end());

        auto torch_in1 = torch::randn(t1shape, this->getTorchConf());
        auto torch_in2 = torch::randn(t2shape, this->getTorchConf());

        auto torch_output = torch::matmul(torch_in1, torch_in2);
        this->fillTensorFromTorch(inputa, torch_in1);
        this->fillTensorFromTorch(inputb, torch_in2);
        this->fillTensorFromTorch(output, torch_output);

        printf("M %d, N %d, K %d\n", to[rank-2], to[rank-1], kk);
        printf("==============================================================\n");
        printf("Input A:\n");
        auto shapea = inputa->getShape();
        printf("%d %d %d %d\n", shapea[0], shapea[1], shapea[2], shapea[3]);
        inputa->print_tensor();
        printf("Input B:\n");
        auto shapeb = inputb->getShape();
        inputb->print_tensor();
#if 0
        reference_matmul(inputa, inputb, output);
#endif
        printf("Output:\n");
        output->print_tensor();
    }
};
}

TEST(MatMulTest, MatMulComprehensiveTest) {
    const std::vector<std::tuple<std::vector<int>, std::vector<int>>> test_cases = {
        {{3, 4, 1}, {3, 1, 6}},
    };
    for (const auto &test_case : test_cases) {
        auto [t1, t2] = test_case;
        MatMulTest<float> mmtest(t1, t2);
        EXPECT_TRUE(mmtest.run_test({mmtest.inputa, mmtest.inputb}, {mmtest.output}));

        MatMulTest<uint16_t> mmtest1(t1, t2);
        EXPECT_TRUE(mmtest1.run_test({mmtest1.inputa, mmtest1.inputb}, {mmtest1.output}));
    }
}