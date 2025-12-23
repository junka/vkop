#include <vector>
#include <random>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "ops/Topk.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;
using vkop::ops::Topk;

namespace {
std::vector<std::pair<int, float>> get_top_k_predictions(const std::vector<float>& probs, int k) {
    std::vector<float> softmax_probs = probs;

    // float max_val = *std::max_element(softmax_probs.begin(), softmax_probs.end());
    // float sum = 0.0F;
    // for (auto& val : softmax_probs) {
    //     val = std::exp(val - max_val);
    //     sum += val;
    // }

    // for (auto& val : softmax_probs) {
    //     val /= sum;
    // }

    std::vector<std::pair<int, float>> indexed_probs;
    indexed_probs.reserve(softmax_probs.size());
    for (size_t i = 0; i < softmax_probs.size(); ++i) {
        indexed_probs.emplace_back(i, softmax_probs[i]);
    }

    std::sort(indexed_probs.begin(), indexed_probs.end(),
              [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                  return a.second > b.second;
              });

    if (indexed_probs.size() > static_cast<size_t>(k)) {
        indexed_probs.resize(k);
    }

    return indexed_probs;
}

class TopkTest : public TestCase {
public:
    std::shared_ptr<Tensor<float>> input;
    std::shared_ptr<Tensor<int>> indexs;
    std::shared_ptr<Tensor<float>> output;
    int k = 5;
    int axis = -1;
    std::unordered_map<std::string, std::string> attrs = {
        {"k", std::to_string(k)},
        {"axis", std::to_string(axis)},
        {"largest", "1"},
        {"sorted", "1"}
    };

    TopkTest():TestCase("TopK") {
        initTestdata();
    }
private:
    void initTestdata()
    {
        std::vector<int> t = {
            400
        };
        auto out_shape = t;
        input = std::make_shared<Tensor<float>>(t);
        axis = axis < 0 ? input->num_dims() + axis : axis;
        out_shape[axis] = k;
        indexs = std::make_shared<Tensor<int>>(out_shape);
        output = std::make_shared<Tensor<float>>(out_shape);
        input->reserveOnCPU();
        indexs->reserveOnCPU();
        output->reserveOnCPU();

        std::random_device rd{};
        std::mt19937 gen{rd()};
        gen.seed(1024);
        std::normal_distribution<> input_dist{0.0F, 1.0F};
        for (int i = 0; i < input->num_elements(); i++) {
            (*input)[i] = input_dist(gen);
        }
        auto ret = get_top_k_predictions(input->data(), k);
        for (size_t i = 0; i < ret.size(); i++) {
            (*output)[i] = ret[i].second;
            (*indexs)[i] = ret[i].first;
            printf("index: %d, value: %f\n", ret[i].first, ret[i].second);
        }
    }
};
}

int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", false);

    TopkTest toptest;
    if (!toptest.run_test<float>({toptest.input}, {toptest.indexs, toptest.output},[&toptest](std::unique_ptr<vkop::ops::Operator> &op) {
        auto *topk_op = dynamic_cast<Topk *>(op.get());
        if (!topk_op) {
            LOG_ERROR("Failed to cast operator to Topk");
            return;
        }
        topk_op->setAttribute(toptest.attrs);
    })) {
        return -1;
    }

    return 0;
}