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

class TopkTest : public TestCase {
public:
    std::shared_ptr<Tensor<float>> input;
    std::shared_ptr<Tensor<int>> indexs;
    std::shared_ptr<Tensor<float>> output;
    int k = 7;
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
            2, 100
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
        for (int i = 0; i < input->num_elements(); i++) {
            printf("input %f\n", (*input)[i]);
        }
        std::vector<std::vector<float>> probs;
        int rows = t[0];
        if (axis == 0) {
            rows = 1;
        }
        for (int i = 0; i < rows; i++) {
            std::vector<float> probs_i(t[axis]);
            for (int j = 0; j < t[axis]; j++) {
                probs_i[j]= ((*input)[(i * t[axis]) + j]);
            }
            probs.push_back(probs_i);
        }
        std::vector<std::vector<std::pair<int, float>>> ret;    
        ret = get_top_k_predictions(probs, k);

        printf("output: %ld\n", ret.size());
        for (size_t c = 0; c < ret.size(); c++) {
            for (size_t i = 0; i < ret[c].size(); i++) {
                (*output)[(c*k) + i] = ret[c][i].second;
                (*indexs)[(c*k) + i] = ret[c][i].first;
                // printf("index: %d, value: %f\n", ret[c][i].first, ret[c][i].second);
            }
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