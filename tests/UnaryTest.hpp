#ifndef UNARY_TEST_HPP
#define UNARY_TEST_HPP

#include "setup.hpp"

namespace vkop {
namespace tests {

template <typename T>
class UnaryTest : public TestCase<T> {
public:
    std::shared_ptr<Tensor<T>> input;
    std::shared_ptr<Tensor<T>> output;
    std::vector<int> shape_;
    at::Tensor torch_input;

    UnaryTest(const std::string& name, std::vector<int> shape):TestCase<T>(name), shape_(std::move(shape)) {
        input = std::make_shared<Tensor<T>>(shape_);
        output = std::make_shared<Tensor<T>>(shape_);

        std::vector<int64_t> inshape(shape_.begin(), shape_.end());
        torch_input = torch::randn(inshape, this->getTorchConf());
        this->fillTensorFromTorch(input, torch_input);
    }

    virtual bool run_test() {
        return TestCase<T>::run_test({input}, {output});
    }
};

} // namespace tests
} // namespace vkop

#endif