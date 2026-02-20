#ifndef BINARY_TEST_HPP
#define BINARY_TEST_HPP

#include "setup.hpp"

namespace vkop {
namespace tests {

template <typename T>
class BinaryTest : public TestCase<T> {
public:
    std::shared_ptr<Tensor<T>> inputa;
    std::shared_ptr<Tensor<T>> inputb;
    std::shared_ptr<Tensor<T>> output;
    std::vector<int> shape_;
    at::Tensor torch_inputa;
    at::Tensor torch_inputb;

    BinaryTest(const std::string& name, std::vector<int> shape): TestCase<T>(name), shape_(std::move(shape)) {
        inputa = std::make_shared<Tensor<T>>(shape_);
        inputb = std::make_shared<Tensor<T>>(shape_);
        output = std::make_shared<Tensor<T>>(shape_);

        std::vector<int64_t> inshape(shape_.begin(), shape_.end());
        torch_inputa = torch::randn(inshape, this->getTorchConf());
        this->fillTensorFromTorch(inputa, torch_inputa);
        torch_inputb = torch::randn(inshape, this->getTorchConf());
        this->fillTensorFromTorch(inputb, torch_inputb);
    }
    
    virtual bool run_test() {
        return TestCase<T>::run_test({inputa, inputb}, {output});
    }
};

} // namespace tests
} // namespace vkop

#endif