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
    std::vector<int> shapea_;
    std::vector<int> shapeb_;
    at::Tensor torch_inputa;
    at::Tensor torch_inputb;

    BinaryTest(const std::string& name, std::vector<int> shape): TestCase<T>(name), shapea_(std::move(shape)) {
        inputa = std::make_shared<Tensor<T>>(shapea_);
        inputb = std::make_shared<Tensor<T>>(shapea_);
        output = std::make_shared<Tensor<T>>(shapea_);

        std::vector<int64_t> inshape(shapea_.begin(), shapea_.end());
        torch_inputa = torch::randn(inshape, this->getTorchConf());
        this->fillTensorFromTorch(inputa, torch_inputa);
        torch_inputb = torch::randn(inshape, this->getTorchConf());
        this->fillTensorFromTorch(inputb, torch_inputb);
    }
    std::vector<int> computeBroadcastShape(const std::vector<int>& shape1, const std::vector<int>& shape2) {
        size_t max_dims = std::max(shape1.size(), shape2.size());
        std::vector<int> result_shape(max_dims, 1);

        // 从后往前填充形状
        for (int i = max_dims - 1; i >= 0; --i) {
            int idx1 = i - (max_dims - shape1.size());
            int idx2 = i - (max_dims - shape2.size());
            
            int dim1 = (idx1 >= 0) ? shape1[idx1] : 1;
            int dim2 = (idx2 >= 0) ? shape2[idx2] : 1;

            // 广播规则：如果任一维度为1或两维度相等，则可广播
            if (dim1 == 1 || dim2 == 1) {
                result_shape[i] = std::max(dim1, dim2);
            } else if (dim1 == dim2) {
                result_shape[i] = dim1;
            } else {
                throw std::runtime_error("Shapes are not broadcast-compatible");
            }
        }
        
        return result_shape;
    }
    BinaryTest(const std::string& name, std::vector<int> shapea, std::vector<int> shapeb): TestCase<T>(name), shapea_(std::move(shapea)), shapeb_(std::move(shapeb)) {
        inputa = std::make_shared<Tensor<T>>(shapea_);
        inputb = std::make_shared<Tensor<T>>(shapeb_);

        std::vector<int64_t> inshapea(shapea_.begin(), shapea_.end());
        torch_inputa = torch::randn(inshapea, this->getTorchConf());
        this->fillTensorFromTorch(inputa, torch_inputa);
        std::vector<int64_t> inshapeb(shapeb_.begin(), shapeb_.end());
        torch_inputb = torch::randn(inshapeb, this->getTorchConf());
        this->fillTensorFromTorch(inputb, torch_inputb);

        auto broadcast_shape = computeBroadcastShape(shapea_, shapeb_);
        output = std::make_shared<Tensor<T>>(broadcast_shape);
    }
    
    virtual bool run_test() {
        return TestCase<T>::run_test({inputa, inputb}, {output});
    }
};

} // namespace tests
} // namespace vkop

#endif