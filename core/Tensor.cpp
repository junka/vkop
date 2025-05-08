#include "Tensor.hpp" 
#include <iostream>

namespace vkop {

Tensor::Tensor() {

}

Tensor::Tensor(int n, int c, int h, int w): n(n), c(c), h(h), w(w)
{
    
}

Tensor::Tensor(std::vector<int> dims): n(dims[0]), c(dims[1]), h(dims[2]), w(dims[3])
{

}

Tensor::Tensor(std::vector<uint32_t> dims): n(dims[0]), c(dims[1]), h(dims[2]), w(dims[3])
{

}

void Tensor::printTensor()
{
    std::cout << n << "," << c << "," << h << "," << w << std::endl;
}

}