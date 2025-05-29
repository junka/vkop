// Copyright 2025 @junka
#include "Tensor.hpp"
#include <iostream>

namespace vkop {

Tensor::Tensor() = default;

Tensor::Tensor(int n, int c, int h, int w) : n_(n), c_(c), h_(h), w_(w) {
    ele_size_ = 4;
    size_ = ele_size_ * n * c * h * w;
}

Tensor::Tensor(std::vector<int> dims)
    : n_(dims[0]), c_(dims[1]), h_(dims[2]), w_(dims[3]) {}

Tensor::Tensor(std::vector<uint32_t> dims)
    : n_(dims[0]), c_(dims[1]), h_(dims[2]), w_(dims[3]) {}

void Tensor::printTensor() const {
    std::cout << n_ << "," << c_ << "," << h_ << "," << w_ << std::endl;
}

void *Tensor::map() { return vulkanobj_; }

void Tensor::unmap() { (void)vulkanobj_; }

} // namespace vkop
