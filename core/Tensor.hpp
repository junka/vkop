// Copyright 2025 @junka
#ifndef CORE_TENSOR_HPP_
#define CORE_TENSOR_HPP_

#include <cstdint>
#include <vector>

namespace vkop {

class Tensor {
  public:
    // empty
    Tensor();

    // nchw
    Tensor(int n, int c, int h, int w);

    // nchw in vector
    explicit Tensor(std::vector<int> dims);

    // nchw in vector
    explicit Tensor(std::vector<uint32_t> dims);

    void printTensor() const;

    void *map();
    void unmap();

    /**
     * @brief Deleted copy constructor to prevent copying of tensor objects.
     */
    Tensor(const Tensor &tensor) = delete;
    /**
     * @brief Deleted move constructor to prevent moving of tensor objects.
     */
    Tensor(const Tensor &&tensor) = delete;
    /**
     * @brief Deleted copy assignment operator to prevent copying of tensor
     * objects.
     */
    Tensor &operator=(const Tensor &) = delete;

    /**
     * @brief Deleted move assignment operator to prevent moving of tensor
     * objects.
     */
    Tensor &operator=(const Tensor &&) = delete;

  private:
    int n_;
    int c_;
    int h_;
    int w_;

    int ele_size_;
    int size_;

    void *vulkanobj_;
};

} // namespace vkop

#endif // CORE_TENSOR_HPP_
