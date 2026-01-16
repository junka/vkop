// Copyright 2025 @junka
#ifndef CORE_FUNCTION_HPP_
#define CORE_FUNCTION_HPP_
#include "core/Tensor.hpp"

namespace vkop {
namespace core {

enum class NormMethod {
    IMAGENET,
    INCEPTION,
    CLIP,
    DEFAULT,
};

class Function {
  public:
    Function();
    ~Function();

    /**
     * @brief Deleted copy constructor to prevent copying of tensor objects.
     */
    Function(const Function &) = delete;
    /**
     * @brief Deleted move constructor to prevent moving of tensor objects.
     */
    Function(const Function &&) = delete;
    /**
     * @brief Deleted copy assignment operator to prevent copying of tensor
     * objects.
     */
    Function &operator=(const Function &) = delete;

    /**
     * @brief Deleted move assignment operator to prevent moving of tensor
     * objects.
     */
    Function &operator=(const Function &&) = delete;

    static void
    preprocess_jpg(const char *input_file,
                   const std::shared_ptr<VulkanCommandPool> &cmdpool,
                   const std::shared_ptr<core::ITensor> &input,
                   NormMethod method = NormMethod::DEFAULT);

    static std::vector<std::pair<int, float>>
    get_top_k_predictions(const std::vector<float> &probs, int k);
};

} // namespace core
} // namespace vkop

#endif // CORE_FUNCTION_HPP_
