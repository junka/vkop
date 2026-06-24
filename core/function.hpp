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

    static std::pair<int, int>
    preprocess_jpg(const char *input_file,
                   const std::shared_ptr<VulkanCommandPool> &cmdpool,
                   const std::shared_ptr<core::ITensor> &input,
                   bool use_letterbox = false,
                   NormMethod method = NormMethod::DEFAULT);

    /**
     * @brief load .npy directly so we could get rid of jpg decoding
     *        resize and normalize
     *        for debugging use
     *
     * npy format as NCHW float32 normalized, shape [1,C,H,W] or [C,H,W]
     * input 张量的 dtype 重新打包成 RGBA（fp16 输入会转 fp16），
     * 复用 preprocess_jpg 的 copyToGPUImage(rgba=true) 上传路径。
     *
     */
    static bool
    preprocess_npy(const std::string &npy_path,
                   const std::shared_ptr<VulkanCommandPool> &cmdpool,
                   const std::shared_ptr<core::ITensor> &input);

    static std::vector<std::pair<int, float>>
    get_top_k_predictions(const std::vector<float> &probs, int k);
};

} // namespace core
} // namespace vkop

#endif // CORE_FUNCTION_HPP_
