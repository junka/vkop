// Copyright 2025 @junka

#include <string>
#include <unordered_map>
#include <vector>

#include "core/Tensor.hpp"
#include "ops/Operator.hpp"

namespace vkop {
namespace core {

class Runtime {
  private:
    int precision_ = 0; // 0: fp32, 1: fp16
    std::shared_ptr<VulkanDevice> m_dev_;
    std::shared_ptr<VulkanCommandPool> m_cmdpool_;
    // Model file path
    std::string model_path_;

    // Cache directory path
    std::string cache_dir_;

    // Input and output tensors mapping by name
    std::unordered_map<std::string, std::shared_ptr<ITensor>> inputs_;
    std::unordered_map<std::string, std::shared_ptr<ITensor>> outputs_;
    // Initializer tensors
    std::unordered_map<std::string, std::shared_ptr<ITensor>> initializers_;

    // Tensor pointers for each node's inputs and outputs
    std::vector<std::unique_ptr<vkop::ops::Operator>> node_ops_;
    std::vector<std::vector<std::shared_ptr<ITensor>>> node_input_tensors_;
    std::vector<std::vector<std::shared_ptr<ITensor>>> node_output_tensors_;

  public:
    // Constructor
    explicit Runtime(std::shared_ptr<VulkanDevice> dev,
                     std::shared_ptr<VulkanCommandPool> cmdpool,
                     const std::string &model_path,
                     const std::string &cache_dir = "");
    ~Runtime() = default;

    // Load cache if available
    void LoadCache();

    void LoadModel();

    // Get input tensor by name
    std::shared_ptr<ITensor> GetInput(const std::string &name);

    // Get output tensor by name
    std::shared_ptr<ITensor> GetOutput(const std::string &name);

    // Get initializer tensor by name, for test only
    std::shared_ptr<ITensor> GetInitializer(const std::string &name);

    void Run();

    void ReadResult();

    void SetPrecision(int val) { precision_ = val; }
    int GetPrecision() const { return precision_; }
};

} // namespace core
} // namespace vkop