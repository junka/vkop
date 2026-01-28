// Copyright 2025 @junka

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/Tensor.hpp"
#include "ops/Operator.hpp"

namespace vkop {
namespace core {

class Runtime {
  private:
    int id_ = 0;
    std::shared_ptr<VulkanCommandBuffer> m_cmds_[vkop::kInflight];
    std::shared_ptr<VulkanCommandPool> m_cmdpool_;
    // Model file path
    std::string model_path_;

    // Cache directory path
    std::string cache_dir_;

    // Input and output tensors mapping by name
    std::unordered_map<std::string, std::shared_ptr<ITensor>> inputs_;
    std::unordered_map<std::string, std::shared_ptr<ITensor>> outputs_;
    std::unordered_map<std::string, std::shared_ptr<ITensor>> real_outputs_;
    // Initializer tensors
    std::unordered_map<std::string, std::shared_ptr<ITensor>> initializers_;

    // Tensor pointers for each node's inputs and outputs
    std::vector<std::unique_ptr<vkop::ops::Operator>> node_ops_;
    std::vector<std::unordered_map<std::string, std::string>> node_attrs_;
    std::vector<std::vector<std::shared_ptr<ITensor>>> node_input_tensors_;
    std::vector<std::vector<std::shared_ptr<ITensor>>> node_output_tensors_;

  public:
    // Constructor
    explicit Runtime(const std::shared_ptr<VulkanCommandPool> &cmdpool,
                     std::string model_path, std::string cache_dir = "");
    ~Runtime();

    // Load cache if available
    void LoadCache();

    void LoadModel();

    // Get input tensor by name
    std::shared_ptr<ITensor> GetInput(const std::string &name = "") const;

    // Get output tensor by name
    std::shared_ptr<ITensor> GetOutput(const std::string &name = "") const;

    // Get initializer tensor by name, for test only
    std::shared_ptr<ITensor> GetInitializer(const std::string &name) const;

    // should be called before loading model
    void TraceNode(const std::string &name);

    double Run();

    void ReadResult();

    void RegisterPostProcess(
        ops::OpType ops,
        const std::unordered_map<std::string, std::string> &attributes,
        const std::vector<std::shared_ptr<ITensor>> &inputs,
        const std::vector<std::shared_ptr<ITensor>> &outputs);
};

} // namespace core
} // namespace vkop