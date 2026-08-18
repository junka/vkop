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
#ifdef FP16
    int precision_ = 1;
#else
    int precision_ = 0; // 0: fp32, 1: fp16
#endif
    // When true, ops are built on the SSBO buffer backend (compact
    // row-major tensors of arbitrary rank) instead of the image backend
    // (NCHW->RGBA). Each op's PIMPL façade selects its BufferImpl when this
    // is set (if a buffer port exists) or falls back to its ImageImpl.
    bool backend_buffer_ = false;
    std::shared_ptr<VulkanCommandPool> m_cmdpool_;

    std::vector<std::vector<size_t>> level_node_indices_;

    // Model file path
    std::string model_path_;

    // Cache directory path
    std::string cache_dir_;

    // Input and output tensors mapping by name
    std::unordered_map<std::string, std::shared_ptr<ITensor>> inputs_;
    std::unordered_map<std::string, std::shared_ptr<ITensor>> outputs_;
    // Persistent view of every named tensor (inputs, initializers, and every
    // node output) for post-Run inspection by name (e.g. dumping intermediates
    // to diagnose NaN propagation). Mirrors the local tensor_map in LoadModel.
    std::unordered_map<std::string, std::shared_ptr<ITensor>> tensor_map_;
    std::unordered_map<std::string, std::shared_ptr<ITensor>> real_outputs_;
    // Initializer tensors
    std::unordered_map<std::string, std::shared_ptr<ITensor>> initializers_;

    // Tensor pointers for each node's inputs and outputs
    std::vector<std::unique_ptr<vkop::ops::Operator>> node_ops_;
    std::vector<std::vector<int>> node_dependency_indices_;
    std::vector<std::unordered_map<std::string, std::string>> node_attrs_;
    std::vector<std::vector<std::shared_ptr<ITensor>>> node_input_tensors_;
    std::vector<std::vector<std::shared_ptr<ITensor>>> node_output_tensors_;
    // Per-node recorded input shapes (from the vkopbin ShapeRef dims). Applied
    // as a pure-logical reshape_view at execute time so each consumer sees the
    // view the converter recorded for IT (e.g. an Unsqueeze fold records a 5-D
    // view [1,8,1,1,128] on the Expand consumer, but the producing Concat
    // leaves the shared tensor 4-D [1,8,1,128]). Resolving at LoadModel time
    // would race: multiple consumers sharing one tensor each reshape it, and
    // the last writer wins — so we store the shapes and apply them right
    // before onExecute in execution order.
    std::vector<std::vector<std::vector<int>>> node_input_shapes_;

  public:
    // Constructor
    Runtime(const std::shared_ptr<VulkanCommandPool> &cmdpool,
            std::string model_path, int precision, std::string cache_dir = "");
    Runtime(const std::shared_ptr<VulkanCommandPool> &cmdpool,
            std::string model_path, std::string cache_dir = "");

    ~Runtime();

    // Load cache if available
    void LoadCache();

    void LoadModel();

    // Get input tensor by name
    std::shared_ptr<ITensor> GetInput(const std::string &name = "") const;

    // Resize a graph input to the caller's actual shape and recreate its SSBO
    // at the new size. Used by the LLM driver: the model records symbolic
    // dims (e.g. past_key_values kv_len=1) but the real prefill tensor has a
    // concrete shape (kv_len=0). Must be called after LoadModel, before Run.
    void ResizeInput(const std::string &name,
                     const std::vector<uint32_t> &dims);

    // Get output tensor by name
    std::shared_ptr<ITensor> GetOutput(const std::string &name = "") const;

    // Get any named tensor (input, initializer, or node output) by name.
    // Returns nullptr if not found. For inspecting intermediates after Run.
    std::shared_ptr<ITensor> GetTensor(const std::string &name) const;

    // Get initializer tensor by name, for test only
    std::shared_ptr<ITensor> GetInitializer(const std::string &name) const;

    // should be called before loading model
    void TraceNode(const std::string &name);

    double Run();

    void ReadResult();

    void setPrecision(int precision) { precision_ = precision; }

    int getPrecision() const { return precision_; }

    // Select the buffer (SSBO) backend for subsequently-loaded operators.
    void set_backend_buffer(bool b) { backend_buffer_ = b; }
    bool get_backend_buffer() const { return backend_buffer_; }

    void RegisterPostProcess(
        ops::OpType ops,
        const std::unordered_map<std::string, std::string> &attributes,
        const std::vector<std::shared_ptr<ITensor>> &inputs,
        const std::vector<std::shared_ptr<ITensor>> &outputs);
};

} // namespace core
} // namespace vkop