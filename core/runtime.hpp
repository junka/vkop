// Copyright 2025 @junka
#ifndef CORE_RUNTIME_HPP_
#define CORE_RUNTIME_HPP_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/ShaderArchive.hpp"
#include "core/Tensor.hpp"
#include "ops/Operator.hpp"

namespace vkop {
namespace core {

class Runtime {
  private:
    // Graph precision for the tensors this Runtime allocates (outputs at load
    // time, plus the fallback for ops whose own dtypes are not informative).
    // The fp16/fp32 shader module is picked per op from the tensor dtype, so
    // this is not a kernel-selection flag; drivers running a half model pass 1.
    int precision_ = 0; // 0: fp32, 1: fp16
    // When true, ops are built on the SSBO buffer backend (compact
    // row-major tensors of arbitrary rank) instead of the image backend
    // (NCHW->RGBA). Each op's PIMPL façade selects its BufferImpl when this
    // is set (if a buffer port exists) or falls back to its ImageImpl.
    bool backend_buffer_ = false;
    std::shared_ptr<VulkanCommandPool> m_cmdpool_;

    std::vector<std::vector<size_t>> level_node_indices_;
    // Per-level op-type sequence, populated only when VKOP_DUMP_LEVEL_SEQ is
    // set (build-time chain-structure analysis for kernel-fusion planning).
    std::vector<std::vector<std::string>> level_op_seq_;

    // Per-level "contains a synchronous readback" flag, learned on round 0
    // (via the queue submit counter advancing during onExecute) and reused to
    // attribute the ~0.43ms forced per-level submit floor. A readback level's
    // copyToCPU does its own cmd.submit+wait on queue0, relying on single-queue
    // FIFO ordering (its producer levels already queued). Stable across decode
    // rounds (an op's readback behavior is determined by its dtype/path, not
    // the round). Printed by VKOP_BATCH_DBG; per-op-type counts by VKOP_OPPROF
    // (rbprof). Batching (VKOP_BATCH_LEVELS) was tried and found net-neutral
    // (readback levels can't batch with pending producers); see
    // readback-per-level-submit-bottleneck.md. The fix is GPU-driven shape-meta
    // (eliminate the readbacks), not batching.
    std::vector<bool> level_readback_;

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

    // Per-node, per-input: whether the input tensor's ELEMENT VALUES vary
    // across decode rounds (converter annotation, load::Shape::value_dynamic).
    // Used by shape-consuming ops (Reshape inputs[1]) to cache readback results
    // across rounds and skip the per-round copyToCPU stall. Only meaningful for
    // int64 shape-meta inputs; false for all others (no cache attempted).
    std::vector<std::vector<bool>> node_input_value_dynamic_;

    // Phase 4-r2 shape-chain host-ification: per-node parsed term list from
    // the converter's `vkop_host_shape` Reshape attribute. Each term is
    // either a constant dim or "the dim_idx-th dim of tensor src". The
    // runtime evaluates the term list from LIVE tensor metadata right before
    // the Reshape executes each round — lazy, because a REF term may
    // reference a tensor produced earlier in the SAME round whose dims are
    // only final once its producer has run — and pushes the result via
    // Operator::set_host_target_shape. Empty list = legacy two-input Reshape
    // (target still read from a GPU int64 chain).
    struct HostShapeTerm {
        bool is_const = false;
        int cval = 0;
        int dim_idx = 0;
        std::shared_ptr<ITensor> src;
    };
    std::vector<std::vector<HostShapeTerm>> node_host_shapes_;

    // Evaluate node_host_shapes_[node_idx] (if any) from live tensor shapes
    // and push the result to the op (set_host_target_shape). No-op for
    // legacy two-input Reshapes (empty term list).
    void eval_host_target_shape(size_t node_idx);

    // Record-once-replay (cuda-graph-style): per-node snapshot of the LIVE
    // input shapes (tensor->getShape()) from the round that earned CACHED. On
    // a later round, if any input's live shape differs, the op is dynamic for
    // that round → force_replay_refresh() (re-record). This is the correctness
    // guard that prevents a stale replay when kv_len grows. Empty entry = no
    // snapshot yet (first round). See KV_INPLACE_PLAN.md Step 4.
    std::vector<std::vector<std::vector<int>>> replay_live_shapes_;
    bool replay_mode_ = false;    // set by VKOP_REPLAY=1
    int replay_cached_count_ = 0; // ops that replayed (CACHED) this Run()
    bool replay_dbg_ = false;     // set by VKOP_REPLAY=2 (log dynamic ops)

    // GPU submit-side profiling (VKOP_SUBMIT_PROF=1). A timestamp query pool
    // with 2 queries per op (begin/end); results read back at Run() end and
    // aggregated per op-type to attribute the submit floor. opprof only
    // measures CPU record time; this measures actual GPU execution.
    VkQueryPool submit_prof_pool_ = VK_NULL_HANDLE;
    bool submit_prof_ = false;
    float timestamp_period_ = 1.0f;
    // Per-node GPU timings from the last profiled Run(), filled when
    // submit_prof_ is on. A driver that wants the same data VKOP_SUBMIT_PROF
    // prints to stderr (but as a structured vector for a JSON dump or a
    // table view) calls enable_submit_profiling() before Run() and reads
    // last_run_timings_ after.
    struct NodeTiming {
        std::string op_type;
        std::string name;
        int level;
        double ms;
    };
    std::vector<NodeTiming> last_run_timings_;

    // Vulkan pipeline cache (VK_PIPELINE_CACHE=1). Serialises compiled
    // VkPipeline binaries to disk so a subsequent LoadModel on the same device
    // can skip per-shader driver compilation. The cache is device-specific:
    // Vulkan's pipeline cache header embeds the device UUID, vendor ID, and
    // driver info, so a mismatched cache is silently discarded by the driver
    // (the cache simply misses, and pipelines rebuild from SPIR-V).
    VkPipelineCache pipeline_cache_ = VK_NULL_HANDLE;
    bool pipeline_cache_enabled_ = false;
    std::string pipeline_cache_path_;

    void LoadPipelineCache();
    void SavePipelineCache();

    // SPIR-V source for this model: the archive the .vkopbin itself carries.
    // onnx2vkop always embeds one, and libvkop is stripped by default, so this
    // is the only source a shipping binary has -- there is deliberately no
    // environment override, which could otherwise substitute different kernels
    // at run time. Null means the model has no archive: every op then keeps
    // its placeholder array and fails by name at build time. The archive is
    // copied out of the model mmap, so only the shader blobs a model actually
    // fires become resident: the trtexec-style "extract only what's needed"
    // property that avoids baking every libvkop shader into every driver.
    std::unique_ptr<ShaderArchive> shader_archive_;

    // The kernel selection the .vkopbin records (Model.kernel_plans), copied
    // out of the transient load::VkModel because ops outlive it. Empty for a
    // file written before the field existed: LoadModel then derives each key
    // from the built-in table in core/ShaderKeys.hpp exactly as it always did.
    std::vector<vkop::core::KernelPlan> kernel_plans_;

    // The key this node's op should be looked up under: the plan's answer when
    // the file carries one, else the built-in OpType table. `plan` is set to
    // the row used (nullptr when the fallback ran) for diagnostics.
    std::string kernel_key_for_node(const load::Node &n, vkop::ops::OpType type,
                                    int op_fp16,
                                    const vkop::core::KernelPlan **plan) const;

  public:
    // Entry count of the loaded archive, 0 when there is none. The bench
    // reports this so an A/B can tell whether the model actually carried one.
    size_t shader_archive_entries() const {
        return shader_archive_ ? shader_archive_->entry_count() : 0;
    }

    // Drop every op's cached recording (back to FRESH) so the next Run()
    // re-records all. Call across a phase boundary where shapes change en masse
    // (e.g. prefill q_len=L → decode q_len=1): partial replay across such a
    // boundary leaves mid-graph tensors at stale shapes and corrupts
    // downstream.
    void invalidate_replay() {
        // Shape-input readback caches (option C) must drop across a phase
        // boundary regardless of replay mode: a value_dynamic=false shape
        // tensor is round-invariant WITHIN a phase, but prefill→decode changes
        // shapes en masse. Cheap (virtual no-op on non-Reshape ops).
        for (auto &op : node_ops_)
            op->invalidate_shape_cache();
        // The per-level readback map is phase-specific (prefill vs decode run
        // different op paths); re-learn it on the first Run() of the new phase.
        level_readback_.clear();
        if (!replay_mode_)
            return;
        for (auto &op : node_ops_)
            op->force_replay_refresh();
        for (auto &s : replay_live_shapes_)
            s.clear();
    }

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

    // Names of every graph input, in the order the vkopbin declared them.
    // Lets a caller (vkop-bench, a debug dumper) iterate all inputs by name
    // without parsing the .vkopbin a second time via load::VkModel.
    std::vector<std::string> ListInputNames() const;

    // Counterpart to ListInputNames for outputs. Same sorted-for-
    // determinism contract.
    std::vector<std::string> ListOutputNames() const;

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

    // List all named tensors (inputs, initializers, node outputs). For
    // driver-side dump-all diagnostics comparing against ORT intermediates.
    std::vector<std::pair<std::string, std::shared_ptr<ITensor>>>
    ListTensors() const;

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

    // Turn on per-op GPU timestamp profiling programmatically. Same path
    // VKOP_SUBMIT_PROF=1 takes; setting the flag here before the first
    // Run() gets the query pool allocated without needing an env var, so a
    // driver can produce a --profile-out JSON dump on demand. Idempotent;
    // the pool survives across Run()s and each call overwrites the last
    // results, so a consumer can read timings from any run.
    void enable_submit_profiling() { submit_prof_ = true; }
    bool submit_profiling_enabled() const { return submit_prof_; }
    const std::vector<NodeTiming> &last_run_timings() const {
        return last_run_timings_;
    }
    // Nanoseconds per Vulkan timestamp tick on this device (from
    // VkPhysicalDeviceLimits::timestampPeriod). Only meaningful once
    // enable_submit_profiling() has been set and Run() has captured
    // timestamps; before that it is the vkop default 1.0f.
    float timestamp_period_ns() const { return timestamp_period_; }

    void RegisterPostProcess(
        ops::OpType ops,
        const std::unordered_map<std::string, std::string> &attributes,
        const std::vector<std::shared_ptr<ITensor>> &inputs,
        const std::vector<std::shared_ptr<ITensor>> &outputs);
};

} // namespace core
} // namespace vkop

#endif // CORE_RUNTIME_HPP_
