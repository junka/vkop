// junka @ 2025
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <queue>

#include "core/runtime.hpp"
#include "model/load.hpp"
#include "ops/OperatorFactory.hpp"
#include "vulkan/VulkanCommandBuffer.hpp"
#include "vulkan/VulkanDevice.hpp"

namespace vkop {
std::shared_ptr<VulkanBuffer> ops::Operator::dummy_buffer_ = nullptr;
std::shared_ptr<VulkanBufferView> ops::Operator::dummy_bufferview_ = nullptr;
std::atomic<int> ops::Operator::instance_count_{0};
namespace core {

Runtime::Runtime(const std::shared_ptr<VulkanCommandPool> &cmdpool,
                 std::string model_path, int precision, std::string cache_dir)
    : precision_(precision), m_cmdpool_(std::move(cmdpool)),
      model_path_(std::move(model_path)), cache_dir_(std::move(cache_dir)) {}

Runtime::Runtime(const std::shared_ptr<VulkanCommandPool> &cmdpool,
                 std::string model_path, std::string cache_dir)
    : m_cmdpool_(std::move(cmdpool)), model_path_(std::move(model_path)),
      cache_dir_(std::move(cache_dir)) {}

Runtime::~Runtime() = default;

void Runtime::LoadCache() {}

void Runtime::LoadModel() {
    auto model = load::VkModel(model_path_);
    model.dump_model();
    std::unordered_map<std::string, std::shared_ptr<ITensor>> tensor_map;

    std::unordered_map<std::string, std::string> inputs_for_node_type;
    std::unordered_map<std::string, std::queue<std::shared_ptr<ITensor>>>
        outshape_tensor_map;

    std::unordered_map<std::string, int> consumers;

    auto dev = m_cmdpool_->getVulkanDevice();
    // preprocess inputs, make sure we know node types for inputs
    for (const auto &n : model.nodes) {
        for (const auto &in_shape : n.inputs) {
            inputs_for_node_type[in_shape.name] = n.op_type;
            consumers[in_shape.name] += 1;
        }
    }
    printf("Total nodes %zu\n", model.nodes.size());
    // Inputs are dtype-aware now that ShapeRef carries dtype. For the buffer
    // (SSBO) backend every input is a storage buffer; for the legacy image
    // backend fp16 inputs stay as input images (matching the original
    // behaviour that add_conv_model and the vision tests rely on).
    for (const auto &i : model.inputs) {
        std::shared_ptr<ITensor> t;
        if (backend_buffer_) {
            if (i.dtype == "int64") {
                auto typed = std::make_shared<Tensor<int64_t>>(i.dims);
                typed->set_ref_cnt_forever();
                typed->as_storage_buffer(dev);
                t = typed;
            } else if (i.dtype == "int32") {
                auto typed = std::make_shared<Tensor<int>>(i.dims);
                typed->set_ref_cnt_forever();
                typed->as_storage_buffer(dev);
                t = typed;
            } else if (i.dtype == "bool" || i.dtype == "int8") {
                // bool/int8 share the int8 storage representation; the LLM's
                // image_pad_mask is bool but buffer ops consume it as bytes.
                auto typed = std::make_shared<Tensor<int8_t>>(i.dims);
                typed->set_ref_cnt_forever();
                typed->as_storage_buffer(dev);
                t = typed;
            } else if (i.dtype == "float32") {
                auto typed = std::make_shared<Tensor<float>>(i.dims);
                typed->set_ref_cnt_forever();
                typed->as_storage_buffer(dev);
                t = typed;
            } else {
                // float16 / unknown -> fp16 (the historical default).
                auto typed = std::make_shared<Tensor<uint16_t>>(i.dims);
                typed->set_ref_cnt_forever();
                typed->as_storage_buffer(dev);
                t = typed;
            }
        } else {
            // legacy image backend: fp16 inputs as images (original path).
            auto t16 = std::make_shared<Tensor<uint16_t>>(i.dims);
            t16->set_ref_cnt_forever();
            t16->as_input_image(dev, nullptr);
            t = t16;
        }
        inputs_[i.name] = t;
        tensor_map[i.name] = t;
    }

    for (const auto &o : model.outputs) {
        if (precision_ == 1) {
            auto t = std::make_shared<Tensor<uint16_t>>(o.dims, true);
            t->set_ref_cnt_forever();
            outputs_[o.name] = t;
            tensor_map[o.name] = t;
            real_outputs_[o.name] = t;
        } else {
            auto t = std::make_shared<Tensor<float>>(o.dims, true);
            t->set_ref_cnt_forever();
            outputs_[o.name] = t;
            tensor_map[o.name] = t;
            real_outputs_[o.name] = t;
        }
    }

    auto handle_floating_point_tensor = [&](const load::Initializer &init,
                                            const uint8_t *src_base,
                                            auto &tensor) {
        using T =
            typename std::remove_reference_t<decltype(*tensor)>::value_type;
        auto *src = const_cast<T *>(reinterpret_cast<const T *>(src_base));
        tensor->set_ref_cnt_forever();
        if (inputs_for_node_type.find(init.name) !=
                inputs_for_node_type.end() &&
            inputs_for_node_type[init.name] == "Conv") {
            tensor->set_transpose();
            if (init.dims.size() == 4 && init.dims[2] == 1 &&
                init.dims[3] == 1) {
                tensor->set_pack();
            }
        }

        if (backend_buffer_) {
            // SSBO backend: every initializer is a storage buffer regardless of
            // rank (row-major compact, no NCHW->RGBA image packing). This is
            // the path the LLM (buffer-only ops) uses.
            tensor->as_storage_buffer(dev);
            tensor->copyToGPU(m_cmdpool_, src);
        } else if (tensor->num_dims() <= 2) {
            if (inputs_for_node_type[init.name] == "Conv" ||
                inputs_for_node_type[init.name] == "BatchNormalization") {
                tensor->as_uniform_bufferview(dev);
            } else {
                tensor->as_storage_buffer(dev);
            }
            // src points into the read-only mmap'd initializer blob; the
            // Tensor upload overloads only read from it.
            tensor->copyToGPU(m_cmdpool_, src);
        } else {
            tensor->as_input_image(dev, nullptr, false, true);
            tensor->copyToGPUImage(m_cmdpool_, src, model.rgba);
        }
        tensor_map[init.name] = tensor;
        initializers_[init.name] = tensor;
    };

    auto handle_unified_tensors = [&](const load::Initializer &init,
                                      const uint8_t *src_base, auto &tensor,
                                      auto &meta, auto &buffer) {
        using T =
            typename std::remove_reference_t<decltype(*tensor)>::value_type;
        auto *src = const_cast<T *>(reinterpret_cast<const T *>(src_base));
        tensor->set_ref_cnt_forever();
        if (inputs_for_node_type.find(init.name) !=
                inputs_for_node_type.end() &&
            inputs_for_node_type[init.name] == "Conv") {
            tensor->set_transpose();
            if (init.dims.size() == 4 && init.dims[2] == 1 &&
                init.dims[3] == 1) {
                tensor->set_pack();
            }
        }

        if (tensor->num_dims() <= 2) {
            tensor->as_uniform_bufferview(dev, buffer, meta.offset());
        } else {
            tensor->as_input_image(dev, nullptr, false, true);
            tensor->copyToGPUImage(m_cmdpool_, src, model.rgba);
        }
        tensor_map[init.name] = tensor;
        initializers_[init.name] = tensor;
    };

    if (model.unified && !model.unified_meta.empty()) {
        // Unified-tensor sub-allocation: one shared GPU uniform buffer holds
        // the whole unified region of the blob; each tensor is a BufferView at
        // meta.offset() into it. (Replaces the legacy unified_metadata /
        // unified_names / unified_tensors magic-initializer scan.)
        size_t unified_base = model.unified_blob_offset;
        const uint8_t *src_ptr = model.initializer_memory + unified_base;
        const char *name_ptr = model.unified_names.data();

        // Derive the unified region's overall dims from the first meta entry
        // (matches legacy behaviour where unified_tensors was a single blob).
        // The shared buffer's element count is the total unified byte size /
        // sizeof(float); runtime only uses it as a backing store.
        size_t unified_total_bytes = 0;
        for (const auto &meta : model.unified_meta) {
            size_t end = static_cast<size_t>(meta.offset()) +
                         static_cast<size_t>(meta.size());
            if (end > unified_total_bytes)
                unified_total_bytes = end;
        }
        std::vector<uint32_t> unified_dims = {
            static_cast<uint32_t>(unified_total_bytes / sizeof(float))};
        auto unified_tensor = std::make_shared<Tensor<float>>(unified_dims);
        unified_tensor->set_ref_cnt_forever();
        auto buffer = unified_tensor->as_uniform_buffer(dev);
        unified_tensor->copyToGPU(
            m_cmdpool_,
            const_cast<float *>(reinterpret_cast<const float *>(src_ptr)));

        size_t name_idx_offset = 0;
        for (const auto &meta : model.unified_meta) {
            auto name =
                std::string(name_ptr + name_idx_offset,
                            name_ptr + name_idx_offset + meta.name_len());
            name_idx_offset += meta.name_len();
            auto init = model.initializers[name];
            if (init.dtype == "float32") {
                auto t = std::make_shared<Tensor<float>>(init.dims);
                handle_unified_tensors(init, src_ptr, t, meta, buffer);
            } else if (init.dtype == "float16") {
                auto t = std::make_shared<Tensor<uint16_t>>(init.dims);
                handle_unified_tensors(init, src_ptr, t, meta, buffer);
            } else if (init.dtype == "int8") {
                auto t = std::make_shared<Tensor<int8_t>>(init.dims);
                handle_unified_tensors(init, src_ptr, t, meta, buffer);
            } else {
                throw std::runtime_error("Unsupported data type: " +
                                         init.dtype);
            }

            model.initializers.erase(name);
        }
    }

    for (const auto &itr : model.initializers) {
        auto init = itr.second;
        size_t offset = model.initializer_offsets[init.name];
        const uint8_t *src_ptr = model.initializer_memory + offset;
        if (init.dtype == "int64") {
            auto t = std::make_shared<Tensor<int64_t>>(init.dims);
            t->set_ref_cnt_forever();
            t->fillToCPU(const_cast<int64_t *>(
                reinterpret_cast<const int64_t *>(src_ptr)));
            // Upload to GPU as an SSBO. Explicit src keeps the CPU copy alive
            // so downstream ops can read it via as_tensor<int64_t>() without a
            // GPU->CPU round trip (int64 is only consumed by buffer-backend
            // CPU-compute branches).
            t->as_storage_buffer(dev);
            t->copyToGPU(m_cmdpool_,
                         const_cast<int64_t *>(
                             reinterpret_cast<const int64_t *>(src_ptr)));
            tensor_map[init.name] = t;
            initializers_[init.name] = t;
        } else if (init.dtype == "int32") {
            auto t = std::make_shared<Tensor<int>>(init.dims);
            t->set_ref_cnt_forever();
            t->fillToCPU(
                const_cast<int *>(reinterpret_cast<const int *>(src_ptr)));
            t->as_storage_buffer(dev);
            t->copyToGPU(
                m_cmdpool_,
                const_cast<int *>(reinterpret_cast<const int *>(src_ptr)));
            tensor_map[init.name] = t;
            initializers_[init.name] = t;
        } else if (init.dtype == "float32") {
            auto t = std::make_shared<Tensor<float>>(init.dims);
            handle_floating_point_tensor(init, src_ptr, t);
        } else if (init.dtype == "float16") {
            auto t = std::make_shared<Tensor<uint16_t>>(init.dims);
            handle_floating_point_tensor(init, src_ptr, t);
        } else if (init.dtype == "int8") {
            auto t = std::make_shared<Tensor<int8_t>>(init.dims);
            handle_floating_point_tensor(init, src_ptr, t);
        } else {
            throw std::runtime_error("Only float32/int32/fp16/int8 initializer "
                                     "is supported for now " +
                                     init.dtype);
        }
    }

    std::unordered_map<std::string, const load::Node *> node_name_map;
    for (const auto &n : model.nodes) {
        node_name_map[n.name] = &n;
    }

    const auto &concurrent_levels = model.getConcurrentExecutionLevels();
    printf("Building execution plan with %zu concurrent levels\n",
           concurrent_levels.size());

    level_node_indices_.resize(concurrent_levels.size());
    size_t global_node_index = 0;
    size_t total_nodes = 0;
    for (const auto &level_nodes : concurrent_levels) {
        total_nodes += level_nodes.size();
    }
    node_dependency_indices_.resize(total_nodes);

    for (size_t level_idx = 0; level_idx < concurrent_levels.size();
         ++level_idx) {
        const auto &level_nodes = concurrent_levels[level_idx];
        printf("Level %zu: %zu nodes\n", level_idx, level_nodes.size());
        level_node_indices_[level_idx].reserve(level_nodes.size());

        for (const auto &node_name : level_nodes) {
            auto node_it = node_name_map.find(node_name);
            if (node_it == node_name_map.end()) {
                printf("Warning: Node %s not found in model\n",
                       node_name.c_str());
                continue;
            }
            const auto &n = *(node_it->second);
            auto type = vkop::ops::convert_opstring_to_enum(n.op_type);
            if (type == vkop::ops::OpType::UNKNOWN) {
                // make it as input for next ops
                continue;
            }

            std::vector<int> current_node_dependencies;
            for (const auto &depend_name : n.dependencies) {
                for (int prev_idx = node_ops_.size() - 1; prev_idx >= 0;
                     --prev_idx) {
                    if (node_ops_[prev_idx]->get_name() == depend_name) {
                        current_node_dependencies.push_back(prev_idx);
                        break;
                    }
                }
            }
            node_dependency_indices_[global_node_index] =
                std::move(current_node_dependencies);

            std::vector<std::shared_ptr<ITensor>> node_inputs;
            std::vector<std::shared_ptr<ITensor>> node_outputs;

            for (const auto &in_shape : n.inputs) {
                if (tensor_map.find(in_shape.name) != tensor_map.end()) {
                    auto t = tensor_map[in_shape.name];
                    if (t->ref_cnt() != std::numeric_limits<uint16_t>::max()) {
                        t->ref_dec();
                    }
                    node_inputs.push_back(tensor_map[in_shape.name]);
                } else if (in_shape.dims.empty()) {
                    node_inputs.push_back(nullptr);
                } else {
                    printf("we should not reach here %s\n",
                           in_shape.name.c_str());
                    assert(false);
                }
            }

            for (const auto &out_shape : n.outputs) {
                if (tensor_map.find(out_shape.name) != tensor_map.end()) {
                    // model output, seperate tensors
                    assert(tensor_map[out_shape.name]->is_on_GPU());
                    node_outputs.push_back(tensor_map[out_shape.name]);
                } else {
                    // Infer the output tensor dtype. ShapeRef carries only
                    // name+dims, so we derive it from op semantics and the
                    // dtypes already present in tensor_map.
                    std::string dtype_marker = "_";
                    if (type == vkop::ops::OpType::SHAPE) {
                        dtype_marker = "_i64_";
                    } else if (type == vkop::ops::OpType::NONZERO) {
                        dtype_marker = "_i64_";
                    } else if (type == vkop::ops::OpType::CAST) {
                        int to = 0;
                        auto it = n.attributes.find("to");
                        if (it != n.attributes.end()) {
                            to = std::stoi(it->second);
                        }
                        if (to == 10) { // fp16
                            dtype_marker = "_f16_";
                        } else {
                            dtype_marker = "_f32_";
                        }
                    } else if (!node_inputs.empty() &&
                               node_inputs[0] != nullptr &&
                               node_inputs[0]->dtype() == typeid(int64_t)) {
                        // dtype follows the DATA input (int64 flows through
                        // Div/Gather/Concat/Equal/Where/Mul/Neg/Add/Slice/
                        // Reshape/Transpose/Expand)
                        dtype_marker = "_i64_";
                    } else if (!node_inputs.empty() &&
                               node_inputs[0] != nullptr &&
                               node_inputs[0]->dtype() == typeid(uint16_t)) {
                        // fp16 data input -> fp16 output.
                        dtype_marker = "_f16_";
                    } else if (!node_inputs.empty() &&
                               node_inputs[0] != nullptr &&
                               node_inputs[0]->dtype() == typeid(float)) {
                        // fp32 data input -> fp32 output. This matters for the
                        // RMSNorm chain: the model wraps Pow/ReduceMean/Add/
                        // Sqrt/Div in Cast(to=1)...Cast(to=10) so the squaring
                        // runs in fp32 and |x|^2 for |x|~200 doesn't overflow
                        // fp16's 65504 max. If we force fp16 here (precision_),
                        // Pow(206,2) overflows to Inf and RMSNorm collapses.
                        dtype_marker = "_f32_";
                    } else {
                        dtype_marker = precision_ == 1 ? "_f16_" : "_f32_";
                    }

                    std::string key = "_";
                    for (const auto &dim : out_shape.dims) {
                        key += std::to_string(dim) + "_";
                    }
                    key += dtype_marker;
                    auto q = outshape_tensor_map[key];
                    if (!q.empty()) {
                        auto t = q.front();
                        q.pop();
                        t->set_ref_cnt(consumers[out_shape.name]);
                        tensor_map[out_shape.name] = t;
                        node_outputs.push_back(t);
                    } else {
                        if (dtype_marker == "_i64_") {
                            // int64 outputs are CPU-computed by the shape
                            // meta-chain ops (Shape/Gather/Slice/...); they
                            // upload to an SSBO themselves. Create off-GPU so
                            // is_on_GPU() doesn't lie about a non-existent
                            // buffer (which would make copyToCPU deref null).
                            auto t = std::make_shared<Tensor<int64_t>>(
                                out_shape.dims, false);
                            t->set_ref_cnt(consumers[out_shape.name]);
                            tensor_map[out_shape.name] = t;
                            node_outputs.push_back(t);
                        } else if (dtype_marker == "_f16_") {
                            auto t = std::make_shared<Tensor<uint16_t>>(
                                out_shape.dims, true);
                            t->set_ref_cnt(consumers[out_shape.name]);
                            tensor_map[out_shape.name] = t;
                            node_outputs.push_back(t);
                        } else {
                            auto t = std::make_shared<Tensor<float>>(
                                out_shape.dims, true);
                            t->set_ref_cnt(consumers[out_shape.name]);
                            tensor_map[out_shape.name] = t;
                            node_outputs.push_back(t);
                        }
                    }
                }
            }
            for (auto &t : node_inputs) {
                if (t && t->ref_cnt() == 0) {
                    // recycle to outshape_tensor_map
                    std::string key = "_";
                    for (const auto &dim : t->getShape()) {
                        key += std::to_string(dim) + "_";
                    }
                    key += "_";
                    if (t->dtype() == typeid(int64_t)) {
                        key += "i64_";
                    } else if (t->dtype() == typeid(uint16_t)) {
                        key += "f16_";
                    } else if (t->dtype() == typeid(float)) {
                        key += "f32_";
                    } else if (t->dtype() == typeid(int)) {
                        key += "i32_";
                    } else {
                        key += "other_";
                    }
                    auto q = outshape_tensor_map[key];
                    q.push(t);
                }
            }
            if (type == vkop::ops::OpType::SOFTMAX) {
                // (Legacy 2-D SSBO auto-path removed — softmax now uses the
                // buffer backend via backend_buffer_, set elsewhere.)
            }
            // For elementwise / unary / reduce / activation ops the fp16 flag
            // must track the DATA input's actual dtype, not the global
            // precision_. The RMSNorm chain runs Pow/ReduceMean/Sqrt/Div in an
            // fp32 domain (wrapped by Cast to=1 ... to=10) so x^2 for |x|~200
            // doesn't overflow fp16. If we force fp16 shaders here, the fp32
            // Cast output gets read as packed half2 garbage and Pow overflows.
            int op_fp16 = precision_;
            if (!node_inputs.empty() && node_inputs[0] != nullptr) {
                switch (type) {
                case vkop::ops::OpType::ADD:
                case vkop::ops::OpType::SUB:
                case vkop::ops::OpType::MUL:
                case vkop::ops::OpType::DIV:
                case vkop::ops::OpType::POW:
                case vkop::ops::OpType::SQRT:
                case vkop::ops::OpType::REDUCE:
                case vkop::ops::OpType::SIGMOID:
                case vkop::ops::OpType::TANH:
                case vkop::ops::OpType::ATAN:
                case vkop::ops::OpType::SOFTPLUS:
                case vkop::ops::OpType::PRELU:
                case vkop::ops::OpType::SOFTMAX:
                case vkop::ops::OpType::SIN:
                case vkop::ops::OpType::COS:
                case vkop::ops::OpType::NEG:
                case vkop::ops::OpType::ERF:
                case vkop::ops::OpType::FLOOR:
                case vkop::ops::OpType::RELU:
                    op_fp16 =
                        (node_inputs[0]->dtype() == typeid(uint16_t)) ? 1 : 0;
                    break;
                default:
                    break;
                }
            }
            auto op = ops::create_from_type(type, op_fp16,
                                            dev->is_support_nv_tensor_core(),
                                            backend_buffer_);
            if (!op) {
                std::cout << "Fail to create operator" << std::endl;
                return;
            }

            op->set_name(n.name);
            op->set_runtime_device(dev, m_cmdpool_);
            op->setAttribute(n.attributes);

            level_node_indices_[level_idx].push_back(global_node_index++);
            node_ops_.push_back(std::move(op));
            node_attrs_.push_back(n.attributes);
            node_input_tensors_.push_back(std::move(node_inputs));
            node_output_tensors_.push_back(std::move(node_outputs));
        }
    }
    printf("Execution plan built with %zu operations\n", node_ops_.size());
    // Persist a name->tensor view of every named tensor for post-Run
    // inspection (driver dumps intermediates to diagnose NaN propagation).
    tensor_map_ = tensor_map;
}

std::shared_ptr<ITensor> Runtime::GetInput(const std::string &name) const {
    if (name.empty() && inputs_.size() > 1) {
        throw std::runtime_error(
            "Input name is empty but there are multiple inputs");
    }
    if (name.empty() && inputs_.size() == 1) {
        return inputs_.begin()->second;
    }
    auto it = inputs_.find(name);
    if (it == inputs_.end()) {
        return nullptr;
    }
    return it->second;
}

void Runtime::ResizeInput(const std::string &name,
                          const std::vector<uint32_t> &dims) {
    auto it = inputs_.find(name);
    if (it == inputs_.end()) {
        throw std::runtime_error("ResizeInput: unknown input " + name);
    }
    auto dev = m_cmdpool_->getVulkanDevice();
    auto &t = it->second;
    if (t->dtype() == typeid(int64_t)) {
        as_tensor<int64_t>(t)->resize(dims);
        as_tensor<int64_t>(t)->recreate_storage_buffer(dev);
    } else if (t->dtype() == typeid(int)) {
        as_tensor<int>(t)->resize(dims);
        as_tensor<int>(t)->recreate_storage_buffer(dev);
    } else if (t->dtype() == typeid(int8_t)) {
        as_tensor<int8_t>(t)->resize(dims);
        as_tensor<int8_t>(t)->recreate_storage_buffer(dev);
    } else if (t->dtype() == typeid(float)) {
        as_tensor<float>(t)->resize(dims);
        as_tensor<float>(t)->recreate_storage_buffer(dev);
    } else {
        as_tensor<uint16_t>(t)->resize(dims);
        as_tensor<uint16_t>(t)->recreate_storage_buffer(dev);
    }
}

std::shared_ptr<ITensor> Runtime::GetOutput(const std::string &name) const {
    if (name.empty() && inputs_.size() > 1) {
        throw std::runtime_error(
            "Output name is empty but there are multiple outputs");
    }
    if (name.empty() && outputs_.size() == 1) {
        return outputs_.begin()->second;
    }
    auto it = outputs_.find(name);
    if (it == outputs_.end()) {
        return nullptr;
    }
    return it->second;
}

std::shared_ptr<ITensor>
Runtime::GetInitializer(const std::string &name) const {
    auto it = initializers_.find(name);
    if (it == initializers_.end()) {
        return nullptr;
    }
    return it->second;
}

std::shared_ptr<ITensor> Runtime::GetTensor(const std::string &name) const {
    auto it = tensor_map_.find(name);
    if (it != tensor_map_.end()) {
        return it->second;
    }
    auto ini = initializers_.find(name);
    if (ini != initializers_.end()) {
        return ini->second;
    }
    auto inp = inputs_.find(name);
    if (inp != inputs_.end()) {
        return inp->second;
    }
    return nullptr;
}

double Runtime::Run() {
    auto dev = m_cmdpool_->getVulkanDevice();
    auto start = std::chrono::steady_clock::now();

    bool single_queue = dev->getNumComputeQueues() <= 1;

    // Debug: VKOP_NAN_SCAN=1 synchronizes after every concurrent level and
    // scans each node's output tensors for NaN/Inf, printing the first node
    // (name + op idx + level) that produces a non-finite value. Slow (forces
    // a full GPU stall per level) but pinpoints where NaN first appears.
    const char *nan_scan_env = std::getenv("VKOP_NAN_SCAN");
    bool nan_scan = nan_scan_env && nan_scan_env[0] == '1';

    if (nan_scan) {
        size_t last_level_index = level_node_indices_.size() - 1;
        for (size_t level_idx = 0; level_idx < level_node_indices_.size();
             level_idx++) {
            const auto &level_nodes = level_node_indices_[level_idx];
            int id = 0;
            std::vector<std::shared_ptr<VulkanCommandBuffer>> cmds;
            for (auto node_idx : level_nodes) {
                node_ops_[node_idx]->onExecute(node_input_tensors_[node_idx],
                                               node_output_tensors_[node_idx],
                                               id);
                auto cmd = node_ops_[node_idx]->get_record();
                for (auto &dep : node_dependency_indices_[node_idx]) {
                    cmd->addWait(
                        node_ops_[dep]->get_record()->getSignalSemaphore(),
                        node_ops_[dep]->get_record()->getSignalValue());
                }
                cmds.push_back(cmd);
                id++;
                id %= vkop::kInflight;
            }
            // Submit + wait this level before proceeding so outputs are ready.
            std::vector<VkSubmitInfo> sis;
            for (auto &c : cmds)
                sis.push_back(c->buildSubmitInfo());
            if (!sis.empty()) {
                VulkanCommandBuffer::submit(dev->getComputeQueue(0), sis);
            }
            for (auto &c : cmds)
                c->wait();
            for (auto &c : cmds) {
                c->clearWaits();
                c->reset();
            }

            // Scan outputs of every node in this level for NaN/Inf.
            for (auto node_idx : level_nodes) {
                const auto &outs = node_output_tensors_[node_idx];
                for (size_t oi = 0; oi < outs.size(); ++oi) {
                    auto &t = outs[oi];
                    if (!t)
                        continue;
                    // Only float-bearing dtypes can hold NaN.
                    bool is_fp16 = (t->dtype() == typeid(uint16_t));
                    bool is_fp32 = (t->dtype() == typeid(float));
                    if (!is_fp16 && !is_fp32)
                        continue;
                    int ne = t->size() / (is_fp16 ? 2 : 4);
                    if (ne <= 0)
                        continue;
                    int nan_cnt = 0, inf_cnt = 0;
                    if (is_fp16) {
                        auto tg = as_tensor<uint16_t>(t);
                        tg->copyToCPU(m_cmdpool_);
                        const uint16_t *p = reinterpret_cast<const uint16_t *>(
                            tg->data().data());
                        for (int i = 0; i < ne; ++i) {
                            float v = ITensor::fp16_to_fp32(p[i]);
                            if (std::isnan(v))
                                nan_cnt++;
                            else if (std::isinf(v))
                                inf_cnt++;
                        }
                    } else {
                        auto tg = as_tensor<float>(t);
                        tg->copyToCPU(m_cmdpool_);
                        const float *p =
                            reinterpret_cast<const float *>(tg->data().data());
                        for (int i = 0; i < ne; ++i) {
                            if (std::isnan(p[i]))
                                nan_cnt++;
                            else if (std::isinf(p[i]))
                                inf_cnt++;
                        }
                    }
                    if (nan_cnt > 0 || inf_cnt > 0) {
                        std::printf(
                            "[NANSCAN] FIRST NaN/Inf at level=%zu nodeidx=%zu "
                            "name=%s out=%zu ne=%d nan=%d inf=%d\n",
                            level_idx, node_idx,
                            node_ops_[node_idx]->get_name().c_str(), oi, ne,
                            nan_cnt, inf_cnt);
                        // Dump input stats so we can tell whether the NaN came
                        // from the op itself or was already in its inputs.
                        const auto &ins = node_input_tensors_[node_idx];
                        for (size_t ii = 0; ii < ins.size(); ++ii) {
                            auto &it = ins[ii];
                            if (!it) {
                                std::printf("  in[%zu]=null\n", ii);
                                continue;
                            }
                            bool ifp16 = (it->dtype() == typeid(uint16_t));
                            bool ifp32 = (it->dtype() == typeid(float));
                            if (!ifp16 && !ifp32) {
                                std::printf("  in[%zu] ne=%d (non-fp)\n", ii,
                                            it->size());
                                continue;
                            }
                            int ine = it->size() / (ifp16 ? 2 : 4);
                            if (ine <= 0) {
                                std::printf("  in[%zu] ne=0\n", ii);
                                continue;
                            }
                            int inan = 0, iinf = 0;
                            float imn = 1e30f, imx = -1e30f;
                            if (ifp16) {
                                auto tg = as_tensor<uint16_t>(it);
                                tg->copyToCPU(m_cmdpool_);
                                const uint16_t *p =
                                    reinterpret_cast<const uint16_t *>(
                                        tg->data().data());
                                for (int i = 0; i < ine; ++i) {
                                    float v = ITensor::fp16_to_fp32(p[i]);
                                    if (std::isnan(v))
                                        inan++;
                                    else if (std::isinf(v))
                                        iinf++;
                                    else {
                                        if (v < imn)
                                            imn = v;
                                        if (v > imx)
                                            imx = v;
                                    }
                                }
                            } else {
                                auto tg = as_tensor<float>(it);
                                tg->copyToCPU(m_cmdpool_);
                                const float *p =
                                    reinterpret_cast<const float *>(
                                        tg->data().data());
                                for (int i = 0; i < ine; ++i) {
                                    if (std::isnan(p[i]))
                                        inan++;
                                    else if (std::isinf(p[i]))
                                        iinf++;
                                    else {
                                        if (p[i] < imn)
                                            imn = p[i];
                                        if (p[i] > imx)
                                            imx = p[i];
                                    }
                                }
                            }
                            std::printf("  in[%zu] ne=%d nan=%d inf=%d "
                                        "min=%.5g max=%.5g\n",
                                        ii, ine, inan, iinf, imn, imx);
                        }
                        // Identify which upstream node produced each input by
                        // pointer-matching against dependency outputs.
                        for (size_t ii = 0; ii < ins.size(); ++ii) {
                            auto &it = ins[ii];
                            if (!it)
                                continue;
                            std::string prod = "<none>";
                            for (auto dep :
                                 node_dependency_indices_[node_idx]) {
                                if (dep < 0 || dep >= (int)node_ops_.size())
                                    continue;
                                const auto &douts = node_output_tensors_[dep];
                                for (size_t k = 0; k < douts.size(); ++k) {
                                    if (douts[k].get() == it.get()) {
                                        prod = node_ops_[dep]->get_name() +
                                               "/out" + std::to_string(k);
                                        break;
                                    }
                                }
                                if (prod != "<none>")
                                    break;
                            }
                            std::printf("  in[%zu] <- %s\n", ii, prod.c_str());
                        }
                        // Dump up to 16 output-NaN indices paired with the
                        // corresponding input[0] and input[1] values, so we
                        // can see exactly which (a,b) pairs go NaN.
                        if (oi == 0 && !ins.empty() && ins[0] &&
                            ins[0]->dtype() == typeid(uint16_t)) {
                            auto a_tg = as_tensor<uint16_t>(ins[0]);
                            a_tg->copyToCPU(m_cmdpool_);
                            const uint16_t *ap =
                                reinterpret_cast<const uint16_t *>(
                                    a_tg->data().data());
                            int ane = ins[0]->size() / 2;
                            const uint16_t *bp = nullptr;
                            int bne = 0;
                            if (ins.size() > 1 && ins[1] &&
                                ins[1]->dtype() == typeid(uint16_t)) {
                                auto b_tg = as_tensor<uint16_t>(ins[1]);
                                b_tg->copyToCPU(m_cmdpool_);
                                bp = reinterpret_cast<const uint16_t *>(
                                    b_tg->data().data());
                                bne = ins[1]->size() / 2;
                            }
                            auto out_tg = as_tensor<uint16_t>(t);
                            out_tg->copyToCPU(m_cmdpool_);
                            const uint16_t *op =
                                reinterpret_cast<const uint16_t *>(
                                    out_tg->data().data());
                            int shown = 0;
                            std::printf("  nanidx (i: a -> out):");
                            for (int i = 0; i < ne && shown < 16; ++i) {
                                float ov = ITensor::fp16_to_fp32(op[i]);
                                if (!std::isnan(ov) && !std::isinf(ov))
                                    continue;
                                float av = (i < ane)
                                               ? ITensor::fp16_to_fp32(ap[i])
                                               : 0.0f / 0.0f;
                                float bv = (bp && i < bne)
                                               ? ITensor::fp16_to_fp32(bp[i])
                                               : 0.0f / 0.0f;
                                std::printf(" [%d: a=%.5g b=%.5g o=%04x]", i,
                                            av, bv, op[i]);
                                shown++;
                            }
                            std::printf("\n");
                        }
                        std::fflush(stdout);
                        (void)last_level_index;
                        auto end = std::chrono::steady_clock::now();
                        std::chrono::duration<double> e = end - start;
                        return e.count() * 1000.0F;
                    }
                }
            }
        }
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        return elapsed.count() * 1000.0F;
    }

    std::vector<std::vector<VkSubmitInfo>> submit_infos(vkop::kInflight);
    std::vector<std::shared_ptr<VulkanCommandBuffer>> last_commands(
        vkop::kInflight);

    size_t last_level_index = level_node_indices_.size() - 1;
    for (size_t level_idx = 0; level_idx < level_node_indices_.size();
         level_idx++) {
        const auto &level_nodes = level_node_indices_[level_idx];
        int id = 0;
        for (auto node_idx : level_nodes) {
            node_ops_[node_idx]->onExecute(node_input_tensors_[node_idx],
                                           node_output_tensors_[node_idx], id);
            auto cmd = node_ops_[node_idx]->get_record();
            auto depends = node_dependency_indices_[node_idx];
            for (auto &dep : depends) {
                cmd->addWait(node_ops_[dep]->get_record()->getSignalSemaphore(),
                             node_ops_[dep]->get_record()->getSignalValue());
            }

            if (single_queue) {
                submit_infos[0].push_back(cmd->buildSubmitInfo());
            } else {
                submit_infos[id].push_back(cmd->buildSubmitInfo());
            }
            if (level_idx == last_level_index) {
                last_commands[single_queue ? 0 : id] = cmd;
            }
            id++;
            id %= vkop::kInflight;
        }
    }

    if (single_queue) {
        // Submit all command buffers in a single vkQueueSubmit call.
        // On single-queue GPUs, splitting into multiple vkQueueSubmit calls
        // creates implicit ordering between submits: the second submit won't
        // begin until the first completes. If a node in submit_infos[0] waits
        // on a semaphore signaled by a node in submit_infos[1], this creates
        // a deadlock (C waits B, but B can't start until C finishes).
        // A single submit lets the GPU schedule based on semaphore
        // dependencies.
        if (!submit_infos[0].empty()) {
            VulkanCommandBuffer::submit(dev->getComputeQueue(0),
                                        submit_infos[0]);
        }
    } else {
        for (int ci = 0; ci < vkop::kInflight; ci++) {
            if (!submit_infos[ci].empty()) {
                VulkanCommandBuffer::submit(dev->getComputeQueue(ci),
                                            submit_infos[ci]);
            }
        }
    }

    for (int ci = 0; ci < vkop::kInflight; ci++) {
        if (last_commands[ci]) {
            last_commands[ci]->wait();
        }
    }
    for (const auto &level_nodes : level_node_indices_) {
        for (auto node_idx : level_nodes) {
            auto cmd = node_ops_[node_idx]->get_record();
            cmd->clearWaits();
            cmd->reset();
        }
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count() * 1000.0F;
}

void Runtime::ReadResult() {
    auto dev = m_cmdpool_->getVulkanDevice();
    dev->wait_all_done();

    for (auto &p : real_outputs_) {
        if (p.second->dtype() == typeid(float)) {
            auto t = vkop::core::as_tensor<float>(p.second);
            t->copyToCPU(m_cmdpool_);
        } else if (p.second->dtype() == typeid(int)) {
            auto t = vkop::core::as_tensor<int>(p.second);
            t->copyToCPU(m_cmdpool_);
        } else if (p.second->dtype() == typeid(uint16_t)) {
            auto t = vkop::core::as_tensor<uint16_t>(p.second);
            t->copyToCPU(m_cmdpool_);
        } else {
            assert(false);
        }
    }
}

void Runtime::RegisterPostProcess(
    ops::OpType ops,
    const std::unordered_map<std::string, std::string> &attributes,
    const std::vector<std::shared_ptr<ITensor>> &inputs,
    const std::vector<std::shared_ptr<ITensor>> &outputs) {

    auto dev = m_cmdpool_->getVulkanDevice();

    auto op = ops::create_from_type(
        ops, precision_, dev->is_support_nv_tensor_core(), backend_buffer_);
    op->set_name("post_" + convert_optype_to_string(ops));
    op->set_runtime_device(dev, m_cmdpool_);
    op->setAttribute(attributes);

    size_t current_op_idx = node_ops_.size();
    node_ops_.push_back(std::move(op));
    node_attrs_.push_back(attributes);
    node_input_tensors_.push_back(std::move(inputs));
    node_output_tensors_.push_back(std::move(outputs));

    std::vector<int> post_process_dependencies;
    if (!level_node_indices_.empty()) {
        const auto &last_level_indices = level_node_indices_.back();
        post_process_dependencies.insert(post_process_dependencies.end(),
                                         last_level_indices.begin(),
                                         last_level_indices.end());
    }
    node_dependency_indices_.resize(current_op_idx + 1);
    node_dependency_indices_[current_op_idx] =
        std::move(post_process_dependencies);

    size_t new_level_idx = level_node_indices_.size();
    level_node_indices_.resize(new_level_idx + 1);
    level_node_indices_[new_level_idx].push_back(node_ops_.size() - 1);
    real_outputs_.clear();
    for (size_t i = 0; i < outputs.size(); ++i) {
        real_outputs_["post_" + convert_optype_to_string(ops) +
                      std::to_string(i)] = outputs[i];
    }
}

void Runtime::TraceNode(const std::string &name) {
    for (auto &op : node_ops_) {
        if (op->get_name() == name) {
            op->enable_trace();
            break;
        }
    }
}

} // namespace core
} // namespace vkop