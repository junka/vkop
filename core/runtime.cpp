// junka @ 2025
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <queue>
#include <unordered_set>

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
        // Sanitize recorded dims: a -1 dynamic sentinel in dims_ would make
        // size_ negative (size_ *= d) in the ctor, breaking the initial
        // as_storage_buffer allocation. Replace -1 with a placeholder 1 — the
        // real shape is supplied later by ResizeInput (the LLM driver calls
        // rt->ResizeInput(name, arr.shape) before Run). 0 (genuinely empty)
        // is kept.
        std::vector<int32_t> in_dims;
        in_dims.reserve(i.dims.size());
        for (auto d : i.dims) {
            in_dims.push_back(d < 0 ? 1 : d);
        }
        std::shared_ptr<ITensor> t;
        if (backend_buffer_) {
            if (i.dtype == "int64") {
                auto typed = std::make_shared<Tensor<int64_t>>(in_dims);
                typed->set_ref_cnt_forever();
                typed->as_storage_buffer(dev);
                t = typed;
            } else if (i.dtype == "int32") {
                auto typed = std::make_shared<Tensor<int>>(in_dims);
                typed->set_ref_cnt_forever();
                typed->as_storage_buffer(dev);
                t = typed;
            } else if (i.dtype == "bool" || i.dtype == "int8") {
                // bool/int8 share the int8 storage representation; the LLM's
                // image_pad_mask is bool but buffer ops consume it as bytes.
                auto typed = std::make_shared<Tensor<int8_t>>(in_dims);
                typed->set_ref_cnt_forever();
                typed->as_storage_buffer(dev);
                t = typed;
            } else if (i.dtype == "float32") {
                auto typed = std::make_shared<Tensor<float>>(in_dims);
                typed->set_ref_cnt_forever();
                typed->as_storage_buffer(dev);
                t = typed;
            } else {
                // float16 / unknown -> fp16 (the historical default).
                auto typed = std::make_shared<Tensor<uint16_t>>(in_dims);
                typed->set_ref_cnt_forever();
                typed->as_storage_buffer(dev);
                t = typed;
            }
        } else {
            // legacy image backend: fp16 inputs as images (original path).
            auto t16 = std::make_shared<Tensor<uint16_t>>(in_dims);
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
            std::vector<std::vector<int>> node_input_shapes;

            for (const auto &in_shape : n.inputs) {
                // Capture the recorded logical shape for execute-time
                // reshape_view (see node_input_shapes_ doc). int dims even
                // though ShapeRef carries unsigned — reshape_view handles both.
                std::vector<int> rec;
                rec.reserve(in_shape.dims.size());
                for (auto d : in_shape.dims) {
                    rec.push_back(static_cast<int>(d));
                }
                node_input_shapes.push_back(std::move(rec));

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
                    // Sanitize recorded dims for the initial Tensor allocation:
                    // a -1 dynamic sentinel in dims_ would make size_ negative
                    // (size_ *= d), breaking num_elements()/dispatch. Replace
                    // -1 with a placeholder 1 — the producing op resize()s to
                    // the concrete shape at execute time. 0 (genuinely empty)
                    // is kept (yields size_=0 -> 16-byte dummy buffer). The
                    // recorded shape (with -1) is still used below for the
                    // outshape_tensor_map cache key and for node_input_shapes_.
                    std::vector<int32_t> alloc_dims;
                    alloc_dims.reserve(out_shape.dims.size());
                    for (auto d : out_shape.dims) {
                        alloc_dims.push_back(d < 0 ? 1 : d);
                    }
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
                                alloc_dims, false);
                            t->set_ref_cnt(consumers[out_shape.name]);
                            tensor_map[out_shape.name] = t;
                            node_outputs.push_back(t);
                        } else if (dtype_marker == "_f16_") {
                            auto t = std::make_shared<Tensor<uint16_t>>(
                                alloc_dims, true);
                            t->set_ref_cnt(consumers[out_shape.name]);
                            tensor_map[out_shape.name] = t;
                            node_outputs.push_back(t);
                        } else {
                            auto t = std::make_shared<Tensor<float>>(alloc_dims,
                                                                     true);
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
                case vkop::ops::OpType::MATMUL:
                case vkop::ops::OpType::TRANSPOSE:
                case vkop::ops::OpType::CONCAT:
                case vkop::ops::OpType::CONV2D:
                case vkop::ops::OpType::ROTARY_EMBEDDING:
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
            node_input_shapes_.push_back(std::move(node_input_shapes));
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

std::vector<std::pair<std::string, std::shared_ptr<ITensor>>>
Runtime::ListTensors() const {
    std::vector<std::pair<std::string, std::shared_ptr<ITensor>>> out;
    out.reserve(tensor_map_.size());
    for (const auto &kv : tensor_map_) {
        out.push_back({kv.first, kv.second});
    }
    return out;
}

double Runtime::Run() {
    auto dev = m_cmdpool_->getVulkanDevice();
    auto start = std::chrono::steady_clock::now();

    bool single_queue = dev->getNumComputeQueues() <= 1;

    // Set of initializer tensor pointers, used by the execute-time
    // reshape_view guard to distinguish a true scalar Constant (an
    // initializer whose ONNX shape was () but whose vkopbin dims_=[1] due to
    // dag.py's np.atleast_1d materialization) from a node-output tensor whose
    // recorded shape is [] merely because the converter had no shape info.
    // Only the former should be reshaped to rank-0: Gather uses the index
    // tensor's rank to compute its output rank, so a scalar index must stay
    // scalar (rank N-1 output). Reshaping a 1-element node output to scalar
    // would crash int64 Concat (in_shape[axis_] on an empty shape).
    std::unordered_set<core::ITensor *> init_ptrs;
    init_ptrs.reserve(initializers_.size());
    for (const auto &kv : initializers_) {
        init_ptrs.insert(kv.second.get());
    }

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
                // Apply this consumer's recorded logical view to each input
                // tensor right before recording. reshape_view is a pure
                // metadata reset (guards on equal element count), so sharing
                // one tensor across consumers with different views is safe:
                // each onExecute sees exactly the shape recorded for IT, in
                // execution order, with no last-writer-wins race.
                const auto &shapes = node_input_shapes_[node_idx];
                const auto &ins = node_input_tensors_[node_idx];
                for (size_t k = 0; k < ins.size() && k < shapes.size(); ++k) {
                    // An empty recorded shape means "scalar" (rank-0). The
                    // converter's dag.py materializes every Constant as a
                    // >=1-D initializer (np.atleast_1d), so an ONNX scalar
                    // index like /rotary_emb/Constant_8 (value 0, shape ())
                    // arrives with dims_=[1] even though the node's recorded
                    // input shape is []. Passing [] here lets reshape_view
                    // restore n_dims_=0 (ne_new=1 == ne_old=1 for the 1-elem
                    // tensor), so Gather computes a rank-(N-1) output matching
                    // ORT instead of inflating the rank by 1. The element-count
                    // guard inside reshape_view rejects this for any genuinely
                    // multi-element tensor, so a stray [] (unknown shape) is
                    // safe.
                    if (!ins[k]) {
                        continue;
                    }
                    if (shapes[k].empty()) {
                        // Scalar Constant: an ONNX scalar whose materialized
                        // dims_=[1] should be rank-0 for Gather's
                        // index-rank→output-rank to match ORT. Only for
                        // initializers (a [] node output means "unknown").
                        if (init_ptrs.count(ins[k].get())) {
                            ins[k]->reshape_view(shapes[k]);
                        }
                    } else {
                        // Skip the recorded-view reset when the recorded shape
                        // carries a -1 "dynamic" sentinel (symbolic dim_param
                        // the converter couldn't resolve). The live tensor
                        // already holds the concrete runtime shape — set by
                        // ResizeInput for graph inputs or recomputed by the
                        // producing op for intermediates. Applying the -1 here
                        // would clobber a concrete batch/seq (e.g. past_key
                        // values resized to [1,2,8,0,128]), and a -1 in dims_
                        // would also make size_ negative, breaking
                        // num_elements(). 0 now means genuinely empty
                        // (kv_len=0) and IS applied. Static (-1-free) recorded
                        // views still apply, so shared-tensor rank
                        // reinterpretation ([4]↔[2,2]) works.
                        bool has_dyn = false;
                        for (int d : shapes[k]) {
                            if (d == -1) {
                                has_dyn = true;
                                break;
                            }
                        }
                        if (!has_dyn) {
                            ins[k]->reshape_view(shapes[k]);
                        }
                    }
                }
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

    // Debug: VKOP_LEVEL_STAT=1 submits per-level (with a CPU wait after each
    // level) and prints min/max/mean/nonzero-count for every node's first fp
    // output, to bisect which level the numerics diverge at. Slow (full GPU
    // stall per level). VKOP_LEVEL_STAT=Lo,Hi restricts output to levels in
    // [Lo,Hi).
    const char *lstat_env = std::getenv("VKOP_LEVEL_STAT");
    bool level_stat = lstat_env && lstat_env[0] == '1';
    size_t lstat_lo = 0, lstat_hi = SIZE_MAX;
    if (lstat_env && std::strchr(lstat_env, ',')) {
        size_t lo = 0, hi = 0;
        if (std::sscanf(lstat_env, "%zu,%zu", &lo, &hi) == 2) {
            lstat_lo = lo;
            lstat_hi = hi;
            level_stat = true;
        }
    }
    if (level_stat) {
        std::vector<std::shared_ptr<VulkanCommandBuffer>> prev_cmds;
        for (size_t level_idx = 0; level_idx < level_node_indices_.size();
             level_idx++) {
            const auto &level_nodes = level_node_indices_[level_idx];
            std::vector<std::shared_ptr<VulkanCommandBuffer>> cur_cmds;
            std::vector<VkSubmitInfo> sis;
            int id = 0;
            for (auto node_idx : level_nodes) {
                const auto &shapes = node_input_shapes_[node_idx];
                const auto &ins = node_input_tensors_[node_idx];
                for (size_t k = 0; k < ins.size() && k < shapes.size(); ++k) {
                    // An empty recorded shape means "scalar" (rank-0). The
                    // converter's dag.py materializes every Constant as a
                    // >=1-D initializer (np.atleast_1d), so an ONNX scalar
                    // index like /rotary_emb/Constant_8 (value 0, shape ())
                    // arrives with dims_=[1] even though the node's recorded
                    // input shape is []. Passing [] here lets reshape_view
                    // restore n_dims_=0 (ne_new=1 == ne_old=1 for the 1-elem
                    // tensor), so Gather computes a rank-(N-1) output matching
                    // ORT instead of inflating the rank by 1. The element-count
                    // guard inside reshape_view rejects this for any genuinely
                    // multi-element tensor, so a stray [] (unknown shape) is
                    // safe.
                    if (!ins[k]) {
                        continue;
                    }
                    if (shapes[k].empty()) {
                        // Scalar Constant: an ONNX scalar whose materialized
                        // dims_=[1] should be rank-0 for Gather's
                        // index-rank→output-rank to match ORT. Only for
                        // initializers (a [] node output means "unknown").
                        if (init_ptrs.count(ins[k].get())) {
                            ins[k]->reshape_view(shapes[k]);
                        }
                    } else {
                        // Skip the recorded-view reset when the recorded shape
                        // carries a -1 "dynamic" sentinel (symbolic dim_param
                        // the converter couldn't resolve). The live tensor
                        // already holds the concrete runtime shape — set by
                        // ResizeInput for graph inputs or recomputed by the
                        // producing op for intermediates. Applying the -1 here
                        // would clobber a concrete batch/seq (e.g. past_key
                        // values resized to [1,2,8,0,128]), and a -1 in dims_
                        // would also make size_ negative, breaking
                        // num_elements(). 0 now means genuinely empty
                        // (kv_len=0) and IS applied. Static (-1-free) recorded
                        // views still apply, so shared-tensor rank
                        // reinterpretation ([4]↔[2,2]) works.
                        bool has_dyn = false;
                        for (int d : shapes[k]) {
                            if (d == -1) {
                                has_dyn = true;
                                break;
                            }
                        }
                        if (!has_dyn) {
                            ins[k]->reshape_view(shapes[k]);
                        }
                    }
                }
                node_ops_[node_idx]->onExecute(node_input_tensors_[node_idx],
                                               node_output_tensors_[node_idx],
                                               id);
                auto cmd = node_ops_[node_idx]->get_record();
                for (auto &dep : node_dependency_indices_[node_idx]) {
                    cmd->addWait(
                        node_ops_[dep]->get_record()->getSignalSemaphore(),
                        node_ops_[dep]->get_record()->getSignalValue());
                }
                for (const auto &pc : prev_cmds) {
                    cmd->addWait(pc->getSignalSemaphore(),
                                 pc->getSignalValue());
                }
                sis.push_back(cmd->buildSubmitInfo());
                cur_cmds.push_back(cmd);
                id++;
                id %= vkop::kInflight;
            }
            if (!sis.empty()) {
                VulkanCommandBuffer::submit(dev->getComputeQueue(0), sis);
            }
            for (auto &c : cur_cmds)
                c->wait();
            bool in_range = (level_idx >= lstat_lo && level_idx < lstat_hi);
            // Helper: compute (min,max,mean,nan,inf,zero) for an fp tensor.
            auto fp_stats = [&](const std::shared_ptr<ITensor> &tt, float &mn,
                                float &mx, float &meanv, int &nan_cnt,
                                int &inf_cnt, int &zero_cnt) -> int {
                mn = 1e30f;
                mx = -1e30f;
                nan_cnt = inf_cnt = zero_cnt = 0;
                double sum = 0;
                bool f16 = (tt->dtype() == typeid(uint16_t));
                bool f32 = (tt->dtype() == typeid(float));
                if (!f16 && !f32)
                    return 0;
                int ne = tt->size() / (f16 ? 2 : 4);
                if (ne <= 0)
                    return 0;
                if (f16) {
                    auto tg = as_tensor<uint16_t>(tt);
                    tg->copyToCPU(m_cmdpool_);
                    const uint16_t *p =
                        reinterpret_cast<const uint16_t *>(tg->data().data());
                    for (int i = 0; i < ne; ++i) {
                        float v = ITensor::fp16_to_fp32(p[i]);
                        if (std::isnan(v)) {
                            nan_cnt++;
                            continue;
                        }
                        if (std::isinf(v)) {
                            inf_cnt++;
                            continue;
                        }
                        if (v == 0.f)
                            zero_cnt++;
                        sum += v;
                        if (v < mn)
                            mn = v;
                        if (v > mx)
                            mx = v;
                    }
                } else {
                    auto tg = as_tensor<float>(tt);
                    tg->copyToCPU(m_cmdpool_);
                    const float *p =
                        reinterpret_cast<const float *>(tg->data().data());
                    for (int i = 0; i < ne; ++i) {
                        float v = p[i];
                        if (std::isnan(v)) {
                            nan_cnt++;
                            continue;
                        }
                        if (std::isinf(v)) {
                            inf_cnt++;
                            continue;
                        }
                        if (v == 0.f)
                            zero_cnt++;
                        sum += v;
                        if (v < mn)
                            mn = v;
                        if (v > mx)
                            mx = v;
                    }
                }
                meanv = (ne - nan_cnt - inf_cnt) > 0
                            ? (float)(sum / (ne - nan_cnt - inf_cnt))
                            : 0.f;
                return ne;
            };
            if (in_range) {
                int ni = 0;
                for (auto node_idx : level_nodes) {
                    const auto &outs = node_output_tensors_[node_idx];
                    auto &t = outs.empty() ? nullptr : outs[0];
                    if (!t)
                        continue;
                    bool is_fp16 = (t->dtype() == typeid(uint16_t));
                    bool is_fp32 = (t->dtype() == typeid(float));
                    if (!is_fp16 && !is_fp32)
                        continue;
                    int ne = t->size() / (is_fp16 ? 2 : 4);
                    if (ne <= 0)
                        continue;
                    int nan_cnt = 0, inf_cnt = 0, zero_cnt = 0;
                    double sum = 0;
                    float mn = 1e30f, mx = -1e30f;
                    if (is_fp16) {
                        auto tg = as_tensor<uint16_t>(t);
                        tg->copyToCPU(m_cmdpool_);
                        const uint16_t *p = reinterpret_cast<const uint16_t *>(
                            tg->data().data());
                        for (int i = 0; i < ne; ++i) {
                            float v = ITensor::fp16_to_fp32(p[i]);
                            if (std::isnan(v)) {
                                nan_cnt++;
                                continue;
                            }
                            if (std::isinf(v)) {
                                inf_cnt++;
                                continue;
                            }
                            if (v == 0.f)
                                zero_cnt++;
                            sum += v;
                            if (v < mn)
                                mn = v;
                            if (v > mx)
                                mx = v;
                        }
                    } else {
                        auto tg = as_tensor<float>(t);
                        tg->copyToCPU(m_cmdpool_);
                        const float *p =
                            reinterpret_cast<const float *>(tg->data().data());
                        for (int i = 0; i < ne; ++i) {
                            float v = p[i];
                            if (std::isnan(v)) {
                                nan_cnt++;
                                continue;
                            }
                            if (std::isinf(v)) {
                                inf_cnt++;
                                continue;
                            }
                            if (v == 0.f)
                                zero_cnt++;
                            sum += v;
                            if (v < mn)
                                mn = v;
                            if (v > mx)
                                mx = v;
                        }
                    }
                    float mean = (ne - nan_cnt - inf_cnt) > 0
                                     ? (float)(sum / (ne - nan_cnt - inf_cnt))
                                     : 0.f;
                    std::printf(
                        "[LSTAT] L=%zu n=%zu ni=%d name=%s ne=%d "
                        "nan=%d inf=%d zero=%d min=%.5g max=%.5g mean=%.5g\n",
                        level_idx, level_nodes.size(), ni,
                        node_ops_[node_idx]->get_name().c_str(), ne, nan_cnt,
                        inf_cnt, zero_cnt, mn, mx, mean);
                    // Dump each fp input's min/max/mean so we can compare a
                    // node's input against its producer's reported output
                    // (catches tensor-recycle buffer aliasing: if an input's
                    // stats here differ from when it was produced, the buffer
                    // was overwritten between the two levels).
                    {
                        const auto &ins = node_input_tensors_[node_idx];
                        for (size_t ii = 0; ii < ins.size(); ++ii) {
                            auto &it = ins[ii];
                            if (!it)
                                continue;
                            float imn, imx, imean;
                            int inan, iinf, izero;
                            int ine = fp_stats(it, imn, imx, imean, inan, iinf,
                                               izero);
                            if (ine <= 0)
                                continue;
                            std::printf("[LIN]  L=%zu %s in[%zu] ne=%d "
                                        "min=%.5g max=%.5g mean=%.5g\n",
                                        level_idx,
                                        node_ops_[node_idx]->get_name().c_str(),
                                        ii, ine, imn, imx, imean);
                        }
                    }
                    // For input_layernorm (block 0) nodes, dump each fp
                    // input's dtype + first values + output dtype, to trace
                    // the fp32-domain RMSNorm chain.
                    std::string nm = node_ops_[node_idx]->get_name();
                    if (nm.find("input_layernorm/") != std::string::npos &&
                        nm.find("input_layernorm_") == std::string::npos) {
                        const auto &ins = node_input_tensors_[node_idx];
                        for (size_t ii = 0; ii < ins.size(); ++ii) {
                            auto &it = ins[ii];
                            if (!it) {
                                std::printf("[LTR] %s in[%zu]=null\n",
                                            nm.c_str(), ii);
                                continue;
                            }
                            bool ifp16 = (it->dtype() == typeid(uint16_t));
                            bool ifp32 = (it->dtype() == typeid(float));
                            if (!ifp16 && !ifp32) {
                                std::printf("[LTR] %s in[%zu] dtype=other\n",
                                            nm.c_str(), ii);
                                continue;
                            }
                            int ine = it->size() / (ifp16 ? 2 : 4);
                            std::printf(
                                "[LTR] %s in[%zu] dtype=%s ne=%d: ", nm.c_str(),
                                ii, ifp16 ? "f16" : "f32", ine);
                            if (ifp16) {
                                auto itg = as_tensor<uint16_t>(it);
                                itg->copyToCPU(m_cmdpool_);
                                for (int i = 0; i < 4 && i < ine; ++i)
                                    std::printf(
                                        "%.4g ",
                                        ITensor::fp16_to_fp32(
                                            reinterpret_cast<const uint16_t *>(
                                                itg->data().data())[i]));
                            } else {
                                auto itg = as_tensor<float>(it);
                                itg->copyToCPU(m_cmdpool_);
                                for (int i = 0; i < 4 && i < ine; ++i)
                                    std::printf("%.4g ",
                                                reinterpret_cast<const float *>(
                                                    itg->data().data())[i]);
                            }
                            std::printf("\n");
                        }
                        std::printf("[LTR] %s out dtype=%s\n", nm.c_str(),
                                    is_fp16 ? "f16" : "f32");
                    }
                    // [ROT] trace the rotary_emb cos/sin chain + the RoPE
                    // application Muls. Dumps output shape/size/dtype + first
                    // values, and each input's shape/size/dtype + first values,
                    // so we can see exactly where cos/sin go wrong.
                    if (nm.find("rotary_emb/") != std::string::npos ||
                        (nm.size() && nm[0] == '/' &&
                         (nm == "/Mul" || nm == "/Mul_1" || nm == "/Mul_2" ||
                          nm == "/Mul_3" || nm == "/Add" || nm == "/Add_1" ||
                          nm == "/Concat_3" || nm == "/Concat_4"))) {
                        auto dump_t = [&](const char *tag,
                                          const std::shared_ptr<ITensor> &tt) {
                            if (!tt) {
                                std::printf("[ROT]   %s=null\n", tag);
                                return;
                            }
                            bool f16 = (tt->dtype() == typeid(uint16_t));
                            bool f32 = (tt->dtype() == typeid(float));
                            auto sh = tt->getShape();
                            std::printf("[ROT]   %s dtype=%s size=%d shape=[",
                                        tag,
                                        f16 ? "f16" : (f32 ? "f32" : "other"),
                                        tt->size());
                            for (size_t d = 0; d < sh.size(); ++d)
                                std::printf("%d%s", sh[d],
                                            d + 1 < sh.size() ? "," : "");
                            std::printf("] vals=");
                            if (f16) {
                                auto tg = as_tensor<uint16_t>(tt);
                                tg->copyToCPU(m_cmdpool_);
                                int ne = (int)(tt->size() / 2);
                                for (int i = 0; i < 8 && i < ne; ++i)
                                    std::printf(
                                        "%.4g ",
                                        ITensor::fp16_to_fp32(
                                            reinterpret_cast<const uint16_t *>(
                                                tg->data().data())[i]));
                            } else if (f32) {
                                auto tg = as_tensor<float>(tt);
                                tg->copyToCPU(m_cmdpool_);
                                int ne = (int)(tt->size() / 4);
                                for (int i = 0; i < 8 && i < ne; ++i)
                                    std::printf("%.4g ",
                                                reinterpret_cast<const float *>(
                                                    tg->data().data())[i]);
                            }
                            std::printf("\n");
                        };
                        std::printf("[ROT] L=%d %s\n", (int)level_idx,
                                    nm.c_str());
                        const auto &rins = node_input_tensors_[node_idx];
                        for (size_t ii = 0; ii < rins.size(); ++ii) {
                            char buf[32];
                            std::snprintf(buf, sizeof(buf), "in[%zu]", ii);
                            dump_t(buf, rins[ii]);
                        }
                        dump_t("out", t);
                    }
                    ni++;
                }
            }
            for (auto &c : cur_cmds) {
                c->clearWaits();
                c->reset();
            }
            prev_cmds = std::move(cur_cmds);
        }
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        return elapsed.count() * 1000.0F;
    }

    std::vector<std::shared_ptr<VulkanCommandBuffer>> last_commands(
        vkop::kInflight);

    // Submission strategy. The graph is submitted ONE LEVEL AT A TIME: each
    // level's command buffers go into their own vkQueueSubmit call, and every
    // command in level N waits (via timeline semaphore) on the commands in
    // level N-1 that produced its inputs (direct data deps) — plus, as an
    // anti-aliasing barrier, on ALL of level N-1's commands.
    //
    // Why not one big vkQueueSubmit for the whole graph (the old strategy):
    // outshape_tensor_map recycles a Tensor (and its backing VkBuffer) the
    // moment its ref_cnt hits 0 during build — i.e. when its last TOPOLOGICAL
    // consumer is processed. An unrelated producer C at a later level can then
    // grab that recycled buffer as its output. C and the last reader B share
    // the SAME VkBuffer but may have no data-dependency edge between them, so
    // the GPU could schedule C's write while B still reads it. On this driver
    // a single vkQueueSubmit carrying thousands of command buffers with
    // timeline-semaphore waits also produced all-zero outputs (the waits were
    // not honored correctly); splitting per-level restored correct results.
    //
    // No deadlock risk from multiple submits on a single queue: each level's
    // commands only wait on EARLIER levels' signal semaphores (deps + the
    // previous-level barrier), never on a later submit, so submits resolve in
    // order. Intra-level parallelism is preserved within each submit.
    //
    // VKOP_LEVEL_SYNC=1 forces a CPU wait+reset after every level (NANSCAN-
    // style, without the NaN scan) for debugging; the default (0/2) just waits
    // for the final level at the end.
    const char *lsync_env = std::getenv("VKOP_LEVEL_SYNC");
    int level_sync = lsync_env ? std::atoi(lsync_env) : 0;
    bool per_level_wait = (level_sync == 1);

    std::vector<std::shared_ptr<VulkanCommandBuffer>> prev_level_cmds;
    size_t last_level_index = level_node_indices_.size() - 1;
    for (size_t level_idx = 0; level_idx < level_node_indices_.size();
         level_idx++) {
        const auto &level_nodes = level_node_indices_[level_idx];
        std::vector<std::shared_ptr<VulkanCommandBuffer>> cur_level_cmds;
        // Per-lane submit batches (multi-queue case); single-queue uses [0].
        std::vector<std::vector<VkSubmitInfo>> sis(vkop::kInflight);
        int id = 0;
        for (auto node_idx : level_nodes) {
            const auto &shapes = node_input_shapes_[node_idx];
            const auto &ins = node_input_tensors_[node_idx];
            for (size_t k = 0; k < ins.size() && k < shapes.size(); ++k) {
                if (!ins[k]) {
                    continue;
                }
                if (shapes[k].empty()) {
                    // A recorded [] shape means "scalar Constant" only when
                    // this tensor is an initializer (ONNX shape () materialized
                    // as dims_=[1] by dag.py's np.atleast_1d). Reshape it back
                    // to rank-0 so Gather's index-rank → output-rank matches
                    // ORT. reshape_view's element-count guard
                    // (ne_new=1==ne_old) makes this a no-op for any
                    // multi-element tensor.
                    if (init_ptrs.count(ins[k].get())) {
                        ins[k]->reshape_view(shapes[k]);
                    }
                } else {
                    // Skip recorded-view reset when the recorded shape carries
                    // a -1 "dynamic" sentinel — the live tensor's concrete
                    // runtime shape is authoritative (see the eager-record path
                    // above for the full rationale). 0 = genuinely empty
                    // (applied).
                    bool has_dyn = false;
                    for (int d : shapes[k]) {
                        if (d == -1) {
                            has_dyn = true;
                            break;
                        }
                    }
                    if (!has_dyn) {
                        ins[k]->reshape_view(shapes[k]);
                    }
                }
            }
            node_ops_[node_idx]->onExecute(node_input_tensors_[node_idx],
                                           node_output_tensors_[node_idx], id);
            auto cmd = node_ops_[node_idx]->get_record();
            for (auto &dep : node_dependency_indices_[node_idx]) {
                cmd->addWait(node_ops_[dep]->get_record()->getSignalSemaphore(),
                             node_ops_[dep]->get_record()->getSignalValue());
            }
            // Anti-aliasing level barrier: wait on every previous-level
            // command (direct data deps are a subset of this for true
            // producers; redundant addWait on an already-waited semaphore is
            // harmless).
            for (const auto &pc : prev_level_cmds) {
                cmd->addWait(pc->getSignalSemaphore(), pc->getSignalValue());
            }

            int lane = single_queue ? 0 : id;
            sis[lane].push_back(cmd->buildSubmitInfo());
            cur_level_cmds.push_back(cmd);
            if (level_idx == last_level_index) {
                last_commands[single_queue ? 0 : id] = cmd;
            }
            id++;
            id %= vkop::kInflight;
        }
        // Submit this level's batches (one vkQueueSubmit per non-empty lane).
        int nlanes = single_queue ? 1 : vkop::kInflight;
        for (int ci = 0; ci < nlanes; ci++) {
            if (!sis[ci].empty()) {
                VulkanCommandBuffer::submit(dev->getComputeQueue(ci), sis[ci]);
            }
        }
        if (per_level_wait) {
            for (auto &c : cur_level_cmds)
                c->wait();
            for (auto &c : cur_level_cmds) {
                c->clearWaits();
                c->reset();
            }
        }
        prev_level_cmds = std::move(cur_level_cmds);
    }

    // Wait for the final level, then reset all command buffers.
    for (int ci = 0; ci < vkop::kInflight; ci++) {
        if (last_commands[ci]) {
            last_commands[ci]->wait();
        }
    }
    // The final-level wait (timeline semaphore above) guarantees the GPU has
    // reached the final level, but on the CPU side the intermediate-level
    // command buffers may still read as "pending" to vkResetCommandBuffer.
    // Resetting a pending buffer leaves the driver with an invalid handle
    // (observed as a segfault inside vkBeginCommandBuffer on the NEXT Run() —
    // e.g. LLM decode round1 reusing round0's command buffers). CPU-wait every
    // command before reset so all buffers are back in the initial state.
    for (const auto &level_nodes : level_node_indices_) {
        for (auto node_idx : level_nodes) {
            auto cmd = node_ops_[node_idx]->get_record();
            cmd->wait();
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

    // The image backend packs tensors as image2DArray (NCHW->RGBA) and needs
    // >=3-D tensors (getGPUShape asserts ndim>=3). A post-process op like
    // Softmax over a 2-D [batch, classes] logits output can't be represented
    // as an image — force the buffer backend for it, mirroring setup.hpp's
    // low-rank Softmax handling. The buffer Softmax handles any rank.
    bool use_buffer = backend_buffer_;
    if (!use_buffer && ops == vkop::ops::OpType::SOFTMAX) {
        for (const auto &in : inputs) {
            if (in && in->num_dims() < 3) {
                use_buffer = true;
                break;
            }
        }
    }
    auto op = ops::create_from_type(
        ops, precision_, dev->is_support_nv_tensor_core(), use_buffer);
    op->set_name("post_" + convert_optype_to_string(ops));
    op->set_runtime_device(dev, m_cmdpool_);
    op->setAttribute(attributes);

    size_t current_op_idx = node_ops_.size();
    // Record each input's current (concrete) shape BEFORE moving inputs away,
    // so Run()'s per-node node_input_shapes_[node_idx] lookup stays in sync
    // with node_ops_ — the two vectors MUST stay parallel (Run indexes them by
    // node_idx). The post-process inputs are already-concrete output tensors,
    // so this just captures their live shape (reshape_view to the same shape
    // is a no-op at execute time).
    std::vector<std::vector<int>> post_input_shapes;
    post_input_shapes.reserve(inputs.size());
    for (const auto &in : inputs) {
        if (in) {
            post_input_shapes.push_back(in->getShape());
        } else {
            post_input_shapes.emplace_back();
        }
    }
    node_ops_.push_back(std::move(op));
    node_attrs_.push_back(attributes);
    node_input_tensors_.push_back(std::move(inputs));
    node_output_tensors_.push_back(std::move(outputs));
    node_input_shapes_.push_back(std::move(post_input_shapes));

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