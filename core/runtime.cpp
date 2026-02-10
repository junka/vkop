// junka @ 2025
#include <chrono>
#include <cstdint>
#include <limits>
#include <queue>

#include "core/runtime.hpp"
#include "model/load.hpp"
#include "ops/OperatorFactory.hpp"
#include "vulkan/VulkanCommandBuffer.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/vulkan_core.h"
#ifdef USE_MEASURE_TIME
#include "include/logger.hpp"
#include "vulkan/VulkanQueryPool.hpp"
#endif

namespace vkop {
std::shared_ptr<VulkanBuffer> ops::Operator::dummy_buffer_ = nullptr;
std::shared_ptr<VulkanBufferView> ops::Operator::dummy_bufferview_ = nullptr;
std::atomic<int> ops::Operator::instance_count_{0};
namespace core {

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
    // we can then use type to tell whether we can use fp16 as option
    for (const auto &n : model.nodes) {
        for (const auto &in_shape : n.inputs) {
            inputs_for_node_type[in_shape.name] = n.op_type;
            consumers[in_shape.name] += 1;
        }
    }
    printf("Total nodes %zu\n", model.nodes.size());
    for (const auto &i : model.inputs) {
        // this is enough for input image
        auto t = std::make_shared<Tensor<uint16_t>>(i.dims);
        t->set_ref_cnt_forever();
        inputs_[i.name] = t;
        tensor_map[i.name] = t;
        t->as_input_image(dev, nullptr);
    }

    for (const auto &o : model.outputs) {
        auto t = std::make_shared<Tensor<float>>(o.dims, true);
        t->set_ref_cnt_forever();
        outputs_[o.name] = t;
        tensor_map[o.name] = t;
        real_outputs_[o.name] = t;
    }

    auto handle_floating_point_tensor = [&](const load::Initializer &init,
                                            auto *src_ptr, auto &tensor) {
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
            if (inputs_for_node_type[init.name] == "Conv" ||
                inputs_for_node_type[init.name] == "BatchNormalization") {
                tensor->as_uniform_bufferview(dev);
            } else {
                tensor->as_storage_buffer(dev);
            }
            tensor->copyToGPU(m_cmdpool_, src_ptr);
        } else {
            tensor->as_input_image(dev, nullptr, false, true);
            tensor->copyToGPUImage(m_cmdpool_, src_ptr, model.rgba);
        }
        tensor_map[init.name] = tensor;
        initializers_[init.name] = tensor;
    };

    auto handle_unified_tensors = [&](const load::Initializer &init,
                                      auto *src_ptr, auto &tensor, auto &meta,
                                      auto &buffer) {
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
            tensor->as_uniform_bufferview(dev, buffer, meta.offset);
        } else {
            tensor->as_input_image(dev, nullptr, false, true);
            tensor->copyToGPUImage(m_cmdpool_, src_ptr, model.rgba);
        }
        tensor_map[init.name] = tensor;
        initializers_[init.name] = tensor;
    };

    if (model.unified && model.initializers.find("unified_tensors") !=
                             model.initializers.end()) {
        size_t meta_offset = model.initializer_offsets["unified_metadata"];
        auto names_offset = model.initializer_offsets["unified_names"];
        auto dims = model.initializers["unified_metadata"].dims;

        int num_metas = dims[0] / 8;
        uint8_t *meta_ptr = model.initializer_memory.data() + meta_offset;
        uint8_t *name_ptr = model.initializer_memory.data() + names_offset;
        std::vector<load::UnifiedMetadata> metas(num_metas);
        std::memcpy(metas.data(), meta_ptr,
                    sizeof(load::UnifiedMetadata) * num_metas);

        auto unified = model.initializers["unified_tensors"];
        size_t offset = model.initializer_offsets["unified_tensors"];
        uint8_t *src_ptr = model.initializer_memory.data() + offset;
        auto unified_tensor = std::make_shared<Tensor<float>>(unified.dims);
        unified_tensor->set_ref_cnt_forever();
        auto buffer = unified_tensor->as_uniform_buffer(dev);
        unified_tensor->copyToGPU(m_cmdpool_,
                                  reinterpret_cast<float *>(src_ptr));

        size_t name_idx_offset = 0;
        for (int i = 0; i < num_metas; ++i) {
            auto &meta = metas[i];
            auto name = std::string(name_ptr + name_idx_offset,
                                    name_ptr + name_idx_offset + meta.name_len);
            name_idx_offset += meta.name_len;
            auto init = model.initializers[name];
            if (init.dtype == "float32") {
                auto t = std::make_shared<Tensor<float>>(init.dims);
                handle_unified_tensors(init, reinterpret_cast<float *>(src_ptr),
                                       t, meta, buffer);
            } else if (init.dtype == "float16") {
                auto t = std::make_shared<Tensor<uint16_t>>(init.dims);
                handle_unified_tensors(init,
                                       reinterpret_cast<uint16_t *>(src_ptr), t,
                                       meta, buffer);
            } else if (init.dtype == "int8") {
                auto t = std::make_shared<Tensor<int8_t>>(init.dims);
                handle_unified_tensors(
                    init, reinterpret_cast<int8_t *>(src_ptr), t, meta, buffer);
            } else {
                throw std::runtime_error("Unsupported data type: " +
                                         init.dtype);
            }

            model.initializers.erase(name);
        }

        model.initializers.erase("unified_tensors");
        model.initializers.erase("unified_metadata");
        model.initializers.erase("unified_names");
    }

    for (const auto &itr : model.initializers) {
        auto init = itr.second;
        size_t offset = model.initializer_offsets[init.name];
        uint8_t *src_ptr = model.initializer_memory.data() + offset;
        if (init.dtype == "int64") {
            auto t = std::make_shared<Tensor<int64_t>>(init.dims);
            t->set_ref_cnt_forever();
            t->fillToCPU(reinterpret_cast<int64_t *>(src_ptr));
            tensor_map[init.name] = t;
            initializers_[init.name] = t;
        } else if (init.dtype == "int32") {
            auto t = std::make_shared<Tensor<int>>(init.dims);
            t->set_ref_cnt_forever();
            t->fillToCPU(reinterpret_cast<int32_t *>(src_ptr));
            tensor_map[init.name] = t;
            initializers_[init.name] = t;
        } else if (init.dtype == "float32") {
            auto t = std::make_shared<Tensor<float>>(init.dims);
            handle_floating_point_tensor(init,
                                         reinterpret_cast<float *>(src_ptr), t);
        } else if (init.dtype == "float16") {
            auto t = std::make_shared<Tensor<uint16_t>>(init.dims);
            handle_floating_point_tensor(
                init, reinterpret_cast<uint16_t *>(src_ptr), t);
        } else if (init.dtype == "int8") {
            auto t = std::make_shared<Tensor<int8_t>>(init.dims);
            handle_floating_point_tensor(
                init, reinterpret_cast<int8_t *>(src_ptr), t);
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
            bool use_ssbo = false;
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
                    std::string key = "_";
                    for (const auto &dim : out_shape.dims) {
                        key += std::to_string(dim) + "_";
                    }
                    auto q = outshape_tensor_map[key];
                    if (!q.empty()) {
                        auto t = q.front();
                        q.pop();
                        t->set_ref_cnt(consumers[out_shape.name]);
                        tensor_map[out_shape.name] = t;
                        node_outputs.push_back(t);
                    } else {
                        auto t = std::make_shared<Tensor<float>>(out_shape.dims,
                                                                 true);
                        t->set_ref_cnt(consumers[out_shape.name]);
                        tensor_map[out_shape.name] = t;
                        node_outputs.push_back(t);
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
                    auto q = outshape_tensor_map[key];
                    q.push(t);
                }
            }
            if (type == vkop::ops::OpType::SOFTMAX) {
                if (node_outputs[0]->num_dims() <= 2) {
                    use_ssbo = true;
                }
            }
            auto op = ops::create_from_type(type, use_ssbo);
            if (!op) {
                std::cout << "Fail to create operator" << std::endl;
                return;
            }

            op->set_runtime_device(dev, m_cmdpool_);
            op->setAttribute(n.attributes);
            op->set_name(n.name);

            level_node_indices_[level_idx].push_back(global_node_index++);
            node_ops_.push_back(std::move(op));
            node_attrs_.push_back(n.attributes);
            node_input_tensors_.push_back(std::move(node_inputs));
            node_output_tensors_.push_back(std::move(node_outputs));
        }
    }
    printf("Execution plan built with %zu operations\n", node_ops_.size());
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

double Runtime::Run() {
    auto dev = m_cmdpool_->getVulkanDevice();
    auto start = std::chrono::steady_clock::now();

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

            submit_infos[id].push_back(cmd->buildSubmitInfo());
            if (level_idx == last_level_index) {
                last_commands[id] = cmd;
            }
            id++;
            id %= vkop::kInflight;
        }
    }
    for (int ci = 0; ci < vkop::kInflight; ci++) {
        if (!submit_infos[ci].empty())
            VulkanCommandBuffer::submit(dev->getComputeQueue(ci),
                                        submit_infos[ci]);
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

    auto op = ops::create_from_type(ops, outputs[0]->num_dims() <= 2);
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