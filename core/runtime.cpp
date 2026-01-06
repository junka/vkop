// junka @ 2025
#include <chrono>
#include <limits>
#include <queue>

#include "core/runtime.hpp"
#include "include/logger.hpp"
#include "model/load.hpp"
#include "ops/OperatorFactory.hpp"
#include "vulkan/VulkanCommandBuffer.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanQueryPool.hpp"

namespace vkop {
namespace core {

Runtime::Runtime(const std::shared_ptr<VulkanCommandPool> &cmdpool,
                 std::string model_path, std::string cache_dir)
    : m_cmdpool_(std::move(cmdpool)), model_path_(std::move(model_path)),
      cache_dir_(std::move(cache_dir)) {
    for (int id = 0; id < vkop::kInflight; id++) {
        m_cmds_[id] =
            std::make_shared<VulkanCommandBuffer>(m_cmdpool_, true, id);
    }
}

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
        t->as_input_image(dev, nullptr, false);
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
            tensor->as_storage_buffer(dev);
            tensor->copyToGPU(m_cmdpool_, src_ptr);
        } else {
            tensor->as_input_image(dev, nullptr);
            tensor->copyToGPU(m_cmdpool_, src_ptr);
        }
        tensor_map[init.name] = tensor;
        initializers_[init.name] = tensor;
    };

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

    for (const auto &n : model.nodes) {
        auto type = vkop::ops::convert_opstring_to_enum(n.op_type);
        if (type == vkop::ops::OpType::UNKNOWN) {
            // make it as input for next ops
            continue;
        }
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
                printf("we should not reach here %s\n", in_shape.name.c_str());
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
                    auto t =
                        std::make_shared<Tensor<float>>(out_shape.dims, true);
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
        auto op = ops::create_from_type(type);
        if (!op) {
            std::cout << "Fail to create operator" << std::endl;
            return;
        }

        op->set_runtime_device(dev, m_cmdpool_);
        op->setAttribute(n.attributes);

        node_ops_.push_back(std::move(op));
        node_attrs_.push_back(n.attributes);
        node_input_tensors_.push_back(std::move(node_inputs));
        node_output_tensors_.push_back(std::move(node_outputs));
    }
}

std::shared_ptr<ITensor> Runtime::GetInput(const std::string &name) {
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

std::shared_ptr<ITensor> Runtime::GetOutput(const std::string &name) {
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

std::shared_ptr<ITensor> Runtime::GetInitializer(const std::string &name) {
    auto it = initializers_.find(name);
    if (it == initializers_.end()) {
        return nullptr;
    }
    return it->second;
}

double Runtime::Run() {
    auto dev = m_cmdpool_->getVulkanDevice();
    auto start = std::chrono::steady_clock::now();
    m_cmds_[id_]->wait(dev->getComputeQueue(id_));
    m_cmds_[id_]->begin();
#ifdef USE_MEASURE_TIME
    VulkanQueryPool query_pool(dev->getLogicalDevice(), 2,
                               VK_QUERY_TYPE_TIMESTAMP);
    query_pool.begin(m_cmds_[id_]->get());
#endif
    for (size_t i = 0; i < node_ops_.size(); ++i) {
        node_ops_[i]->onExecute(node_input_tensors_[i], node_output_tensors_[i],
                                m_cmds_[id_], id_);
    }
#ifdef USE_MEASURE_TIME
    query_pool.end(m_cmds_[id_]->get());
#endif
    m_cmds_[id_]->end();
    m_cmds_[id_]->submit(dev->getComputeQueue(id_));
#ifdef USE_MEASURE_TIME
    auto r = query_pool.getResults();
    LOG_INFO("Time: %f s", static_cast<double>(r[1] - r[0]) * (1e-9) *
                               dev->getTimestampPeriod());
#endif
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    id_++;
    id_ %= vkop::kInflight;
    return elapsed.count() * 1000.0F;
}

void Runtime::ReadResult() {
    auto dev = m_cmdpool_->getVulkanDevice();
    for (int i = 0; i < vkop::kInflight; ++i) {
        m_cmds_[i]->wait(dev->getComputeQueue(i));
    }
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
    node_ops_.push_back(std::move(op));
    node_attrs_.push_back(attributes);
    node_input_tensors_.push_back(std::move(inputs));
    node_output_tensors_.push_back(std::move(outputs));
    real_outputs_.clear();
    for (size_t i = 0; i < outputs.size(); ++i) {
        real_outputs_["post_" + convert_optype_to_string(ops) +
                      std::to_string(i)] = outputs[i];
    }
}

} // namespace core
} // namespace vkop