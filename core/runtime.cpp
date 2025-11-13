// junka @ 2025
#include <chrono>

#include "core/runtime.hpp"
#include "model/load.hpp"
#include "ops/OperatorFactory.hpp"

namespace vkop {
namespace core {
Runtime::Runtime(std::shared_ptr<VulkanDevice> dev,
                 std::shared_ptr<VulkanCommandPool> cmdpool,
                 const std::string &model_path, const std::string &cache_dir)
    : m_dev_(std::move(dev)), m_cmdpool_(std::move(cmdpool)),
      model_path_(std::move(model_path)), cache_dir_(std::move(cache_dir)) {}

void Runtime::LoadCache() {}

void Runtime::LoadModel() {
    auto model = load::VkModel(model_path_);
    std::unordered_map<std::string, std::shared_ptr<ITensor>> tensor_map;
    std::unordered_map<std::shared_ptr<ITensor>, std::string> tensor_name_map;

    for (const auto &i : model.inputs) {
        auto t = std::make_shared<Tensor<float>>(i.dims);
        inputs_[i.name] = t;
        tensor_map[i.name] = t;
        tensor_name_map[t] = i.name;
    }

    for (const auto &o : model.outputs) {
        auto t = std::make_shared<Tensor<float>>(o.dims);
        t->toGPU();
        outputs_[o.name] = t;
        tensor_map[o.name] = t;
        tensor_name_map[t] = o.name;
    }

    for (const auto &itr : model.initializers) {
        auto init = itr.second;
        auto t = std::make_shared<Tensor<float>>(init.dims);
        if (init.dtype == "int64") {
            auto t = std::make_shared<Tensor<int64_t>>(init.dims);
            for (int i = 0; i < t->num_elements(); ++i) {
                t->data()[i] = static_cast<float>(init.dataii[i]);
            }
            tensor_map[init.name] = t;
            tensor_name_map[t] = init.name;
            initializers_[init.name] = t;
        } else if (init.dtype == "int32") {
            auto t = std::make_shared<Tensor<int>>(init.dims);
            for (int i = 0; i < t->num_elements(); ++i) {
                t->data()[i] = static_cast<float>(init.dataii[i]);
            }
            tensor_map[init.name] = t;
            tensor_name_map[t] = init.name;
            initializers_[init.name] = t;
        } else if (init.dtype == "float32") {
            auto t = std::make_shared<Tensor<float>>(init.dims);
            std::memcpy(t->data(), init.dataf.data(),
                        t->num_elements() * sizeof(float));
            tensor_map[init.name] = t;
            tensor_name_map[t] = init.name;
            initializers_[init.name] = t;
        } else {
            throw std::runtime_error(
                "Only float32/int32 initializer is supported for now " +
                init.dtype);
        }
    }

    for (const auto &n : model.nodes) {
        auto t = vkop::ops::convert_opstring_to_enum(n.op_type);
        if (t == vkop::ops::OpType::CONSTANT ||
            t == vkop::ops::OpType::UNKNOWN) {
            // make it as input for next ops
            continue;
        }
        std::vector<std::shared_ptr<ITensor>> node_inputs;
        std::vector<std::shared_ptr<ITensor>> node_outputs;

        for (const auto &out_shape : n.outputs) {
            if (tensor_map.find(out_shape.name) != tensor_map.end()) {
                tensor_map[out_shape.name]->toGPU();
                node_outputs.push_back(tensor_map[out_shape.name]);
            } else {
                auto t = std::make_shared<Tensor<float>>(out_shape.dims);
                t->toGPU();
                tensor_map[out_shape.name] = t;
                tensor_name_map[t] = out_shape.name;
                node_outputs.push_back(t);
            }
        }
        for (const auto &in_shape : n.inputs) {
            if (tensor_map.find(in_shape.name) != tensor_map.end()) {
                node_inputs.push_back(tensor_map[in_shape.name]);
            } else if (in_shape.dims.empty()) {
                node_inputs.push_back(nullptr);
            }
        }

        auto op = ops::OperatorFactory::get_instance().create(t);
        if (!op) {
            std::cout << "Fail to create operator" << std::endl;
            return;
        }

        op->set_runtime_device(m_dev_, m_cmdpool_);
        if (!n.attributes.empty()) {
            op->setAttribute(n.attributes);
        }
        node_ops_.push_back(std::move(op));
        node_input_tensors_.push_back(std::move(node_inputs));
        node_output_tensors_.push_back(std::move(node_outputs));
    }

    for (auto &init : initializers_) {
        if (!init.second || init.second->num_dims() < 4) {
            continue;
        }
        auto t = vkop::core::as_tensor<float>(init.second);
        t->as_input_image(m_dev_, m_cmdpool_);
        t->copyToGPU(m_dev_, m_cmdpool_);
    }
}

std::shared_ptr<ITensor> Runtime::GetInput(const std::string &name) {
    auto it = inputs_.find(name);
    if (it == inputs_.end()) {
        return nullptr;
    }
    return it->second;
}

std::shared_ptr<ITensor> Runtime::GetOutput(const std::string &name) {
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

void Runtime::Run() {
    for (auto &p : inputs_) {
        auto t = vkop::core::as_tensor<float>(p.second);
        t->copyToGPU(m_dev_, m_cmdpool_);
    }
    auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < node_ops_.size(); ++i) {
        // printf("ops %s input tensors %ld\n",
        // vkop::ops::convert_openum_to_string(ops_all[i]->get_type()).c_str(),
        // inputs_all[i].size());
        node_ops_[i]->apply(node_input_tensors_[i], node_output_tensors_[i]);
        node_ops_[i]->execute(node_input_tensors_[i], node_output_tensors_[i]);
    }
    for (auto &p : outputs_) {
        auto t = vkop::core::as_tensor<float>(p.second);
        t->copyToCPU(m_dev_, m_cmdpool_);
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "inference time:" << elapsed.count() << " s" << std::endl;
}

} // namespace core
} // namespace vkop