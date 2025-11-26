// junka @ 2025
#include <chrono>
#include <numeric>

#include "core/runtime.hpp"
#include "model/load.hpp"
#include "ops/OperatorFactory.hpp"
#include "vulkan/VulkanCommandBuffer.hpp"

namespace vkop {
namespace core {

Runtime::Runtime(std::shared_ptr<VulkanDevice> dev,
                 std::shared_ptr<VulkanCommandPool> cmdpool,
                 const std::string &model_path, const std::string &cache_dir)
    : m_dev_(std::move(dev)), m_cmdpool_(std::move(cmdpool)),
      model_path_(std::move(model_path)), cache_dir_(std::move(cache_dir)) {
    m_cmd_ = std::make_shared<VulkanCommandBuffer>(m_dev_, m_cmdpool_);
}

void Runtime::LoadCache() {}

void Runtime::LoadModel() {
    auto model = load::VkModel(model_path_);
    std::unordered_map<std::string, std::shared_ptr<ITensor>> tensor_map;
    std::unordered_map<std::shared_ptr<ITensor>, std::string> tensor_name_map;

    std::unordered_map<std::string, std::string> inputs_for_node_type;

    // preprocess inputs, make sure we know node types for inputs
    // we can then use type to tell whether we can use fp16 as option
    for (const auto &n : model.nodes) {
        for (const auto &in_shape : n.inputs) {
            inputs_for_node_type[in_shape.name] = n.op_type;
        }
    }
    for (const auto &i : model.inputs) {
#ifdef FP16
        // not really needed, TODO: make it more flexible
        if (inputs_for_node_type.find(i.name) != inputs_for_node_type.end() &&
            (inputs_for_node_type[i.name] == "Conv" ||
             inputs_for_node_type[i.name] == "Add")) {
            auto t = std::make_shared<Tensor<uint16_t>>(i.dims);
            t->set_ref_cnt_forever();
            inputs_[i.name] = t;
            tensor_map[i.name] = t;
            tensor_name_map[t] = i.name;
            t->as_input_image(m_dev_, nullptr);
        } else {
#endif
            auto t = std::make_shared<Tensor<float>>(i.dims);
            t->set_ref_cnt_forever();
            inputs_[i.name] = t;
            tensor_map[i.name] = t;
            tensor_name_map[t] = i.name;
            t->as_input_image(m_dev_, nullptr);
#ifdef FP16
        }
#endif
    }

    for (const auto &o : model.outputs) {
        auto t = std::make_shared<Tensor<float>>(o.dims);
        t->set_ref_cnt_forever();
        t->toGPU();
        outputs_[o.name] = t;
        tensor_map[o.name] = t;
        tensor_name_map[t] = o.name;
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
            tensor_name_map[t] = init.name;
            initializers_[init.name] = t;
        } else if (init.dtype == "int32") {
            auto t = std::make_shared<Tensor<int>>(init.dims);
            t->set_ref_cnt_forever();
            t->fillToCPU(reinterpret_cast<int32_t *>(src_ptr));
            tensor_map[init.name] = t;
            tensor_name_map[t] = init.name;
            initializers_[init.name] = t;
        } else if (init.dtype == "float32") {
#ifdef FP16
            if (inputs_for_node_type.find(init.name) !=
                    inputs_for_node_type.end() &&
                (inputs_for_node_type[init.name] == "Conv" ||
                 inputs_for_node_type[init.name] == "Add")) {
                // printf("save as fp16 %s\n", init.name.c_str());
                auto t = std::make_shared<Tensor<uint16_t>>(init.dims);
                t->set_ref_cnt_forever();
                if (t->num_dims() == 2 || t->num_dims() == 1) {
                    if ((t->num_dims() == 1 && t->num_elements() <= 4)) {
                        t->fillFP32ToCPU(reinterpret_cast<float *>(src_ptr));
                    } else {
                        t->as_storage_buffer(m_dev_);
                        std::vector<uint16_t> t_datah(t->num_elements());
                        for (int i = 0; i < t->num_elements(); i++) {
                            t_datah[i] = ITensor::fp32_to_fp16(
                                reinterpret_cast<float *>(src_ptr)[i]);
                        }

                        t->copyToGPU(m_dev_, m_cmdpool_, t_datah.data());
                    }
                } else {
                    t->as_input_image(m_dev_, nullptr);
                    std::vector<uint16_t> t_datah(t->num_elements());
                    for (int i = 0; i < t->num_elements(); i++) {
                        t_datah[i] = ITensor::fp32_to_fp16(
                            reinterpret_cast<float *>(src_ptr)[i]);
                    }
                    t->copyToGPU(m_dev_, m_cmdpool_, t_datah.data());
                }
                tensor_map[init.name] = t;
                tensor_name_map[t] = init.name;
                initializers_[init.name] = t;
            } else {
#endif
                auto t = std::make_shared<Tensor<float>>(init.dims);
                t->set_ref_cnt_forever();
                if (t->num_dims() == 2 || t->num_dims() == 1) {
                    if ((t->num_dims() == 1 && t->num_elements() <= 4)) {
                        t->fillToCPU(reinterpret_cast<float *>(src_ptr));
                    } else {
                        t->as_storage_buffer(m_dev_);
                        t->copyToGPU(m_dev_, m_cmdpool_,
                                     reinterpret_cast<float *>(src_ptr));
                    }
                } else {
                    t->as_input_image(m_dev_, nullptr);
                    t->copyToGPU(m_dev_, m_cmdpool_,
                                 reinterpret_cast<float *>(src_ptr));
                }
                tensor_map[init.name] = t;
                tensor_name_map[t] = init.name;
                initializers_[init.name] = t;
#ifdef FP16
            }
#endif
        } else if (init.dtype == "float16") {
            auto t = std::make_shared<Tensor<uint16_t>>(init.dims);
            t->set_ref_cnt_forever();
            if (t->num_dims() == 2 || t->num_dims() == 1) {
                if ((t->num_dims() == 1 && t->num_elements() <= 4)) {
                    t->fillToCPU(reinterpret_cast<uint16_t *>(src_ptr));
                } else {
                    t->as_storage_buffer(m_dev_);
                    t->copyToGPU(m_dev_, m_cmdpool_,
                                 reinterpret_cast<uint16_t *>(src_ptr));
                }
            } else {
                t->as_input_image(m_dev_, nullptr);
                t->copyToGPU(m_dev_, m_cmdpool_,
                             reinterpret_cast<uint16_t *>(src_ptr));
            }
            tensor_map[init.name] = t;
            tensor_name_map[t] = init.name;
            initializers_[init.name] = t;
        } else {
            throw std::runtime_error(
                "Only float32/int32/fp16 initializer is supported for now " +
                init.dtype);
        }
    }

    for (const auto &n : model.nodes) {
        auto t = vkop::ops::convert_opstring_to_enum(n.op_type);
        if (t == vkop::ops::OpType::UNKNOWN) {
            // make it as input for next ops
            continue;
        }
        std::vector<std::shared_ptr<ITensor>> node_inputs;
        std::vector<std::shared_ptr<ITensor>> node_outputs;

        for (const auto &out_shape : n.outputs) {
            if (tensor_map.find(out_shape.name) != tensor_map.end()) {
                assert(tensor_map[out_shape.name]->is_on_GPU());
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
                auto t = tensor_map[in_shape.name];
                if (t->ref_cnt() != std::numeric_limits<uint16_t>::max()) {
                    t->ref_inc();
                }
                node_inputs.push_back(tensor_map[in_shape.name]);
            } else if (in_shape.dims.empty()) {
                node_inputs.push_back(nullptr);
            } else {
                printf("we should not reach here\n");
                assert(false);
            }
        }

        auto op = ops::OperatorFactory::get_instance().create(t);
        if (!op) {
            std::cout << "Fail to create operator" << std::endl;
            return;
        }

        op->set_runtime_device(m_dev_, m_cmdpool_, m_cmd_);

        node_ops_.push_back(std::move(op));
        node_attrs_.push_back(n.attributes);
        node_input_tensors_.push_back(std::move(node_inputs));
        node_output_tensors_.push_back(std::move(node_outputs));
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
        if (p.second->dtype() == typeid(float)) {
            auto t = vkop::core::as_tensor<float>(p.second);
            t->copyToGPU(m_dev_, m_cmdpool_);
        } else if (p.second->dtype() == typeid(uint16_t)) {
            auto t = vkop::core::as_tensor<uint16_t>(p.second);
            t->copyToGPU(m_dev_, m_cmdpool_);
        }
    }
    auto start = std::chrono::steady_clock::now();
    m_cmd_->wait();
    m_cmd_->begin();
    for (size_t i = 0; i < node_ops_.size(); ++i) {
        // printf("ops %s input tensors %ld\n",
        //        vkop::ops::convert_openum_to_string(node_ops_[i]->get_type())
        //            .c_str(),
        //        node_input_tensors_[i].size());

        node_ops_[i]->setAttribute(node_attrs_[i]);
        node_ops_[i]->onExecute(node_input_tensors_[i],
                                node_output_tensors_[i]);
        for (auto &it : node_input_tensors_[i]) {
            auto t = vkop::core::as_tensor<float>(it);
            if (t && t->ref_cnt() == 1) {
                // t->resize(0);
            }
        }
    }
    m_cmd_->end();
    m_cmd_->submit(m_dev_->getComputeQueue());
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "inference time:" << elapsed.count() << " s" << std::endl;
}

void Runtime::ReadResult() {
    for (auto &p : outputs_) {
        auto t = vkop::core::as_tensor<float>(p.second);
        t->copyToCPU(m_dev_, m_cmdpool_);
    }
}

} // namespace core
} // namespace vkop