// junka @ 2025
#include <chrono>
#include <limits>
#include <numeric>
#include <queue>

#include "core/runtime.hpp"
#include "include/logger.hpp"
#include "model/load.hpp"
#include "ops/OperatorFactory.hpp"
#include "vulkan/VulkanCommandBuffer.hpp"
#include "vulkan/VulkanPipeline.hpp"
#include "vulkan/VulkanQueryPool.hpp"

namespace vkop {
namespace core {

Runtime::Runtime(std::shared_ptr<VulkanCommandPool> cmdpool,
                 std::string model_path, std::string cache_dir)
    : m_cmdpool_(std::move(cmdpool)), model_path_(std::move(model_path)),
      cache_dir_(std::move(cache_dir)) {
    for (int id = 0; id < vkop::kInflight; id++) {
        m_cmds_[id] =
            std::make_shared<VulkanCommandBuffer>(m_cmdpool_, true, id);
    }
}

Runtime::~Runtime() {
    auto dev = m_cmdpool_->getVulkanDevice();
    for (int id = 0; id < vkop::kInflight; id++) {
        m_cmds_[id]->wait(dev->getComputeQueue(id));
        m_cmds_[id].reset();
    }
}

void Runtime::LoadCache() {}

void Runtime::LoadModel() {
    auto model = load::VkModel(model_path_);
    // model.dump_model();
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
        auto t = std::make_shared<Tensor<float>>(i.dims);
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

        if (tensor->num_dims() == 2 || tensor->num_dims() == 1) {
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
                printf("we should not reach here\n");
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

        auto op = ops::OperatorFactory::get_instance().create(t);
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
    auto dev = m_cmdpool_->getVulkanDevice();
    static int id = 0;
    for (auto &p : inputs_) {
        if (p.second->dtype() == typeid(float)) {
            auto t = vkop::core::as_tensor<float>(p.second);
            t->copyToGPU(m_cmdpool_);
        } else if (p.second->dtype() == typeid(uint16_t)) {
            auto t = vkop::core::as_tensor<uint16_t>(p.second);
            t->copyToGPU(m_cmdpool_);
        }
    }
    auto start = std::chrono::steady_clock::now();
    m_cmds_[id]->wait(dev->getComputeQueue(id));
    m_cmds_[id]->begin();
#ifdef USE_MEASURE_TIME
    VulkanQueryPool query_pool(dev->getLogicalDevice(), 2,
                               VK_QUERY_TYPE_TIMESTAMP);
    query_pool.begin(m_cmds_[id]->get());
#endif
    for (size_t i = 0; i < node_ops_.size(); ++i) {
        // printf("ops %s input tensors %ld\n",
        //        vkop::ops::convert_openum_to_string(node_ops_[i]->get_type())
        //            .c_str(),
        //        node_input_tensors_[i].size());

        node_ops_[i]->onExecute(node_input_tensors_[i], node_output_tensors_[i],
                                m_cmds_[id], id);
        for (auto &it : node_input_tensors_[i]) {
            auto t = vkop::core::as_tensor<float>(it);
            if (t && t->ref_cnt() == 1) {
                // t->resize(0);
            }
        }
    }
#ifdef USE_MEASURE_TIME
    query_pool.end(m_cmds_[id]->get());
#endif
    m_cmds_[id]->end();
    m_cmds_[id]->submit(dev->getComputeQueue(id));
#ifdef USE_MEASURE_TIME
    auto r = query_pool.getResults();
    LOG_INFO("Time: %f s", static_cast<double>(r[1] - r[0]) * (1e-9) *
                               dev->getTimestampPeriod());
#endif
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "inference time:" << elapsed.count() * 1000.0F << " ms"
              << std::endl;
    id++;
    id %= vkop::kInflight;
}

void Runtime::ReadResult() {
    for (auto &p : outputs_) {
        auto t = vkop::core::as_tensor<float>(p.second);
        t->copyToCPU(m_cmdpool_);
    }
}

} // namespace core
} // namespace vkop