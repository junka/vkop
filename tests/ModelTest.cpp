#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "include/logger.hpp"
#include "core/Tensor.hpp"
#include "model/load.hpp"
#include "ops/OperatorFactory.hpp"
#include "ops/Ops.hpp"

#include <unordered_set>
#include <random>
#include <vector>
#include <cmath>

#include <vulkan/vulkan_core.h>

using vkop::VulkanInstance;
using vkop::VulkanDevice;
using vkop::core::Tensor;
using vkop::core::ITensor;
using vkop::load::VkModel;
using vkop::ops::OperatorFactory;

namespace {
class ModelTest {
public:
    std::vector<int> input_shape_ = {
        1, 3, 224, 224
    };
    std::vector<float> expectedOutput;

    ModelTest() = default;
    void initTestData(const std::shared_ptr<Tensor<float>>& ta, const std::shared_ptr<Tensor<float>>& tb) {
        ta->printTensorShape();
        auto *inputa_ptr = ta->data();
        auto *inputb_ptr = tb->data();
        expectedOutput.resize(ta->num_elements());
        
        std::random_device rd{};
        std::mt19937 gen{rd()};
        gen.seed(1024);
        std::normal_distribution<> inputa_dist{-1.0F, 1.0F};
        std::normal_distribution<> inputb_dist{1.0F, 2.0F};
        for (int i = 0; i < ta->num_elements(); i++) {
            auto a = inputa_dist(gen);
            auto b = inputb_dist(gen);
            inputa_ptr[i] = a;
            inputb_ptr[i] = b;
            expectedOutput[i] = a+b;
            if (i == 0) {
                LOG_INFO("inputa: %f, inputb: %f, expectedOutput: %f", inputa_ptr[i], inputb_ptr[i], expectedOutput[i]);
            }
        }
    }
};
}


int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", true);
    VkPhysicalDevice phydev = VK_NULL_HANDLE;
    std::shared_ptr<VulkanDevice> dev;
    try {
        auto phydevs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
        for (auto *pdev : phydevs) {
            dev = std::make_shared<VulkanDevice>(pdev);
            if (dev->getDeviceName().find("llvmpipe") != std::string::npos) {
                continue;
            }
            phydev = pdev;
            LOG_INFO("%s",dev->getDeviceName().c_str());
        }
    } catch (const std::exception &e) {
        LOG_ERROR("%s", e.what());
        return EXIT_FAILURE;
    }
    auto *device = dev->getLogicalDevice();
    auto cmdpool = std::make_shared<vkop::VulkanCommandPool>(device, dev->getComputeQueueFamilyIndex());
    std::string binary_file_path = TEST_DATA_PATH"/add_model.bin";
    VkModel model(binary_file_path);
    VkModel::dump_model(model);

    std::vector<std::shared_ptr<ITensor>> inputs;
    std::vector<std::shared_ptr<ITensor>> outputs;
    std::unordered_set<std::shared_ptr<ITensor>> output_tensor_set;
    std::unordered_map<std::string, std::shared_ptr<ITensor>> tensor_map;

    std::vector<std::unique_ptr<vkop::ops::Operator>> ops_all;
    std::vector<std::vector<std::shared_ptr<ITensor>>> inputs_all;
    std::vector<std::vector<std::shared_ptr<ITensor>>> outputs_all;
    for (const auto& i : model.inputs) {
        auto t = std::make_shared<Tensor<float>>(i.dims);
        inputs.push_back(t);
        tensor_map[i.name] = t;
    }

    for (const auto& o: model.outputs) {
        auto t = std::make_shared<Tensor<float>>(o.dims);
        t->toGPU();
        outputs.push_back(t);
        output_tensor_set.insert(t);
        tensor_map[o.name] = t;
    }

    for (const auto& n: model.nodes) {
        auto t = vkop::ops::convert_opstring_to_enum(n.op_type);
        if (t == vkop::ops::OpType::CONSTANT || t == vkop::ops::OpType::UNKNOWN) {
            continue;
        }
        std::vector<std::shared_ptr<ITensor>> node_inputs;
        std::vector<std::shared_ptr<ITensor>> node_outputs;

        for (const auto& in_shape : n.inputs) {
            if (tensor_map.find(in_shape.name) != tensor_map.end()) {
                node_inputs.push_back(tensor_map[in_shape.name]);
            } else {
                if (in_shape.dims.empty()) {
                    node_inputs.push_back(nullptr);
                    continue;
                }
                if (model.initializers.find(in_shape.name) != model.initializers.end()) {
                    auto& init = model.initializers.at(in_shape.name);
                    if (init.dims != in_shape.dims) {
                        throw std::runtime_error("Initializer dims do not match for " + in_shape.name);
                    }
                    if (init.dtype == "int64") {
                        auto t = std::make_shared<Tensor<int64_t>>(in_shape.dims);
                        for (int i = 0; i < t->num_elements(); ++i) {
                            t->data()[i] = static_cast<float>(init.dataii[i]);
                        }
                        tensor_map[in_shape.name] = t;
                        node_inputs.push_back(t);
                    } else if (init.dtype == "int32") {
                        auto t = std::make_shared<Tensor<int>>(in_shape.dims);
                        for (int i = 0; i < t->num_elements(); ++i) {
                            t->data()[i] = static_cast<float>(init.dataii[i]);
                        }
                        tensor_map[in_shape.name] = t;
                        node_inputs.push_back(t);
                    } else if (init.dtype == "float32") {
                        auto t = std::make_shared<Tensor<float>>(in_shape.dims);
                        std::memcpy(t->data(), init.dataf.data(), t->num_elements() * sizeof(float));
                        tensor_map[in_shape.name] = t;
                        node_inputs.push_back(t);
                    } else {
                        throw std::runtime_error("Only float32 initializer is supported for now " + init.dtype);
                    }
                }
            }
        }

        for (const auto& out_shape : n.outputs) {
            if (tensor_map.find(out_shape.name) != tensor_map.end()) {
                node_outputs.push_back(tensor_map[out_shape.name]);
            } else {
                auto t = std::make_shared<Tensor<float>>(out_shape.dims);
                t->toGPU();
                tensor_map[out_shape.name] = t;
                node_outputs.push_back(t);
            }
        }

        auto op = OperatorFactory::get_instance().create(t);
        if (!op) {
            std::cout << "Fail to create operator" << std::endl;
            return 1;
        }

        op->set_runtime_device(phydev, dev, cmdpool);
        if (!n.attributes.empty()) {
            op->setAttribute(n.attributes);
        }
        ops_all.push_back(std::move(op));
        inputs_all.push_back(node_inputs);
        outputs_all.push_back(node_outputs);
    }

    auto t1 = vkop::core::as_tensor<float>(inputs[0]);
    auto t2 = vkop::core::as_tensor<float>(inputs[1]);
    ModelTest test;
    test.initTestData(t1, t2);

    for (size_t i = 0; i < ops_all.size(); ++i) {
        ops_all[i]->apply(inputs_all[i], outputs_all[i]);
        for (auto &p: inputs_all[i]) {
            if (!p || p->num_dims() < 3) {
                continue;
            }
            if (p->is_on_GPU()) {
                printf("already on GPU\n");
            } else {
                printf("on CPU\n");
                auto t = vkop::core::as_tensor<float>(p);
                t->copyToGPU(dev, cmdpool);
            }
        }
        ops_all[i]->execute(inputs_all[i], outputs_all[i]);
        for (auto & j : outputs_all[i]) {
            if (output_tensor_set.find(j) != output_tensor_set.end()) {
                auto t = vkop::core::as_tensor<float>(j);
                t->copyToCPU(dev, cmdpool);
            }
        }
        auto result = vkop::core::as_tensor<float>(outputs[0]);
        for (int i = 0; i < result->num_elements(); ++i) {
            // printf("%f %f\n", result->data()[i], test.expectedOutput[i]);
            if (std::fabs(result->data()[i] - test.expectedOutput[i]) > 1e-5) {
                printf("Failed\n");
                return 1;
            }
        }
    }
    return 0;
}