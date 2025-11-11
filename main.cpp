#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "include/logger.hpp"
#include "core/Tensor.hpp"
#include "model/load.hpp"
#include "ops/OperatorFactory.hpp"
#include "ops/Ops.hpp"

#include <algorithm>
#include <bits/stdint-intn.h>
#include <cstdint>
#include <memory>
#include <random>
#include <chrono>
#include <cmath>

#include <sys/types.h>
#include <unistd.h>
#include <vulkan/vulkan_core.h>

using vkop::VulkanInstance;
using vkop::VulkanDevice;
using vkop::core::Tensor;
using vkop::core::ITensor;
using vkop::load::VkModel;
using vkop::ops::OperatorFactory;



int main(int argc, char *argv[]) {
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

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <binary_file_path>" << std::endl;
        return 1;
    }
    try {
        std::string binary_file_path = argv[1];
        VkModel model(binary_file_path);

        std::cout << "Inputs:" << std::endl;
        for (const auto& input : model.inputs) {
            std::cout << "  Name: " << input.name << ", Shape: [";
            for (size_t i = 0; i < input.dims.size(); ++i) {
                std::cout << input.dims[i] << (i + 1 < input.dims.size() ? ", " : "");
            }
            std::cout << "]" << std::endl;
        }

        std::cout << "Outputs:" << std::endl;
        for (const auto& output : model.outputs) {
            std::cout << "  Name: " << output.name << ", Shape: [";
            for (size_t i = 0; i < output.dims.size(); ++i) {
                std::cout << output.dims[i] << (i + 1 < output.dims.size() ? ", " : "");
            }
            std::cout << "]" << std::endl;
        }

        std::cout << "Nodes:" << std::endl;
        for (const auto& node : model.nodes) {
            std::cout << "  OpType: " << node.op_type;
            std::cout << "  Name: " << node.name;
            if (!node.attributes.empty()) {
                std::cout << ", Attributes: {";
                for (const auto& attr : node.attributes) {
                    std::cout << attr.first << ": " << attr.second << ", ";
                }
                std::cout << "}";
            }
            std::cout << "  Inputs: " ;
            for (const auto& input : node.inputs) {
                std::cout << input.name << ", [";
                for (size_t i = 0; i < input.dims.size(); ++i) {
                    std::cout << input.dims[i] << (i + 1 < input.dims.size() ? ", " : "");
                }
                std::cout << "]" << std::endl;
            }

            std::cout << "  Outputs: ";
            for (const auto& output : node.outputs) {
                std::cout << output.name << ", [";
                for (size_t i = 0; i < output.dims.size(); ++i) {
                    std::cout << output.dims[i] << (i + 1 < output.dims.size() ? ", " : "");
                }
                std::cout << "]" << std::endl;
            }
            std::cout << std::endl;
        }

        // std::cout << "Initializers:" << std::endl;
        // for (const auto& [name, initializer] : model.initializers) {
        //     std::cout << name << ", [";
        //     for (size_t i = 0; i < initializer.dims.size(); ++i) {
        //         std::cout << initializer.dims[i] << (i + 1 < initializer.dims.size() ? ", " : "");
        //     }
        //     std::cout << "], DType: " << initializer.dtype << std::endl;
        // }

        std::vector<std::shared_ptr<ITensor>> inputs;
        std::vector<std::shared_ptr<ITensor>> outputs;
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
            printf("Output %s\n", o.name.c_str());
            auto t = std::make_shared<Tensor<float>>(o.dims);
            outputs.push_back(t);
            tensor_map[o.name] = t;
        }

        for (const auto& n: model.nodes) {
            if (n.op_type == "Identity") {
                // this should be optimized out in onnx graph
                continue;
            }
            auto t = vkop::ops::convert_opstring_to_enum(n.op_type);
            if (t == vkop::ops::OpType::CONSTANT || t == vkop::ops::OpType::UNKNOWN) {
                // make it as input for next ops
                continue;
            }
            std::vector<std::shared_ptr<ITensor>> node_inputs;
            std::vector<std::shared_ptr<ITensor>> node_outputs;

            for (const auto& in_shape : n.inputs) {
                if (tensor_map.find(in_shape.name) != tensor_map.end()) {
                    node_inputs.push_back(tensor_map[in_shape.name]);
                    // std::cout << "find input tensor " << in_shape.name << " for op " << n.op_type << std::endl;
                } else {
                    std::cout << "create empty tensor " << in_shape.name << " for op " << n.op_type << std::endl;
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
                            printf("load int64 initializer %s for op %s\n", in_shape.name.c_str(), n.op_type.c_str());
                            auto t = std::make_shared<Tensor<int64_t>>(in_shape.dims);
                            for (int i = 0; i < t->num_elements(); ++i) {
                                t->data()[i] = static_cast<float>(init.dataii[i]);
                                printf("%ld ", t->data()[i]);
                            }
                            tensor_map[in_shape.name] = t;
                            node_inputs.push_back(t);
                            // std::cout << "load int64 initializer " << in_shape.name << " for op " << n.op_type << std::endl;
                        } else if (init.dtype == "int32") {
                            auto t = std::make_shared<Tensor<int>>(in_shape.dims);
                            for (int i = 0; i < t->num_elements(); ++i) {
                                t->data()[i] = static_cast<float>(init.dataii[i]);
                            }
                            tensor_map[in_shape.name] = t;
                            node_inputs.push_back(t);
                            // std::cout << "load int32 initializer " << in_shape.name << " for op " << n.op_type << std::endl;
                        } else if (init.dtype == "float32") {
                            auto t = std::make_shared<Tensor<float>>(in_shape.dims);
                            std::memcpy(t->data(), init.dataf.data(), t->num_elements() * sizeof(float));
                            tensor_map[in_shape.name] = t;
                            node_inputs.push_back(t);
                            // std::cout << "load float32 initializer " << in_shape.name << " for op " << n.op_type << std::endl;
                        } else {
                            throw std::runtime_error("Only float32 initializer is supported for now " + init.dtype);
                        }
                    }
                }
            }

            for (const auto& out_shape : n.outputs) {
                if (tensor_map.find(out_shape.name) != tensor_map.end()) {
                    node_outputs.push_back(tensor_map[out_shape.name]);
                    // std::cout << "find output tensor " << out_shape.name << " for op " << n.op_type << std::endl;
                } else {
                    // std::cout << "create empty out tensor " << out_shape.name << " for op " << n.op_type << std::endl;
                    auto t = std::make_shared<Tensor<float>>(out_shape.dims);
                    tensor_map[out_shape.name] = t;
                    node_outputs.push_back(t);
                }
            }

            auto op = OperatorFactory::get_instance().create(t);
            if (!op) {
                std::cout << "Fail to create operator" << std::endl;
                return 1;
            }
            auto *device = dev->getLogicalDevice();
            auto cmdpool = std::make_shared<vkop::VulkanCommandPool>(device, dev->getComputeQueueFamilyIndex());

            op->set_runtime_device(phydev, dev, cmdpool);
            if (!n.attributes.empty()) {
                op->setAttribute(n.attributes);
            }
            ops_all.push_back(std::move(op));
            inputs_all.push_back(node_inputs);
            outputs_all.push_back(node_outputs);
        }


        // fill input with image
        for (size_t i = 0; i < ops_all.size(); ++i) {
            printf("execute op %s\n", convert_openum_to_string(ops_all[i]->get_type()).c_str());
            ops_all[i]->execute(inputs_all[i], outputs_all[i]);
        }
        // for (size_t i = 0; i < outputs.size(); ++i) {
        //     std::cout << "output " << i << ": " << " " << outputs[i]->num_elements() << std::endl;
        // }

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
    return EXIT_SUCCESS;
}