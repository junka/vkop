#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "include/logger.hpp"
#include "core/Tensor.hpp"
#include "model/load.hpp"
#include "ops/OperatorFactory.hpp"
#include "ops/Ops.hpp"

#include <cstdint>
#include <memory>
#include <random>
#include <chrono>
#include <cmath>

#include <sys/types.h>
#include <unistd.h>

using vkop::VulkanInstance;
using vkop::VulkanDevice;
using vkop::core::Tensor;
using vkop::load::VkModel;
using vkop::ops::OperatorFactory;

static vkop::ops::OpType convert_opstring_to_enum(const std::string &name) {
    if (name == "Add") return vkop::ops::OpType::ADD;
    if (name == "Sub") return vkop::ops::OpType::SUB;
    if (name == "Mul") return vkop::ops::OpType::MUL;
    if (name == "Div") return vkop::ops::OpType::DIV;
    if (name == "Atan") return vkop::ops::OpType::ATAN;
    if (name == "Erf") return vkop::ops::OpType::ERF;
    if (name == "Pow") return vkop::ops::OpType::POW;
    if (name == "BatchNormalization") return vkop::ops::OpType::BATCHNORM;
    if (name == "Relu") return vkop::ops::OpType::RELU;
    if (name == "Softmax") return vkop::ops::OpType::SOFTMAX;
    if (name == "Tanh") return vkop::ops::OpType::TANH;
    if (name == "MatMul") return vkop::ops::OpType::MATMUL;
    if (name == "Conv2d" || name == "Conv") return vkop::ops::OpType::CONV2D;
    if (name == "MaxPool2d"||name == "MaxPool") return vkop::ops::OpType::MAXPOOL2D;
    if (name == "AvgPool2d") return vkop::ops::OpType::AVGPOOL2D;
    if (name == "Upsample2d") return vkop::ops::OpType::UPSAMPLE2D;
    if (name == "GridSample") return vkop::ops::OpType::GRIDSAMPLE;
    if (name == "Constant") return vkop::ops::OpType::CONSTANT;
    if (name == "Floor") return vkop::ops::OpType::FLOOR;
    if (name == "Resize") return vkop::ops::OpType::RESIZE;
    printf("Unknown op type: %s\n", name.c_str());
    return vkop::ops::OpType::UNKNOWN;
}

int main(int argc, char *argv[]) {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", true);
    try {
        auto phydevs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
        for (auto *pdev : phydevs) {
            auto dev = std::make_shared<VulkanDevice>(pdev);
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
                std::cout << "]";
            }

            std::cout << "  Outputs: ";
            for (const auto& output : node.outputs) {
                std::cout << output.name << ", [";
                for (size_t i = 0; i < output.dims.size(); ++i) {
                    std::cout << output.dims[i] << (i + 1 < output.dims.size() ? ", " : "");
                }
                std::cout << "]";
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

        std::vector<std::shared_ptr<Tensor<float>>> inputs;
        std::vector<std::shared_ptr<Tensor<float>>> outputs;
        std::unordered_map<std::string, std::shared_ptr<Tensor<float>>> tensor_map;
        for (const auto& i : model.inputs) {
            auto t = std::make_shared<Tensor<float>>(i.dims);
            inputs.push_back(t);
            tensor_map[i.name] = t;
        }
        for (const auto& o: model.outputs) {
            auto t = std::make_shared<Tensor<float>>(o.dims);
            outputs.push_back(t);
            tensor_map[o.name] = t;
        }

        for (const auto& n: model.nodes) {
            if (n.op_type == "Identity") {
                // this should be optimized out in onnx graph
                continue;
            }
            auto t = convert_opstring_to_enum(n.op_type);
            if (t == vkop::ops::OpType::CONSTANT || t == vkop::ops::OpType::UNKNOWN) {
                // make it as input for next ops
                continue;
            }
            std::vector<std::shared_ptr<Tensor<float>>> node_inputs;
            std::vector<std::shared_ptr<Tensor<float>>> node_outputs;
            if (!n.inputs.empty()) {
                for (const auto& in_shape : n.inputs) {
                    if (tensor_map.find(in_shape.name) != tensor_map.end()) {
                        node_inputs.push_back(tensor_map[in_shape.name]);
                        std::cout << "find input tensor " << in_shape.name << " for op " << n.op_type << std::endl;
                    } else {
                        if (model.initializers.find(in_shape.name) != model.initializers.end()) {
                            auto& init = model.initializers.at(in_shape.name);
                            if (init.dims != in_shape.dims) {
                                throw std::runtime_error("Initializer dims do not match for " + in_shape.name);
                            }
                            if (init.dtype == "int64") {
                                auto t = std::make_shared<Tensor<float>>(in_shape.dims);
                                for (int i = 0; i < t->num_elements(); ++i) {
                                    t->data()[i] = static_cast<float>(init.dataii[i]);
                                }
                                tensor_map[in_shape.name] = t;
                                node_inputs.push_back(t);
                                std::cout << "load int64 initializer " << in_shape.name << " for op " << n.op_type << std::endl;
                            } else if (init.dtype == "int32") {
                                auto t = std::make_shared<Tensor<float>>(in_shape.dims);
                                for (int i = 0; i < t->num_elements(); ++i) {
                                    t->data()[i] = static_cast<float>(init.dataii[i]);
                                }
                                tensor_map[in_shape.name] = t;
                                node_inputs.push_back(t);
                                std::cout << "load int32 initializer " << in_shape.name << " for op " << n.op_type << std::endl;
                            } else if (init.dtype == "float32") {
                                auto t = std::make_shared<Tensor<float>>(in_shape.dims);
                                std::memcpy(t->data(), init.dataf.data(), t->num_elements() * sizeof(float));
                                tensor_map[in_shape.name] = t;
                                node_inputs.push_back(t);
                                std::cout << "load float32 initializer " << in_shape.name << " for op " << n.op_type << std::endl;
                            } else {
                                throw std::runtime_error("Only float32 initializer is supported for now " + init.dtype);
                            }
                        }
                        std::cout << "create empty tensor " << in_shape.name << " for op " << n.op_type << std::endl;
                    }
                }
            }
            if (!n.outputs.empty()) {
                for (const auto& out_shape : n.outputs) {
                    if (tensor_map.find(out_shape.name) != tensor_map.end()) {
                        node_outputs.push_back(tensor_map[out_shape.name]);
                        std::cout << "find output tensor " << out_shape.name << " for op " << n.op_type << std::endl;
                    } else {
                        auto t = std::make_shared<Tensor<float>>(out_shape.dims);
                        tensor_map[out_shape.name] = t;
                        node_outputs.push_back(t);
                        std::cout << "create empty tensor " << out_shape.name << " for op " << n.op_type << std::endl;
                    }
                }
            }
            auto op = OperatorFactory::get_instance().create(t);
            if (!n.attributes.empty()) {
                op->setAttribute(n.attributes);
            }
            op->execute(node_inputs, node_outputs);
            std::cout << "run ops " << n.op_type << std::endl;
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
    sleep(100000);
    return EXIT_SUCCESS;
}
