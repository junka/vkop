#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "include/logger.hpp"
#include "core/Tensor.hpp"
#include "model/load.hpp"
#include "ops/OperatorFactory.hpp"

#include <cstdint>
#include <memory>
#include <random>
#include <chrono>
#include <cmath>

#include <sys/types.h>

using vkop::VulkanInstance;
using vkop::VulkanDevice;
using vkop::core::Tensor;
using vkop::load::VkModel;
using vkop::ops::OperatorFactory;

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
            if (!node.attributes.empty()) {
                std::cout << ", Attributes: {";
                for (const auto& attr : node.attributes) {
                    std::cout << attr.first << ": " << attr.second << ", ";
                }
                std::cout << "}";
            }
            std::cout << std::endl;
            std::cout << "Inputs:" << std::endl;
            for (const auto& input : node.inputs) {
                std::cout << "  Name: " << input.name << ", Shape: [";
                for (size_t i = 0; i < input.dims.size(); ++i) {
                    std::cout << input.dims[i] << (i + 1 < input.dims.size() ? ", " : "");
                }
                std::cout << "]" << std::endl;
            }

            std::cout << "Outputs:" << std::endl;
            for (const auto& output : node.outputs) {
                std::cout << "  Name: " << output.name << ", Shape: [";
                for (size_t i = 0; i < output.dims.size(); ++i) {
                    std::cout << output.dims[i] << (i + 1 < output.dims.size() ? ", " : "");
                }
                std::cout << "]" << std::endl;
            }
        }

        std::cout << "Initializers:" << std::endl;
        for (const auto& [name, initializer] : model.initializers) {
            std::cout << "  Name: " << name << ", Shape: [";
            for (size_t i = 0; i < initializer.dims.size(); ++i) {
                std::cout << initializer.dims[i] << (i + 1 < initializer.dims.size() ? ", " : "");
            }
            std::cout << "], DType: " << initializer.dtype << std::endl;
        }

        std::vector<std::shared_ptr<Tensor<float>>> inputs;
        std::vector<std::shared_ptr<Tensor<float>>> outputs;
        for (const auto& i : model.inputs) {
            auto t = std::make_shared<Tensor<float>>(i.dims);
            inputs.push_back(t);
        }
        for (const auto& i : inputs) {
            i->printTensorShape();
        }
        for (const auto& o: model.outputs) {
            auto t = std::make_shared<Tensor<float>>(o.dims);
            outputs.push_back(t);
        }

        for (const auto& n: model.nodes) {
            auto op = OperatorFactory::get_instance().create(n.op_type);
            // op->execute(inputs, outputs);
            std::cout << "run ops " << n.op_type << std::endl;
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    return EXIT_SUCCESS;
}