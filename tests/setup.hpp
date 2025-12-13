#ifndef VKOP_TESTS_HPP_
#define VKOP_TESTS_HPP_

#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>

#include "core/Tensor.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "include/logger.hpp"
#include "ops/OperatorFactory.hpp"
#include "ops/Ops.hpp"

#include "Python.h"

namespace vkop {
namespace tests {

using vkop::core::Tensor;

class TestCase {
private:
    std::string name_;
public:
    TestCase() = delete;
    explicit TestCase(std::string name): name_(std::move(name)) {
        initialize_pyenv();
    }
    ~TestCase() {
        finalize_pyenv();
    }
    TestCase(const TestCase &test) = delete;
    TestCase(const TestCase &&test) = delete;
    TestCase &operator=(const TestCase &) = delete;
    TestCase &operator=(const TestCase &&) = delete;

    template <typename T>
    bool run_test(const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &expect_outputs,
        const std::function<void(std::unique_ptr<ops::Operator> &)> &attribute_func = nullptr)
    {
        try {
            auto phydevs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
            for (auto *pdev : phydevs) {
                auto dev = std::make_shared<VulkanDevice>(pdev);
                if (dev->getDeviceName().find("llvmpipe") != std::string::npos) {
                    continue;
                }
                LOG_INFO("%s",dev->getDeviceName().c_str());
                auto cmdpool = std::make_shared<VulkanCommandPool>(dev);
                auto cmd = std::make_shared<VulkanCommandBuffer>(cmdpool);

                auto op = ops::OperatorFactory::get_instance().create(vkop::ops::convert_opstring_to_enum(name_));
                if (!op) {
                    LOG_ERROR("Fail to create operator");
                    return false;
                }
                op->set_runtime_device(dev, cmdpool);

                // Apply the attribute function callback if provided
                if (attribute_func) {
                    attribute_func(op);
                }
                std::vector<std::shared_ptr<core::ITensor>> outputs;
                for (size_t i = 0; i < expect_outputs.size(); i++) {
                    auto output = std::make_shared<Tensor<T>>(true);
                    outputs.push_back(output);
                }
                for (const auto &input : inputs) {
                    if (!input || input->num_dims() < 2) {
                        continue;
                    }
                    if (input->num_dims() == 2) {
                        auto t = core::as_tensor<T>(input);
                        t->as_storage_buffer(dev);
                        t->copyToGPU(cmdpool);
                        continue;
                    }
                    auto t = core::as_tensor<T>(input);
                    t->as_input_image(dev, nullptr);
                    t->copyToGPU(cmdpool);
                }
                cmd->wait(dev->getComputeQueue());
                cmd->begin();
                op->onExecute(inputs, outputs, cmd, 0);
                cmd->end();
                cmd->submit(dev->getComputeQueue());
                for (size_t idx = 0; idx < outputs.size(); idx++) {
                    auto output = core::as_tensor<T>(outputs[idx]);
                    output->copyToCPU(cmdpool);
                    auto oshape = output->getShape();
                    printf("output shape: %ld\n", oshape.size());
                    #if 0
                    if (oshape.size() == 4) {
                        for (int i = 0; i < oshape[0]; i++) {
                            printf("[\n");
                            for (int j = 0; j < oshape[1]; j++) {
                                printf("[\n");
                                for (int k = 0; k < oshape[2]; k++) {
                                    printf("[");
                                    for (int l = 0; l < oshape[3]; l++) {
                                        int idx = i * oshape[1] * oshape[2] * oshape[3] + j * oshape[2] * oshape[3] +
                                            k * oshape[3] + l;
                                        printf("%.4f, ", (*output)[idx]);
                                    }
                                    printf("]\n");
                                }
                                printf("]\n");
                            }
                            printf("]\n");
                        }
                    } else if (oshape.size() == 2) {
                        for (int i = 0; i < oshape[0]; i++) {
                            printf("[");
                            for (int j = 0; j < oshape[1]; j++) {
                                int idx = i * oshape[1] + j;
                                printf("%.4f, ", (*output)[idx]);
                            }
                            printf("]\n");
                        }
                        printf("]\n");
                    }
                    #endif
                    auto expect = core::as_tensor<T>(expect_outputs[idx]);
                    for (int i = 0; i < output->num_elements(); i++) {
                        if (sizeof(T) == 2) {
                            std::cout << i<< ": " << core::ITensor::fp16_to_fp32((*output)[i]) << " vs " << core::ITensor::fp16_to_fp32((*expect)[i]) << std::endl;
                            if (std::fabs(core::ITensor::fp16_to_fp32((*output)[i]) - core::ITensor::fp16_to_fp32((*expect)[i])) > 0.02) {
                                LOG_ERROR("Test Fail at1 (%d): %f, %f", i, core::ITensor::fp16_to_fp32((*output)[i]), core::ITensor::fp16_to_fp32((*expect)[i]));
                                return false;
                            }
                        } else {
                            std::cout << i<< ": " << (*output)[i] << " vs " << (*expect)[i] << std::endl;
                            if (std::fabs((*output)[i] - (*expect)[i]) > 1e-3) {
                                LOG_ERROR("Test Fail at (%d): %f, %f", i, (*output)[i], (*expect)[i]);
                                return false;
                            }
                        }
                    }
                }
                LOG_INFO("Test Passed for operator: %s", name_.c_str());
            }
        } catch (const std::exception &e) {
            LOG_ERROR("%s\n", e.what());
            return false;
        }
        return true;
    }

    static void* initialize_pyenv() {
        // Initialize Python environment and load necessary libraries
        Py_Initialize();
        try {
            PyObject* torch_name = PyUnicode_DecodeFSDefault("torch");
            PyObject* torch_module = PyImport_Import(torch_name);
            Py_DECREF(torch_name);

            if (!torch_module) {
                PyErr_Print();
                fprintf(stderr, "Failed to load torch module.\n");
                return nullptr;
            }

            // Store the modules for later use
            Py_XINCREF(torch_module);

            printf("Python environment initialized. Numpy and Torch loaded successfully.\n");
        } catch (...) {
            fprintf(stderr, "An exception occurred while initializing Python environment.\n");
        }
        return nullptr;
    }

    static void finalize_pyenv() {
        // Finalize Python environment
        try {
            Py_Finalize();
            printf("Python environment finalized.\n");
        } catch (...) {
            fprintf(stderr, "An exception occurred while finalizing Python environment.\n");
        }
    }
    static std::unordered_map<std::string, std::string> onnx_to_pytorch_attrs(
        const std::unordered_map<std::string, std::string>& onnx_attrs
    ) {
        std::unordered_map<std::string, std::string> pytorch_attrs;

        for (const auto& attr : onnx_attrs) {
            const std::string& name = attr.first;
            const std::string& value = attr.second;

            if (name == "kernel_shape") {
                // ONNX: "3,3" → PyTorch: kernel_size="3,3"
                // pytorch_attrs["kernel_size"] = value;
            }
            else if (name == "strides") {
                pytorch_attrs["stride"] = value;
            }
            else if (name == "dilations") {
                pytorch_attrs["dilation"] = value;
            }
            else if (name == "group") {
                pytorch_attrs["groups"] = value;
            }
            else if (name == "auto_pad") {
                if (value == "SAME_UPPER" || value == "SAME_LOWER") {
                    pytorch_attrs["padding"] = "same";  // PyTorch 支持 'same'
                }
                // "VALID" → padding=0，但通常 ONNX 用 pads=0
            }
            else if (name == "pads") {
                // ONNX: "1,1,1,1" → [x1,y1,x2,y2]
                // PyTorch: 通常用对称 padding (x,y)
                std::vector<int> pads;
                std::stringstream ss(value);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    pads.push_back(std::stoi(item));
                }

                if (pads.size() == 4) {
                    int px = (pads[0] + pads[2]) / 2;  // 平均左右
                    int py = (pads[1] + pads[3]) / 2;  // 平均上下
                    pytorch_attrs["padding"] = std::to_string(px) + "," + std::to_string(py);
                } else if (pads.size() == 2) {
                    pytorch_attrs["padding"] = value;  // 直接用
                }
            }
            // 其他通用属性
            else {
                pytorch_attrs[name] = value;
            }
        }

        return pytorch_attrs;
    }

    static std::tuple<std::vector<std::vector<float>>, std::vector<int>>
    execute_torch_operator(const std::string& op_name,
                        const std::vector<std::vector<int>>& shapes,
                        const std::unordered_map<std::string, std::string>& attributes) {
        std::vector<std::vector<float>> ret;
        std::vector<int> output_shape;

        if (shapes.empty()) {
            throw std::invalid_argument("At least one shape (input) must be provided");
        }

        auto input_shape = shapes[0];
        std::vector<int> weight_shape;
        std::vector<int> bias_shape;
        bool has_weight = false;
        bool has_bias = false;
        bool has_laynernorm = false;

        if (shapes.size() > 1) {
            weight_shape = shapes[1];
            has_weight = true;
        }
        if (shapes.size() > 2) {
            bias_shape = shapes[2];
            has_bias = true;
        }

        // 检查 Python 是否初始化
        if (!Py_IsInitialized()) {
            PyErr_SetString(PyExc_RuntimeError, "Python not initialized");
            return {ret, output_shape};
        }

        PyObject* torch_module = nullptr;
        PyObject* functional_module = nullptr;
        PyObject* torch_input = nullptr;
        PyObject* torch_weight = nullptr;
        PyObject* torch_bias = nullptr;
        PyObject* op_func = nullptr;
        PyObject* kwargs = nullptr;
        PyObject* torch_output = nullptr;
        PyObject* norm_shape = nullptr;

        auto cleanup = [&]() {
            Py_XDECREF(torch_module);
            Py_XDECREF(functional_module);
            Py_XDECREF(torch_input);
            Py_XDECREF(torch_weight);
            Py_XDECREF(torch_bias);
            Py_XDECREF(op_func);
            Py_XDECREF(kwargs);
            Py_XDECREF(torch_output);
            Py_XDECREF(norm_shape);
        };

        auto attrs = onnx_to_pytorch_attrs(attributes);

        try {
            // === 1. Import torch and torch.nn.functional ===
            torch_module = PyImport_ImportModule("torch");
            if (!torch_module) {
                PyErr_Print();
                throw std::runtime_error("Failed to import torch");
            }

            functional_module = PyImport_ImportModule("torch.nn.functional");
            if (!functional_module) {
                PyErr_Print();
                throw std::runtime_error("Failed to import torch.nn.functional");
            }

            // === 2. Create random tensors ===
            auto create_random_tensor = [&torch_module](const std::vector<int>& shape) -> PyObject* {
                PyObject* shape_list = PyList_New(shape.size());
                for (size_t i = 0; i < shape.size(); ++i) {
                    PyList_SetItem(shape_list, i, PyLong_FromLong(shape[i]));
                }
                PyObject* rand_func = PyObject_GetAttrString(torch_module, "rand");
                PyObject* tensor = PyObject_CallFunctionObjArgs(rand_func, shape_list, nullptr);
                Py_DECREF(shape_list);
                Py_DECREF(rand_func);
                if (!tensor) {
                    throw std::runtime_error("Failed to create random tensor");
                }
                return tensor;
            }

            torch_input = create_random_tensor(input_shape);
            Py_INCREF(torch_input);
            if (has_weight) {
                torch_weight = create_random_tensor(weight_shape);
                Py_INCREF(torch_weight);
            }

            if (has_bias) {
                torch_bias = create_random_tensor(bias_shape);
                Py_INCREF(torch_bias);
            }

            // === 3. Get operator function ===
            op_func = PyObject_GetAttrString(functional_module, op_name.c_str());
            if (!op_func) {
                PyErr_Print();
                throw std::runtime_error("Failed to find operator: " + op_name);
            }

            // === 4. Build kwargs dictionary ===
            kwargs = PyDict_New();
            for (const auto& attr : attrs) {
                const std::string& key = attr.first;
                const std::string& val_str = attr.second;
                PyObject* value = nullptr;

                if (key == "inplace" || key == "bias" || key == "align_corners" || key == "antialias") {
                    value = (val_str == "True" || val_str == "true") ? Py_True : Py_False;
                    Py_INCREF(value);
                } else if (key == "alpha" || key == "p" || key == "value") {
                    value = PyFloat_FromDouble(std::stod(val_str));
                } else if (key == "padding" && (val_str == "same" || val_str == "valid")) {
                    value = PyUnicode_FromString(val_str.c_str());
                } else if (key == "momentum" || key == "eps") {
                    value = PyFloat_FromDouble(std::stof(val_str));
                } else if (key == "normalized_shape") {
                    std::vector<int> normalized_shape;
                    if (val_str.front() == '[' && val_str.back() == ']') {
                        std::string content = val_str.substr(1, val_str.size() - 2);
                        std::stringstream ss(content);
                        std::string item;
                        while (std::getline(ss, item, ',')) {
                            normalized_shape.push_back(std::stoi(item));
                        }
                    }
                    printf("Setting normalized_shape: %d\n", normalized_shape[0]);
                    has_laynernorm = true;
                    PyObject* shape_tuple = PyTuple_New(normalized_shape.size());
                    for (size_t i = 0; i < normalized_shape.size(); ++i) {
                        PyTuple_SetItem(shape_tuple, i, PyLong_FromLong(normalized_shape[i]));
                    }
                    norm_shape = shape_tuple;
                    continue;
                } else if (key == "size" || key == "scale_factor") {
                    std::vector<int> resize_shape;
                    if (val_str.front() == '[' && val_str.back() == ']') {
                        std::string content = val_str.substr(1, val_str.size() - 2);
                        std::stringstream ss(content);
                        std::string item;
                        while (std::getline(ss, item, ',')) {
                            resize_shape.push_back(std::stoi(item));
                        }
                    }
                    PyObject* shape_tuple = PyTuple_New(resize_shape.size());
                    for (size_t i = 0; i < resize_shape.size(); ++i) {
                        PyTuple_SetItem(shape_tuple, i, PyLong_FromLong(resize_shape[i]));
                    }
                    value = shape_tuple;
                } else {
                    try {
                        value = PyLong_FromLong(std::stol(val_str));
                    } catch (...) {
                        try {
                            value = PyFloat_FromDouble(std::stod(val_str));
                        } catch (...) {
                            value = PyUnicode_FromString(val_str.c_str());
                        }
                    }
                }
                PyDict_SetItemString(kwargs, key.c_str(), value);
                Py_XDECREF(value);
            }

            // === 5. Prepare positional arguments dynamically ===
            std::vector<PyObject*> arg_list;
            arg_list.push_back(torch_input);
            if (has_laynernorm) arg_list.push_back(norm_shape);
            if (has_weight) arg_list.push_back(torch_weight);
            if (has_bias) arg_list.push_back(torch_bias);

            PyObject* args = PyTuple_New(arg_list.size());
            for (size_t i = 0; i < arg_list.size(); ++i) {
                PyTuple_SetItem(args, i, arg_list[i]);
            }

            // === 6. Call the operator ===
            torch_output = PyObject_Call(op_func, args, kwargs);
            Py_DECREF(args);
            if (!torch_output) {
                PyErr_Print();
                throw std::runtime_error("Failed to execute operator: " + op_name);
            }
            // === 7. Extract output data ===
            PyObject* torch_output_shape = PyObject_GetAttrString(torch_output, "shape");
            if (torch_output_shape && PyTuple_Check(torch_output_shape)) {
                Py_ssize_t ndim = PyTuple_Size(torch_output_shape);
                for (Py_ssize_t i = 0; i < ndim; ++i) {
                    output_shape.push_back(PyLong_AsLong(PyTuple_GetItem(torch_output_shape, i)));
                }
            } else {
                throw std::runtime_error("Failed to get output shape");
            }
            auto flatten_values = [](PyObject* tensor) -> std::vector<float> {
                PyObject* detached = PyObject_CallMethod(tensor, "detach", nullptr);
                if (!detached) {
                    PyErr_Print();
                    detached = tensor;
                    Py_INCREF(detached);
                }
                PyObject* cpu_tensor = PyObject_CallMethod(detached, "cpu", nullptr);
                if (!cpu_tensor) {
                    PyErr_Print();
                    throw std::runtime_error("cpu() failed");
                }
                PyObject* contiguous = PyObject_CallMethod(cpu_tensor, "contiguous", nullptr);
                Py_DECREF(cpu_tensor);
                if (!contiguous) {
                    PyErr_Print();
                    throw std::runtime_error("contiguous() failed");
                }
                PyObject* numel_obj = PyObject_CallMethod(contiguous, "numel", nullptr);
                int64_t numel = PyLong_AsLong(numel_obj);
                Py_DECREF(numel_obj);
                PyObject* ptr_obj = PyObject_CallMethod(contiguous, "data_ptr", nullptr);
                uint64_t ptr_val = PyLong_AsLongLong(ptr_obj);
                Py_DECREF(ptr_obj);

                auto* data_ptr = reinterpret_cast<float*>(ptr_val);
                std::vector<float> result(data_ptr, data_ptr + numel);
                Py_DECREF(contiguous);

                return result;
            }

            ret.emplace_back(flatten_values(torch_output));
            ret.emplace_back(flatten_values(torch_input));
            if (has_weight) ret.emplace_back(flatten_values(torch_weight));
            if (has_bias) ret.emplace_back(flatten_values(torch_bias));

        } catch (const std::exception& e) {
            fprintf(stderr, "C++ Exception: %s\n", e.what());
            cleanup();
        } catch (...) {
            cleanup();
            fprintf(stderr, "Unknown exception in execute_torch_operator\n");
        }

        cleanup();

        return {ret, output_shape};
    }
};

} // namespace tests
} // namespace vkop

#endif // VKOP_TESTS_HPP_
