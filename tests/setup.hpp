#ifndef VKOP_TESTS_HPP_
#define VKOP_TESTS_HPP_

#include <cmath>
#include <cstring>
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

    static vkop::ops::OpType convert_opstring_to_enum(const std::string &name) {
        if (name == "Add") return vkop::ops::OpType::ADD;
        if (name == "Sub") return vkop::ops::OpType::SUB;
        if (name == "Mul") return vkop::ops::OpType::MUL;
        if (name == "Div") return vkop::ops::OpType::DIV;
        if (name == "Atan") return vkop::ops::OpType::ATAN;
        if (name == "Erf") return vkop::ops::OpType::ERF;
        if (name == "Pow") return vkop::ops::OpType::POW;
        if (name == "BatchNorm") return vkop::ops::OpType::BATCHNORM;
        if (name == "Relu") return vkop::ops::OpType::RELU;
        if (name == "Softmax") return vkop::ops::OpType::SOFTMAX;
        if (name == "Tanh") return vkop::ops::OpType::TANH;
        if (name == "MatMul") return vkop::ops::OpType::MATMUL;
        if (name == "Conv2d" || name == "Conv") return vkop::ops::OpType::CONV2D;
        if (name == "MaxPool2d" || name == "MaxPool") return vkop::ops::OpType::MAXPOOL2D;
        if (name == "AvgPool2d") return vkop::ops::OpType::AVGPOOL2D;
        if (name == "Upsample2d") return vkop::ops::OpType::UPSAMPLE2D;
        if (name == "GridSample") return vkop::ops::OpType::GRIDSAMPLE;
        if (name == "Constant") return vkop::ops::OpType::CONSTANT;
        if (name == "Floor") return vkop::ops::OpType::FLOOR;
        if (name == "Resize") return vkop::ops::OpType::RESIZE;
        return vkop::ops::OpType::UNKNOWN;
    }

    template <typename T>
    bool run_test(const std::vector<std::shared_ptr<Tensor<T>>> &inputs,
        const std::vector<T> &expectedOutput,
        const std::function<void(std::unique_ptr<ops::Operator> &)> &attribute_func)
    {
        try {
            auto phydevs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
            for (auto *pdev : phydevs) {
                auto dev = std::make_shared<VulkanDevice>(pdev);
                if (dev->getDeviceName().find("llvmpipe") != std::string::npos) {
                    continue;
                }
                auto *device = dev->getLogicalDevice();
                auto cmdpool = std::make_shared<VulkanCommandPool>(device, dev->getComputeQueueFamilyIndex());

                auto op = ops::OperatorFactory::get_instance().create(convert_opstring_to_enum(name_));
                if (!op) {
                    LOG_ERROR("Fail to create operator");
                    return false;
                }
                op->set_runtime_device(pdev, dev, cmdpool);

                // Apply the attribute function callback if provided
                if (attribute_func) {
                    attribute_func(op);
                }

                auto output = std::make_shared<Tensor<T>>();
                op->execute(inputs, std::vector<std::shared_ptr<Tensor<T>>> {output});
                auto *out_ptr = output->data();
                for (int i = 0; i < output->num_elements(); i++) {
                    std::cout << i<< ": " << out_ptr[i] << " vs " <<expectedOutput[i] << std::endl;
                    if (std::fabs(out_ptr[i] - expectedOutput[i]) > 0.001) {
                        LOG_ERROR("Test Fail at (%d): %f, %f", i, out_ptr[i], expectedOutput[i]);
                        return false;
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

    template <typename T>
    bool run_test(const std::vector<std::shared_ptr<Tensor<T>>> &inputs,
        const std::vector<T> &expectedOutput)
    {
        try {
            auto phydevs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
            for (auto *pdev : phydevs) {
                auto dev = std::make_shared<VulkanDevice>(pdev);
                if (dev->getDeviceName().find("llvmpipe") != std::string::npos) {
                    continue;
                }
                auto *device = dev->getLogicalDevice();
                auto cmdpool = std::make_shared<VulkanCommandPool>(device, dev->getComputeQueueFamilyIndex());

                auto op = ops::OperatorFactory::get_instance().create(convert_opstring_to_enum(name_));
                if (!op) {
                    LOG_ERROR("Fail to create operator");
                    return false;
                }
                op->set_runtime_device(pdev, dev, cmdpool);

                auto output = std::make_shared<Tensor<T>>();
                op->execute(inputs, std::vector<std::shared_ptr<Tensor<T>>> {output});
                auto *out_ptr = output->data();
                for (int i = 0; i < output->num_elements(); i++) {
                    if (sizeof(T) == 2) {
                        if (std::fabs(float16_to_float32(out_ptr[i]) - float16_to_float32(expectedOutput[i])) > 0.01) {
                            LOG_ERROR("Test Fail at1 (%d): %f, %f", i, float16_to_float32(out_ptr[i]), float16_to_float32(expectedOutput[i]));
                            return false;
                        }
                    } else {
                        if (std::fabs(out_ptr[i] - expectedOutput[i]) > 0.001) {
                            LOG_ERROR("Test Fail at (%d): %f, %f", i, out_ptr[i], expectedOutput[i]);
                            return false;
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

    static uint16_t float32_to_float16(float f) {
        uint32_t x;
        std::memcpy(&x, &f, sizeof(x));

        uint32_t sign = (x >> 16) & 0x8000; // 符号位 (bit 15)
        uint32_t exp = (x >> 23) & 0xFF;    // 指数 (8 bits)
        uint32_t mantissa = x & 0x7FFFFF;   // 尾数 (23 bits)

        uint16_t h = 0;

        if (exp == 0) {
            // FP32 zero or denormal
            h = static_cast<uint16_t>(sign | (mantissa != 0 ? 1 : 0)); // 保持为 0 或最小正数
        } else if (exp == 0xFF) {
            // Inf or NaN
            h = static_cast<uint16_t>(sign | 0x7C00 | (mantissa ? 0x200 : 0));
        } else {
            int new_exp = exp - 127 + 15; // 调整偏移

            if (new_exp >= 31) {
                // 溢出 → Inf
                h = static_cast<uint16_t>(sign | 0x7C00);
            } else if (new_exp <= 0) {
                // 下溢 → 非规格化或 0
                if (new_exp < -10) {
                    h = static_cast<uint16_t>(sign); // underflow to zero
                } else {
                    // 生成非规格化数
                    uint32_t shifted_mantissa = mantissa | 0x800000; // 添加隐含位
                    shifted_mantissa >>= (1 - new_exp); // 右移
                    h = static_cast<uint16_t>(sign | (shifted_mantissa >> 13));
                }
            } else {
                // 正规数
                h = static_cast<uint16_t>(sign | (new_exp << 10) | (mantissa >> 13));
            }
        }

        return h;
    }

    static float float16_to_float32(uint16_t h) {
        uint32_t sign = (h & 0x8000) << 16; // 符号位
        uint32_t exponent = (h & 0x7C00);    // 指数位
        uint32_t mantissa = (h & 0x03FF);    // 尾数位
    
        if (exponent == 0x7C00) { // Inf 或 NaN
            exponent = 0x3FC00; // FP32 的 Inf/Nan 指数
        } else if (exponent != 0) { // 正规数
            exponent = (exponent >> 10) + (127 - 15); // 指数偏移调整
            exponent <<= 23;
        } else if (mantissa != 0) { // 非规格化数
            // 处理非规格化数（denormal）
            int shift = __builtin_clz(mantissa) - 22; // GCC 内置函数
            mantissa <<= shift;
            exponent = (127 - 15 - shift + 1) << 23;
        }
        // 否则为 0
    
        mantissa <<= 13; // 尾数左移
        auto ret = (sign | exponent | mantissa);
        float retfloat;
        std::memcpy(&retfloat, &ret, 4);
        return retfloat;
    }

    static void initialize_pyenv() {
        // Initialize Python environment and load necessary libraries
        Py_Initialize();
        try {
            // Import numpy and torch modules
            PyObject* numpy_name = PyUnicode_DecodeFSDefault("numpy");
            PyObject* numpy_module = PyImport_Import(numpy_name);
            Py_DECREF(numpy_name);

            PyObject* torch_name = PyUnicode_DecodeFSDefault("torch");
            PyObject* torch_module = PyImport_Import(torch_name);
            Py_DECREF(torch_name);

            if (!numpy_module || !torch_module) {
                PyErr_Print();
                fprintf(stderr, "Failed to load numpy or torch module.\n");
                return;
            }

            // Store the modules for later use
            Py_XINCREF(numpy_module);
            Py_XINCREF(torch_module);

            printf("Python environment initialized. Numpy and Torch loaded successfully.\n");
        } catch (...) {
            fprintf(stderr, "An exception occurred while initializing Python environment.\n");
        }
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

    static std::tuple<std::vector<float>, std::vector<float>, std::vector<int>>
    execute_torch_operator(const std::string& op_name,
                        const std::vector<std::vector<int>>& shapes,
                        const std::unordered_map<std::string, std::string>& attributes) {
        std::vector<float> input_values;
        std::vector<float> output_values;
        std::vector<int> output_shape;
        auto input_shape = shapes[0];
        std::vector<int> weight_shape;
        std::vector<int> bias_shape;
        if (shapes.size() > 1) {
            weight_shape = shapes[1];
        }
        if (shapes.size() > 2) {
            bias_shape = shapes[2];
        }

        // 初始化返回值
        if (!Py_IsInitialized()) {
            PyErr_SetString(PyExc_RuntimeError, "Python not initialized");
            return {input_values, output_values, output_shape};
        }

        // 作用域内管理 PyObject*，避免泄漏
        PyObject* numpy_module = nullptr;
        PyObject* torch_module = nullptr;
        PyObject* functional_module = nullptr;
        PyObject* numpy_random = nullptr;
        PyObject* numpy_rand_func = nullptr;
        PyObject* shape_tuple = nullptr;
        PyObject* weight_tuple = nullptr;
        PyObject* bias_tuple = nullptr;
        PyObject* numpy_input = nullptr;
        PyObject* numpy_weight = nullptr;
        PyObject* numpy_bias = nullptr;
        PyObject* torch_input = nullptr;
        PyObject* torch_weight = nullptr;
        PyObject* torch_bias = nullptr;
        PyObject* op_func = nullptr;
        PyObject* kwargs = nullptr;
        PyObject*torch_output = nullptr;
        PyObject* numpy_output = nullptr;
        PyObject* numpy_output_shape = nullptr;
        PyObject* numpy_flat = nullptr;
        PyObject* numpy_input_flat = nullptr;

        auto attrs = onnx_to_pytorch_attrs(attributes);
        try {
            // === 1. Import numpy ===
            numpy_module = PyImport_ImportModule("numpy");
            if (!numpy_module) {
                PyErr_Print();
                throw std::runtime_error("Failed to import numpy");
            }

            // === 2. Import torch and torch.nn.functional ===
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

            // === 3. Create input tensor via numpy.random.rand ===
            numpy_random = PyObject_GetAttrString(numpy_module, "random");
            if (!numpy_random) {
                PyErr_Print();
                throw std::runtime_error("Failed to get numpy.random");
            }

            numpy_rand_func = PyObject_GetAttrString(numpy_random, "rand");
            if (!numpy_rand_func) {
                PyErr_Print();
                throw std::runtime_error("Failed to get numpy.random.rand");
            }

            shape_tuple = PyTuple_New(input_shape.size());
            for (size_t i = 0; i < input_shape.size(); ++i) {
                // PyTuple_SetItem steals reference, so we don't DECREF after
                PyObject* item = PyLong_FromLong(input_shape[i]);
                PyTuple_SetItem(shape_tuple, i, item);  // steals ref
            }

            weight_tuple = PyTuple_New(weight_shape.size());
            for (size_t i = 0; i < weight_shape.size(); ++i) {
                // PyTuple_SetItem steals reference, so we don't DECREF after
                PyObject* item = PyLong_FromLong(weight_shape[i]);
                PyTuple_SetItem(weight_tuple, i, item);  // steals ref
            }

            bias_tuple = PyTuple_New(bias_shape.size());
            for (size_t i = 0; i < bias_shape.size(); ++i) {
                // PyTuple_SetItem steals reference, so we don't DECREF after
                PyObject* item = PyLong_FromLong(bias_shape[i]);
                PyTuple_SetItem(bias_tuple, i, item);  // steals ref
            }

            numpy_input = PyObject_CallObject(numpy_rand_func, shape_tuple);
            if (!numpy_input) {
                PyErr_Print();
                throw std::runtime_error("Failed to generate random input");
            }
            numpy_weight = PyObject_CallObject(numpy_rand_func, weight_tuple);
            if (!numpy_weight) {
                PyErr_Print();
                throw std::runtime_error("Failed to generate random weight");
            }
            numpy_bias = PyObject_CallObject(numpy_rand_func, bias_tuple);
            if (!numpy_bias) {
                PyErr_Print();
                throw std::runtime_error("Failed to generate random bias");
            }

            // === 4. Convert numpy array to torch tensor ===
            PyObject* torch_from_numpy = PyObject_GetAttrString(torch_module, "from_numpy");
            if (!torch_from_numpy) {
                PyErr_Print();
                throw std::runtime_error("Failed to get torch.from_numpy");
            }

            torch_input = PyObject_CallFunctionObjArgs(torch_from_numpy, numpy_input, nullptr);
            Py_DECREF(torch_from_numpy);
            if (!torch_input) {
                PyErr_Print();
                throw std::runtime_error("Failed to convert numpy array to torch tensor");
            }

            torch_weight = PyObject_CallFunctionObjArgs(torch_from_numpy, numpy_weight, nullptr);
            Py_DECREF(torch_from_numpy);
            if (!torch_weight) {
                PyErr_Print();
                throw std::runtime_error("Failed to convert numpy array to torch tensor");
            }
            torch_bias = PyObject_CallFunctionObjArgs(torch_from_numpy, numpy_bias, nullptr);
            Py_DECREF(torch_from_numpy);
            if (!torch_bias) {
                PyErr_Print();
                throw std::runtime_error("Failed to convert numpy array to torch tensor");
            }

            // === 5. Get operator from torch.nn.functional ===
            op_func = PyObject_GetAttrString(functional_module, op_name.c_str());
            if (!op_func) {
                PyErr_Print();
                throw std::runtime_error("Failed to find operator in torch.nn.functional: " + op_name);
            }

            // === 6. Build kwargs ===
            kwargs = PyDict_New();
            for (const auto& attr : attrs) {
                PyObject* value = nullptr;
                const std::string& key = attr.first;
                const std::string& val_str = attr.second;
                printf("parseing %s, %s\n", key.c_str(), val_str.c_str());

                if (key == "padding" && (val_str == "same" || val_str == "valid")) {
                    value = PyUnicode_FromString(val_str.c_str());
                } else if (key == "inplace" || key == "bias") {
                    value = (val_str == "True" || val_str == "true") ? Py_True : Py_False;
                    Py_INCREF(value);
                } else if (key == "eps" || key == "momentum" || key == "alpha" || key == "p") {
                    value = PyFloat_FromDouble(std::stod(val_str));
                } else {
                    // 默认：尝试 int → float → string
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
            }

            std::vector<PyObject*> pos_args;
            // === 7. Call operator: op_func(input, **kwargs) ===
            PyObject* args = PyTuple_Pack(3, torch_input, torch_weight, torch_bias);

            torch_output = PyObject_Call(op_func, args, kwargs);
            Py_DECREF(args);
            if (!torch_output) {
                PyErr_Print();
                throw std::runtime_error("Failed to execute operator: " + op_name);
            }

            // === 8. Convert output to numpy for extraction ===
            numpy_output = PyObject_CallMethod(torch_output, "cpu", nullptr);  // Ensure CPU
            if (!numpy_output) {
                PyErr_Print();
                numpy_output = torch_output;  // fallback
                Py_INCREF(numpy_output);
            } else {
                Py_DECREF(numpy_output);
                numpy_output = PyObject_CallMethod(torch_output, "numpy", nullptr);
            }

            if (!numpy_output) {
                PyErr_Print();
                throw std::runtime_error("Failed to convert output to numpy");
            }

            // Get shape
            numpy_output_shape = PyObject_GetAttrString(numpy_output, "shape");
            if (numpy_output_shape && PyTuple_Check(numpy_output_shape)) {
                Py_ssize_t ndim = PyTuple_Size(numpy_output_shape);
                for (Py_ssize_t i = 0; i < ndim; ++i) {
                    output_shape.push_back(PyLong_AsLong(PyTuple_GetItem(numpy_output_shape, i)));
                }
            } else {
                throw std::runtime_error("Failed to get output shape");
            }

            // PyObject* torch_output_contig = PyObject_CallMethod(numpy_output, "contiguous", nullptr);
            // auto *numpy_output1 = PyObject_CallMethod(torch_output_contig, "numpy", nullptr);
            // // Py_DECREF(torch_output_contig);

            // // Flatten and extract output values
            // numpy_flat = PyObject_CallMethod(numpy_output, "flatten", nullptr);
            // if (!numpy_flat || !PyList_Check(numpy_flat)) {
            //     throw std::runtime_error("Failed to flatten output");
            // }

            // Py_ssize_t num_out = PyObject_Length(numpy_flat);
            // for (Py_ssize_t i = 0; i < num_out; ++i) {
            //     PyObject* item = PyList_GetItem(numpy_flat, i);  // borrowed ref!
            //     output_values.push_back(PyFloat_AsDouble(item));
            // }

            // Extract input values
            // numpy_input_flat = PyObject_CallMethod(numpy_input, "flatten", nullptr);
            // if (!numpy_input_flat || !PyList_Check(numpy_input_flat)) {
            //     throw std::runtime_error("Failed to flatten input");
            // }

            // Py_ssize_t num_in = PyObject_Length(numpy_input_flat);
            // for (Py_ssize_t i = 0; i < num_in; ++i) {
            //     PyObject* item = PyList_GetItem(numpy_input_flat, i);  // borrowed ref!
            //     input_values.push_back(PyFloat_AsDouble(item));
            // }

        } catch (const std::exception& e) {
            fprintf(stderr, "C++ Exception: %s\n", e.what());
        } catch (...) {
            fprintf(stderr, "Unknown exception in execute_torch_operator\n");
        }

        // === Cleanup: Safe to DECREF null pointers? No, so check ===
        #define SAFE_DECREF(obj) if (obj) { Py_DECREF(obj); obj = nullptr; }
        SAFE_DECREF(numpy_module);
        SAFE_DECREF(torch_module);
        SAFE_DECREF(functional_module);
        SAFE_DECREF(numpy_random);
        SAFE_DECREF(numpy_rand_func);
        SAFE_DECREF(shape_tuple);
        SAFE_DECREF(numpy_input);
        SAFE_DECREF(torch_input);
        SAFE_DECREF(op_func);
        SAFE_DECREF(kwargs);
        SAFE_DECREF(torch_output);
        SAFE_DECREF(numpy_output);
        SAFE_DECREF(numpy_output_shape);
        SAFE_DECREF(numpy_flat);
        SAFE_DECREF(numpy_input_flat);
        #undef SAFE_DECREF

        return {input_values, output_values, output_shape};
    }
};

} // namespace tests
} // namespace vkop

#endif // VKOP_TESTS_HPP_
