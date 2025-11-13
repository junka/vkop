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
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

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

                auto output = std::make_shared<Tensor<T>>();
                auto outputs = std::vector<std::shared_ptr<core::ITensor>> {output};
                op->apply(inputs, outputs);
                for (const auto &input : inputs) {
                    if (!input || input->num_dims() < 3) {
                        continue;
                    }
                    auto t = core::as_tensor<T>(input);
                    t->copyToGPU(dev, cmdpool);
                }
                // op->copyTensorToImages<T>(inputs);
                op->execute(inputs, outputs);
                // op->copyImageToTensor<T>(output);
                output->copyToCPU(dev, cmdpool);
                auto *out_ptr = output->data();
                auto oshape = output->getTensorShape();
                for (int i = 0; i < oshape[0]; i++) {
                    printf("[\n");
                    for (int j = 0; j < oshape[1]; j++) {
                        printf("[\n");
                        for (int k = 0; k < oshape[2]; k++) {
                            printf("[");
                            for (int l = 0; l < oshape[3]; l++) {
                                int idx = i * oshape[1] * oshape[2] * oshape[3] + j * oshape[2] * oshape[3] +
                                    k * oshape[3] + l;
                                printf("%.4f, ", out_ptr[idx]);
                            }
                            printf("]\n");
                        }
                        printf("]\n");
                    }
                    printf("]\n");
                }
                for (int i = 0; i < output->num_elements(); i++) {
                    std::cout << i<< ": " << out_ptr[i] << " vs " <<expectedOutput[i] << std::endl;
                    if (std::fabs(out_ptr[i] - expectedOutput[i]) > 1e-4) {
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
    bool run_test(const std::vector<std::shared_ptr<core::ITensor>> &inputs,
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

                auto op = ops::OperatorFactory::get_instance().create(vkop::ops::convert_opstring_to_enum(name_));
                if (!op) {
                    LOG_ERROR("Fail to create operator");
                    return false;
                }
                op->set_runtime_device(dev, cmdpool);

                auto output = std::make_shared<Tensor<T>>();
                auto outputs = std::vector<std::shared_ptr<core::ITensor>> {output};
                op->apply(inputs, outputs);
                for (const auto &input : inputs) {
                    if (!input || input->num_dims() < 3) {
                        continue;
                    }
                    auto t = core::as_tensor<T>(input);
                    t->copyToGPU(dev, cmdpool);
                }
                // op->copyTensorToImages<T>(inputs);
                op->execute(inputs, std::vector<std::shared_ptr<core::ITensor>> {output});
                // op->copyImageToTensor<T>(output);
                output->copyToCPU(dev, cmdpool);
                auto *out_ptr = output->data();
                for (int i = 0; i < output->num_elements(); i++) {
                    if (sizeof(T) == 2) {
                        if (std::fabs(float16_to_float32(out_ptr[i]) - float16_to_float32(expectedOutput[i])) > 0.01) {
                            LOG_ERROR("Test Fail at1 (%d): %f, %f", i, float16_to_float32(out_ptr[i]), float16_to_float32(expectedOutput[i]));
                            return false;
                        }
                    } else {
                        if (std::fabs(out_ptr[i] - expectedOutput[i]) > 1e-3) {
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

    static void* initialize_pyenv() {
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
                return nullptr;
            }

            import_array();

            // Store the modules for later use
            Py_XINCREF(numpy_module);
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

    // 工具函数：按指定维度打印 NumPy 数组（支持任意形状，按行展开）
    static void print_numpy_array(PyObject* numpy_array, const std::string& name, const std::vector<int>& shape) {
        if (!numpy_array || !PyArray_Check((PyArrayObject*)numpy_array)) {
            printf("%s is not a valid numpy array\n", name.c_str());
            return;
        }

        auto* data = static_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(numpy_array)));
        printf("=== %s (shape: [", name.c_str());
        for (size_t i = 0; i < shape.size(); ++i) {
            printf("%d", shape[i]);
            if (i != shape.size() - 1) printf(", ");
        }
        printf("]) ===\n");

        // 使用递归方式按维度分组打印
        std::function<void(float*, int, int, const std::vector<int>&, int)> print_recursive =
            [&](float* ptr, int depth, int flat_idx, const std::vector<int>& dims, int leading_spaces) {
                if (depth == static_cast<int>(dims.size())) {
                    printf("%.4f ", ptr[flat_idx]);
                    return;
                }

                int stride = 1;
                for (size_t i = depth + 1; i < dims.size(); ++i) {
                    stride *= dims[i];
                }

                printf("[");
                for (int i = 0; i < dims[depth]; ++i) {
                    // 前导空格表示维度层级
                    for (int s = 0; s < depth * 2; ++s) printf(" ");
                    print_recursive(ptr, depth + 1, flat_idx + i * stride, dims, leading_spaces + 2);
                }
                printf("]\n");
            };

        print_recursive(data, 0, 0, shape, 0);
        printf("\n");
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

        // 所有 PyObject 指针初始化为 nullptr，便于安全清理
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
        PyObject* torch_output = nullptr;
        PyObject* numpy_output = nullptr;
        PyObject* numpy_output_shape = nullptr;
        PyObject* numpy_flat = nullptr;
        PyObject* numpy_input_flat = nullptr;
        PyObject* nomarlized_shape = nullptr;

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

            // === 3. Create shape tuples ===
            auto create_shape_tuple = [] (std::vector<int> shapes) ->PyObject * {
                PyObject * shape_tuple = PyTuple_New(shapes.size());
                for (size_t i = 0; i < shapes.size(); ++i) {
                    PyObject* item = PyLong_FromLong(shapes[i]);
                    PyTuple_SetItem(shape_tuple, i, item);
                }
                return shape_tuple;
            };
 
            shape_tuple = create_shape_tuple(input_shape);
            if (has_weight) {
                weight_tuple = create_shape_tuple(weight_shape);
            }

            if (has_bias) {
                bias_tuple = create_shape_tuple(bias_shape);
            }

            numpy_random = PyObject_GetAttrString(numpy_module, "random");
            if (!numpy_random) throw std::runtime_error("Failed to get numpy.random");

            numpy_rand_func = PyObject_GetAttrString(numpy_random, "rand");
            if (!numpy_rand_func) throw std::runtime_error("Failed to get numpy.random.rand");

            auto numpy_rand_as_float = [&numpy_module, &numpy_rand_func] (PyObject* tuple) {
                auto *input = PyObject_CallObject(numpy_rand_func, tuple);
                if (!input) throw std::runtime_error("Failed to generate random input");
                input = PyObject_CallMethod(input, "astype", "(O)", PyObject_GetAttrString(numpy_module, "float32"));
                return input;
            };
            numpy_input = numpy_rand_as_float(shape_tuple);

            if (has_weight) {
                numpy_weight = numpy_rand_as_float(weight_tuple);
            }

            if (has_bias) {
                numpy_bias = numpy_rand_as_float(bias_tuple);
            }

            PyObject* torch_from_numpy = PyObject_GetAttrString(torch_module, "from_numpy");
            if (!torch_from_numpy) throw std::runtime_error("Failed to get torch.from_numpy");

            torch_input = PyObject_CallFunctionObjArgs(torch_from_numpy, numpy_input, nullptr);
            if (!torch_input) throw std::runtime_error("Failed to convert input to torch tensor");
            Py_DECREF(torch_from_numpy);

            if (has_weight) {
                torch_from_numpy = PyObject_GetAttrString(torch_module, "from_numpy");
                torch_weight = PyObject_CallFunctionObjArgs(torch_from_numpy, numpy_weight, nullptr);
                if (!torch_weight) throw std::runtime_error("Failed to convert weight to torch tensor");
                Py_DECREF(torch_from_numpy);
            }

            if (has_bias) {
                torch_from_numpy = PyObject_GetAttrString(torch_module, "from_numpy");
                torch_bias = PyObject_CallFunctionObjArgs(torch_from_numpy, numpy_bias, nullptr);
                if (!torch_bias) throw std::runtime_error("Failed to convert bias to torch tensor");
                Py_DECREF(torch_from_numpy);
            }
            op_func = PyObject_GetAttrString(functional_module, op_name.c_str());
            if (!op_func) {
                PyErr_Print();
                throw std::runtime_error("Failed to find operator: " + op_name);
            }

            // === 7. Build kwargs dictionary ===
            kwargs = PyDict_New();
            for (const auto& attr : attrs) {
                const std::string& key = attr.first;
                const std::string& val_str = attr.second;
                PyObject* value = nullptr;

                printf("Parsing attribute: %s = %s\n", key.c_str(), val_str.c_str());
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
                    nomarlized_shape = create_shape_tuple(normalized_shape);
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
                    value = create_shape_tuple(resize_shape);
                } else {
                    // Try int first, then float, then string
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
                Py_XDECREF(value); // PyDict_SetItemString does NOT steal, so we must DECREF
            }

            // === 8. Prepare positional arguments dynamically ===
            std::vector<PyObject*> arg_list;
            arg_list.push_back(torch_input);
            if (has_laynernorm) arg_list.push_back(nomarlized_shape);
            if (has_weight) arg_list.push_back(torch_weight);
            if (has_bias) arg_list.push_back(torch_bias);

            PyObject* args = PyTuple_New(arg_list.size());
            for (size_t i = 0; i < arg_list.size(); ++i) {
                PyTuple_SetItem(args, i, arg_list[i]); // Steals reference
            }

            // === 9. Call the operator: op_func(*args, **kwargs) ===
            torch_output = PyObject_Call(op_func, args, kwargs);
            Py_DECREF(args);
            if (!torch_output) {
                PyErr_Print();
                throw std::runtime_error("Failed to execute operator: " + op_name);
            }
            printf("Operator %s executed successfully.\n", op_name.c_str());

            // === 10. Convert output to numpy for data extraction ===
            // Ensure it's on CPU and convert to numpy
            PyObject* cpu_output = PyObject_CallMethod(torch_output, "cpu", nullptr);
            if (!cpu_output) {
                PyErr_Print();
                cpu_output = torch_output;
                Py_INCREF(cpu_output);
            }

            numpy_output = PyObject_CallMethod(cpu_output, "numpy", nullptr);
            Py_DECREF(cpu_output);
            if (!numpy_output) {
                PyErr_Print();
                throw std::runtime_error("Failed to convert output tensor to numpy array");
            }

            numpy_output_shape = PyObject_GetAttrString(numpy_output, "shape");
            if (numpy_output_shape && PyTuple_Check(numpy_output_shape)) {
                Py_ssize_t ndim = PyTuple_Size(numpy_output_shape);
                for (Py_ssize_t i = 0; i < ndim; ++i) {
                    output_shape.push_back(PyLong_AsLong(PyTuple_GetItem(numpy_output_shape, i)));
                }
            } else {
                throw std::runtime_error("Failed to get output shape");
            }
            // print_numpy_array(numpy_input, "NumPy Input", input_shape);
            // print_numpy_array(numpy_output, "NumPy Output", output_shape);

            auto flatten_values = [] (PyObject * values) -> std::vector<float> {
                std::vector<float> output;
                auto *flat = PyObject_CallMethod(values, "flatten", nullptr);
                if (!flat || !PyArray_Check((PyArrayObject*)flat)) {
                    throw std::runtime_error("Failed to flatten output array");
                }
                auto* out_data = static_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(flat)));
                npy_intp num_out = PyArray_SIZE((PyArrayObject*)flat);
                output.resize(num_out);
                std::copy(out_data, out_data + num_out, output.begin());
                return output;
            };

            ret.emplace_back(flatten_values(numpy_output));
            ret.emplace_back(flatten_values(numpy_input));
            if (has_weight) ret.emplace_back(flatten_values(numpy_weight));
            if (has_bias) ret.emplace_back(flatten_values(numpy_bias));

        } catch (const std::exception& e) {
            fprintf(stderr, "C++ Exception: %s\n", e.what());
        } catch (...) {
            fprintf(stderr, "Unknown exception in execute_torch_operator\n");
        }

        // === Cleanup: Safe DECREF ===
    #define SAFE_DECREF(obj) do { if (obj) { Py_DECREF(obj); (obj) = nullptr; } } while(0)
        SAFE_DECREF(numpy_module);
        SAFE_DECREF(torch_module);
        SAFE_DECREF(functional_module);
        SAFE_DECREF(numpy_random);
        SAFE_DECREF(numpy_rand_func);
        SAFE_DECREF(shape_tuple);
        SAFE_DECREF(weight_tuple);
        SAFE_DECREF(bias_tuple);
        SAFE_DECREF(numpy_input);
        SAFE_DECREF(numpy_weight);
        SAFE_DECREF(numpy_bias);
        SAFE_DECREF(torch_input);
        SAFE_DECREF(torch_weight);
        SAFE_DECREF(torch_bias);
        SAFE_DECREF(op_func);
        SAFE_DECREF(kwargs);
        SAFE_DECREF(torch_output);
        SAFE_DECREF(numpy_output);
        SAFE_DECREF(numpy_output_shape);
        SAFE_DECREF(numpy_flat);
        SAFE_DECREF(numpy_input_flat);
    #undef SAFE_DECREF

        return {ret, output_shape};
    }
};

} // namespace tests
} // namespace vkop

#endif // VKOP_TESTS_HPP_
