// Copyright 2025 @junka
#ifndef MODEL_LOAD_HPP_
#define MODEL_LOAD_HPP_

#include <cassert>
#include <stdexcept>
#include <vector>
#include <string>
#include <cstdint>
#include <unordered_map>
#include <cstring>
#include <iostream>

namespace vkop {

namespace load {
struct Shape {
    std::string name;
    std::vector<uint32_t> dims;
};

struct Node {
    std::string op_type;
    std::string name;
    std::unordered_map<std::string, std::string> attributes;
    std::vector<Shape> inputs;
    std::vector<Shape> outputs;
};

struct Initializer {
    std::string name;
    std::string dtype;
    std::vector<uint32_t> dims;
};

class VkModel {
public:
    std::vector<Shape> inputs;
    std::vector<Shape> outputs;
    std::vector<Node> nodes;
    std::unordered_map<std::string, Initializer> initializers;
    std::unordered_map<std::string, size_t> initializer_offsets;
    std::vector<uint8_t> initializer_memory;

    explicit VkModel(const std::string& filePath);
    void dump_model() {
        std::cout << "Inputs:" << std::endl;
        for (const auto& input : this->inputs) {
            std::cout << "  Name: " << input.name << ", Shape: [";
            for (size_t i = 0; i < input.dims.size(); ++i) {
                std::cout << input.dims[i] << (i + 1 < input.dims.size() ? ", " : "");
            }
            std::cout << "]" << std::endl;
        }

        std::cout << "Outputs:" << std::endl;
        for (const auto& output : this->outputs) {
            std::cout << "  Name: " << output.name << ", Shape: [";
            for (size_t i = 0; i < output.dims.size(); ++i) {
                std::cout << output.dims[i] << (i + 1 < output.dims.size() ? ", " : "");
            }
            std::cout << "]" << std::endl;
        }

        std::cout << "Nodes:" << std::endl;
        for (const auto& node : this->nodes) {
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

        std::cout << "Initializers:" << std::endl;
        for (const auto& [name, initializer] : this->initializers) {
            std::cout << name << ", [";
            for (size_t i = 0; i < initializer.dims.size(); ++i) {
                std::cout << initializer.dims[i] << (i + 1 < initializer.dims.size() ? ", " : "");
            }
            std::cout << "], DType: " << initializer.dtype << std::endl;
        }
    }
private:
    void loadFromBinary(const std::string& filePath);

    static uint32_t readUint32(const char*& ptr, const char* end) {
        if (ptr + sizeof(uint32_t) > end) throw std::runtime_error("Unexpected end of file u32");
        uint32_t value;
        std::memcpy(&value, ptr, sizeof(uint32_t));
        ptr += sizeof(uint32_t);
        return value;
    }

    static int64_t readint64(const char*& ptr, const char* end) {
        if (ptr + sizeof(int64_t) > end) throw std::runtime_error("Unexpected end of file i64");
        int64_t value;
        std::memcpy(&value, ptr, sizeof(int64_t));
        ptr += sizeof(int64_t);
        return value;
    }

    static float readFloat32(const char*& ptr, const char* end) {
        if (ptr + sizeof(float) > end) throw std::runtime_error("Unexpected end of file f32");
        float value;
        std::memcpy(&value, ptr, sizeof(float));
        ptr += sizeof(float);
        return value;
    }

    static double readFloat64(const char*& ptr, const char* end) {
        if (ptr + sizeof(double) > end) throw std::runtime_error("Unexpected end of file f64");
        double value;
        std::memcpy(&value, ptr, sizeof(double));
        ptr += sizeof(double);
        return value;
    }

    static std::string readString(const char*& ptr, const char* end) {
        uint32_t length = readUint32(ptr, end);
        if (ptr + length > end) throw std::runtime_error("Unexpected end of file str");
        std::string str(ptr, length);
        ptr += length;
        return str;
    }
    
    static std::vector<uint32_t> readDims(const char*& ptr, const char* end) {
        uint32_t num_dims = readUint32(ptr, end);
        std::vector<uint32_t> dims(num_dims);
        for (uint32_t i = 0; i < num_dims; ++i) {
            dims[i] = readUint32(ptr, end);
        }
        return dims;
    }

    template<typename T>
    static std::vector<T> readList(const char*& ptr, const char* end) {
        uint32_t count = readUint32(ptr, end);
        std::vector<T> list(count);
        for (uint32_t i = 0; i < count; ++i) {
            if (std::is_same<T, float>::value) {
                list[i] = static_cast<T>(readFloat32(ptr, end));
            } else if (std::is_same<T, uint32_t>::value) {
                list[i] = static_cast<T>(readUint32(ptr, end));
            } else {
                throw std::runtime_error("Unsupported type for readList");
            }
        }
        return list;
    }

    static std::vector<Shape> readListWithShapes(const char*& ptr, const char* end) {
        uint32_t count = readUint32(ptr, end);
        std::vector<Shape> shapes(count);
        assert(count > 0);
        for (uint32_t i = 0; i < count; ++i) {
            shapes[i].name = readString(ptr, end);
            shapes[i].dims = readDims(ptr, end);
        }
        return shapes;
    }
    static std::unordered_map<std::string, std::string> readDict(const char*& ptr, const char* end) {
        uint32_t count = readUint32(ptr, end);
        std::unordered_map<std::string, std::string> dict;
        for (uint32_t i = 0; i < count; ++i) {
            std::string key = readString(ptr, end);
            std::string value;
            uint8_t tag = *ptr;
            ptr++;
            if (tag == 0) {
                value = readString(ptr, end);
            } else if (tag == 1) {
                value = std::to_string(readint64(ptr, end));
            } else if (tag == 2) {
                value = std::to_string(readFloat32(ptr, end));
            } else if (tag == 3) {
                auto l = readList<uint32_t>(ptr, end);
                value = "[" + std::to_string(l[0]);
                for (size_t i = 1; i < l.size(); ++i) {
                    value += ", " + std::to_string(l[i]);
                }
                value += "]";
            } else if (tag == 4) {
                auto l = readList<float>(ptr, end);
                value = "[" + std::to_string(l[0]);
                for (size_t i = 1; i < l.size(); ++i) {
                    value += ", " + std::to_string(l[i]);
                }
                value += "]";
            } else {
                throw std::runtime_error("Unknown attribute type tag");
            }
            dict[key] = value;
        }
        return dict;
    }
    static std::vector<uint8_t> readArray(const char*& ptr, const char* end) {
        std::string dtype = readString(ptr, end);
        std::vector<uint32_t> shape = readDims(ptr, end);
        int64_t size = readint64(ptr, end);

        if (ptr + size > end) throw std::runtime_error("Unexpected end of file array");
        std::vector<uint8_t> data(size);
        std::memcpy(data.data(), ptr, size);
        ptr += size;

        return data;
    }

};

} // namespace load
} // namespace vkop
#endif /* MODEL_LOAD_HPP_ */
