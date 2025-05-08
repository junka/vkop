#ifndef __LOAD_MODEL_HPP__
#define __LOAD_MODEL_HPP__

#include <vector>
#include <string>
#include <cstdint>
#include <unordered_map>

namespace vkop {

struct Shape {
    std::string name;
    std::vector<uint32_t> dims;
};

struct Node {
    std::string op_type;
    std::unordered_map<std::string, std::string> attributes;
    std::vector<Shape> inputs;
    std::vector<Shape> outputs;
};

struct Initializer {
    std::string name;
    std::string dtype;
    std::vector<uint32_t> dims;
    const void* data;
};

class VkModel {
public:
    std::vector<Shape> inputs;
    std::vector<Shape> outputs;
    std::vector<Node> nodes;
    std::unordered_map<std::string, Initializer> initializers;

    VkModel(const std::string& filePath);
private:
    void loadFromBinary(const std::string& filePath);
    uint32_t readUint32(const char*& ptr, const char* end);
    uint64_t readUint64(const char*& ptr, const char* end);
    float readFloat32(const char*& ptr, const char* end);
    double readFloat64(const char*& ptr, const char* end);
    std::string readString(const char*& ptr, const char* end);
    std::vector<uint32_t> readDims(const char*& ptr, const char* end);
    std::vector<Shape> readListWithShapes(const char*& ptr, const char* end);
    std::unordered_map<std::string, std::string> readDict(const char*& ptr, const char* end);
};


}

#endif