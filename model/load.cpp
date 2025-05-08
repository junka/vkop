#include <iostream>
#include <string>
#include <cstdint>
#include <cstring>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "load.hpp"

namespace vkop {

VkModel::VkModel(const std::string& filePath) {
    loadFromBinary(filePath);
}

void VkModel::loadFromBinary(const std::string& filePath) {

    // Open the file
    int fd = open(filePath.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("Failed to open file: " + filePath);
    }

    // Get the file size
    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        throw std::runtime_error("Failed to get file size: " + filePath);
    }

    size_t fileSize = st.st_size;

    // Memory map the file
    void* mappedData = mmap(nullptr, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mappedData == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("Failed to mmap file: " + filePath);
    }

    close(fd);

    // Parse the binary data
    const char* ptr = static_cast<const char*>(mappedData);
    const char* end = ptr + fileSize;

    this->inputs = readListWithShapes(ptr, end);
    this->outputs = readListWithShapes(ptr, end);

    // Read nodes
    uint32_t numNodes = readUint32(ptr, end);
    for (uint32_t i = 0; i < numNodes; ++i) {
        Node node;
        node.op_type = readString(ptr, end);
        node.attributes = readDict(ptr, end);
        node.inputs = readListWithShapes(ptr, end);
        node.outputs = readListWithShapes(ptr, end);
        this->nodes.push_back(std::move(node));
    }

    // Read initializers
    uint32_t numInitializers = readUint32(ptr, end);
    for (uint32_t i = 0; i < numInitializers; ++i) {
        Initializer initializer;
        initializer.name = readString(ptr, end);
        initializer.dtype = readString(ptr, end);
        initializer.dims = readDims(ptr, end);
        uint64_t dataSize = readUint64(ptr, end);
        initializer.data = ptr;
        ptr += dataSize;
        this->initializers[initializer.name] = std::move(initializer);
    }

    // Unmap the file
    munmap(const_cast<char*>(static_cast<const char*>(mappedData)), fileSize);

}

uint32_t VkModel::readUint32(const char*& ptr, const char* end) {
    if (ptr + sizeof(uint32_t) > end) throw std::runtime_error("Unexpected end of file");
    uint32_t value;
    std::memcpy(&value, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    return value;
}

uint64_t VkModel::readUint64(const char*& ptr, const char* end) {
    if (ptr + sizeof(uint64_t) > end) throw std::runtime_error("Unexpected end of file");
    uint64_t value;
    std::memcpy(&value, ptr, sizeof(uint64_t));
    ptr += sizeof(uint64_t);
    return value;
}

float VkModel::readFloat32(const char*& ptr, const char* end) {
    if (ptr + sizeof(float) > end) throw std::runtime_error("Unexpected end of file");
    float value;
    std::memcpy(&value, ptr, sizeof(float));
    ptr += sizeof(float);
    return value;
}

double VkModel::readFloat64(const char*& ptr, const char* end) {
    if (ptr + sizeof(double) > end) throw std::runtime_error("Unexpected end of file");
    double value;
    std::memcpy(&value, ptr, sizeof(double));
    ptr += sizeof(double);
    return value;
}

std::string VkModel::readString(const char*& ptr, const char* end) {
    uint32_t length = readUint32(ptr, end);
    if (ptr + length > end) throw std::runtime_error("Unexpected end of file");
    std::string str(ptr, length);
    ptr += length;
    return str;
}

std::vector<uint32_t> VkModel::readDims(const char*& ptr, const char* end) {
    uint32_t numDims = readUint32(ptr, end);
    std::vector<uint32_t> dims(numDims);
    for (uint32_t i = 0; i < numDims; ++i) {
        dims[i] = readUint32(ptr, end);
    }
    return dims;
}

std::vector<Shape> VkModel::readListWithShapes(const char*& ptr, const char* end) {
    uint32_t count = readUint32(ptr, end);
    std::vector<Shape> shapes(count);
    for (uint32_t i = 0; i < count; ++i) {
        shapes[i].name = readString(ptr, end);
        shapes[i].dims = readDims(ptr, end);
    }
    return shapes;
}

std::unordered_map<std::string, std::string> VkModel::readDict(const char*& ptr, const char* end) {
    uint32_t count = readUint32(ptr, end);
    std::unordered_map<std::string, std::string> dict;
    for (uint32_t i = 0; i < count; ++i) {
        std::string key = readString(ptr, end);
        std::string value;
        uint8_t tag = *ptr;
        ptr ++;
        if (tag == 0) {
            value = readString(ptr, end);
        } else if (tag == 1) {
            value = std::to_string(readUint64(ptr, end));
        } else if (tag == 2) {
            value = std::to_string(readFloat64(ptr, end));
        }
        dict[key] = value;
    }
    return dict;
}

}
