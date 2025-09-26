// Copyright 2025 @junka
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
namespace load {

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

    size_t file_size = st.st_size;

    // Memory map the file
    void* mapped_data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped_data == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("Failed to mmap file: " + filePath);
    }

    close(fd);

    // Parse the binary data
    const char* ptr = static_cast<const char*>(mapped_data);
    const char* end = ptr + file_size;

    this->inputs = readListWithShapes(ptr, end);
    this->outputs = readListWithShapes(ptr, end);

    // Read nodes
    uint32_t num_nodes = readUint32(ptr, end);
    for (uint32_t i = 0; i < num_nodes; ++i) {
        Node node;
        node.op_type = readString(ptr, end);
        node.name = readString(ptr, end);
        node.attributes = readDict(ptr, end);
        node.inputs = readListWithShapes(ptr, end);
        node.outputs = readListWithShapes(ptr, end);
        this->nodes.push_back(std::move(node));
    }

    // Read initializers
    uint32_t num_initializers = readUint32(ptr, end);
    for (uint32_t i = 0; i < num_initializers; ++i) {
        Initializer initializer;
        initializer.name = readString(ptr, end);
        initializer.dtype = readString(ptr, end);
        initializer.dims = readDims(ptr, end);
        uint64_t data_size = readUint64(ptr, end);
        printf("Initializer %s, dtype: %s, dim %d, %d, %d, %d, size: %lu bytes\n", initializer.name.c_str(), initializer.dtype.c_str(), initializer.dims[0], initializer.dims[1], initializer.dims[2], initializer.dims[3], data_size);
        if (initializer.dtype == "float32") {
            uint64_t num_elements = data_size / sizeof(float);
            initializer.dataf.resize(num_elements);
            printf("Expect %lu float32 elements\n", num_elements);
            std::memcpy(initializer.dataf.data(), ptr, data_size);
        } else if (initializer.dtype == "int32") {
            uint64_t num_elements = data_size / sizeof(int32_t);
            initializer.datai.resize(num_elements);
            std::memcpy(initializer.datai.data(), ptr, data_size);
        } else if (initializer.dtype == "int64") {
            uint64_t num_elements = data_size / sizeof(int64_t);
            initializer.dataii.resize(num_elements);
            std::memcpy(initializer.dataii.data(), ptr, data_size);
        } else {
            throw std::runtime_error("Unsupported initializer dtype: " + initializer.dtype);
        }
        ptr += data_size;
        this->initializers[initializer.name] = std::move(initializer);
    }

    // Unmap the file
    munmap(const_cast<char*>(static_cast<const char*>(mapped_data)), file_size);
}

} // namespace load
} // namespace vkop
