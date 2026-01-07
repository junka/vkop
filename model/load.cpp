// Copyright 2025 @junka
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

    // Calculate total memory size needed for all initializers
    size_t total_memory_size = 0;
    // Align to 64 bytes to avoid memory fragmentation
    const size_t alignment = 64; 

    uint32_t num_initializers = readUint32(ptr, end);
    const char *init_start = ptr;
    for (uint32_t i = 0; i < num_initializers; ++i) {
        Initializer initializer;
        initializer.name = readString(ptr, end);
        initializer.dtype = readString(ptr, end);
        initializer.dims = readDims(ptr, end);
        int64_t data_size = readint64(ptr, end);
        ptr += data_size;
        // Align the offset
        size_t aligned_offset =
            (total_memory_size + alignment - 1) & ~(alignment - 1);
        this->initializer_offsets[initializer.name] = aligned_offset;
        total_memory_size = aligned_offset + data_size;
        this->initializers[initializer.name] = std::move(initializer);
    }

    printf("Total memory allocated for initializers: %zu bytes\n",
           total_memory_size);

    // Allocate a single large memory block
    initializer_memory.reserve(total_memory_size);

    // second pass to read initializers
    ptr = init_start;
    // Copy initializer data into the allocated memory
    for (uint32_t i = 0; i < num_initializers; ++i) {
        auto name = readString(ptr, end);
        auto dtype = readString(ptr, end);
        auto dims = readDims(ptr, end);
        int64_t data_size = readint64(ptr, end);

        size_t offset = initializer_offsets[name];
        uint8_t *dest_ptr = initializer_memory.data() + offset;

        std::memcpy(dest_ptr, ptr, data_size);
        ptr += data_size;
    }

    if (this->initializers.find("unified_metadata") != this->initializers.end()) {
        size_t meta_offset = initializer_offsets["unified_metadata"];
        auto names_offset = this->initializer_offsets["unified_names"];
        auto tensors_offset = this->initializer_offsets["unified_tensors"];
        auto dims = this->initializers["unified_metadata"].dims;
        int num_metas = dims[0]/8;
        uint8_t *meta_ptr = initializer_memory.data() + meta_offset;
        uint8_t *name_ptr = initializer_memory.data() + names_offset;
        std::vector<struct UnifiedMetadata> metas(num_metas);
        std::memcpy(metas.data(), meta_ptr, sizeof(struct UnifiedMetadata) * num_metas);
        size_t name_idx_offset = 0;
        std::string datatyep_map[] = {
            "none",
            "float32",
            "uint8",
            "int8",
            "uint16",
            "int16",
            "int32",
            "int64",
            "string",
            "bool",
            "float16",
            "float64",
            "uint32",
            "uint64",
            "complex64",
            "complex128",
            "bfloat16"
        };
        for (int i = 0; i < num_metas; ++i) {
            auto &meta = metas[i];
            Initializer initializer;
            initializer.name = std::string(name_ptr + name_idx_offset, name_ptr + name_idx_offset + meta.name_len);
            initializer.dtype = datatyep_map[meta.dtype];
            for (unsigned int dim : meta.dims) {
                if (dim == 0) {
                    break;
                }
                initializer.dims.push_back(dim);
            }
            initializer_offsets[initializer.name] = tensors_offset + meta.offset;
            this->initializers[initializer.name] = std::move(initializer);
            name_idx_offset += meta.name_len;
        }
        this->initializers.erase("unified_metadata");
        this->initializers.erase("unified_names");
        this->initializers.erase("unified_tensors");

    }

    // Unmap the file
    munmap(const_cast<char*>(static_cast<const char*>(mapped_data)), file_size);
}


} // namespace load
} // namespace vkop
