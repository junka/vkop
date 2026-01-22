// Copyright 2025 @junka
#include <string>
#include <cstdint>
#include <cstring>

#ifdef _WIN32
    #include <windows.h>
    #include <io.h>
#else
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <unistd.h>
#endif
#include "load.hpp"

namespace vkop {
namespace load {

VkModel::VkModel(const std::string& filePath) {
    loadFromBinary(filePath);
}

struct FileMapping {
    void* data = nullptr;
    size_t size = 0;

#ifdef _WIN32
    HANDLE hFile = INVALID_HANDLE_VALUE;
    HANDLE hMapping = nullptr;
#else
    int fd = -1;
#endif

    ~FileMapping() {
#ifdef _WIN32
        if (data) UnmapViewOfFile(data);
        if (hMapping != nullptr) CloseHandle(hMapping);
        if (hFile != INVALID_HANDLE_VALUE) CloseHandle(hFile);
#else
        if (data && data != MAP_FAILED) {
            munmap(data, size);
        }
        if (fd >= 0) close(fd);
#endif
    }

    bool map_file(const std::string& path) {
#ifdef _WIN32
        hFile = CreateFileA(path.c_str(),
                            GENERIC_READ,
                            FILE_SHARE_READ,
                            nullptr,
                            OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL,
                            nullptr);
        if (hFile == INVALID_HANDLE_VALUE) {
            return false;
        }

        LARGE_INTEGER li;
        if (!GetFileSizeEx(hFile, &li) || li.QuadPart > SIZE_MAX) {
            return false;
        }
        size = static_cast<size_t>(li.QuadPart);

        if (size == 0) {
            data = nullptr;
            return true;
        }

        hMapping = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!hMapping) {
            return false;
        }

        data = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, size);
        return data != nullptr;
#else
        fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;

        struct stat st;
        if (fstat(fd, &st) < 0) return false;
        size = static_cast<size_t>(st.st_size);

        if (size == 0) {
            data = nullptr;
            return true;
        }

        data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        return data != MAP_FAILED;
#endif
    }
};

void VkModel::loadFromBinary(const std::string& filePath) {
    FileMapping mapping;
    if (!mapping.map_file(filePath)) {
        throw std::runtime_error("Failed to map file: " + filePath);
    }

    const char* ptr = static_cast<const char*>(mapping.data);
    const char* end = ptr + mapping.size;

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
        unified = true;
    }

    if (this->initializers.find("rgba_conversion_metadata") != this->initializers.end()) {
        printf("Found RGBA conversion metadata, restoring original shapes...\n");
        size_t meta_offset = initializer_offsets["rgba_conversion_metadata"];
        auto names_offset = this->initializer_offsets["rgba_conversion_names"];
        auto dims = this->initializers["rgba_conversion_metadata"].dims;
        int num_metas = dims[0]/8;
        printf("Number of RGBA conversion metas: %d\n", dims[0]);
        uint8_t *meta_ptr = initializer_memory.data() + meta_offset;
        uint8_t *name_ptr = initializer_memory.data() + names_offset;
        std::vector<struct RGBAConversion> metas(num_metas);
        std::memcpy(metas.data(), meta_ptr, sizeof(struct RGBAConversion) * num_metas);
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
            auto name = std::string(name_ptr + name_idx_offset, name_ptr + name_idx_offset + meta.name_len);
            auto dtype = datatyep_map[meta.dtype];
            this->initializers[name].dims.resize(4);
            for (int i = 0; i < 4; ++i) {
                this->initializers[name].dims[i] = meta.dims[i];
            }
            name_idx_offset += meta.name_len;
        }
        this->initializers.erase("rgba_conversion_metadata");
        this->initializers.erase("rgba_conversion_names");
        rgba = true;
    }
}


} // namespace load
} // namespace vkop
