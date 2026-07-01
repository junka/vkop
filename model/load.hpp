// Copyright 2025 @junka
#ifndef MODEL_LOAD_HPP_
#define MODEL_LOAD_HPP_

#include <cassert>
#include <memory>
#include <stdexcept>
#include <vector>
#include <string>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <cstring>
#include <iostream>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <unistd.h>
#endif

#include "generated/vkop_model_generated.h"

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
    std::unordered_set<std::string> dependencies;
    std::unordered_set<std::string> dependents;
};

struct Initializer {
    std::string name;
    std::string dtype;
    std::vector<uint32_t> dims;
};

// RAII handle over a memory-mapped file. Owned by VkModel so that the
// initializer blob view (initializer_memory) stays valid for the model's
// lifetime. Zero-copy: the loader never memcpy's weight data out of the mmap.
struct FileMapping {
    void* data = nullptr;
    size_t size = 0;

#ifdef _WIN32
    HANDLE hFile = INVALID_HANDLE_VALUE;
    HANDLE hMapping = nullptr;
#else
    int fd = -1;
#endif

    FileMapping() = default;
    FileMapping(const FileMapping&) = delete;
    FileMapping& operator=(const FileMapping&) = delete;
    FileMapping(FileMapping&&) noexcept = default;
    FileMapping& operator=(FileMapping&&) noexcept = default;

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


class VkModel {
public:
    std::vector<Shape> inputs;
    std::vector<Shape> outputs;
    std::vector<Node> nodes;
    std::unordered_map<std::string, Initializer> initializers;
    bool rgba = false;
    bool unified = false;
    std::unordered_map<std::string, size_t> initializer_offsets;

    // Zero-copy view into the FlatBuffer's initializer_blob, which itself
    // lives inside the mmap'd file held by file_mapping_. Read-only: runtime
    // only ever reads through this pointer (uploads to GPU / memcpy into
    // host staging). Offsets in initializer_offsets are 64-byte-aligned
    // absolute byte offsets into this block, computed by the Python writer.
    const uint8_t* initializer_memory = nullptr;
    size_t initializer_memory_size = 0;

    // Unified-tensor sub-allocation metadata (replaces the legacy
    // unified_metadata/unified_names/unified_tensors magic-initializer hack).
    // Copied out of the FlatBuffer struct array at load time (cheap: N x 32B);
    // runtime indexes this directly instead of re-parsing the blob.
    std::vector<vkop::model::UnifiedMeta> unified_meta;
    std::vector<vkop::model::RGBAConversionMeta> rgba_meta;
    std::string unified_names;
    std::string rgba_names;
    size_t unified_blob_offset = 0;

    std::vector<std::vector<std::string>> concurrent_execution_levels;

    explicit VkModel(const std::string& filePath);

    const std::vector<std::vector<std::string>>& getConcurrentExecutionLevels() const {
        return concurrent_execution_levels;
    }

    void dump_model() {
        std::cout << "Inputs:" << std::endl;
        for (const auto &input : this->inputs) {
            std::cout << "  Name: " << input.name << ", Shape: [";
            for (size_t i = 0; i < input.dims.size(); ++i) {
                std::cout << input.dims[i] << (i + 1 < input.dims.size() ? ", " : "");
            }
            std::cout << "]" << std::endl;
        }

        std::cout << "Outputs:" << std::endl;
        for (const auto &output : this->outputs) {
            std::cout << "  Name: " << output.name << ", Shape: [";
            for (size_t i = 0; i < output.dims.size(); ++i) {
                std::cout << output.dims[i] << (i + 1 < output.dims.size() ? ", " : "");
            }
            std::cout << "]" << std::endl;
        }

        std::cout << "Nodes:" << std::endl;
        for (const auto &node : this->nodes) {
            std::cout << "  OpType: " << node.op_type;
            std::cout << "  Name: " << node.name;
            if (!node.attributes.empty()) {
                std::cout << ", Attributes: {";
                for (const auto &attr : node.attributes) {
                    std::cout << attr.first << ": " << attr.second << ", ";
                }
                std::cout << "}";
            }
            std::cout << "  Inputs: " ;
            for (const auto &input : node.inputs) {
                std::cout << input.name << ", [";
                for (size_t i = 0; i < input.dims.size(); ++i) {
                    std::cout << input.dims[i] << (i + 1 < input.dims.size() ? ", " : "");
                }
                std::cout << "]" << std::endl;
            }

            std::cout << "  Outputs: ";
            for (const auto &output : node.outputs) {
                std::cout << output.name << ", [";
                for (size_t i = 0; i < output.dims.size(); ++i) {
                    std::cout << output.dims[i] << (i + 1 < output.dims.size() ? ", " : "");
                }
                std::cout << "]" << std::endl;
            }
            if (!node.dependencies.empty()) {
                std::cout << "  Dependencies: {";
                for (const auto &dep : node.dependencies) {
                    std::cout << dep << ", ";
                }
                std::cout << "}" << std::endl;
            }

            if (!node.dependents.empty()) {
                std::cout << "  Dependents: {";
                for (const auto &dep : node.dependents) {
                    std::cout << dep << ", ";
                }
                std::cout << "}" << std::endl;
            }
            std::cout << std::endl;
        }

        std::cout << "Initializers:" << std::endl;
        for (const auto & [name, initializer] : this->initializers) {
            std::cout << name << ", [";
            for (size_t i = 0; i < initializer.dims.size(); ++i) {
                std::cout << initializer.dims[i] << (i + 1 < initializer.dims.size() ? ", " : "");
            }
            std::cout << "], DType: " << initializer.dtype << std::endl;
        }

        std::cout << "Concurrent Execution Levels:" << std::endl;
        for (size_t level_idx = 0; level_idx < concurrent_execution_levels.size(); ++level_idx) {
            std::cout << "  Level " << level_idx << ": {";
            for (size_t i = 0; i < concurrent_execution_levels[level_idx].size(); ++i) {
                std::cout << concurrent_execution_levels[level_idx][i];
                if (i + 1 < concurrent_execution_levels[level_idx].size()) {
                    std::cout << ", ";
                }
            }
            std::cout << "}" << std::endl;
        }
    }

private:
    // Owns the mmap for the model's lifetime; initializer_memory points into it.
    std::unique_ptr<FileMapping> file_mapping_;

    void loadFromBinary(const std::string& filePath);

    // --- FlatBuffers reader ---
    void loadFromFlatbuffer(const uint8_t* buf, size_t size);

    // Helper: stringify a typed Attribute into the unordered_map<string,string>
    // contract that ops/*.hpp::setAttribute expects (mirrors legacy readDict).
    static std::string attrValueToString(const vkop::model::Attribute* attr);
};

} // namespace load
} // namespace vkop
#endif /* MODEL_LOAD_HPP_ */
