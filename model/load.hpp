// Copyright 2025 @junka
#ifndef MODEL_LOAD_HPP_
#define MODEL_LOAD_HPP_

#include <cassert>
#include <cerrno>
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
    std::vector<int32_t> dims; // -1 = dynamic sentinel, 0 = empty (see schema)
    std::string dtype; // "" if the writer didn't record one
    // True when this int64 shape-meta tensor's element VALUES vary across
    // decode rounds (depend on a dynamic graph-input dim like kv_len). When
    // false, consumers (Reshape/Unsqueeze/Slice/...) may cache the readback-
    // derived values and skip per-round copyToCPU. See schema ShapeRef.
    bool value_dynamic = false;
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
    // Payload size in BYTES as the writer recorded it. Kept separate from dims
    // because the two are only interchangeable while every element is a whole
    // number of bytes: a packed format (int4, fp4) holds several elements per
    // byte, so the runtime must take the byte count from the file and check it
    // against dims rather than derive one from the other.
    size_t size = 0;
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

    // Hint that this fd will be swept front-to-back. Linux/FreeBSD only: the
    // macOS SDK has no posix_fadvise at all (POSIX_FADV_SEQUENTIAL is not even
    // declared), and there the per-file readahead detector keys off the
    // ascending pread offsets instead, so the hint is not needed to get the
    // same effect. Failure is ignored on purpose — a rejected hint only costs
    // bandwidth, never correctness.
    void advise_sequential() const {
#if defined(__linux__) || defined(__FreeBSD__)
        (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif
    }

    // Read len bytes at a file offset without going through the mapping.
    // Touching a multi-GB blob through mmap faults it in one 16KB page at a
    // time (~1.4 GB/s measured on this volume); large preads run ~13 GB/s.
    // Throws on a short read: a partially filled staging buffer would leave
    // the rest of an initializer silently zeroed on the GPU.
    void read_at(void* dst, size_t len, size_t offset) const {
        if (len == 0) return;
        if (offset > size || len > size - offset) {
            throw std::runtime_error(
                "read_at out of bounds: offset=" + std::to_string(offset) +
                " len=" + std::to_string(len) +
                " file_size=" + std::to_string(size));
        }
        char* out = static_cast<char*>(dst);
        size_t done = 0;
#ifdef _WIN32
        OVERLAPPED ov = {};
        LARGE_INTEGER pos;
        while (done < len) {
            pos.QuadPart = static_cast<LONGLONG>(offset + done);
            ov.Offset = pos.LowPart;
            ov.OffsetHigh = pos.HighPart;
            DWORD got = 0;
            if (!ReadFile(hFile, out + done, static_cast<DWORD>(len - done),
                          &got, &ov) || got == 0) {
                throw std::runtime_error("ReadFile failed at offset " +
                                         std::to_string(offset + done));
            }
            done += got;
        }
#else
        while (done < len) {
            ssize_t n = ::pread(fd, out + done, len - done,
                                static_cast<off_t>(offset + done));
            if (n < 0) {
                if (errno == EINTR) continue;
                throw std::runtime_error("pread failed at offset " +
                                         std::to_string(offset + done) + ": " +
                                         std::strerror(errno));
            }
            if (n == 0) {
                throw std::runtime_error("pread short read at offset " +
                                         std::to_string(offset + done) +
                                         " (" + std::to_string(len - done) +
                                         " bytes requested)");
            }
            done += static_cast<size_t>(n);
        }
#endif
    }

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

    // Byte offset within the file that initializer_memory points at, so a
    // recorded blob offset can be turned into an absolute pread offset (see
    // FileMapping::read_at for why the loader streams weights that way).
    size_t initializer_file_offset = 0;

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

    // Backing mapping for initializer_memory / initializer_offsets. Only valid
    // while this VkModel is alive, and only for models that loaded without
    // throwing (loadFromBinary resets it on every failure path).
    const FileMapping& fileMapping() const {
        assert(file_mapping_);
        return *file_mapping_;
    }

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
