#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

struct Tensor {
    std::string name;
    std::vector<uint64_t> shape;
};

struct Node {
    std::string op_type;
    std::unordered_map<std::string, std::string> attributes;
    std::vector<Tensor> inputs;
    std::vector<Tensor> outputs;
};

struct Initializer {
    std::string name;
    std::string dtype;
    std::vector<uint64_t> shape;
    const void* data;
};

class CustomModel {
public:
    std::vector<Tensor> inputs;
    std::vector<Tensor> outputs;
    std::vector<Node> nodes;
    std::unordered_map<std::string, Initializer> initializers;

    static CustomModel loadFromBinary(const std::string& filePath) {
        CustomModel model;

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

        model.inputs = readListWithShapes(ptr, end);
        model.outputs = readListWithShapes(ptr, end);

        // Read nodes
        uint32_t numNodes = readUint32(ptr, end);
        for (uint32_t i = 0; i < numNodes; ++i) {
            Node node;
            node.op_type = readString(ptr, end);
            node.attributes = readDict(ptr, end);
            node.inputs = readListWithShapes(ptr, end);
            node.outputs = readListWithShapes(ptr, end);
            model.nodes.push_back(std::move(node));
        }

        // Read initializers
        uint32_t numInitializers = readUint32(ptr, end);
        for (uint32_t i = 0; i < numInitializers; ++i) {
            Initializer initializer;
            initializer.name = readString(ptr, end);
            initializer.dtype = readString(ptr, end);
            initializer.shape = readShape(ptr, end);
            uint64_t dataSize = readUint64(ptr, end);
            initializer.data = ptr;
            ptr += dataSize;
            model.initializers[initializer.name] = std::move(initializer);
        }

        // Unmap the file
        munmap(const_cast<char*>(static_cast<const char*>(mappedData)), fileSize);

        return model;
    }

private:
    static uint32_t readUint32(const char*& ptr, const char* end) {
        if (ptr + sizeof(uint32_t) > end) throw std::runtime_error("Unexpected end of file");
        uint32_t value;
        std::memcpy(&value, ptr, sizeof(uint32_t));
        ptr += sizeof(uint32_t);
        return value;
    }

    static uint64_t readUint64(const char*& ptr, const char* end) {
        if (ptr + sizeof(uint64_t) > end) throw std::runtime_error("Unexpected end of file");
        uint64_t value;
        std::memcpy(&value, ptr, sizeof(uint64_t));
        ptr += sizeof(uint64_t);
        return value;
    }

    static std::string readString(const char*& ptr, const char* end) {
        uint32_t length = readUint32(ptr, end);
        if (ptr + length > end) throw std::runtime_error("Unexpected end of file");
        std::string str(ptr, length);
        ptr += length;
        return str;
    }

    static std::vector<uint64_t> readShape(const char*& ptr, const char* end) {
        uint32_t numDims = readUint32(ptr, end);
        std::vector<uint64_t> shape(numDims);
        for (uint32_t i = 0; i < numDims; ++i) {
            shape[i] = readUint64(ptr, end);
        }
        return shape;
    }

    static std::vector<Tensor> readListWithShapes(const char*& ptr, const char* end) {
        uint32_t count = readUint32(ptr, end);
        std::vector<Tensor> tensors(count);
        for (uint32_t i = 0; i < count; ++i) {
            tensors[i].name = readString(ptr, end);
            tensors[i].shape = readShape(ptr, end);
        }
        return tensors;
    }

    static std::unordered_map<std::string, std::string> readDict(const char*& ptr, const char* end) {
        uint32_t count = readUint32(ptr, end);
        std::unordered_map<std::string, std::string> dict;
        for (uint32_t i = 0; i < count; ++i) {
            std::string key = readString(ptr, end);
            std::string value = readString(ptr, end);
            dict[key] = value;
        }
        return dict;
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <binary_file_path>" << std::endl;
        return 1;
    }

    try {
        std::string binaryFilePath = argv[1];
        CustomModel model = CustomModel::loadFromBinary(binaryFilePath);

        std::cout << "Inputs:" << std::endl;
        for (const auto& input : model.inputs) {
            std::cout << "  Name: " << input.name << ", Shape: [";
            for (size_t i = 0; i < input.shape.size(); ++i) {
                std::cout << input.shape[i] << (i + 1 < input.shape.size() ? ", " : "");
            }
            std::cout << "]" << std::endl;
        }

        std::cout << "Outputs:" << std::endl;
        for (const auto& output : model.outputs) {
            std::cout << "  Name: " << output.name << ", Shape: [";
            for (size_t i = 0; i < output.shape.size(); ++i) {
                std::cout << output.shape[i] << (i + 1 < output.shape.size() ? ", " : "");
            }
            std::cout << "]" << std::endl;
        }

        std::cout << "Nodes:" << std::endl;
        for (const auto& node : model.nodes) {
            std::cout << "  OpType: " << node.op_type << std::endl;
        }

        std::cout << "Initializers:" << std::endl;
        for (const auto& [name, initializer] : model.initializers) {
            std::cout << "  Name: " << name << ", Shape: [";
            for (size_t i = 0; i < initializer.shape.size(); ++i) {
                std::cout << initializer.shape[i] << (i + 1 < initializer.shape.size() ? ", " : "");
            }
            std::cout << "], DType: " << initializer.dtype << std::endl;
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}