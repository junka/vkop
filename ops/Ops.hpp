// Copyright 2025 @junka
#ifndef OPS_OPS_HPP_
#define OPS_OPS_HPP_

#include <string>
#include <vector>

namespace vkop {
namespace ops {

enum class OpType {
    UNKNOWN,
    ADD,
    ATAN,
    AVERAGEPOOL,
    BATCHNORM,
    COL2IM,
    CONCAT,
    CONV2D,
    DIV,
    ERF,
    FLOOR,
    GEMM,
    GLOBALAVERAGEPOOL,
    GRIDSAMPLE,
    LAYERNORM,
    MATMUL,
    MAXPOOL2D,
    MUL,
    POW,
    PRELU,
    REDUCE,
    RELU,
    RESHAPE,
    RESIZE,
    SIGMOID,
    SLICE,
    SOFTMAX,
    SOFTPLUS,
    SPLIT,
    SUB,
    TOPK,
    TRANSPOSE,
    TOTAL_NUM
};

inline std::string convert_optype_to_string(const OpType &type) {
    std::vector<std::string> names = {
        "Unknown",           // UNKNOWN = 0
        "Add",               // ADD = 1
        "Atan",              // ATAN = 2
        "AveragePool",       // AVERAGEPOOL = 3
        "BatchNorm",         // BATCHNORM = 4
        "Col2Im",            // COL2IM = 5
        "Concat",            // CONCAT = 6
        "Conv2d",            // CONV2D = 7
        "Div",               // DIV = 8
        "Erf",               // ERF = 9
        "Floor",             // FLOOR = 10
        "Gemm",              // GEMM = 11
        "GlobalAveragePool", // GLOBALAVERAGEPOOL = 12
        "GridSample",        // GRIDSAMPLE = 13
        "LayerNorm",         // LAYERNORM = 14
        "MatMul",            // MATMUL = 15
        "MaxPool2d",         // MAXPOOL2D = 16
        "Mul",               // MUL = 17
        "Pow",               // POW = 18
        "PRelu",             // PRELU = 19
        "Reduce",            // REDUCE = 20
        "Relu",              // RELU = 21
        "Reshape",           // RESHAPE = 22
        "Resize",            // RESIZE = 23
        "Sigmoid",           // SIGMOID = 24
        "Slice",             // SLICE = 25
        "Softmax",           // SOFTMAX = 26
        "Softplus",          // SOFTPLUS = 27
        "Split",             // SPLIT = 28
        "Sub",               // SUB = 29
        "TopK",              // TOPK = 30
        "Transpose",         // TRANSPOSE = 31
        ""};                 // TOTAL_NUM = 32 (should not be accessed)
    if (type >= OpType::TOTAL_NUM)
        return names[0];
    return names[static_cast<int>(type)];
}

inline OpType convert_opstring_to_enum(const std::string &name) {
    // Alphabetical order for better readability and maintenance
    if (name == "Add")
        return vkop::ops::OpType::ADD;
    if (name == "Atan")
        return vkop::ops::OpType::ATAN;
    if (name == "AveragePool")
        return vkop::ops::OpType::AVERAGEPOOL;
    if (name == "BatchNormalization" || name == "BatchNorm")
        return vkop::ops::OpType::BATCHNORM;
    if (name == "Col2Im")
        return vkop::ops::OpType::COL2IM;
    if (name == "Concat")
        return vkop::ops::OpType::CONCAT;
    if (name == "Conv2d" || name == "Conv")
        return vkop::ops::OpType::CONV2D;
    if (name == "Div")
        return vkop::ops::OpType::DIV;
    if (name == "Erf")
        return vkop::ops::OpType::ERF;
    if (name == "Floor")
        return vkop::ops::OpType::FLOOR;
    if (name == "Gemm")
        return vkop::ops::OpType::GEMM;
    if (name == "GlobalAveragePool")
        return vkop::ops::OpType::GLOBALAVERAGEPOOL;
    if (name == "GridSample")
        return vkop::ops::OpType::GRIDSAMPLE;
    if (name == "LayerNormalization" || name == "LayerNorm")
        return vkop::ops::OpType::LAYERNORM;
    if (name == "MatMul")
        return vkop::ops::OpType::MATMUL;
    if (name == "MaxPool2d" || name == "MaxPool")
        return vkop::ops::OpType::MAXPOOL2D;
    if (name == "Mul")
        return vkop::ops::OpType::MUL;
    if (name == "Pow")
        return vkop::ops::OpType::POW;
    if (name == "PRelu")
        return vkop::ops::OpType::PRELU;
    if (name == "Reduce")
        return vkop::ops::OpType::REDUCE;
    if (name == "Relu")
        return vkop::ops::OpType::RELU;
    if (name == "Reshape")
        return vkop::ops::OpType::RESHAPE;
    if (name == "Resize")
        return vkop::ops::OpType::RESIZE;
    if (name == "Sigmoid")
        return vkop::ops::OpType::SIGMOID;
    if (name == "Slice")
        return vkop::ops::OpType::SLICE;
    if (name == "Softmax")
        return vkop::ops::OpType::SOFTMAX;
    if (name == "Softplus")
        return vkop::ops::OpType::SOFTPLUS;
    if (name == "Split")
        return vkop::ops::OpType::SPLIT;
    if (name == "Sub")
        return vkop::ops::OpType::SUB;
    if (name == "TopK")
        return vkop::ops::OpType::TOPK;
    if (name == "Transpose")
        return vkop::ops::OpType::TRANSPOSE;
    printf("Unknown op type: %s\n", name.c_str());
    return vkop::ops::OpType::UNKNOWN;
}

} // namespace ops
} // namespace vkop

#endif /* OPS_OPS_HPP_ */