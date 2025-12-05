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
    SUB,
    MUL,
    DIV,
    ATAN,
    ERF,
    POW,
    BATCHNORM,
    LAYERNORM,
    RELU,
    SOFTMAX,
    TANH,
    MATMUL,
    CONV2D,
    MAXPOOL2D,
    AVGPOOL2D,
    UPSAMPLE2D,
    GRIDSAMPLE,
    CONSTANT,
    FLOOR,
    FLATTEN,
    RESIZE,
    CONCAT,
    SLICE,
    UNSQUEEZE,
    SQUEEZE,
    COL2IM,
    IM2COL,
    PRELU,
    SIGMOID,
    SOFTPLUS,
    GEMM,
    REDUCE,
    SPLIT,
    RESHAPE,
    TOTAL_NUM
};

inline std::string convert_openum_to_string(const OpType &type) {
    std::vector<std::string> names = {
        "Unknown",   "Add",       "Sub",        "Mul",        "Div",
        "Atan",      "Erf",       "Pow",        "BatchNorm",  "LayerNorm",
        "Relu",      "Softmax",   "Tanh",       "MatMul",     "Conv2d",
        "MaxPool2d", "AvgPool2d", "Upsample2d", "GridSample", "Constant",
        "Floor",     "Flatten",   "Resize",     "Concat",     "Slice",
        "Unsqueeze", "Squeeze",   "Col2Im",     "Im2Col",     "PRelu",
        "Sigmoid",   "Softplus",  "Gemm",       "Reduce",     "Split",
        "Reshape"};
    if (type >= OpType::TOTAL_NUM)
        return names[0];
    return names[static_cast<int>(type)];
}

inline OpType convert_opstring_to_enum(const std::string &name) {
    if (name == "Add")
        return vkop::ops::OpType::ADD;
    if (name == "Sub")
        return vkop::ops::OpType::SUB;
    if (name == "Mul")
        return vkop::ops::OpType::MUL;
    if (name == "Div")
        return vkop::ops::OpType::DIV;
    if (name == "Atan")
        return vkop::ops::OpType::ATAN;
    if (name == "Erf")
        return vkop::ops::OpType::ERF;
    if (name == "Pow")
        return vkop::ops::OpType::POW;
    if (name == "BatchNormalization" || name == "BatchNorm")
        return vkop::ops::OpType::BATCHNORM;
    if (name == "LayerNormalization" || name == "LayerNorm")
        return vkop::ops::OpType::LAYERNORM;
    if (name == "Relu")
        return vkop::ops::OpType::RELU;
    if (name == "Softmax")
        return vkop::ops::OpType::SOFTMAX;
    if (name == "Tanh")
        return vkop::ops::OpType::TANH;
    if (name == "MatMul")
        return vkop::ops::OpType::MATMUL;
    if (name == "Conv2d" || name == "Conv")
        return vkop::ops::OpType::CONV2D;
    if (name == "MaxPool2d" || name == "MaxPool")
        return vkop::ops::OpType::MAXPOOL2D;
    if (name == "AvgPool2d")
        return vkop::ops::OpType::AVGPOOL2D;
    if (name == "Upsample2d")
        return vkop::ops::OpType::UPSAMPLE2D;
    if (name == "GridSample")
        return vkop::ops::OpType::GRIDSAMPLE;
    if (name == "Constant")
        return vkop::ops::OpType::CONSTANT;
    if (name == "Floor")
        return vkop::ops::OpType::FLOOR;
    if (name == "Resize")
        return vkop::ops::OpType::RESIZE;
    if (name == "PRelu")
        return vkop::ops::OpType::PRELU;
    if (name == "Flatten")
        return vkop::ops::OpType::FLATTEN;
    if (name == "Concat")
        return vkop::ops::OpType::CONCAT;
    if (name == "Slice")
        return vkop::ops::OpType::SLICE;
    if (name == "Unsqueeze")
        return vkop::ops::OpType::UNSQUEEZE;
    if (name == "Squeeze")
        return vkop::ops::OpType::SQUEEZE;
    if (name == "Col2Im")
        return vkop::ops::OpType::COL2IM;
    if (name == "Im2Col")
        return vkop::ops::OpType::IM2COL;
    if (name == "Sigmoid")
        return vkop::ops::OpType::SIGMOID;
    if (name == "Softplus")
        return vkop::ops::OpType::SOFTPLUS;
    if (name == "Gemm")
        return vkop::ops::OpType::GEMM;
    if (name == "Reduce")
        return vkop::ops::OpType::REDUCE;
    if (name == "Split")
        return vkop::ops::OpType::SPLIT;
    if (name == "Reshape")
        return vkop::ops::OpType::RESHAPE;
    printf("Unknown op type: %s\n", name.c_str());
    return vkop::ops::OpType::UNKNOWN;
}

} // namespace ops
} // namespace vkop

#endif /* OPS_OPS_HPP_ */