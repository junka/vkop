// Copyright 2025 @junka
#ifndef OPS_OPS_HPP_
#define OPS_OPS_HPP_

#include <string>

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
};

inline std::string convert_openum_to_string(const OpType &type) {
    switch (type) {
    case OpType::ADD:
        return "Add";
    case OpType::SUB:
        return "Sub";
    case OpType::MUL:
        return "Mul";
    case OpType::DIV:
        return "Div";
    case OpType::ATAN:
        return "Atan";
    case OpType::ERF:
        return "Erf";
    case OpType::POW:
        return "Pow";
    case OpType::BATCHNORM:
        return "BatchNorm";
    case OpType::LAYERNORM:
        return "LayerNorm";
    case OpType::RELU:
        return "Relu";
    case OpType::SOFTMAX:
        return "Softmax";
    case OpType::TANH:
        return "Tanh";
    case OpType::MATMUL:
        return "MatMul";
    case OpType::CONV2D:
        return "Conv2d";
    case OpType::MAXPOOL2D:
        return "MaxPool2d";
    case OpType::AVGPOOL2D:
        return "AvgPool2d";
    case OpType::UPSAMPLE2D:
        return "Upsample2d";
    case OpType::GRIDSAMPLE:
        return "GridSample";
    case OpType::CONSTANT:
        return "Constant";
    case OpType::FLOOR:
        return "Floor";
    case OpType::FLATTEN:
        return "Flatten";
    case OpType::RESIZE:
        return "Resize";
    case OpType::CONCAT:
        return "Concat";
    case OpType::SLICE:
        return "Slice";
    case OpType::UNSQUEEZE:
        return "Unsqueeze";
    case OpType::SQUEEZE:
        return "Squeeze";
    case OpType::COL2IM:
        return "Col2Im";
    case OpType::IM2COL:
        return "Im2Col";
    default:
        return "Unknown";
    }
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
    printf("Unknown op type: %s\n", name.c_str());
    return vkop::ops::OpType::UNKNOWN;
}

} // namespace ops
} // namespace vkop

#endif /* OPS_OPS_HPP_ */