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
    NMS,
    GATHER,
    RANGE,
    EXPAND,
    SQRT,
    SIN,
    COS,
    NEG,
    WHERE,
    TANH,
    EMBEDDING_FORWARD,
    EQUAL,
    NONZERO,
    PAD,
    SCATTER_ELEMENTS,
    SHAPE,
    CAST,
    CLIP,
    SCATTER_ND,
    SQUEEZE,
    UNSQUEEZE,
    ROTARY_EMBEDDING,
    FUSED_ELEMWISE,
    RMSNORM,
    REDUCEMEAN,
    MIN,
    MAX,
    MOD,
    QUANTIZE_LINEAR,
    DEQUANTIZE_LINEAR,
    ATTENTION,
    CUMSUM,
    CONV1D,
    LINEAR_ATTENTION,
    TOTAL_NUM
};

inline std::string convert_optype_to_string(const OpType &type) {
    std::vector<std::string> names = {
        "Unknown",           // UNKNOWN = 0
        "Add",               // ADD = 1
        "Atan",              // ATAN = 2
        "AveragePool",       // AVERAGEPOOL = 3
        "BatchNorm",         // BATCHNORM = 4
        "Concat",            // CONCAT = 5
        "Conv2d",            // CONV2D = 6
        "Div",               // DIV = 7
        "Erf",               // ERF = 8
        "Floor",             // FLOOR = 9
        "Gemm",              // GEMM = 10
        "GlobalAveragePool", // GLOBALAVERAGEPOOL = 11
        "GridSample",        // GRIDSAMPLE = 12
        "LayerNorm",         // LAYERNORM = 13
        "MatMul",            // MATMUL = 14
        "MaxPool2d",         // MAXPOOL2D = 15
        "Mul",               // MUL = 16
        "Pow",               // POW = 17
        "PRelu",             // PRELU = 18
        "Reduce",            // REDUCE = 19
        "Relu",              // RELU = 20
        "Reshape",           // RESHAPE = 21
        "Resize",            // RESIZE = 22
        "Sigmoid",           // SIGMOID = 23
        "Slice",             // SLICE = 24
        "Softmax",           // SOFTMAX = 25
        "Softplus",          // SOFTPLUS = 26
        "Split",             // SPLIT = 27
        "Sub",               // SUB = 28
        "TopK",              // TOPK = 29
        "Transpose",         // TRANSPOSE = 30
        "Nms",               // NMS = 31
        "Gather",            // GATHER = 32
        "Range",             // RANGE = 33
        "Expand",            // EXPAND = 34
        "Sqrt",              // SQRT = 35
        "Sin",               // SIN = 36
        "Cos",               // COS = 37
        "Neg",               // NEG = 38
        "Where",             // WHERE = 39
        "Tanh",              // TANH = 40
        "EmbeddingForward",  // EMBEDDING_FORWARD = 41
        "Equal",             // EQUAL = 42
        "NonZero",           // NONZERO = 43
        "Pad",               // PAD = 44
        "ScatterElements",   // SCATTER_ELEMENTS = 45
        "Shape",             // SHAPE = 46
        "Cast",              // CAST = 47
        "Clip",              // CLIP = 48
        "ScatterND",         // SCATTER_ND = 49
        "Squeeze",           // SQUEEZE = 50
        "Unsqueeze",         // UNSQUEEZE = 51
        "RotaryEmbedding",   // ROTARY_EMBEDDING = 52
        "FusedElemwise",     // FUSED_ELEMWISE = 53
        "RMSNorm",           // RMSNORM = 54
        "ReduceMean",        // REDUCEMEAN = 55
        "Min",               // MIN = 56
        "Max",               // MAX = 57
        "Mod",               // MOD = 58
        "QuantizeLinear",    // QUANTIZE_LINEAR = 59
        "DequantizeLinear",  // DEQUANTIZE_LINEAR = 60
        "Attention",         // ATTENTION = 61
        "CumSum",            // CUMSUM = 62
        "Conv1d",            // CONV1D = 63
        "LinearAttention",   // LINEAR_ATTENTION = 64
        ""};                 // TOTAL_NUM (should not be accessed)
    if (type >= OpType::TOTAL_NUM)
        return names[0];
    return names[static_cast<int>(type)];
}

inline OpType convert_opstring_to_enum(const std::string &name) {    // Alphabetical order for better readability and maintenance
    if (name == "Add")
        return vkop::ops::OpType::ADD;
    if (name == "Atan")
        return vkop::ops::OpType::ATAN;
    if (name == "Attention")
        return vkop::ops::OpType::ATTENTION;
    if (name == "AveragePool")
        return vkop::ops::OpType::AVERAGEPOOL;
    if (name == "BatchNormalization" || name == "BatchNorm")
        return vkop::ops::OpType::BATCHNORM;
    if (name == "Cast")
        return vkop::ops::OpType::CAST;
    if (name == "Clip")
        return vkop::ops::OpType::CLIP;
    if (name == "Concat")
        return vkop::ops::OpType::CONCAT;
    if (name == "Conv1d" || name == "Conv1D")
        return vkop::ops::OpType::CONV1D;
    if (name == "Conv2d" || name == "Conv")
        return vkop::ops::OpType::CONV2D;
    if (name == "Cos")
        return vkop::ops::OpType::COS;
    if (name == "CumSum" || name == "Cumsum")
        return vkop::ops::OpType::CUMSUM;
    if (name == "DequantizeLinear")
        return vkop::ops::OpType::DEQUANTIZE_LINEAR;
    if (name == "Div")
        return vkop::ops::OpType::DIV;
    if (name == "EmbeddingForward")
        return vkop::ops::OpType::EMBEDDING_FORWARD;
    if (name == "Equal")
        return vkop::ops::OpType::EQUAL;
    if (name == "Erf")
        return vkop::ops::OpType::ERF;
    if (name == "Expand")
        return vkop::ops::OpType::EXPAND;
    if (name == "Floor")
        return vkop::ops::OpType::FLOOR;
    if (name == "FusedElemwise")
        return vkop::ops::OpType::FUSED_ELEMWISE;
    if (name == "Gather")
        return vkop::ops::OpType::GATHER;
    if (name == "Gemm")
        return vkop::ops::OpType::GEMM;
    if (name == "GlobalAveragePool")
        return vkop::ops::OpType::GLOBALAVERAGEPOOL;
    if (name == "GridSample")
        return vkop::ops::OpType::GRIDSAMPLE;
    if (name == "LayerNormalization" || name == "LayerNorm")
        return vkop::ops::OpType::LAYERNORM;
    if (name == "LinearAttention" || name == "GatedDeltaNet")
        return vkop::ops::OpType::LINEAR_ATTENTION;
    if (name == "MatMul")
        return vkop::ops::OpType::MATMUL;
    if (name == "Max")
        return vkop::ops::OpType::MAX;
    if (name == "MaxPool2d" || name == "MaxPool")
        return vkop::ops::OpType::MAXPOOL2D;
    if (name == "Min")
        return vkop::ops::OpType::MIN;
    if (name == "Mod")
        return vkop::ops::OpType::MOD;
    if (name == "Mul")
        return vkop::ops::OpType::MUL;
    if (name == "Neg")
        return vkop::ops::OpType::NEG;
    if (name == "Nms")
        return vkop::ops::OpType::NMS;
    if (name == "NonZero")
        return vkop::ops::OpType::NONZERO;
    if (name == "PRelu")
        return vkop::ops::OpType::PRELU;
    if (name == "Pad")
        return vkop::ops::OpType::PAD;
    if (name == "Pow")
        return vkop::ops::OpType::POW;
    if (name == "QuantizeLinear")
        return vkop::ops::OpType::QUANTIZE_LINEAR;
    if (name == "RMSNorm")
        return vkop::ops::OpType::RMSNORM;
    if (name == "Range")
        return vkop::ops::OpType::RANGE;
    if (name == "Reduce")
        return vkop::ops::OpType::REDUCE;
    if (name == "ReduceMean")
        return vkop::ops::OpType::REDUCEMEAN;
    if (name == "Relu")
        return vkop::ops::OpType::RELU;
    if (name == "Reshape")
        return vkop::ops::OpType::RESHAPE;
    if (name == "Resize")
        return vkop::ops::OpType::RESIZE;
    if (name == "RotaryEmbedding")
        return vkop::ops::OpType::ROTARY_EMBEDDING;
    if (name == "ScatterElements")
        return vkop::ops::OpType::SCATTER_ELEMENTS;
    if (name == "ScatterND")
        return vkop::ops::OpType::SCATTER_ND;
    if (name == "Shape")
        return vkop::ops::OpType::SHAPE;
    if (name == "Sigmoid")
        return vkop::ops::OpType::SIGMOID;
    if (name == "Sin")
        return vkop::ops::OpType::SIN;
    if (name == "Slice")
        return vkop::ops::OpType::SLICE;
    if (name == "Softmax")
        return vkop::ops::OpType::SOFTMAX;
    if (name == "Softplus")
        return vkop::ops::OpType::SOFTPLUS;
    if (name == "Split")
        return vkop::ops::OpType::SPLIT;
    if (name == "Sqrt")
        return vkop::ops::OpType::SQRT;
    if (name == "Squeeze")
        return vkop::ops::OpType::SQUEEZE;
    if (name == "Sub")
        return vkop::ops::OpType::SUB;
    if (name == "Tanh")
        return vkop::ops::OpType::TANH;
    if (name == "TopK")
        return vkop::ops::OpType::TOPK;
    if (name == "Transpose")
        return vkop::ops::OpType::TRANSPOSE;
    if (name == "Unsqueeze")
        return vkop::ops::OpType::UNSQUEEZE;
    if (name == "Where")
        return vkop::ops::OpType::WHERE;
    printf("Unknown op type: %s\n", name.c_str());
    return vkop::ops::OpType::UNKNOWN;
}

} // namespace ops
} // namespace vkop

#endif /* OPS_OPS_HPP_ */