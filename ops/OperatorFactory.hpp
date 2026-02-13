// Copyright 2025 @junka
#ifndef OPS_OPERATOR_FACTORY_HPP_
#define OPS_OPERATOR_FACTORY_HPP_

#include "Atan.hpp"
#include "BatchNorm.hpp"
#include "Erf.hpp"
#include "Floor.hpp"
#include "LayerNorm.hpp"
#include "Relu.hpp"
#include "Sigmoid.hpp"
#include "Softplus.hpp"
#include "ops/Operator.hpp"

#include "Add.hpp"
#include "Div.hpp"
#include "Mul.hpp"
#include "PRelu.hpp"
#include "Pow.hpp"
#include "Sub.hpp"

#include "Col2Im.hpp"
#include "Conv2d.hpp"
#include "Gemm.hpp"
#include "GridSample.hpp"
#include "Matmul.hpp"
#include "Maxpool2d.hpp"
#include "Reduce.hpp"
#include "Resize.hpp"
#include "Softmax.hpp"

#include "AveragePool.hpp"
#include "Concat.hpp"
#include "GlobalAveragePool.hpp"
#include "Reshape.hpp"
#include "Slice.hpp"
#include "Split.hpp"
#include "Topk.hpp"
#include "Transpose.hpp"

namespace vkop {

namespace ops {

static inline std::unique_ptr<Operator>
create_from_type(OpType type, bool use_ssbo = false, int fp16 = 0) {
    switch (type) {
    case OpType::ADD:
        return std::make_unique<Add>();
    case OpType::ATAN:
        return std::make_unique<Atan>();
    case OpType::AVERAGEPOOL:
        return std::make_unique<AveragePool>();
    case OpType::BATCHNORM:
        return std::make_unique<BatchNorm>();
    case OpType::COL2IM:
        return std::make_unique<Col2Im>();
    case OpType::CONCAT:
        return std::make_unique<Concat>();
    case OpType::CONV2D:
        return std::make_unique<Conv2d>();
    case OpType::DIV:
        return std::make_unique<Div>();
    case OpType::ERF:
        return std::make_unique<Erf>();
    case OpType::FLOOR:
        return std::make_unique<Floor>();
    case OpType::GEMM:
        return std::make_unique<Gemm>();
    case OpType::GLOBALAVERAGEPOOL:
        return std::make_unique<GlobalAveragePool>();
    case OpType::GRIDSAMPLE:
        return std::make_unique<GridSample>();
    case OpType::LAYERNORM:
        return std::make_unique<LayerNorm>();
    case OpType::MATMUL:
        return std::make_unique<MatMul>();
    case OpType::MAXPOOL2D:
        return std::make_unique<Maxpool2d>();
    case OpType::MUL:
        return std::make_unique<Mul>();
    case OpType::POW:
        return std::make_unique<Pow>();
    case OpType::PRELU:
        return std::make_unique<PRelu>();
    case OpType::REDUCE:
        return std::make_unique<Reduce>();
    case OpType::RELU:
        return std::make_unique<Relu>();
    case OpType::RESHAPE:
        return std::make_unique<Reshape>();
    case OpType::RESIZE:
        return std::make_unique<Resize>();
    case OpType::SIGMOID:
        return std::make_unique<Sigmoid>();
    case OpType::SLICE:
        return std::make_unique<Slice>();
    case OpType::SOFTPLUS:
        return std::make_unique<Softplus>();
    case OpType::SPLIT:
        return std::make_unique<Split>();
    case OpType::SUB:
        return std::make_unique<Sub>();
    case OpType::TOPK:
        return std::make_unique<Topk>(fp16);
    case OpType::TRANSPOSE:
        return std::make_unique<Transpose>();
    case OpType::SOFTMAX:
        return std::make_unique<Softmax>(use_ssbo);

    default:
        return nullptr;
    }
}

} // namespace ops
} // namespace vkop

#endif /* OPS_OPERATOR_FACTORY_HPP_ */
