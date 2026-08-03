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
#include "Nms.hpp"
#include "Reshape.hpp"
#include "Slice.hpp"
#include "Split.hpp"
#include "Topk.hpp"
#include "Transpose.hpp"

#include "Expand.hpp"
#include "Gather.hpp"
#include "Range.hpp"

#include "Cos.hpp"
#include "EmbeddingForward.hpp"
#include "Neg.hpp"
#include "Sin.hpp"
#include "Sqrt.hpp"
#include "Tanh.hpp"
#include "Where.hpp"

namespace vkop {

namespace ops {

static inline std::unique_ptr<Operator>
create_from_type(OpType type, int fp16 = 0, int use_tensorcore = 0,
                 bool backend_buffer = false) {
    // The image/buffer choice now lives inside each op's PIMPL façade ctor
    // (it picks the BufferImpl when backend_buffer is set and a buffer port
    // exists, else the ImageImpl). This factory just constructs the façade
    // with (fp16, backend_buffer) — or (use_tensorcore, backend_buffer) for
    // MatMul.
    switch (type) {
    case OpType::ADD:
        return std::make_unique<Add>(fp16, backend_buffer);
    case OpType::ATAN:
        return std::make_unique<Atan>(fp16, backend_buffer);
    case OpType::AVERAGEPOOL:
        return std::make_unique<AveragePool>(fp16, backend_buffer);
    case OpType::BATCHNORM:
        return std::make_unique<BatchNorm>(fp16, backend_buffer);
    case OpType::COL2IM:
        return std::make_unique<Col2Im>(fp16, backend_buffer);
    case OpType::CONCAT:
        return std::make_unique<Concat>(fp16, backend_buffer);
    case OpType::CONV2D:
        return std::make_unique<Conv2d>(fp16, backend_buffer);
    case OpType::DIV:
        return std::make_unique<Div>(fp16, backend_buffer);
    case OpType::EMBEDDING_FORWARD:
        return std::make_unique<EmbeddingForward>(fp16, backend_buffer);
    case OpType::ERF:
        return std::make_unique<Erf>(fp16, backend_buffer);
    case OpType::FLOOR:
        return std::make_unique<Floor>(fp16, backend_buffer);
    case OpType::GEMM:
        return std::make_unique<Gemm>(fp16, backend_buffer);
    case OpType::GLOBALAVERAGEPOOL:
        return std::make_unique<GlobalAveragePool>(fp16, backend_buffer);
    case OpType::GRIDSAMPLE:
        return std::make_unique<GridSample>(fp16, backend_buffer);
    case OpType::LAYERNORM:
        return std::make_unique<LayerNorm>(fp16, backend_buffer);
    case OpType::MATMUL:
        return std::make_unique<MatMul>(use_tensorcore, fp16, backend_buffer);
    case OpType::MAXPOOL2D:
        return std::make_unique<Maxpool2d>(fp16, backend_buffer);
    case OpType::MUL:
        return std::make_unique<Mul>(fp16, backend_buffer);
    case OpType::POW:
        return std::make_unique<Pow>(fp16, backend_buffer);
    case OpType::PRELU:
        return std::make_unique<PRelu>(fp16, backend_buffer);
    case OpType::REDUCE:
        return std::make_unique<Reduce>(fp16, backend_buffer);
    case OpType::RELU:
        return std::make_unique<Relu>(fp16, backend_buffer);
    case OpType::RESHAPE:
        return std::make_unique<Reshape>(fp16, backend_buffer);
    case OpType::RESIZE:
        return std::make_unique<Resize>(fp16, backend_buffer);
    case OpType::SIGMOID:
        return std::make_unique<Sigmoid>(fp16, backend_buffer);
    case OpType::SLICE:
        return std::make_unique<Slice>(fp16, backend_buffer);
    case OpType::SOFTPLUS:
        return std::make_unique<Softplus>(fp16, backend_buffer);
    case OpType::SPLIT:
        return std::make_unique<Split>(fp16, backend_buffer);
    case OpType::SUB:
        return std::make_unique<Sub>(fp16, backend_buffer);
    case OpType::TOPK:
        return std::make_unique<Topk>(fp16, backend_buffer);
    case OpType::TRANSPOSE:
        return std::make_unique<Transpose>(fp16, backend_buffer);
    case OpType::SOFTMAX:
        return std::make_unique<Softmax>(fp16, backend_buffer);
    case OpType::NMS:
        return std::make_unique<Nms>(fp16, backend_buffer);
    case OpType::GATHER:
        return std::make_unique<Gather>(fp16, backend_buffer);
    case OpType::RANGE:
        return std::make_unique<Range>(fp16, backend_buffer);
    case OpType::EXPAND:
        return std::make_unique<Expand>(fp16, backend_buffer);
    case OpType::SIN:
        return std::make_unique<Sin>(fp16, backend_buffer);
    case OpType::COS:
        return std::make_unique<Cos>(fp16, backend_buffer);
    case OpType::NEG:
        return std::make_unique<Neg>(fp16, backend_buffer);
    case OpType::SQRT:
        return std::make_unique<Sqrt>(fp16, backend_buffer);
    case OpType::WHERE:
        return std::make_unique<Where>(fp16, backend_buffer);
    case OpType::TANH:
        return std::make_unique<Tanh>(fp16, backend_buffer);
    default:
        return nullptr;
    }
}

} // namespace ops
} // namespace vkop

#endif /* OPS_OPERATOR_FACTORY_HPP_ */
