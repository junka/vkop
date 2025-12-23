#include "Atan.hpp"
#include "BatchNorm.hpp"
#include "Erf.hpp"
#include "Floor.hpp"
#include "LayerNorm.hpp"
#include "Relu.hpp"
#include "Sigmoid.hpp"
#include "Softplus.hpp"

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

#include "OperatorFactory.hpp"
#include "Ops.hpp"

namespace vkop {
namespace ops {

REGISTER_OPERATOR(OpType::ATAN, Atan);
REGISTER_OPERATOR(OpType::ERF, Erf);
REGISTER_OPERATOR(OpType::RELU, Relu);
REGISTER_OPERATOR(OpType::BATCHNORM, BatchNorm);
REGISTER_OPERATOR(OpType::LAYERNORM, LayerNorm);
REGISTER_OPERATOR(OpType::FLOOR, Floor);
REGISTER_OPERATOR(OpType::SOFTPLUS, Softplus);
REGISTER_OPERATOR(OpType::SIGMOID, Sigmoid);

REGISTER_OPERATOR(OpType::ADD, Add);
REGISTER_OPERATOR(OpType::SUB, Sub);
REGISTER_OPERATOR(OpType::MUL, Mul);
REGISTER_OPERATOR(OpType::DIV, Div);
REGISTER_OPERATOR(OpType::POW, Pow);
REGISTER_OPERATOR(OpType::PRELU, PRelu);

REGISTER_OPERATOR(OpType::MAXPOOL2D, Maxpool2d);
REGISTER_OPERATOR(OpType::COL2IM, Col2im);
REGISTER_OPERATOR(OpType::CONV2D, Conv2d);
REGISTER_OPERATOR(OpType::GRIDSAMPLE, GridSample);
REGISTER_OPERATOR(OpType::RESIZE, Resize);
REGISTER_OPERATOR(OpType::SOFTMAX, Softmax);
REGISTER_OPERATOR(OpType::MATMUL, MatMul);
REGISTER_OPERATOR(OpType::GEMM, Gemm);
REGISTER_OPERATOR(OpType::REDUCE, Reduce);

REGISTER_OPERATOR(OpType::RESHAPE, Reshape);
REGISTER_OPERATOR(OpType::SLICE, Slice);
REGISTER_OPERATOR(OpType::TRANSPOSE, Transpose);
REGISTER_OPERATOR(OpType::CONCAT, Concat);
REGISTER_OPERATOR(OpType::SPLIT, Split);

REGISTER_OPERATOR(OpType::GLOBALAVERAGEPOOL, GlobalAveragePool);
REGISTER_OPERATOR(OpType::AVERAGEPOOL, AveragePool);
REGISTER_OPERATOR(OpType::TOPK, Topk);

} // namespace ops
} // namespace vkop