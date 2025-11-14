#include "Atan.hpp"
#include "BatchNorm2d.hpp"
#include "Erf.hpp"
#include "Floor.hpp"
#include "LayerNorm.hpp"
#include "Relu.hpp"
#include "Sigmoid.hpp"
#include "Softplus.hpp"

#include "OperatorFactory.hpp"
#include "ops/Ops.hpp"

namespace vkop {
namespace ops {

REGISTER_OPERATOR(OpType::ATAN, Atan);
REGISTER_OPERATOR(OpType::ERF, Erf);
REGISTER_OPERATOR(OpType::RELU, Relu);
REGISTER_OPERATOR(OpType::BATCHNORM, BatchNorm2d);
REGISTER_OPERATOR(OpType::LAYERNORM, LayerNorm);
REGISTER_OPERATOR(OpType::FLOOR, Floor);
REGISTER_OPERATOR(OpType::SOFTPLUS, Softplus);
REGISTER_OPERATOR(OpType::SIGMOID, Sigmoid);

} // namespace ops
} // namespace vkop