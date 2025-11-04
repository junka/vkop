#include "Atan.hpp"
#include "BatchNorm2d.hpp"
#include "Erf.hpp"
#include "Floor.hpp"
#include "LayerNorm.hpp"
#include "Relu.hpp"

#include "OperatorFactory.hpp"

namespace vkop {
namespace ops {

REGISTER_OPERATOR(OpType::ATAN, Atan);
REGISTER_OPERATOR(OpType::ERF, Erf);
REGISTER_OPERATOR(OpType::RELU, Relu);
REGISTER_OPERATOR(OpType::BATCHNORM, BatchNorm2d);
REGISTER_OPERATOR(OpType::LAYERNORM, LayerNorm);
REGISTER_OPERATOR(OpType::FLOOR, Floor);

} // namespace ops
} // namespace vkop