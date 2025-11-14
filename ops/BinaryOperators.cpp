#include "Add.hpp"
#include "Div.hpp"
#include "Mul.hpp"
#include "PRelu.hpp"
#include "Pow.hpp"
#include "Sub.hpp"

#include "OperatorFactory.hpp"
#include "Ops.hpp"

namespace vkop {
namespace ops {

REGISTER_OPERATOR(OpType::ADD, Add);
REGISTER_OPERATOR(OpType::SUB, Sub);
REGISTER_OPERATOR(OpType::MUL, Mul);
REGISTER_OPERATOR(OpType::DIV, Div);
REGISTER_OPERATOR(OpType::POW, Pow);
REGISTER_OPERATOR(OpType::PRELU, PRelu);

} // namespace ops
} // namespace vkop