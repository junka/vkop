#include "Add.hpp"
#include "Div.hpp"
#include "Mul.hpp"
#include "Sub.hpp"

#include "OperatorFactory.hpp"

namespace vkop {
namespace ops {

REGISTER_OPERATOR(Add);
REGISTER_OPERATOR(Sub);
REGISTER_OPERATOR(Mul);
REGISTER_OPERATOR(Div);

} // namespace ops
} // namespace vkop