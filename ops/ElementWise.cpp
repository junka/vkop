#include "Add.hpp"
#include "Div.hpp"
#include "Mul.hpp"
#include "Sub.hpp"

#include "OperatorFactory.hpp"

namespace vkop {
namespace ops {

REGISTER_OPERATOR("Add", Add);
REGISTER_OPERATOR("Sub", Sub);
REGISTER_OPERATOR("Mul", Mul);
REGISTER_OPERATOR("Div", Div);

} // namespace ops
} // namespace vkop