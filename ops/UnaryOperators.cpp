#include "Atan.hpp"
#include "Erf.hpp"
#include "Relu.hpp"

#include "OperatorFactory.hpp"

namespace vkop {
namespace ops {

REGISTER_OPERATOR(Atan);
REGISTER_OPERATOR(Erf);
REGISTER_OPERATOR(Relu);

} // namespace ops
} // namespace vkop