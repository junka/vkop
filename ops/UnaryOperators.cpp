#include "Atan.hpp"
#include "Erf.hpp"
#include "Relu.hpp"
#include "Softmax.hpp"

#include "OperatorFactory.hpp"

namespace vkop {
namespace ops {

REGISTER_OPERATOR(Atan);
REGISTER_OPERATOR(Erf);
REGISTER_OPERATOR(Relu);
REGISTER_OPERATOR(Softmax);

} // namespace ops
} // namespace vkop