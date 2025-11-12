// Copyright 2025 @junka
#include "Col2Im.hpp"
#include "Conv2d.hpp"
#include "GridSample.hpp"
#include "Maxpool2d.hpp"
#include "Resize.hpp"
#include "Softmax.hpp"

#include "OperatorFactory.hpp"
#include "Ops.hpp"
#include "include/logger.hpp"

namespace vkop {
namespace ops {

namespace {
REGISTER_OPERATOR(OpType::MAXPOOL2D, Maxpool2d);
REGISTER_OPERATOR(OpType::COL2IM, Col2im);
REGISTER_OPERATOR(OpType::CONV2D, Conv2d);
REGISTER_OPERATOR(OpType::GRIDSAMPLE, GridSample);
REGISTER_OPERATOR(OpType::RESIZE, Resize);
REGISTER_OPERATOR(OpType::SOFTMAX, Softmax);
} // namespace

} // namespace ops
} // namespace vkop
