// Copyright 2025 @junka
#ifndef OPS_OPS_HPP_
#define OPS_OPS_HPP_

#include <string>

namespace vkop {
namespace ops {

enum class OpType {
    UNKNOWN,
    ADD,
    SUB,
    MUL,
    DIV,
    ATAN,
    ERF,
    POW,
    BATCHNORM,
    RELU,
    SOFTMAX,
    TANH,
    MATMUL,
    CONV2D,
    MAXPOOL2D,
    AVGPOOL2D,
    UPSAMPLE2D,
    GRIDSAMPLE,
    CONSTANT,
    FLOOR,
};

} // namespace ops
} // namespace vkop

#endif /* OPS_OPS_HPP_ */