// Copyright 2025 @junka
#ifndef OPS_SIGMOID_HPP_
#define OPS_SIGMOID_HPP_

#include "UnaryFactory.hpp"

extern "C" {
extern unsigned char sigmoid_spv[];
extern unsigned int sigmoid_spv_len;
};

namespace vkop {
namespace ops {

class Sigmoid : public UnaryFactory {
  public:
    Sigmoid() : UnaryFactory(OpType::SIGMOID, sigmoid_spv, sigmoid_spv_len) {}
};

} // namespace ops
} // namespace vkop
#endif // OPS_SIGMOID_HPP_
