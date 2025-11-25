// Copyright 2025 @junka
#ifndef OPS_PRELU_HPP_
#define OPS_PRELU_HPP_

#include "BinaryFactory.hpp"
#include "ops/Ops.hpp"

extern unsigned char prelu_spv[];
extern unsigned int prelu_spv_len;

namespace vkop {
namespace ops {

class PRelu : public BinaryFactory {
  public:
    PRelu() : BinaryFactory(OpType::PRELU, prelu_spv, prelu_spv_len) {}
};

} // namespace ops
} // namespace vkop
#endif // OPS_PRELU_HPP_
