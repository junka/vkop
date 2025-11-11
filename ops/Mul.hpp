// Copyright 2025 @junka
#ifndef OPS_MUL_HPP_
#define OPS_MUL_HPP_

#include "BinaryFactory.hpp"

extern unsigned char mul_spv[];
extern unsigned int mul_spv_len;

namespace vkop {
namespace ops {

class Mul : public BinaryFactory {
  public:
    Mul() : BinaryFactory(OpType::MUL) { set_vulkan_spv(mul_spv, mul_spv_len); }
};

} // namespace ops
} // namespace vkop
#endif // OPS_MUL_HPP_
