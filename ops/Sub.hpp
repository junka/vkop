// Copyright 2025 @junka
#ifndef OPS_SUB_HPP_
#define OPS_SUB_HPP_

#include "BinaryFactory.hpp"

extern unsigned char sub_spv[];
extern unsigned int sub_spv_len;

namespace vkop {
namespace ops {

class Sub : public BinaryFactory {
  public:
    Sub() : BinaryFactory(OpType::SUB, sub_spv, sub_spv_len) {}
};

} // namespace ops
} // namespace vkop
#endif // OPS_SUB_HPP_
