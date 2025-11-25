// Copyright 2025 @junka
#ifndef OPS_ATAN_HPP_
#define OPS_ATAN_HPP_

#include "UnaryFactory.hpp"

extern unsigned char atan_spv[];
extern unsigned int atan_spv_len;

namespace vkop {
namespace ops {

class Atan : public UnaryFactory {
  public:
    Atan() : UnaryFactory(OpType::ATAN, atan_spv, atan_spv_len) {}
};

} // namespace ops
} // namespace vkop
#endif // OPS_ATAN_HPP_
