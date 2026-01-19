// Copyright 2025 @junka
#ifndef OPS_FLOOR_HPP_
#define OPS_FLOOR_HPP_

#include "UnaryFactory.hpp"

extern "C" {
extern unsigned char floor_spv[];
extern unsigned int floor_spv_len;
};

namespace vkop {
namespace ops {

class Floor : public UnaryFactory {
  public:
    Floor() : UnaryFactory(OpType::FLOOR, floor_spv, floor_spv_len) {}
};

} // namespace ops
} // namespace vkop
#endif // OPS_FLOOR_HPP_
