// Copyright 2026 @junka
#ifndef OPS_SQRT_HPP_
#define OPS_SQRT_HPP_

#include "UnaryFactory.hpp"
extern "C" {
extern unsigned char sqrt_spv[];
extern unsigned int sqrt_spv_len;
}
namespace vkop {
namespace ops {

class Sqrt : public UnaryFactory {
  public:
    Sqrt() : UnaryFactory(OpType::SQRT, sqrt_spv, sqrt_spv_len) {}
};

} // namespace ops
} // namespace vkop
#endif // OPS_SQRT_HPP_
