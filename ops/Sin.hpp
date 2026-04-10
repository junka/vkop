// Copyright 2026 @junka
#ifndef OPS_SIN_HPP_
#define OPS_SIN_HPP_

#include "UnaryFactory.hpp"
extern "C" {
extern unsigned char sin_spv[];
extern unsigned int sin_spv_len;
}
namespace vkop {
namespace ops {

class Sin : public UnaryFactory {
  public:
    Sin() : UnaryFactory(OpType::SIN, sin_spv, sin_spv_len) {}
};

} // namespace ops
} // namespace vkop
#endif // OPS_SIN_HPP_
