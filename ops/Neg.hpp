// Copyright 2026 @junka
#ifndef OPS_NEG_HPP_
#define OPS_NEG_HPP_

#include "UnaryFactory.hpp"
extern "C" {
extern unsigned char neg_spv[];
extern unsigned int neg_spv_len;
}
namespace vkop {
namespace ops {

class Neg : public UnaryFactory {
  public:
    Neg() : UnaryFactory(OpType::NEG, neg_spv, neg_spv_len) {}
};

} // namespace ops
} // namespace vkop
#endif // OPS_NEG_HPP_
