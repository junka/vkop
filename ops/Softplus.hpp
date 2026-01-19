// Copyright 2025 @junka
#ifndef OPS_SOFTPLUS_HPP_
#define OPS_SOFTPLUS_HPP_

#include "UnaryFactory.hpp"

extern "C" {
extern unsigned char softplus_spv[];
extern unsigned int softplus_spv_len;
};

namespace vkop {
namespace ops {

class Softplus : public UnaryFactory {
  public:
    Softplus()
        : UnaryFactory(OpType::SOFTPLUS, softplus_spv, softplus_spv_len) {}
};

} // namespace ops
} // namespace vkop
#endif // OPS_SOFTPLUS_HPP_
