// Copyright 2025 @junka
#ifndef OPS_ERF_HPP_
#define OPS_ERF_HPP_

#include "UnaryFactory.hpp"
#include "ops/Ops.hpp"

extern unsigned char erf_spv[];
extern unsigned int erf_spv_len;

namespace vkop {
namespace ops {

class Erf : public UnaryFactory {
  public:
    Erf() : UnaryFactory(OpType::ERF) { set_vulkan_spv(erf_spv, erf_spv_len); }
};

} // namespace ops
} // namespace vkop
#endif // OPS_ERF_HPP_
