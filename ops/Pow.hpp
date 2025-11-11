// Copyright 2025 @junka
#ifndef OPS_POW_HPP_
#define OPS_POW_HPP_

#include "BinaryFactory.hpp"

extern unsigned char pow_spv[];
extern unsigned int pow_spv_len;

namespace vkop {
namespace ops {

class Pow : public BinaryFactory {
  public:
    Pow() : BinaryFactory(OpType::POW) { set_vulkan_spv(pow_spv, pow_spv_len); }
};

} // namespace ops
} // namespace vkop
#endif // OPS_POW_HPP_
