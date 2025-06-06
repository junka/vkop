// Copyright 2025 @junka
#ifndef OPS_DIV_HPP_
#define OPS_DIV_HPP_

#include "ElementWiseFactory.hpp"

extern unsigned char div_spv[];
extern unsigned int div_spv_len;

namespace vkop {
namespace ops {

class Div : public ElementWiseFactory {
  public:
    Div() { set_vulkan_spv(div_spv, div_spv_len); }
};

} // namespace ops
} // namespace vkop
#endif // OPS_DIV_HPP_
