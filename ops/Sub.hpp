// Copyright 2025 @junka
#ifndef OPS_SUB_HPP_
#define OPS_SUB_HPP_

#include "ElementWiseFactory.hpp"

extern unsigned char sub_spv[];
extern unsigned int sub_spv_len;

namespace vkop {
namespace ops {

class Sub : public ElementWiseFactory {
  public:
    Sub() { set_vulkan_spv(sub_spv, sub_spv_len); }
};

} // namespace ops
} // namespace vkop
#endif // OPS_SUB_HPP_
