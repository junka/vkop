// Copyright 2025 @junka
#ifndef OPS_RELU_HPP_
#define OPS_RELU_HPP_

#include "UnaryFactory.hpp"

extern unsigned char relu_spv[];
extern unsigned int relu_spv_len;

namespace vkop {
namespace ops {

class Relu : public UnaryFactory {
  public:
    Relu() { set_vulkan_spv(relu_spv, relu_spv_len); }
};

} // namespace ops
} // namespace vkop
#endif // OPS_RELU_HPP_
