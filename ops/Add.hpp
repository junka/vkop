// Copyright 2025 @junka
#ifndef OPS_ADD_HPP_
#define OPS_ADD_HPP_

#include "BinaryFactory.hpp"
#include "ops/Ops.hpp"

extern unsigned char add_spv[];
extern unsigned int add_spv_len;

namespace vkop {
namespace ops {

class Add : public BinaryFactory {
  public:
    Add() : BinaryFactory(OpType::ADD) { set_vulkan_spv(add_spv, add_spv_len); }
};

} // namespace ops
} // namespace vkop
#endif // OPS_ADD_HPP_
