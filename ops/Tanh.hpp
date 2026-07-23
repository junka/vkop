// Copyright 2025 @junka
#ifndef OPS_TANH_HPP_
#define OPS_TANH_HPP_

#include "UnaryFactory.hpp"
extern "C" {
extern unsigned char tanh_spv[];
extern unsigned int tanh_spv_len;
}
namespace vkop {
namespace ops {

// element-wise tanh。走 UnaryFactory（与 Erf/Sin/Cos/Neg 同路径），shader 用
// GLSL 内建 tanh()。主要用于 Qwen3-VL 视觉 block 的 GELU-tanh 近似激活。

class Tanh : public UnaryFactory {
  public:
    Tanh() : UnaryFactory(OpType::TANH, tanh_spv, tanh_spv_len) {}
};

} // namespace ops
} // namespace vkop
#endif // OPS_TANH_HPP_
