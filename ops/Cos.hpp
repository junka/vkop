// Copyright 2026 @junka
#ifndef OPS_COS_HPP_
#define OPS_COS_HPP_

#include "UnaryFactory.hpp"
extern "C" {
extern unsigned char cos_spv[];
extern unsigned int cos_spv_len;
}
namespace vkop {
namespace ops {

class Cos : public UnaryFactory {
  public:
    Cos() : UnaryFactory(OpType::COS, cos_spv, cos_spv_len) {}
};

} // namespace ops
} // namespace vkop
#endif // OPS_COS_HPP_
