// Copyright 2025 @junka
#ifndef OPS_MUL_HPP_
#define OPS_MUL_HPP_

#include "BinaryFactory.hpp"
#include "ops/BufferBinaryFactory.hpp"
#include "ops/PimplFacade.hpp"
extern "C" {
extern unsigned char image_mul_spv[];
extern unsigned int image_mul_spv_len;
extern unsigned char buffer_mul_spv[];
extern unsigned int buffer_mul_spv_len;
}
namespace vkop {
namespace ops {

class MulImage : public BinaryFactory {
  public:
    MulImage() : BinaryFactory(OpType::MUL, image_mul_spv, image_mul_spv_len) {}
};

class MulBuffer : public BufferBinaryFactory {
  public:
    explicit MulBuffer(int fp16)
        : BufferBinaryFactory(OpType::MUL, buffer_mul_spv, buffer_mul_spv_len,
                              fp16) {}
};

class Mul : public PimplFacade {
  public:
    Mul(int fp16, bool backend_buffer) : PimplFacade(OpType::MUL) {
        impl_ =
            backend_buffer
                ? std::unique_ptr<Operator>(std::make_unique<MulBuffer>(fp16))
                : std::make_unique<MulImage>();
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_MUL_HPP_
