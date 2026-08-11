// Copyright 2025 @junka
#ifndef OPS_POW_HPP_
#define OPS_POW_HPP_

#include "BinaryFactory.hpp"
#include "ops/BufferBinaryFactory.hpp"
#include "ops/PimplFacade.hpp"
extern "C" {
extern unsigned char image_pow_spv[];
extern unsigned int image_pow_spv_len;
extern unsigned char buffer_pow_spv[];
extern unsigned int buffer_pow_spv_len;
extern unsigned char buffer_pow_fp16_spv[];
extern unsigned int buffer_pow_fp16_spv_len;
}
namespace vkop {
namespace ops {

class PowImage : public BinaryFactory {
  public:
    PowImage() : BinaryFactory(OpType::POW, image_pow_spv, image_pow_spv_len) {}
};

class PowBuffer : public BufferBinaryFactory {
  public:
    explicit PowBuffer(int fp16)
        : BufferBinaryFactory(
              OpType::POW, fp16 ? buffer_pow_fp16_spv : buffer_pow_spv,
              fp16 ? buffer_pow_fp16_spv_len : buffer_pow_spv_len, fp16) {}
};

class Pow : public PimplFacade {
  public:
    Pow(int fp16, bool backend_buffer) : PimplFacade(OpType::POW) {
        impl_ =
            backend_buffer
                ? std::unique_ptr<Operator>(std::make_unique<PowBuffer>(fp16))
                : std::make_unique<PowImage>();
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_POW_HPP_
