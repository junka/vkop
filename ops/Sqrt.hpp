// Copyright 2025 @junka
#ifndef OPS_SQRT_HPP_
#define OPS_SQRT_HPP_

#include "UnaryFactory.hpp"
#include "ops/BufferUnaryFactory.hpp"
#include "ops/PimplFacade.hpp"
extern "C" {
extern unsigned char image_sqrt_spv[];
extern unsigned int image_sqrt_spv_len;
extern unsigned char buffer_sqrt_spv[];
extern unsigned int buffer_sqrt_spv_len;
extern unsigned char buffer_sqrt_fp16_spv[];
extern unsigned int buffer_sqrt_fp16_spv_len;
}
namespace vkop {
namespace ops {

// Image (image2DArray NCHW->RGBA) implementation.
class SqrtImage : public UnaryFactory {
  public:
    SqrtImage()
        : UnaryFactory(OpType::SQRT, image_sqrt_spv, image_sqrt_spv_len) {}
};

// Buffer (SSBO, compact row-major) implementation.
class SqrtBuffer : public BufferUnaryFactory {
  public:
    explicit SqrtBuffer(int fp16)
        : BufferUnaryFactory(
              OpType::SQRT, fp16 ? buffer_sqrt_fp16_spv : buffer_sqrt_spv,
              fp16 ? buffer_sqrt_fp16_spv_len : buffer_sqrt_spv_len, fp16) {}
};

// PIMPL façade: picks the buffer SSBO impl when backend_buffer is set,
// else the image impl.
class Sqrt : public PimplFacade {
  public:
    Sqrt(int fp16, bool backend_buffer) : PimplFacade(OpType::SQRT) {
        impl_ =
            backend_buffer
                ? std::unique_ptr<Operator>(std::make_unique<SqrtBuffer>(fp16))
                : std::make_unique<SqrtImage>();
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_SQRT_HPP_
