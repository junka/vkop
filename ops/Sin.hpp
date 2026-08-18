// Copyright 2025 @junka
#ifndef OPS_SIN_HPP_
#define OPS_SIN_HPP_

#include "UnaryFactory.hpp"
#include "ops/BufferUnaryFactory.hpp"
#include "ops/PimplFacade.hpp"
extern "C" {
extern unsigned char image_sin_spv[];
extern unsigned int image_sin_spv_len;
extern unsigned char buffer_sin_spv[];
extern unsigned int buffer_sin_spv_len;
extern unsigned char buffer_sin_fp16_spv[];
extern unsigned int buffer_sin_fp16_spv_len;
}
namespace vkop {
namespace ops {

// Image (image2DArray NCHW->RGBA) implementation.
class SinImage : public UnaryFactory {
  public:
    SinImage() : UnaryFactory(OpType::SIN, image_sin_spv, image_sin_spv_len) {}
};

// Buffer (SSBO, compact row-major) implementation.
class SinBuffer : public BufferUnaryFactory {
  public:
    explicit SinBuffer(int fp16)
        : BufferUnaryFactory(
              OpType::SIN, fp16 ? buffer_sin_fp16_spv : buffer_sin_spv,
              fp16 ? buffer_sin_fp16_spv_len : buffer_sin_spv_len, fp16) {}
};

// PIMPL façade: picks the buffer SSBO impl when backend_buffer is set,
// else the image impl.
class Sin : public PimplFacade {
  public:
    Sin(int fp16, bool backend_buffer) : PimplFacade(OpType::SIN) {
        impl_ =
            backend_buffer
                ? std::unique_ptr<Operator>(std::make_unique<SinBuffer>(fp16))
                : std::make_unique<SinImage>();
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_SIN_HPP_
