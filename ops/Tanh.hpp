// Copyright 2025 @junka
#ifndef OPS_TANH_HPP_
#define OPS_TANH_HPP_

#include "UnaryFactory.hpp"
#include "ops/BufferUnaryFactory.hpp"
#include "ops/PimplFacade.hpp"
extern "C" {
extern unsigned char image_tanh_spv[];
extern unsigned int image_tanh_spv_len;
extern unsigned char buffer_tanh_spv[];
extern unsigned int buffer_tanh_spv_len;
extern unsigned char buffer_tanh_fp16_spv[];
extern unsigned int buffer_tanh_fp16_spv_len;
}
namespace vkop {
namespace ops {

// Image (image2DArray NCHW->RGBA) implementation.
class TanhImage : public UnaryFactory {
  public:
    TanhImage()
        : UnaryFactory(OpType::TANH, image_tanh_spv, image_tanh_spv_len) {}
};

// Buffer (SSBO, compact row-major) implementation.
class TanhBuffer : public BufferUnaryFactory {
  public:
    explicit TanhBuffer(int fp16)
        : BufferUnaryFactory(
              OpType::TANH, fp16 ? buffer_tanh_fp16_spv : buffer_tanh_spv,
              fp16 ? buffer_tanh_fp16_spv_len : buffer_tanh_spv_len, fp16) {}
};

// PIMPL façade: picks the buffer SSBO impl when backend_buffer is set,
// else the image impl.
class Tanh : public PimplFacade {
  public:
    Tanh(int fp16, bool backend_buffer) : PimplFacade(OpType::TANH) {
        impl_ =
            backend_buffer
                ? std::unique_ptr<Operator>(std::make_unique<TanhBuffer>(fp16))
                : std::make_unique<TanhImage>();
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_TANH_HPP_
