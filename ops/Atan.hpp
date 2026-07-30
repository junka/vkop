// Copyright 2025 @junka
#ifndef OPS_ATAN_HPP_
#define OPS_ATAN_HPP_

#include "UnaryFactory.hpp"
#include "ops/BufferUnaryFactory.hpp"
#include "ops/PimplFacade.hpp"
extern "C" {
extern unsigned char image_atan_spv[];
extern unsigned int image_atan_spv_len;
extern unsigned char buffer_atan_spv[];
extern unsigned int buffer_atan_spv_len;
extern unsigned char buffer_atan_fp16_spv[];
extern unsigned int buffer_atan_fp16_spv_len;
}
namespace vkop {
namespace ops {

// Image (image2DArray NCHW->RGBA) implementation.
class AtanImage : public UnaryFactory {
  public:
    AtanImage()
        : UnaryFactory(OpType::ATAN, image_atan_spv, image_atan_spv_len) {}
};

// Buffer (SSBO, compact row-major) implementation.
class AtanBuffer : public BufferUnaryFactory {
  public:
    explicit AtanBuffer(int fp16)
        : BufferUnaryFactory(
              OpType::ATAN, fp16 ? buffer_atan_fp16_spv : buffer_atan_spv,
              fp16 ? buffer_atan_fp16_spv_len : buffer_atan_spv_len, fp16) {}
};

// PIMPL façade: picks the buffer SSBO impl when backend_buffer is set,
// else the image impl.
class Atan : public PimplFacade {
  public:
    Atan(int fp16, bool backend_buffer) : PimplFacade(OpType::ATAN) {
        impl_ =
            backend_buffer
                ? std::unique_ptr<Operator>(std::make_unique<AtanBuffer>(fp16))
                : std::make_unique<AtanImage>();
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_ATAN_HPP_
