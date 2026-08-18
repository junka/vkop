// Copyright 2025 @junka
#ifndef OPS_COS_HPP_
#define OPS_COS_HPP_

#include "UnaryFactory.hpp"
#include "ops/BufferUnaryFactory.hpp"
#include "ops/PimplFacade.hpp"
extern "C" {
extern unsigned char image_cos_spv[];
extern unsigned int image_cos_spv_len;
extern unsigned char buffer_cos_spv[];
extern unsigned int buffer_cos_spv_len;
extern unsigned char buffer_cos_fp16_spv[];
extern unsigned int buffer_cos_fp16_spv_len;
}
namespace vkop {
namespace ops {

// Image (image2DArray NCHW->RGBA) implementation.
class CosImage : public UnaryFactory {
  public:
    CosImage() : UnaryFactory(OpType::COS, image_cos_spv, image_cos_spv_len) {}
};

// Buffer (SSBO, compact row-major) implementation.
class CosBuffer : public BufferUnaryFactory {
  public:
    explicit CosBuffer(int fp16)
        : BufferUnaryFactory(
              OpType::COS, fp16 ? buffer_cos_fp16_spv : buffer_cos_spv,
              fp16 ? buffer_cos_fp16_spv_len : buffer_cos_spv_len, fp16) {}
};

// PIMPL façade: picks the buffer SSBO impl when backend_buffer is set,
// else the image impl.
class Cos : public PimplFacade {
  public:
    Cos(int fp16, bool backend_buffer) : PimplFacade(OpType::COS) {
        impl_ =
            backend_buffer
                ? std::unique_ptr<Operator>(std::make_unique<CosBuffer>(fp16))
                : std::make_unique<CosImage>();
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_COS_HPP_
