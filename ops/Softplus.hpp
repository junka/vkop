// Copyright 2025 @junka
#ifndef OPS_SOFTPLUS_HPP_
#define OPS_SOFTPLUS_HPP_

#include "UnaryFactory.hpp"
#include "ops/BufferUnaryFactory.hpp"
#include "ops/PimplFacade.hpp"
extern "C" {
extern unsigned char image_softplus_spv[];
extern unsigned int image_softplus_spv_len;
extern unsigned char buffer_softplus_spv[];
extern unsigned int buffer_softplus_spv_len;
extern unsigned char buffer_softplus_fp16_spv[];
extern unsigned int buffer_softplus_fp16_spv_len;
}
namespace vkop {
namespace ops {

// Image (image2DArray NCHW->RGBA) implementation.
class SoftplusImage : public UnaryFactory {
  public:
    SoftplusImage()
        : UnaryFactory(OpType::SOFTPLUS, image_softplus_spv,
                       image_softplus_spv_len) {}
};

// Buffer (SSBO, compact row-major) implementation.
class SoftplusBuffer : public BufferUnaryFactory {
  public:
    explicit SoftplusBuffer(int fp16)
        : BufferUnaryFactory(
              OpType::SOFTPLUS,
              fp16 ? buffer_softplus_fp16_spv : buffer_softplus_spv,
              fp16 ? buffer_softplus_fp16_spv_len : buffer_softplus_spv_len,
              fp16) {}
};

// PIMPL façade: picks the buffer SSBO impl when backend_buffer is set,
// else the image impl.
class Softplus : public PimplFacade {
  public:
    Softplus(int fp16, bool backend_buffer) : PimplFacade(OpType::SOFTPLUS) {
        impl_ = backend_buffer ? std::unique_ptr<Operator>(
                                     std::make_unique<SoftplusBuffer>(fp16))
                               : std::make_unique<SoftplusImage>();
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_SOFTPLUS_HPP_
