// Copyright 2025 @junka
#ifndef OPS_SIGMOID_HPP_
#define OPS_SIGMOID_HPP_

#include "UnaryFactory.hpp"
#include "ops/BufferUnaryFactory.hpp"
#include "ops/PimplFacade.hpp"
extern "C" {
extern unsigned char image_sigmoid_spv[];
extern unsigned int image_sigmoid_spv_len;
extern unsigned char buffer_sigmoid_spv[];
extern unsigned int buffer_sigmoid_spv_len;
extern unsigned char buffer_sigmoid_fp16_spv[];
extern unsigned int buffer_sigmoid_fp16_spv_len;
}
namespace vkop {
namespace ops {

// Image (image2DArray NCHW->RGBA) implementation.
class SigmoidImage : public UnaryFactory {
  public:
    SigmoidImage()
        : UnaryFactory(OpType::SIGMOID, image_sigmoid_spv,
                       image_sigmoid_spv_len) {}
};

// Buffer (SSBO, compact row-major) implementation.
class SigmoidBuffer : public BufferUnaryFactory {
  public:
    explicit SigmoidBuffer(int fp16)
        : BufferUnaryFactory(
              OpType::SIGMOID,
              fp16 ? buffer_sigmoid_fp16_spv : buffer_sigmoid_spv,
              fp16 ? buffer_sigmoid_fp16_spv_len : buffer_sigmoid_spv_len,
              fp16) {}
};

// PIMPL façade: picks the buffer SSBO impl when backend_buffer is set,
// else the image impl.
class Sigmoid : public PimplFacade {
  public:
    Sigmoid(int fp16, bool backend_buffer) : PimplFacade(OpType::SIGMOID) {
        impl_ = backend_buffer ? std::unique_ptr<Operator>(
                                     std::make_unique<SigmoidBuffer>(fp16))
                               : std::make_unique<SigmoidImage>();
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_SIGMOID_HPP_
