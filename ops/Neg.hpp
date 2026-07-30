// Copyright 2025 @junka
#ifndef OPS_NEG_HPP_
#define OPS_NEG_HPP_

#include "UnaryFactory.hpp"
#include "ops/BufferUnaryFactory.hpp"
#include "ops/PimplFacade.hpp"
extern "C" {
extern unsigned char image_neg_spv[];
extern unsigned int image_neg_spv_len;
extern unsigned char buffer_neg_spv[];
extern unsigned int buffer_neg_spv_len;
}
namespace vkop {
namespace ops {

// Image (image2DArray NCHW->RGBA) implementation.
class NegImage : public UnaryFactory {
  public:
    NegImage() : UnaryFactory(OpType::NEG, image_neg_spv, image_neg_spv_len) {}
};

// Buffer (SSBO, compact row-major) implementation.
class NegBuffer : public BufferUnaryFactory {
  public:
    explicit NegBuffer(int fp16)
        : BufferUnaryFactory(OpType::NEG, buffer_neg_spv, buffer_neg_spv_len,
                             fp16) {}
};

// PIMPL façade: picks the buffer SSBO impl when backend_buffer is set,
// else the image impl.
class Neg : public PimplFacade {
  public:
    Neg(int fp16, bool backend_buffer) : PimplFacade(OpType::NEG) {
        impl_ =
            backend_buffer
                ? std::unique_ptr<Operator>(std::make_unique<NegBuffer>(fp16))
                : std::make_unique<NegImage>();
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_NEG_HPP_
