// Copyright 2025 @junka
#ifndef OPS_ATAN_HPP_
#define OPS_ATAN_HPP_

#include "UnaryFactory.hpp"
#include "ops/BufferUnaryFactory.hpp"
#include "ops/PimplFacade.hpp"
namespace vkop {
namespace ops {

// Image (image2DArray NCHW->RGBA) implementation.
class AtanImage : public UnaryFactory {
  public:
    AtanImage() : UnaryFactory(OpType::ATAN) {}
};

// Buffer (SSBO, compact row-major) implementation.
class AtanBuffer : public BufferUnaryFactory {
  public:
    explicit AtanBuffer(int fp16) : BufferUnaryFactory(OpType::ATAN, fp16) {}
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
