// Copyright 2025 @junka
#ifndef OPS_SIN_HPP_
#define OPS_SIN_HPP_

#include "UnaryFactory.hpp"
#include "ops/BufferUnaryFactory.hpp"
#include "ops/PimplFacade.hpp"
namespace vkop {
namespace ops {

// Image (image2DArray NCHW->RGBA) implementation.
class SinImage : public UnaryFactory {
  public:
    SinImage() : UnaryFactory(OpType::SIN) {}
};

// Buffer (SSBO, compact row-major) implementation.
class SinBuffer : public BufferUnaryFactory {
  public:
    explicit SinBuffer(int fp16) : BufferUnaryFactory(OpType::SIN, fp16) {}
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
