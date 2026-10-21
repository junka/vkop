// Copyright 2025 @junka
#ifndef OPS_ERF_HPP_
#define OPS_ERF_HPP_

#include "UnaryFactory.hpp"
#include "ops/BufferUnaryFactory.hpp"
#include "ops/PimplFacade.hpp"
namespace vkop {
namespace ops {

// Image (image2DArray NCHW->RGBA) implementation.
class ErfImage : public UnaryFactory {
  public:
    ErfImage() : UnaryFactory(OpType::ERF) {}
};

// Buffer (SSBO, compact row-major) implementation.
class ErfBuffer : public BufferUnaryFactory {
  public:
    explicit ErfBuffer(int fp16) : BufferUnaryFactory(OpType::ERF, fp16) {}
};

// PIMPL façade: picks the buffer SSBO impl when backend_buffer is set,
// else the image impl.
class Erf : public PimplFacade {
  public:
    Erf(int fp16, bool backend_buffer) : PimplFacade(OpType::ERF) {
        impl_ =
            backend_buffer
                ? std::unique_ptr<Operator>(std::make_unique<ErfBuffer>(fp16))
                : std::make_unique<ErfImage>();
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_ERF_HPP_
