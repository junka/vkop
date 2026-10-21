// Copyright 2025 @junka
#ifndef OPS_RELU_HPP_
#define OPS_RELU_HPP_

#include "UnaryFactory.hpp"
#include "ops/BufferUnaryFactory.hpp"
#include "ops/PimplFacade.hpp"
namespace vkop {
namespace ops {

// Image (image2DArray NCHW->RGBA) implementation.
class ReluImage : public UnaryFactory {
  public:
    ReluImage() : UnaryFactory(OpType::RELU) {}
};

// Buffer (SSBO, compact row-major) implementation.
class ReluBuffer : public BufferUnaryFactory {
  public:
    explicit ReluBuffer(int fp16) : BufferUnaryFactory(OpType::RELU, fp16) {}
};

// PIMPL façade: picks the buffer SSBO impl when backend_buffer is set,
// else the image impl.
class Relu : public PimplFacade {
  public:
    Relu(int fp16, bool backend_buffer) : PimplFacade(OpType::RELU) {
        impl_ =
            backend_buffer
                ? std::unique_ptr<Operator>(std::make_unique<ReluBuffer>(fp16))
                : std::make_unique<ReluImage>();
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_RELU_HPP_
