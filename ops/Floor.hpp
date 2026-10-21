// Copyright 2025 @junka
#ifndef OPS_FLOOR_HPP_
#define OPS_FLOOR_HPP_

#include "UnaryFactory.hpp"
#include "ops/BufferUnaryFactory.hpp"
#include "ops/PimplFacade.hpp"
namespace vkop {
namespace ops {

// Image (image2DArray NCHW->RGBA) implementation.
class FloorImage : public UnaryFactory {
  public:
    FloorImage() : UnaryFactory(OpType::FLOOR) {}
};

// Buffer (SSBO, compact row-major) implementation.
class FloorBuffer : public BufferUnaryFactory {
  public:
    explicit FloorBuffer(int fp16) : BufferUnaryFactory(OpType::FLOOR, fp16) {}
};

// PIMPL façade: picks the buffer SSBO impl when backend_buffer is set,
// else the image impl.
class Floor : public PimplFacade {
  public:
    Floor(int fp16, bool backend_buffer) : PimplFacade(OpType::FLOOR) {
        impl_ =
            backend_buffer
                ? std::unique_ptr<Operator>(std::make_unique<FloorBuffer>(fp16))
                : std::make_unique<FloorImage>();
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_FLOOR_HPP_
