// Copyright 2025 @junka
#ifndef OPS_FLOOR_HPP_
#define OPS_FLOOR_HPP_

#include "UnaryFactory.hpp"
#include "ops/BufferUnaryFactory.hpp"
#include "ops/PimplFacade.hpp"
extern "C" {
extern unsigned char image_floor_spv[];
extern unsigned int image_floor_spv_len;
extern unsigned char buffer_floor_spv[];
extern unsigned int buffer_floor_spv_len;
}
namespace vkop {
namespace ops {

// Image (image2DArray NCHW->RGBA) implementation.
class FloorImage : public UnaryFactory {
  public:
    FloorImage()
        : UnaryFactory(OpType::FLOOR, image_floor_spv, image_floor_spv_len) {}
};

// Buffer (SSBO, compact row-major) implementation.
class FloorBuffer : public BufferUnaryFactory {
  public:
    explicit FloorBuffer(int fp16)
        : BufferUnaryFactory(OpType::FLOOR, buffer_floor_spv,
                             buffer_floor_spv_len, fp16) {}
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
