// Copyright 2025 @junka
#ifndef OPS_ADD_HPP_
#define OPS_ADD_HPP_

#include "BinaryFactory.hpp"
#include "ops/BufferBinaryFactory.hpp"
#include "ops/PimplFacade.hpp"
extern "C" {
extern unsigned char image_add_spv[];
extern unsigned int image_add_spv_len;
extern unsigned char buffer_add_spv[];
extern unsigned int buffer_add_spv_len;
extern unsigned char buffer_add_fp16_spv[];
extern unsigned int buffer_add_fp16_spv_len;
}
namespace vkop {
namespace ops {

class AddImage : public BinaryFactory {
  public:
    AddImage() : BinaryFactory(OpType::ADD, image_add_spv, image_add_spv_len) {}
};

class AddBuffer : public BufferBinaryFactory {
  public:
    explicit AddBuffer(int fp16)
        : BufferBinaryFactory(
              OpType::ADD, fp16 ? buffer_add_fp16_spv : buffer_add_spv,
              fp16 ? buffer_add_fp16_spv_len : buffer_add_spv_len, fp16) {}
};

class Add : public PimplFacade {
  public:
    Add(int fp16, bool backend_buffer) : PimplFacade(OpType::ADD) {
        impl_ =
            backend_buffer
                ? std::unique_ptr<Operator>(std::make_unique<AddBuffer>(fp16))
                : std::make_unique<AddImage>();
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_ADD_HPP_
