// Copyright 2025 @junka
#ifndef OPS_SUB_HPP_
#define OPS_SUB_HPP_

#include "BinaryFactory.hpp"
#include "ops/BufferBinaryFactory.hpp"
#include "ops/PimplFacade.hpp"
extern "C" {
extern unsigned char image_sub_spv[];
extern unsigned int image_sub_spv_len;
extern unsigned char buffer_sub_spv[];
extern unsigned int buffer_sub_spv_len;
extern unsigned char buffer_sub_fp16_spv[];
extern unsigned int buffer_sub_fp16_spv_len;
}
namespace vkop {
namespace ops {

class SubImage : public BinaryFactory {
  public:
    SubImage() : BinaryFactory(OpType::SUB, image_sub_spv, image_sub_spv_len) {}
};

class SubBuffer : public BufferBinaryFactory {
  public:
    explicit SubBuffer(int fp16)
        : BufferBinaryFactory(
              OpType::SUB, fp16 ? buffer_sub_fp16_spv : buffer_sub_spv,
              fp16 ? buffer_sub_fp16_spv_len : buffer_sub_spv_len, fp16) {}
};

class Sub : public PimplFacade {
  public:
    Sub(int fp16, bool backend_buffer) : PimplFacade(OpType::SUB) {
        impl_ =
            backend_buffer
                ? std::unique_ptr<Operator>(std::make_unique<SubBuffer>(fp16))
                : std::make_unique<SubImage>();
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_SUB_HPP_
