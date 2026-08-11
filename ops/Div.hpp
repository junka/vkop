// Copyright 2025 @junka
#ifndef OPS_DIV_HPP_
#define OPS_DIV_HPP_

#include "BinaryFactory.hpp"
#include "ops/BufferBinaryFactory.hpp"
#include "ops/PimplFacade.hpp"
extern "C" {
extern unsigned char image_div_spv[];
extern unsigned int image_div_spv_len;
extern unsigned char buffer_div_spv[];
extern unsigned int buffer_div_spv_len;
extern unsigned char buffer_div_fp16_spv[];
extern unsigned int buffer_div_fp16_spv_len;
}
namespace vkop {
namespace ops {

class DivImage : public BinaryFactory {
  public:
    DivImage() : BinaryFactory(OpType::DIV, image_div_spv, image_div_spv_len) {}
};

class DivBuffer : public BufferBinaryFactory {
  public:
    explicit DivBuffer(int fp16)
        : BufferBinaryFactory(
              OpType::DIV, fp16 ? buffer_div_fp16_spv : buffer_div_spv,
              fp16 ? buffer_div_fp16_spv_len : buffer_div_spv_len, fp16) {}
};

class Div : public PimplFacade {
  public:
    Div(int fp16, bool backend_buffer) : PimplFacade(OpType::DIV) {
        impl_ =
            backend_buffer
                ? std::unique_ptr<Operator>(std::make_unique<DivBuffer>(fp16))
                : std::make_unique<DivImage>();
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_DIV_HPP_
