// Copyright 2025 @junka
#ifndef OPS_MUL_HPP_
#define OPS_MUL_HPP_

#include "BinaryFactory.hpp"
#include "ops/BufferBinaryFactory.hpp"
#include "ops/PimplFacade.hpp"
namespace vkop {
namespace ops {

class MulImage : public BinaryFactory {
  public:
    MulImage() : BinaryFactory(OpType::MUL) {}
};

class MulBuffer : public BufferBinaryFactory {
  public:
    explicit MulBuffer(int fp16) : BufferBinaryFactory(OpType::MUL, fp16) {}
};

class Mul : public PimplFacade {
  public:
    Mul(int fp16, bool backend_buffer) : PimplFacade(OpType::MUL) {
        impl_ =
            backend_buffer
                ? std::unique_ptr<Operator>(std::make_unique<MulBuffer>(fp16))
                : std::make_unique<MulImage>();
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_MUL_HPP_
