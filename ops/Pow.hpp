// Copyright 2025 @junka
#ifndef OPS_POW_HPP_
#define OPS_POW_HPP_

#include "BinaryFactory.hpp"
#include "ops/BufferBinaryFactory.hpp"
#include "ops/PimplFacade.hpp"
namespace vkop {
namespace ops {

class PowImage : public BinaryFactory {
  public:
    PowImage() : BinaryFactory(OpType::POW) {}
};

class PowBuffer : public BufferBinaryFactory {
  public:
    explicit PowBuffer(int fp16) : BufferBinaryFactory(OpType::POW, fp16) {}
};

class Pow : public PimplFacade {
  public:
    Pow(int fp16, bool backend_buffer) : PimplFacade(OpType::POW) {
        impl_ =
            backend_buffer
                ? std::unique_ptr<Operator>(std::make_unique<PowBuffer>(fp16))
                : std::make_unique<PowImage>();
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_POW_HPP_
