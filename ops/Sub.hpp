// Copyright 2025 @junka
#ifndef OPS_SUB_HPP_
#define OPS_SUB_HPP_

#include "BinaryFactory.hpp"
#include "ops/BufferBinaryFactory.hpp"
#include "ops/PimplFacade.hpp"
namespace vkop {
namespace ops {

class SubImage : public BinaryFactory {
  public:
    SubImage() : BinaryFactory(OpType::SUB) {}
};

class SubBuffer : public BufferBinaryFactory {
  public:
    explicit SubBuffer(int fp16) : BufferBinaryFactory(OpType::SUB, fp16) {}
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
