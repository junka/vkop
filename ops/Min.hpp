// Copyright 2025 @junka
#ifndef OPS_MIN_HPP_
#define OPS_MIN_HPP_

#include "BinaryFactory.hpp"
#include "ops/BufferBinaryFactory.hpp"
#include "ops/PimplFacade.hpp"
namespace vkop {
namespace ops {

class MinBuffer : public BufferBinaryFactory {
  public:
    explicit MinBuffer(int fp16) : BufferBinaryFactory(OpType::MIN, fp16) {}
};

class Min : public PimplFacade {
  public:
    Min(int fp16, bool backend_buffer) : PimplFacade(OpType::MIN) {
        impl_ =
            backend_buffer
                ? std::unique_ptr<Operator>(std::make_unique<MinBuffer>(fp16))
                : std::unique_ptr<Operator>(std::make_unique<MinBuffer>(fp16));
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_MIN_HPP_
