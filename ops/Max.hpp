// Copyright 2025 @junka
#ifndef OPS_MAX_HPP_
#define OPS_MAX_HPP_

#include "BinaryFactory.hpp"
#include "ops/BufferBinaryFactory.hpp"
#include "ops/PimplFacade.hpp"
namespace vkop {
namespace ops {

class MaxBuffer : public BufferBinaryFactory {
  public:
    explicit MaxBuffer(int fp16) : BufferBinaryFactory(OpType::MAX, fp16) {}
};

class Max : public PimplFacade {
  public:
    Max(int fp16, bool backend_buffer) : PimplFacade(OpType::MAX) {
        impl_ =
            backend_buffer
                ? std::unique_ptr<Operator>(std::make_unique<MaxBuffer>(fp16))
                : std::unique_ptr<Operator>(std::make_unique<MaxBuffer>(fp16));
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_MAX_HPP_
