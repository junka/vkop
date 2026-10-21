// Copyright 2025 @junka
#ifndef OPS_ADD_HPP_
#define OPS_ADD_HPP_

#include "BinaryFactory.hpp"
#include "ops/BufferBinaryFactory.hpp"
#include "ops/PimplFacade.hpp"
namespace vkop {
namespace ops {

class AddImage : public BinaryFactory {
  public:
    explicit AddImage(int fp16) : BinaryFactory(OpType::ADD, fp16) {}
};

class AddBuffer : public BufferBinaryFactory {
  public:
    explicit AddBuffer(int fp16) : BufferBinaryFactory(OpType::ADD, fp16) {}
};

class Add : public PimplFacade {
  public:
    Add(int fp16, bool backend_buffer) : PimplFacade(OpType::ADD) {
        impl_ =
            backend_buffer
                ? std::unique_ptr<Operator>(std::make_unique<AddBuffer>(fp16))
                : std::make_unique<AddImage>(fp16);
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_ADD_HPP_
