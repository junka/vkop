// Copyright 2025 @junka
#ifndef OPS_MOD_HPP_
#define OPS_MOD_HPP_

#include "BinaryFactory.hpp"
#include "ops/BufferBinaryFactory.hpp"
#include "ops/PimplFacade.hpp"
namespace vkop {
namespace ops {

class ModBuffer : public BufferBinaryFactory {
  public:
    explicit ModBuffer(int fp16) : BufferBinaryFactory(OpType::MOD, fp16) {}
};

class Mod : public PimplFacade {
  public:
    Mod(int fp16, bool backend_buffer) : PimplFacade(OpType::MOD) {
        impl_ =
            backend_buffer
                ? std::unique_ptr<Operator>(std::make_unique<ModBuffer>(fp16))
                : std::unique_ptr<Operator>(std::make_unique<ModBuffer>(fp16));
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_MOD_HPP_
