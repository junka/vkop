// Copyright 2025 @junka
#ifndef OPS_OPERATOR_HPP_
#define OPS_OPERATOR_HPP_

namespace vkop {

namespace ops {

class Operator {
  public:
    Operator() = default;
    virtual ~Operator() = default;
    Operator(const Operator &) = delete;
    Operator &operator=(const Operator &) = delete;
    Operator(Operator &&) = delete;
    Operator &operator=(Operator &&) = delete;
};

} // namespace ops

} // namespace vkop
#endif // OPS_OPERATOR_HPP_
