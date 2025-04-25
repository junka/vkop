#ifndef OPERATOR_HPP
#define OPERATOR_HPP

namespace vkop {

namespace ops {

class Operator {
public:
    Operator() = default;
    virtual ~Operator() = default;
    Operator(const Operator&) = delete;
    Operator& operator=(const Operator&) = delete;
    Operator(Operator&&) = delete;
    Operator& operator=(Operator&&) = delete;
};


}

}
#endif // OPERATOR_HPP