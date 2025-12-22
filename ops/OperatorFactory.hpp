// Copyright 2025 @junka
#ifndef OPS_OPERATOR_FACTORY_HPP_
#define OPS_OPERATOR_FACTORY_HPP_

#include <functional>
#include <string>
#include <unordered_map>

#include "ops/Operator.hpp"

namespace vkop {

namespace ops {

using Creator = std::function<std::unique_ptr<Operator>()>;

class OperatorFactory {
  public:
    static OperatorFactory &get_instance() {
        static OperatorFactory instance;
        return instance;
    }

    void register_operator(OpType type, Creator creator) {
        creators_[type] = std::move(creator);
    }

    std::unique_ptr<Operator> create(OpType type) const {
        auto it = creators_.find(type);
        if (it != creators_.end()) {
            return it->second();
        }
        throw std::runtime_error("Unknown operator: " +
                                 std::to_string(static_cast<int>(type)));
    }

  private:
    std::unordered_map<OpType, Creator> creators_;
};

} // namespace ops
} // namespace vkop

#define REGISTER_OPERATOR(type, name)                                          \
    static bool register_##name##_dummy = []() {                               \
        OperatorFactory::get_instance().register_operator(                     \
            type, []() { return std::make_unique<name>(); });                  \
        return true;                                                           \
    }();

#endif /* OPS_OPERATOR_FACTORY_HPP_ */
