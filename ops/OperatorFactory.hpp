// Copyright 2025 @junka
#ifndef OPS_OPERATOR_FACTORY_HPP_
#define OPS_OPERATOR_FACTORY_HPP_

#include <functional>
#include <unordered_map>

#include "Operator.hpp"

namespace vkop {

namespace ops {

using Creator = std::function<std::unique_ptr<Operator>()>;

class OperatorFactory {
  public:
    static OperatorFactory &get_instance() {
        static OperatorFactory instance;
        return instance;
    }

    void register_operator(const std::string &name, Creator creator) {
        creators_[name] = std::move(creator);
    }

    std::unique_ptr<Operator> create(const std::string &name) const {
        auto it = creators_.find(name);
        if (it != creators_.end()) {
            return it->second();
        }
        throw std::runtime_error("Unknown operator: " + name);
    }

  private:
    std::unordered_map<std::string, Creator> creators_;
};

} // namespace ops
} // namespace vkop

#define REGISTER_OPERATOR(name)                                                \
    bool register_##name##_dummy = []() {                                      \
        OperatorFactory::get_instance().register_operator(                     \
            #name, []() { return std::make_unique<name>(); });                 \
        return true;                                                           \
    }();

#endif /* OPS_OPERATOR_FACTORY_HPP_ */
