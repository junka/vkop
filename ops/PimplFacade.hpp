// Copyright 2025 @junka
#ifndef OPS_PIMPL_FACADE_HPP_
#define OPS_PIMPL_FACADE_HPP_

#include "ops/Operator.hpp"
#include <memory>
#include <utility>

// PIMPL façade base for op headers. Each public op class (Relu, Add, ...)
// derives from PimplFacade and holds a `std::unique_ptr<Operator> impl_`
// that is either the ImageImpl (e.g. ReluImage) or the BufferImpl
// (e.g. <Op>Buffer (e.g. ReluBuffer)), selected at construction. PimplFacade
// forwards the full public API the runtime/tests drive to impl_:
// set_runtime_device, setAttribute, onExecute, get_record, get_type,
// set_name/get_name, enable_trace/disable_trace,
// set_required_subgroup_size. The impl owns its own
// pipeline/command-buffer/descriptor state; the façade's stays null.
//
// The façade passes a no-op (spv=nullptr, len=0) Operator base ctor so it
// never builds a pipeline of its own — all dispatch flows through impl_.

namespace vkop {
namespace ops {

class PimplFacade : public Operator {
  public:
    // Empty façade: spv=nullptr/0 so set_runtime_device's pipeline build is
    // skipped (the guard `if (spv_len_ > 0 && spv_)` in
    // Operator::set_runtime_device skips it). impl_ is assigned by the derived
    // class ctor body.
    PimplFacade(OpType type)
        : Operator(type, nullptr, 0, std::vector<VkDescriptorType>{}, 0, 0) {}

    void set_runtime_device(
        const std::shared_ptr<VulkanDevice> &dev,
        const std::shared_ptr<VulkanCommandPool> &cmdpool) override {
        impl_->set_runtime_device(dev, cmdpool);
    }
    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        impl_->setAttribute(attributes);
    }
    void onExecute(const std::vector<std::shared_ptr<core::ITensor>> &inputs,
                   const std::vector<std::shared_ptr<core::ITensor>> &outputs,
                   int id) override {
        impl_->onExecute(inputs, outputs, id);
    }
    std::shared_ptr<VulkanCommandBuffer> get_record() override {
        return impl_->get_record();
    }
    OpType get_type() override { return impl_->get_type(); }
    void set_name(const std::string &name) override { impl_->set_name(name); }
    std::string get_name() const override { return impl_->get_name(); }
    void enable_trace() override { impl_->enable_trace(); }
    void disable_trace() override { impl_->disable_trace(); }
    void set_required_subgroup_size(uint32_t size) override {
        impl_->set_required_subgroup_size(size);
    }

  protected:
    std::unique_ptr<Operator> impl_;

  private:
    // PimplFacade is not itself an op — forward execute() to impl_. (The
    // runtime drives ops through onExecute, which the façade already
    // forwards; this is only needed to satisfy the pure-virtual contract
    // so the façade class is concrete when constructed standalone in tests.)
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        impl_->onExecute(inputs, outputs, 0);
    }
};

} // namespace ops
} // namespace vkop

#endif // OPS_PIMPL_FACADE_HPP_
