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
// The façade passes an empty Operator base ctor so it never builds a pipeline
// of its own — all dispatch flows through impl_.

namespace vkop {
namespace ops {

class PimplFacade : public Operator {
  public:
    // Empty façade: no descriptor layout and no shader, so set_runtime_device
    // is overridden below to forward to impl_ and the base class's
    // resolve_primary_spv()/pipeline build never runs here. impl_ is assigned
    // by the derived class ctor body.
    PimplFacade(OpType type)
        : Operator(type, std::vector<VkDescriptorType>{}, 0, 0) {}

    void set_runtime_device(const std::shared_ptr<VulkanDevice> &dev,
                            const std::shared_ptr<VulkanCommandPool> &cmdpool,
                            VkPipelineCache cache = VK_NULL_HANDLE) override {
        impl_->set_runtime_device(dev, cmdpool, cache);
    }
    // The facade's own spv_ is null/0 (it builds no pipeline), so overriding
    // the base without forwarding would just silently write fields nobody
    // reads. Route to the impl_ that actually builds the pipeline.
    void override_spv(const uint32_t *spv, uint32_t spv_len) override {
        impl_->override_spv(spv, spv_len);
    }
    void set_shader_archive(const vkop::core::ShaderArchive *archive) override {
        impl_->set_shader_archive(archive);
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
    bool uses_buffer_backend() const override {
        return impl_->uses_buffer_backend();
    }
    void set_name(const std::string &name) override { impl_->set_name(name); }
    std::string get_name() const override { return impl_->get_name(); }
    void enable_trace() override { impl_->enable_trace(); }
    void disable_trace() override { impl_->disable_trace(); }
    void set_required_subgroup_size(uint32_t size) override {
        impl_->set_required_subgroup_size(size);
    }
    void enable_replay(bool v) override { impl_->enable_replay(v); }
    bool replay_cached() const override { return impl_->replay_cached(); }
    void force_replay_refresh() override { impl_->force_replay_refresh(); }
    void invalidate_shape_cache() override { impl_->invalidate_shape_cache(); }
    void set_input_value_dynamic(const std::vector<bool> &vd) override {
        impl_->set_input_value_dynamic(vd);
    }
    void set_host_target_shape(const std::vector<int> &dims) override {
        impl_->set_host_target_shape(dims);
    }
    void set_prof_query(VkQueryPool pool, uint32_t base) override {
        impl_->set_prof_query(pool, base);
    }
    void
    set_graph_cmd(const std::shared_ptr<VulkanCommandBuffer> &cmd) override {
        impl_->set_graph_cmd(cmd);
    }
    void repoint_graph_cmd(
        const std::shared_ptr<VulkanCommandBuffer> &cmd) override {
        impl_->repoint_graph_cmd(cmd);
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
