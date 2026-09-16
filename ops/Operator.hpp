// Copyright 2025 @junka
#ifndef OPS_OPERATOR_HPP_
#define OPS_OPERATOR_HPP_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/Tensor.hpp"
#include "ops/Ops.hpp"

namespace vkop {
namespace ops {

class Operator {
  public:
    explicit Operator(OpType type, uint8_t *spv, uint32_t spv_len,
                      const std::vector<VkDescriptorType> &types,
                      size_t pc_size = 0, int fp16 = 0)
        : type_(type), pc_size_(pc_size), spv_(spv), spv_len_(spv_len),
          types_(std::move(types)), fp16_(fp16) {
        assert(pc_size_ <= 128);
        instance_count_.fetch_add(1);

        objs_.reserve(types_.size());
        writes_.resize(types_.size());
    }

    // Pin the compute pipeline's subgroup size. Set to the device's reported
    // subgroup size before set_runtime_device() for shaders that hard-assume a
    // fixed numSubgroups (e.g. softmax.comp's cross-subgroup reduce).
    virtual void set_required_subgroup_size(uint32_t size) {
        required_subgroup_size_ = size;
    }

    virtual ~Operator() {
        for (auto &m_d : m_ds_) {
            if (m_d)
                pipeline_->freeDescriptorSets(m_d);
        }
        auto cnt = instance_count_.fetch_sub(1);
        if (cnt == 1 && dummy_buffer_ != nullptr &&
            dummy_bufferview_ != nullptr) {
            dummy_bufferview_.reset();
            dummy_buffer_.reset();
            dummy_buffer_ = nullptr;
            dummy_bufferview_ = nullptr;
        }
        m_cmdpool_ = nullptr;
        m_dev_ = nullptr;
    };
    Operator(const Operator &) = delete;
    Operator &operator=(const Operator &) = delete;
    Operator(Operator &&) = delete;
    Operator &operator=(Operator &&) = delete;

    virtual void
    set_runtime_device(const std::shared_ptr<VulkanDevice> &dev,
                       const std::shared_ptr<VulkanCommandPool> &cmdpool) {
        m_dev_ = dev;
        m_cmdpool_ = cmdpool;
        if (spv_len_ > 0 && spv_) {
            bool use_uab = update_after_bind_ &&
                           m_dev_->is_support_descriptor_update_after_bind();
            pipeline_ = std::make_unique<VulkanPipeline>(
                m_dev_->getLogicalDevice(), types_, pc_size_,
                reinterpret_cast<const uint32_t *>(spv_), spv_len_, use_uab,
                required_subgroup_size_);
            for (auto &ds : m_ds_) {
                ds = pipeline_->allocDescriptorSets();
            }
        }
        if (!dummy_buffer_) {
            dummy_buffer_ = std::make_shared<VulkanBuffer>(
                m_dev_, 16, UNIFORM | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            assert(dummy_buffer_);
            dummy_bufferview_ = std::make_shared<VulkanBufferView>(
                m_dev_, dummy_buffer_, VK_FORMAT_R32G32B32A32_SFLOAT, 16, 0);
            assert(dummy_bufferview_);
        }
    }

    virtual void setAttribute(
        const std::unordered_map<std::string, std::string> &attributes) {
        if (!attributes.empty()) {
            for (const auto &attr : attributes) {
                std::cout << "attribute " << attr.first << ": " << attr.second
                          << " in operator." << std::endl;
            }
        }
    }
    template <typename T>
    std::vector<T> parse_attr_list(const std::string &str) {
        std::vector<T> result;
        if (str.front() == '[' && str.back() == ']') {
            std::string content = str.substr(1, str.size() - 2);
            size_t comma_count = 0;
            for (char c : content) {
                if (c == ',')
                    ++comma_count;
            }
            size_t estimated_size = comma_count + 1;
            result.reserve(estimated_size);
            std::stringstream ss(content);
            std::string item;
            while (std::getline(ss, item, ',')) {
                try {
                    if (std::is_same<T, float>::value) {
                        result.emplace_back(std::stof(item));
                    } else {
                        result.emplace_back(std::stol(item));
                    }
                } catch (const std::invalid_argument &e) {
                    throw std::runtime_error(
                        "Invalid number in attribute list: " + item);
                } catch (const std::out_of_range &e) {
                    throw std::runtime_error(
                        "Number out of range in attribute list: " + item);
                }
            }
        }
        return result;
    }

    virtual OpType get_type() { return type_; }

    // Record-once-replay control (cuda-graph-style). The Runtime enables this
    // for the LLM decode loop; see onExecute for the state machine. An op is
    // FRESH until its first onExecute stores a fingerprint and flips it to
    // CACHED; thereafter onExecute skips recording and reuses the kept cmd.
    // force_refresh() drops back to FRESH so the next onExecute re-records
    // (used when the Runtime detects an input shape change for this op).
    virtual void enable_replay(bool v) { replay_enabled_ = v; }
    virtual bool replay_cached() const {
        return replay_state_ == ReplayState::CACHED;
    }
    virtual void force_replay_refresh() {
        replay_state_ = ReplayState::FRESH;
        fp_stored_.clear();
    }

    // Drop any cached shape-input readback (option C: converter static
    // annotation). Called by Runtime across a phase boundary (prefill→decode)
    // where, even for value_dynamic=false shape tensors, the resolved dim[]
    // could differ. Default no-op; only shape-consuming ops (Reshape) override.
    virtual void invalidate_shape_cache() {}

    // These members are the public API the runtime/tests drive. They are
    // virtual so a PIMPL façade op can forward each to its image/buffer
    // impl (which owns the real pipeline/command-buffer state).
    virtual std::shared_ptr<VulkanCommandBuffer> get_record() { return m_cmd_; }

    virtual void
    onExecute(const std::vector<std::shared_ptr<core::ITensor>> &inputs,
              const std::vector<std::shared_ptr<core::ITensor>> &outputs,
              int id) {
        if (!m_cmd_) {
            m_cmd_ = std::make_shared<VulkanCommandBuffer>(m_cmdpool_, id);
        }
        m_id_ = id;

        // Record-once-replay (cuda-graph-style). The LLM decode loop re-runs
        // Run() every round; for INVARIANT ops (weight Gemms, RMSNorm,
        // elementwise on fixed-shape tensors) the recorded command buffer is
        // byte-identical every round. We skip re-recording those:
        //   - Every recording pass builds a fingerprint (push-constant bytes,
        //     dispatch dims, bound resource handles) in fp_passes_.
        //   - After recording, if the new fingerprint matches the stored one
        //     from the previous round, the just-recorded cmd is identical to
        //     last round's; we mark the op CACHED and KEEP its cmd buffer (not
        //     reset) so the NEXT round can replay it verbatim.
        //   - If the fingerprint differs (dynamic op: KV Concat, attention
        //     MatMul whose kv_len grew), keep re-recording every round and
        //     update the stored fingerprint — the op never enters CACHED.
        // Warmup: round 0 records + stores fp (FRESH). Round 1 records +比对;
        //         match → CACHED (cmd kept). Round 2+ replays (no recording).
        //         A CACHED op whose fp later changes (Runtime forced refresh)
        //         drops to FRESH and re-warms.
        if (replay_enabled_ && replay_state_ == ReplayState::CACHED) {
            // Reuse the kept cmd buffer. No begin/execute/end, no descriptor
            // rewrites, no dispatch reissue. m_cmd_ holds the recording from
            // the round that earned CACHED; Run() re-submits it
            // (SIMULTANEOUS_USE).
            replay_hits_++;
            replay_total_hits_.fetch_add(1);
            return;
        }

        objs_.clear();
        fp_passes_.clear();
        // When replay is enabled, record with SIMULTANEOUS_USE so that if this
        // recording is later kept (CACHED), it can be re-submitted verbatim on
        // following rounds. (The first two rounds still record; round 2+
        // replay the CACHED recording.)
        if (replay_enabled_) {
            m_cmd_->set_replayable(true);
        }
        m_cmd_->begin();
        execute(inputs, outputs);
        m_cmd_->end();

        if (replay_enabled_) {
            if (fp_stored_.empty()) {
                // Round 0: store canonical fingerprint. Not yet CACHED — round
                // 1 must still record to confirm the fingerprint is stable.
                fp_stored_ = fp_passes_;
            } else if (fingerprint_matches()) {
                // Round 1+ and the recording matches last round's: the op is
                // invariant. Keep this cmd buffer (Runtime skips reset) and
                // become CACHED so the NEXT round replays it.
                fp_stored_ = fp_passes_;
                replay_state_ = ReplayState::CACHED;
            } else {
                // Dynamic for this round: refresh the stored fingerprint so
                // the next round compares against this round's recording.
                fp_stored_ = fp_passes_;
            }
        }

        if (trace_) {
            m_cmd_->wait();
            m_dev_->wait_all_done();
            printf("%s: %s\n", convert_optype_to_string(type_).c_str(),
                   name_.c_str());
            int i = 0;
            for (auto input : inputs) {
                printf("input %d:\n", i);
                if (input->dtype() == typeid(int64_t)) {
                    continue;
                }
                dispatch_by_dtype(input->dtype(), [&](auto dummy) {
                    using T = decltype(dummy);
                    auto in = core::as_tensor<T>(input);
                    in->copyToCPU(m_cmdpool_);
                    in->print_tensor();
                    in->toGPU();
                });
                i++;
            }
            printf("output 0:\n");
            dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
                using T = decltype(dummy);
                auto output = core::as_tensor<T>(outputs[0]);
                output->copyToCPU(m_cmdpool_);
                output->print_tensor();
                output->toGPU();
            });
        }
    }
    virtual void enable_trace() { trace_ = true; }
    virtual void disable_trace() { trace_ = false; }

    virtual void set_name(const std::string &name) { name_ = name; }
    virtual std::string get_name() const { return name_; }

    // Per-input "value_dynamic" annotation (converter-side: whether this
    // input's ELEMENT VALUES vary across decode rounds — e.g. a shape tensor
    // derived from kv_len is value_dynamic=true; a folded Constant shape is
    // false). The runtime sets this before onExecute for shape-consuming ops
    // (Reshape inputs[1]) so they can cache readback results across rounds and
    // skip the per-round copyToCPU stall. Default no-op: only ops that read a
    // shape input need override it. See core/runtime.cpp
    // node_input_value_dynamic_.
    virtual void set_input_value_dynamic(const std::vector<bool> &vd) {
        (void)vd;
    }

    static std::shared_ptr<VulkanBuffer> dummy_buffer_;
    static std::shared_ptr<VulkanBufferView> dummy_bufferview_;
    static std::atomic<int> instance_count_;
    static std::atomic<long>
        replay_total_hits_; // onExecute CACHED-return count

  protected:
    std::shared_ptr<VulkanDevice> m_dev_;
    std::shared_ptr<VulkanCommandPool> m_cmdpool_;
    std::shared_ptr<VulkanCommandBuffer> m_cmd_ = nullptr;

    std::unique_ptr<VulkanPipeline> pipeline_;
    VkDescriptorSet m_ds_[vkop::kInflight] = {nullptr};
    std::vector<VkWriteDescriptorSet> writes_;
    std::vector<VkDescriptorBufferInfo> buffer_infos_;
    std::vector<VkDescriptorImageInfo> image_infos_;
    int m_id_;
    OpType type_;
    size_t pc_size_ = 0;
    uint8_t *spv_ = nullptr;
    uint32_t spv_len_ = 0;
    bool update_after_bind_ = false;
    bool trace_ = false;
    std::string name_;
    uint32_t required_subgroup_size_ = 0;

    // we should release objs_ here, since for some intermediate tensor, we will
    // release them in the end of the execution.
    std::vector<std::shared_ptr<VulkanResource>> objs_;
    std::vector<VkDescriptorType> types_;
    int fp16_ = 0; // 0: fp32, 1: fp16

    // --- Record-once-replay (cuda-graph-style) ---------------------------
    // The LLM decode loop calls Run() every round with q_len=1; past_kv grows
    // by one token per round but, for INVARIANT ops (weight Gemms, RMSNorm,
    // elementwise on fixed-shape tensors), the push-constant bytes, dispatch
    // dimensions, and bound buffer handles are identical every round. Such an
    // op's recorded command buffer can be replayed verbatim — skipping the
    // ~0.5ms onExecute re-record that dominates the ~1.8s/round bottleneck
    // (3270 ops × re-record = 1771ms; see KV_INPLACE_PLAN.md Step 4).
    //
    // State machine (per op, keyed by m_id_/inflight lane):
    //   FRESH  : first recording. onExecute records normally; each submit()
    //            call appends a PassFingerprint. At onExecute end the op is
    //            marked CACHED and the cmd buffer is kept (NOT reset at Run()
    //            end) with SIMULTANEOUS_USE so it can be re-submitted.
    //   CACHED : a later round. onExecute builds a NEW fingerprint and
    //            compares to the stored one. Match  -> REPLAY (skip recording,
    //            reuse m_cmd_). Mismatch -> the op is dynamic for this round;
    //            re-record (REFRESH) and update the stored fingerprint.
    //
    // A fingerprint covers everything submit() records: pc bytes, dispatch
    // (w,h,layers), and the VkBuffer/VkImage handle of every bound resource
    // (handle, not contents — preallocated KV buffers keep a stable handle
    // across rounds even as their contents change). Buffer *contents* changing
    // is fine (the shader reads whatever is in the buffer at submit time); only
    // a handle/shape/dispatch change forces a re-record.
    struct PassFingerprint {
        std::vector<uint8_t> pc;
        int w = 0, h = 0, layers = 0;
        std::vector<uint64_t>
            handles; // VkBuffer/VkImage handles, in binding order
    };
    std::vector<PassFingerprint> fp_passes_; // accumulated during one onExecute
    std::vector<PassFingerprint>
        fp_stored_; // from the round that earned CACHED
    enum class ReplayState { FRESH, CACHED };
    ReplayState replay_state_ = ReplayState::FRESH;
    bool replay_enabled_ = false; // set true by Runtime when cache mode on
    long replay_hits_ = 0;        // times onExecute took the CACHED return
    // Accumulate the fingerprint of one submit() call. Called from submit()/
    // submit_per_ds() when replay_enabled_. Cheap: one small alloc + handle
    // reads (handles are already materialized in objs_).
    void record_fingerprint(void *ptr, int width, int height, int layers) {
        PassFingerprint p;
        p.w = width;
        p.h = height;
        p.layers = layers;
        if (ptr && pc_size_ > 0) {
            p.pc.assign(static_cast<uint8_t *>(ptr),
                        static_cast<uint8_t *>(ptr) + pc_size_);
        }
        p.handles.reserve(objs_.size());
        for (const auto &r : objs_) {
            if (!r) {
                p.handles.push_back(0);
                continue;
            }
            // VulkanResource::getDescriptorInfo returns the handle-bearing
            // variant; for buffers pBufferInfo->buffer, for images
            // pImageInfo... For a stable fingerprint we just use the resource
            // pointer identity — stable across rounds iff the Tensor (and its
            // backing VkBuffer) is reused, which is exactly the invariant we
            // care about. (Handle value would be more precise but requires
            // pulling VkBuffer out of the variant; pointer identity suffices
            // because a recycled Tensor gets a new VulkanResource.)
            p.handles.push_back(reinterpret_cast<uint64_t>(r.get()));
        }
        fp_passes_.push_back(std::move(p));
    }
    bool fingerprint_matches() const {
        if (fp_passes_.size() != fp_stored_.size())
            return false;
        for (size_t i = 0; i < fp_passes_.size(); i++) {
            const auto &a = fp_passes_[i], &b = fp_stored_[i];
            if (a.w != b.w || a.h != b.h || a.layers != b.layers)
                return false;
            if (a.pc != b.pc)
                return false;
            if (a.handles != b.handles)
                return false;
        }
        return true;
    }

    using SupportedTypes = std::tuple<float, uint16_t, int, int64_t, int8_t>;
    template <typename Func>
    void dispatch_by_dtype(const std::type_info &dtype, Func &&func) {
        bool dispatched = false;

        std::apply(
            [&](auto... types) {
                (([&] {
                     using T = decltype(types);
                     if (dtype == typeid(T)) {
                         func.template operator()<T>(T{});
                         dispatched = true;
                     }
                 }()),
                 ...);
            },
            SupportedTypes{});

        if (!dispatched) {
            throw std::runtime_error("Unsupported dtype: " +
                                     std::string(dtype.name()));
        }
    }

    virtual void submit(void *ptr, int width, int height, int layers) {
        if (!m_ds_[m_id_]) {
            m_ds_[m_id_] = pipeline_->allocDescriptorSets();
        }
        fillWriteDescriptorSets(m_ds_[m_id_]);
        pipeline_->updateDescriptorSets(writes_);

        m_cmd_->bind(*pipeline_, m_ds_[m_id_]);
        if (ptr) {
            m_cmd_->push_constants(*pipeline_, static_cast<uint32_t>(pc_size_),
                                   ptr);
        }
        m_cmd_->dispatch(width, height, layers);
        if (replay_enabled_) {
            record_fingerprint(ptr, width, height, layers);
        }
    }

    // Variant of submit() where the dispatch dimensions come from a GPU buffer
    // (vkCmdDispatchIndirect) instead of CPU-provided w/h/layers. Used by
    // dynamic-dispatch ops whose thread count depends on a shape value that
    // lives on the GPU (e.g. kv_len-driven attention) — avoids a GPU->CPU
    // readback solely to know the dispatch dims. `dispatch_buf` must hold a
    // VkDispatchIndirectCommand{w,h,z} at `dispatch_off`, written and
    // barriered to INDIRECT_READ by a prior shader in the same cmd buffer.
    // The fingerprint stores 0,0,0 for dims (the indirect buffer's contents
    // aren't visible to the host at record time; replay of a dynamic op must
    // re-run the shape->dispatch shader, so it never reaches CACHED anyway).
    void submit_indirect(void *ptr, VkBuffer dispatch_buf,
                         VkDeviceSize dispatch_off) {
        if (!m_ds_[m_id_]) {
            m_ds_[m_id_] = pipeline_->allocDescriptorSets();
        }
        fillWriteDescriptorSets(m_ds_[m_id_]);
        pipeline_->updateDescriptorSets(writes_);

        m_cmd_->bind(*pipeline_, m_ds_[m_id_]);
        if (ptr) {
            m_cmd_->push_constants(*pipeline_, static_cast<uint32_t>(pc_size_),
                                   ptr);
        }
        m_cmd_->dispatch_indirect(dispatch_buf, dispatch_off);
        if (replay_enabled_) {
            record_fingerprint(ptr, 0, 0, 0);
        }
    }
    // Each pass must use its own descriptor set because vkUpdateDescriptorSets
    // modifies the set immediately — all dispatches referencing the same set
    // see the LAST update's bindings, causing earlier passes to read wrong
    // buffers on some drivers (e.g. Intel ANV).
    void submit_per_ds(VkDescriptorSet ds, void *ptr, int width, int height,
                       int layers) {
        fillWriteDescriptorSets(ds);
        pipeline_->updateDescriptorSets(writes_);

        m_cmd_->bind(*pipeline_, ds);
        if (ptr) {
            m_cmd_->push_constants(*pipeline_, static_cast<uint32_t>(pc_size_),
                                   ptr);
        }
        m_cmd_->dispatch(width, height, layers);
        if (replay_enabled_) {
            record_fingerprint(ptr, width, height, layers);
        }
    }

    // Allocate a fresh descriptor set from the pipeline's pool.
    VkDescriptorSet allocPassDescriptorSet() {
        return pipeline_->allocDescriptorSets();
    }

    // Return a per-pass descriptor set to the pool.
    void freePassDescriptorSet(VkDescriptorSet ds) {
        pipeline_->freeDescriptorSets(ds);
    }

  protected:
    // fillWriteDescriptorSets is overridden by façade ops (e.g. Gather's int64
    // submit override calls it to bind the int64 pipeline's descriptor set).
    virtual void fillWriteDescriptorSets(VkDescriptorSet ds) {
        // Write one descriptor per BOUND object. The pipeline layout may
        // declare MORE bindings than we bind (e.g. Phase 3 binary shaders
        // declare optional shape-SSBO bindings 3/4/5 that are only bound when
        // an input carries shape_ssbo_). Under UPDATE_AFTER_BIND the unbound
        // bindings are legal and the shader only reads them when its dispatch
        // mode selects the SSBO path — so we never write descriptors we didn't
        // bind.
        size_t n = std::min(types_.size(), objs_.size());
        writes_.resize(
            types_.size()); // ensure capacity for all declared bindings
        for (size_t i = 0; i < n; i++) {
            writes_[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes_[i].dstSet = ds;
            writes_[i].dstBinding = static_cast<uint32_t>(i);
            writes_[i].dstArrayElement = 0;
            writes_[i].descriptorCount = 1;
            writes_[i].descriptorType = types_[i];
            switch (objs_[i]->getResourceType()) {
            case ResourceType::VK_BUFFER_VIEW:
                writes_[i].pTexelBufferView =
                    std::get<VkBufferView *>(objs_[i]->getDescriptorInfo());
                break;
            case ResourceType::VK_BUFFER:
                writes_[i].pBufferInfo = std::get<VkDescriptorBufferInfo *>(
                    objs_[i]->getDescriptorInfo());
                break;
            case ResourceType::VK_IMAGE:
                writes_[i].pImageInfo = std::get<VkDescriptorImageInfo *>(
                    objs_[i]->getDescriptorInfo());
                break;
            default:
                break;
            }
        }
        // Truncate to the bound-object count so updateDescriptorSets() only
        // sees the initialized writes (see the note above on optional
        // bindings).
        writes_.resize(n);
    }

    virtual void
    execute(const std::vector<std::shared_ptr<core::ITensor>> &inputs,
            const std::vector<std::shared_ptr<core::ITensor>> &outputs) = 0;
};

} // namespace ops
} // namespace vkop
#endif // OPS_OPERATOR_HPP_
