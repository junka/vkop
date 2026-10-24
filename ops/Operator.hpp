// Copyright 2025 @junka
#ifndef OPS_OPERATOR_HPP_
#define OPS_OPERATOR_HPP_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/Tensor.hpp"
#include "ops/Ops.hpp"

namespace vkop {
namespace core {
class ShaderArchive;
}
namespace ops {

// VKOP_HOST_SHAPE evaluates the int64 shape/metadata chain entirely on the
// host instead of via GPU shaders whose outputs must be read back to the CPU
// by downstream shape-consuming ops (Reshape/Expand/Mul/Cast). The chain is
// derived purely from tensor shape metadata + int64 constants — no hidden
// state — so host evaluation is exact. Producers mark their int64 outputs
// host-authoritative (Tensor::set_host_authoritative) so a downstream
// copyToCPU returns the already-correct host bytes with no GPU round trip;
// the bytes are still mirrored to the GPU via copyToGPUDeferred for GPU
// consumers. On by default; VKOP_HOST_SHAPE=0 restores the GPU-shader baseline
// for A/B comparison.
inline bool host_shape_enabled() {
    static const bool enabled = [] {
        const char *env = std::getenv("VKOP_HOST_SHAPE");
        return env == nullptr || env[0] != '0';
    }();
    return enabled;
}

class Operator {
  public:
    // Ops take no shader symbol: their primary SPIR-V is resolved from the
    // archive the .vkopbin carries, keyed by shader_key_for(type_,
    // uses_buffer_backend(), fp16_) — see set_runtime_device. Only ops with
    // secondary pipelines (Matmul/Conv2d) ever name a key themselves.
    explicit Operator(OpType type, const std::vector<VkDescriptorType> &types,
                      size_t pc_size = 0, int fp16 = 0)
        : type_(type), pc_size_(pc_size), types_(std::move(types)),
          fp16_(fp16) {
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

    // Replace the SPIR-V this op builds its pipeline from, called by the
    // Runtime after create_from_type() and before set_runtime_device().
    // libvkop links in no shader bytes of its own: the model's .vkopbin is the
    // only source, and this is how its per-op entry reaches the op. Passing
    // nullptr leaves whatever the op resolved for itself alone (used when the
    // archive does not have an entry for this op — e.g. an int64 shape-meta
    // path that reaches the archive by a different key).
    //
    // `spv_len == 0` is the "no shader" sentinel (a PimplFacade shell whose
    // impl_ owns the real pipeline). Both fields are updated together so a
    // caller can't leave spv_ and spv_len_ in disagreeing states.
    virtual void override_spv(const uint32_t *spv, uint32_t spv_len) {
        if (!spv || spv_len == 0) {
            return;
        }
        spv_ = reinterpret_cast<uint8_t *>(const_cast<uint32_t *>(spv));
        spv_len_ = spv_len;
    }

    // Ops whose set_runtime_device override builds secondary pipelines
    // (Matmul's coop/pack/coop_i8, Conv2d's pack) call this to reach the
    // archive from inside those builds. Runtime calls set_shader_archive()
    // immediately after create_from_type(), before any override_spv() or
    // set_runtime_device(), so an archive is available whenever the op needs
    // one. Null means the op was built with no model behind it; it then falls
    // back to ShaderArchive::process_default(), which only the dev binaries
    // install.
    virtual void set_shader_archive(const vkop::core::ShaderArchive *archive) {
        shader_archive_ = archive;
    }

    virtual void
    set_runtime_device(const std::shared_ptr<VulkanDevice> &dev,
                       const std::shared_ptr<VulkanCommandPool> &cmdpool,
                       VkPipelineCache cache = VK_NULL_HANDLE) {
        m_dev_ = dev;
        m_cmdpool_ = cmdpool;
        // Nothing supplied this op's shader yet: Runtime swaps archive bytes in
        // via override_spv() before calling this for ops it builds from a model
        // file, and ops built straight through create_from_type() (unit-test
        // fixtures, example drivers) never get that call. Resolve this op's own
        // key against whatever archive is reachable, and throw a named error if
        // that fails rather than letting pipeline_ stay null and every later
        // submit() crash on pipeline_->allocDescriptorSets(). A host-only op
        // (Shape, SqueezeUnsqueeze) has no key and legitimately builds none.
        if (spv_len_ == 0) {
            resolve_primary_spv();
        }
        if (spv_len_ > 0 && spv_) {
            bool use_uab = update_after_bind_ &&
                           m_dev_->is_support_descriptor_update_after_bind();
            pipeline_ = std::make_unique<VulkanPipeline>(
                m_dev_->getLogicalDevice(), types_, pc_size_,
                reinterpret_cast<const uint32_t *>(spv_), spv_len_, use_uab,
                required_subgroup_size_, cache);
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

    // Which backend this instance actually implements. The PIMPL façades pick
    // image/buffer at construction and a request for a missing backend falls
    // back silently to the other one, so callers (tests) cannot infer it from
    // the flag they passed — they have to ask the constructed op.
    virtual bool uses_buffer_backend() const { return false; }

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

    // GPU submit-side profiling (VKOP_SUBMIT_PROF). The Runtime assigns each
    // op a pair of timestamp-query indices (base, base+1) in a shared pool;
    // onExecute writes a TOP_OF_PIPE timestamp right after begin() and a
    // BOTTOM_OF_PIPE timestamp right before end(), so the delta (×
    // timestampPeriod) is this op's GPU execution time. No-op when pool is
    // null. CACHED-replay ops keep their recorded timestamps (the recording
    // from the round that earned CACHED carries them) and re-emit them on
    // re-submission. Forwarded by PimplFacade to the impl.
    virtual void set_prof_query(VkQueryPool pool, uint32_t base) {
        prof_query_pool_ = pool;
        prof_query_base_ = base;
    }

    // Print the recorded replay fingerprint to stdout (one line per
    // submit()). The fingerprint is exactly what the cache compares to decide
    // CACHED -- pc bytes, dispatch dims, bound handles -- so it is a direct,
    // Vulkan-layer-free readout of WHICH KERNEL RAN and with what geometry.
    // Two passes at dispatch (n/32, m) are the coopmat MMA + pack; one pass at
    // (n/4, m) is the 4-column ksplit GEMV; (n/64, ceil(m/64)) is tiled. Call
    // it right after onExecute (the list is cleared at the top of the next
    // one) or after a CACHED round (the list is empty then, by design).
    // Virtual so a PIMPL façade op forwards these to its impl_ -- the façade
    // never dispatches, so its own fp_passes_ stays permanently empty.
    virtual void dump_fingerprint() const {
        printf("fingerprint: %zu submit(s)\n", fp_passes_.size());
        for (size_t i = 0; i < fp_passes_.size(); ++i) {
            const auto &p = fp_passes_[i];
            printf("  [%zu] w=%d h=%d z=%d pc=%zuB\n", i, p.w, p.h, p.layers,
                   p.pc.size());
        }
    }

    // These members are the public API the runtime/tests drive. They are
    // virtual so a PIMPL façade op can forward each to its image/buffer
    // impl (which owns the real pipeline/command-buffer state).
    virtual std::shared_ptr<VulkanCommandBuffer> get_record() { return m_cmd_; }

    // Graph-submit mode (VKOP_GRAPH_SUBMIT). The Runtime records every level's
    // ops into ONE shared command buffer so ~980 per-level vkQueueSubmit calls
    // collapse into a handful. While a non-null graph cmd is set, onExecute
    // records into it (borrowed — the Runtime owns begin/end/submit) and the
    // per-op replay cache is bypassed, since a shared recording target cannot
    // be cached per op. Passing nullptr restores the normal per-op path.
    virtual void
    set_graph_cmd(const std::shared_ptr<VulkanCommandBuffer> &cmd) {
        graph_cmd_ = cmd;
    }

    // Repoint the recording target mid-onExecute. The Runtime's readback hook
    // (VulkanCommandBuffer::pre_readback_hook) closes the open segment right
    // before an op does a synchronous GPU->CPU readback, then calls this so the
    // remainder of the op's recording lands in the freshly opened segment.
    virtual void
    repoint_graph_cmd(const std::shared_ptr<VulkanCommandBuffer> &cmd) {
        graph_cmd_ = cmd;
        m_cmd_ = cmd;
    }

    virtual void
    onExecute(const std::vector<std::shared_ptr<core::ITensor>> &inputs,
              const std::vector<std::shared_ptr<core::ITensor>> &outputs,
              int id); // defined below the class: its body is long enough that
                       // gcc declines to emit a weak copy for every TU that
                       // includes this header (see the definition for why)

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

    // Phase 4-r2 shape-chain host-ification: the runtime evaluates the
    // converter's `vkop_host_shape` term list from LIVE tensor metadata each
    // round and pushes the resulting target dims here, right before
    // onExecute. The Reshape then has NO inputs[1] (single-input form — the
    // converter deleted the int64 Shape/Gather/Concat chain) and takes its
    // target shape from this call instead. Default no-op; only ops that
    // consume a shape input override it.
    virtual void set_host_target_shape(const std::vector<int> &dims) {
        (void)dims;
    }

    static std::shared_ptr<VulkanBuffer> dummy_buffer_;
    static std::shared_ptr<VulkanBufferView> dummy_bufferview_;
    static std::atomic<int> instance_count_;
    static std::atomic<long>
        replay_total_hits_; // onExecute CACHED-return count

  public:
    // Read-only view of the passes recorded by the last onExecute: one entry
    // per submit() call, carrying the pc bytes, the dispatch dims and the bound
    // resource handles. The replay cache compares these to decide CACHED, so
    // they are exactly "what ran" for a recording -- which makes them the only
    // reliable way for a harness to identify the kernel without a Vulkan debug
    // layer or a per-shader marker (/tmp/bench_tc.cpp).
    struct PassInfo {
        std::vector<uint8_t> pc;
        int w = 0, h = 0, layers = 0;
    };
    virtual std::vector<PassInfo> recorded_passes() const {
        std::vector<PassInfo> out;
        out.reserve(fp_passes_.size());
        for (const auto &p : fp_passes_) {
            out.push_back(PassInfo{p.pc, p.w, p.h, p.layers});
        }
        return out;
    }

  protected:
    std::shared_ptr<VulkanDevice> m_dev_;
    std::shared_ptr<VulkanCommandPool> m_cmdpool_;
    std::shared_ptr<VulkanCommandBuffer> m_cmd_ = nullptr;
    // Non-null only in graph-submit mode; borrowed from the Runtime (see
    // set_graph_cmd). When set, onExecute records into it instead of m_cmd_.
    std::shared_ptr<VulkanCommandBuffer> graph_cmd_ = nullptr;

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
    // When Runtime loaded a vkop_shaders.bin archive, this points at it so
    // ops with secondary pipelines (Matmul/Conv2d) can consult it inside
    // their set_runtime_device overrides. Null otherwise.
    const vkop::core::ShaderArchive *shader_archive_ = nullptr;
    bool update_after_bind_ = false;
    bool trace_ = false;
    std::string name_;
    uint32_t required_subgroup_size_ = 0;

    // Secondary-pipeline helper for ops like Matmul/Conv2d that build more
    // than one VulkanPipeline inside their set_runtime_device override, and so
    // name a variant key the op's primary lookup does not cover. Runtime's
    // post-construct override_spv() can only reach the op's primary spv_; these
    // extra pipelines need to consult the archive directly. Returns the mapped
    // blob range for `key`, or throws a missing-variant error naming it: there
    // is no fallback, because libvkop links in no SPIR-V of its own.
    std::pair<const uint32_t *, uint32_t>
    resolve_spv(const std::string &key) const;

    // Recovery of this op's PRIMARY shader when nothing supplied one: looks up
    // shader_key_for(type_, uses_buffer_backend(), fp16_) on the op's own
    // archive pointer, or on ShaderArchive::process_default() when Runtime
    // never set one, and feeds the result through override_spv(). An empty key
    // means a host-only op (Shape, SqueezeUnsqueeze) and returns quietly;
    // otherwise a miss throws the named "carries no built-in shaders …
    // reconvert" error. Lives in Operator.cpp: it needs ShaderKeys.hpp and the
    // complete ShaderArchive type, both of which the header avoids.
    void resolve_primary_spv();

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
    // GPU timestamp profiling (VKOP_SUBMIT_PROF). Assigned once by Runtime
    // (set_prof_query); null when profiling is off (no overhead).
    VkQueryPool prof_query_pool_ = VK_NULL_HANDLE;
    uint32_t prof_query_base_ = 0;
    // Accumulate the fingerprint of ONE DISPATCH -- not of one submit() call.
    // Called from every dispatch site (submit / submit_per_ds / dispatch_on /
    // the branches that bind a secondary pipeline and dispatch directly) when
    // replay_enabled_. That distinction is the whole point: a fingerprint with
    // one entry per submit() misses every dispatch issued outside submit(), and
    // then two recordings of DIFFERENT pipelines look identical. The concrete
    // case: fp16 MatMul at M=3..8 takes the ksplit path here while M=9 goes
    // through coopmat, and the coopmat branch's MMA dispatch is a direct
    // m_cmd_->dispatch, so submit()'s fingerprint saw only the pack pass --
    // making an M=3 ksplit recording and an M=9 coopmat recording compare
    // EQUAL. A dynamic op whose dispatch geometry grows (attention's kv_len)
    // would then be told it is invariant and would replay a stale recording at
    // the wrong size. Nothing ever DECLARED replay to be per-submit; the cache
    // just observed one entry per submit on the paths anyone had exercised.
    //
    // Cheap: one small alloc + handle reads (handles are already materialized
    // in objs_).
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
        dispatch_and_record(width, height, layers, ptr);
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
        dispatch_and_record(width, height, layers, ptr);
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
    // The single place every dispatch goes through, so that the replay
    // fingerprint covers what actually went into the command buffer rather
    // than only the dispatches that came from submit(). Call it immediately
    // before m_cmd_->dispatch(...); it reads the current push constants from
    // pc_ptr_/pc_size_ and the resource handles from objs_, both of which are
    // materialized by the bind/push that precedes every dispatch.
    void dispatch_and_record(int w, int h, int z, const void *pc) {
        m_cmd_->dispatch(w, h, z);
        if (replay_enabled_) {
            record_fingerprint(const_cast<void *>(pc), w, h, z);
        }
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
            if (!objs_[i]) {
                // A null entry means some op bound an input whose storage was
                // already created as the other backend's resource kind; the
                // dereference below would segfault with no hint of which op or
                // which binding, so name both.
                throw std::runtime_error(
                    "vkop: " +
                    (name_.empty()
                         ? std::string(convert_optype_to_string(type_))
                         : name_) +
                    " has no resource for binding " + std::to_string(i) +
                    "; an input was bound for the other backend");
            }
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
