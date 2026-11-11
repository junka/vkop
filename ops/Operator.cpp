// Copyright 2025 @junka
// Out-of-line definitions for Operator's methods that need the full
// ShaderArchive type. Operator.hpp forward-declares the class so ops headers
// don't have to pull in model/load.hpp; anything that dereferences the
// archive pointer lives here.

#include "ops/Operator.hpp"

#include <cstdio>
#include <cstdlib>

#include "core/ShaderArchive.hpp"
#include "core/ShaderKeys.hpp"
#include "vulkan/VulkanBuffer.hpp"

namespace vkop {
namespace ops {

namespace {
// The archive this op should consult: the pointer Runtime set when it built
// the op from a model file, otherwise the process-wide one dev binaries
// install (include/dev_shaders.hpp). Null when neither applies -- an op built
// with no model and no dev archive, which then fails by name.
const vkop::core::ShaderArchive *
effective_archive(const vkop::core::ShaderArchive *own) {
    return own ? own : vkop::core::ShaderArchive::process_default();
}

// The failure every unresolved lookup produces. libvkop links in no SPIR-V, so
// the only fix is a model whose archive covers the key.
[[noreturn]] void missing_shader(const std::string &name,
                                 const std::string &key, const char *what) {
    throw std::runtime_error(
        "vkop: " + name + " needs the shader " + what + " '" + key +
        "' and nothing supplied it: this libvkop carries no built-in SPIR-V "
        "and the archive the .vkopbin carries does not have that key. "
        "Reconvert the model with onnx2vkop -- it embeds every key its kernel "
        "plan lists -- against a complete vkop_shaders.bin "
        "(--master-archive).");
}
} // namespace

// VKOP_BDA gates every shader-side variant (the allocation flag and buffer
// usage are automatic). Resolved once: a driver that reports BDA but
// miscompiles PhysicalStorageBuffer cannot change model behaviour until it has
// been validated, and VKOP_NO_BDA remains the global kill switch.
static bool bda_shaders_enabled() {
    static const bool enabled = []() {
        const char *e = std::getenv("VKOP_BDA");
        return e != nullptr && e[0] == '1';
    }();
    return enabled;
}

void Operator::resolve_primary_spv() {
    std::string key = core::shader_key_for(type_, uses_buffer_backend(), fp16_);
    if (key.empty()) {
        // Host-only op (Shape, SqueezeUnsqueeze): no pipeline to build.
        return;
    }
    const auto *archive = effective_archive(shader_archive_);
    if (archive) {
        // The norm/softmax/attention modules ship a -DSUBGROUP build under
        // <key>_sg (workgroup reduce via subgroupAdd/Max instead of the
        // 256-slot tree). Runtime swaps the key in for models it loads; this is
        // the path for ops built straight through create_from_type() (test
        // fixtures, example drivers). Both the missing feature and the missing
        // key (an older archive) mean the base module, so nothing here can
        // fail.
        if (m_dev_ && m_dev_->supportsSubgroupArithmetic()) {
            const std::string sg_key = key + "_sg";
            if (archive->lookup(sg_key).first != nullptr) {
                key = sg_key;
            }
        }
        auto [spv, len] = archive->lookup(key);
        if (spv) {
            override_spv(spv, len);
            if (std::getenv("VKOP_SHADER_DEBUG")) {
                std::fprintf(stderr,
                             "[shaders] self-resolved key=%s len=%u op=%s\n",
                             key.c_str(), len, name_.c_str());
            }
            return;
        }
        if (std::getenv("VKOP_SHADER_DEBUG")) {
            std::fprintf(stderr,
                         "[shaders] archive has no key=%s for op=%s -- this "
                         "archive does not cover the op\n",
                         key.c_str(), name_.c_str());
        }
    }
    missing_shader(name_.empty() ? std::string(convert_optype_to_string(type_))
                                 : name_,
                   key, "variant");
}

std::pair<const uint32_t *, uint32_t>
Operator::resolve_spv(const std::string &key) const {
    const auto *archive = effective_archive(shader_archive_);
    if (archive) {
        auto r = archive->lookup(key);
        if (r.first) {
            return r;
        }
    }
    missing_shader(name_.empty() ? std::string(convert_optype_to_string(type_))
                                 : name_,
                   key, "variant");
}

bool Operator::archive_has_spv(const std::string &key) const {
    const auto *archive = effective_archive(shader_archive_);
    if (!archive) {
        return false;
    }
    return archive->lookup(key).first != nullptr;
}

void Operator::try_build_bda_pipeline(VkPipelineCache cache) {
    // Step 2 rollout gate: the host-side capability (allocation flag, buffer
    // usage) is automatic, but a shader variant is only selected when the
    // caller opts in. A driver that reports BDA yet miscompiles
    // PhysicalStorageBuffer therefore cannot change model behaviour until
    // that driver has been validated; VKOP_NO_BDA remains the global kill
    // switch.
    if (!bda_shaders_enabled() || bda_pipeline_ || !supports_bda_variant() ||
        !m_dev_ || !m_dev_->is_support_buffer_device_address()) {
        return;
    }
    // Packed fp16 kernels load uint words. Descriptor offsets (the only way a
    // zero-copy view is exposed) are multiples of
    // minStorageBufferOffsetAlignment, so require that to be word-aligned;
    // otherwise a 2-byte view offset would become an unaligned 32-bit load.
    // Devices that fail this simply keep the descriptor variant.
    if (m_dev_->minStorageBufferOffsetAlignment() % 4 != 0) {
        return;
    }
    const size_t bpc = bda_pc_size();
    // Where's fp32 shader has no push constants; a table op with pc_size_ 0 is
    // legitimate (submit pushes nothing). Direct-address ops still need > 0.
    if ((bpc == 0 && bda_address_table_slots() == 0) || bpc > 128) {
        return;
    }
    const auto *archive = effective_archive(shader_archive_);
    if (!archive) {
        return;
    }
    // "<base>_bda" (fp32) / "<base>_bda_fp16" (dual builds), matching the
    // archive keys CMake produces for the -DBDA build of buffer/<op>.comp.
    const std::string base = core::shader_key_for(type_, true, 0);
    if (base.empty()) {
        return;
    }
    std::string key = base + "_bda";
    if (fp16_ != 0) {
        const std::string fp16_base = core::shader_key_for(type_, true, 1);
        if (fp16_base.empty() || fp16_base == base) {
            // An op with no dual fp16 build normally has no distinct fp16
            // kernel to derive a variant from; running the fp32 BDA module on
            // packed half data would misread every word. Cast/QDQ are the
            // exception: one module handles both dtypes (the PC's mode/fmt
            // selects the layout), so they opt in to the single BDA key.
            if (!bda_variant_ignores_fp16()) {
                return;
            }
        } else {
            key += "_fp16";
        }
    }
    // The runtime prefers <base>[_fp16]_sg when the device advertises subgroup
    // arithmetic. Offer the matching BDA module so enabling BDA does not
    // silently drop the subgroup reduction (attention/norm/softmax/matmul).
    if (m_dev_->supportsSubgroupArithmetic()) {
        const std::string sg_base =
            fp16_ != 0 ? base + "_fp16_sg" : base + "_sg";
        const std::string sg_bda =
            fp16_ != 0 ? base + "_fp16_sg_bda" : base + "_sg_bda";
        if (archive->lookup(sg_bda).first != nullptr) {
            key = sg_bda;
        } else if (std::getenv("VKOP_BDA_DEBUG") &&
                   archive->lookup(sg_base).first != nullptr) {
            // Only worth warning about when the descriptor path DOES take the
            // subgroup module: that is a real (perf) downgrade, not an op with
            // no _sg build at all.
            std::fprintf(stderr,
                         "[bda] WARNING: %s exists but %s does not; using %s\n",
                         sg_base.c_str(), sg_bda.c_str(), key.c_str());
        }
    }
    auto [spv, len] = archive->lookup(key);
    if (!spv) {
        return;
    }
    const std::string label =
        name_.empty() ? convert_optype_to_string(type_) : name_;
    try {
        const size_t table_slots = bda_address_table_slots();
        std::vector<VkDescriptorType> bda_types = bda_descriptor_types();
        if (table_slots > 0 && bda_types.empty()) {
            bda_types.push_back(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        }
        const bool bind_table = !bda_types.empty();
        bda_pipeline_ = std::make_unique<VulkanPipeline>(
            m_dev_->getLogicalDevice(), std::move(bda_types), bpc,
            reinterpret_cast<const uint32_t *>(spv), static_cast<int>(len),
            /*update_after_bind=*/false, required_subgroup_size_, cache);
        bda_pc_size_ = bpc;
        if (bind_table) {
            bda_ds_ = bda_pipeline_->allocDescriptorSets();
            if (table_slots > 0) {
                // One uint64 per data binding, DEVICE_LOCAL + TRANSFER_DST:
                // fill_bda_table rewrites it before every dispatch.
                bda_table_buf_ = std::make_shared<VulkanBuffer>(
                    m_dev_, table_slots * sizeof(VkDeviceAddress),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                auto *info = std::get<VkDescriptorBufferInfo *>(
                    bda_table_buf_->getDescriptorInfo());
                VkWriteDescriptorSet write{};
                write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write.dstSet = bda_ds_;
                write.dstBinding = 0;
                write.dstArrayElement = 0;
                write.descriptorCount = 1;
                write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                write.pBufferInfo = info;
                bda_pipeline_->updateDescriptorSets({write});
            }
        }
        if (std::getenv("VKOP_BDA_DEBUG")) {
            std::fprintf(stderr, "[bda] op=%s key=%s pc=%zu table_ds=%d\n",
                         label.c_str(), key.c_str(), bpc, bind_table ? 1 : 0);
        }
    } catch (const std::exception &e) {
        // A driver that advertises BDA but rejects this module (e.g. an
        // unimplemented PhysicalStorageBuffer path in the shader compiler)
        // must not take the op down: the descriptor pipeline still works.
        bda_pipeline_.reset();
        bda_pc_size_ = 0;
        // Drop any table the failed attempt allocated: bda_ready() is false so
        // nothing would use it, but leaving it bound to a dead set only leaks
        // pool memory on a retry-less op.
        bda_ds_ = VK_NULL_HANDLE;
        bda_table_buf_.reset();
        std::fprintf(
            stderr,
            "[bda] pipeline for %s (%s) failed: %s; using descriptors\n",
            label.c_str(), key.c_str(), e.what());
    }
}

// Rewrite the generic address table from the first N objs_ entries. Returns
// false when any slot is missing or belongs to a host-imported buffer (no
// device address), in which case the caller keeps the descriptor pipeline.
bool Operator::fill_bda_table() {
    const size_t n = bda_address_table_slots();
    if (n == 0 || !bda_table_buf_ || !m_cmd_) {
        return false;
    }
    std::vector<VkDeviceAddress> addrs(n, 0);
    for (size_t i = 0; i < n; ++i) {
        if (i >= objs_.size() || !objs_[i]) {
            return false;
        }
        auto buf = std::dynamic_pointer_cast<VulkanBuffer>(objs_[i]);
        if (!buf || !buf->hasDeviceAddress()) {
            return false;
        }
        addrs[i] = buf->getDeviceAddress();
    }
    // Set the tracked access to SHADER_READ *before* the update so
    // updateBuffer's post-update barrier is TRANSFER_WRITE -> SHADER_READ.
    // On a fresh table m_access_ is 0, which would restore dstAccess=0 and
    // leave the vkCmdUpdateBuffer write without a visibility edge to the
    // dispatch's read.
    const VkDeviceSize bytes = n * sizeof(VkDeviceAddress);
    bda_table_buf_->readBarrier(m_cmd_->get(), bytes, 0);
    bda_table_buf_->updateBuffer(m_cmd_->get(), addrs.data(), bytes);
    return true;
}

bool Operator::build_bda_table_module(const std::string &key, size_t pc_bytes,
                                      size_t slots, BdaTableModule &m,
                                      VkPipelineCache cache) {
    m.pipeline.reset();
    m.ds = VK_NULL_HANDLE;
    m.table.reset();
    m.pc_bytes = 0;
    m.slots = 0;
    if (!bda_shaders_enabled() || !m_dev_ ||
        !m_dev_->is_support_buffer_device_address() ||
        m_dev_->minStorageBufferOffsetAlignment() % 4 != 0 || pc_bytes == 0 ||
        pc_bytes > 128 || slots == 0) {
        return false;
    }
    const auto *archive = effective_archive(shader_archive_);
    if (!archive) {
        return false;
    }
    auto [spv, len] = archive->lookup(key);
    if (!spv) {
        return false;
    }
    try {
        m.pipeline = std::make_unique<VulkanPipeline>(
            m_dev_->getLogicalDevice(),
            std::vector<VkDescriptorType>{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
            pc_bytes, reinterpret_cast<const uint32_t *>(spv),
            static_cast<int>(len), /*update_after_bind=*/false,
            required_subgroup_size_, cache);
        m.ds = m.pipeline->allocDescriptorSets();
        m.table = std::make_shared<VulkanBuffer>(
            m_dev_, slots * sizeof(VkDeviceAddress),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        auto *info =
            std::get<VkDescriptorBufferInfo *>(m.table->getDescriptorInfo());
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = m.ds;
        write.dstBinding = 0;
        write.dstArrayElement = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = info;
        m.pipeline->updateDescriptorSets({write});
        m.pc_bytes = pc_bytes;
        m.slots = slots;
        if (std::getenv("VKOP_BDA_DEBUG")) {
            std::fprintf(stderr, "[bda] module key=%s pc=%zu slots=%zu\n",
                         key.c_str(), pc_bytes, slots);
        }
        return true;
    } catch (const std::exception &e) {
        std::fprintf(stderr,
                     "[bda] secondary module %s failed: %s; using "
                     "descriptors\n",
                     key.c_str(), e.what());
        m.pipeline.reset();
        m.ds = VK_NULL_HANDLE;
        m.table.reset();
        return false;
    }
}

bool Operator::fill_bda_table_module(BdaTableModule &m) {
    return fill_bda_table_module(m, objs_);
}

bool Operator::fill_bda_table_module(
    BdaTableModule &m,
    const std::vector<std::shared_ptr<VulkanResource>> &bufs) {
    if (!m.table || m.ds == VK_NULL_HANDLE || m.slots == 0 || !m_cmd_) {
        return false;
    }
    std::vector<VkDeviceAddress> addrs(m.slots, 0);
    for (size_t i = 0; i < m.slots; ++i) {
        if (i >= bufs.size() || !bufs[i]) {
            return false;
        }
        auto buf = std::dynamic_pointer_cast<VulkanBuffer>(bufs[i]);
        if (!buf || !buf->hasDeviceAddress()) {
            return false;
        }
        addrs[i] = buf->getDeviceAddress();
    }
    const VkDeviceSize bytes = m.slots * sizeof(VkDeviceAddress);
    // Pre-set SHADER_READ so updateBuffer's restore barrier is a real
    // TRANSFER_WRITE -> SHADER_READ visibility edge (see fill_bda_table).
    m.table->readBarrier(m_cmd_->get(), bytes, 0);
    m.table->updateBuffer(m_cmd_->get(), addrs.data(), bytes);
    return true;
}

bool Operator::submit_bda_table_module(BdaTableModule &m, const void *pc,
                                       int width, int height, int layers) {
    return submit_bda_table_module(m, objs_, pc, width, height, layers);
}

bool Operator::submit_bda_table_module(
    BdaTableModule &m,
    const std::vector<std::shared_ptr<VulkanResource>> &bufs, const void *pc,
    int width, int height, int layers) {
    if (!m.pipeline || m.ds == VK_NULL_HANDLE ||
        !fill_bda_table_module(m, bufs)) {
        return false;
    }
    m_cmd_->bind(*m.pipeline, m.ds);
    if (pc) {
        m_cmd_->push_constants(*m.pipeline,
                               static_cast<uint32_t>(m.pc_bytes), pc);
    }
    bda_dispatched_ = true;
    dispatch_and_record(width, height, layers, pc, m.pc_bytes);
    return true;
}

// Defined out of line on purpose: gcc will not emit a weak copy of a large
// inline virtual for every TU that includes this header, so a harness that
// links only libvkop.a would otherwise get a silent no-op object from its own
// TU. Exactly one emitted copy is the point (onExecute is overridden by ops
// that need different dispatch, and the base body is not small).
void Operator::onExecute(
    const std::vector<std::shared_ptr<core::ITensor>> &inputs,
    const std::vector<std::shared_ptr<core::ITensor>> &outputs, int id) {
    if (!m_cmd_ && !graph_cmd_) {
        m_cmd_ = std::make_shared<VulkanCommandBuffer>(m_cmdpool_, id);
    }
    m_id_ = id;

    // Tag every dispatch this op records with its graph node name, so a
    // RenderDoc capture reads "level 437 / Matmul /q_proj" instead of
    // anonymous vkCmdDispatch entries.
    if (VulkanDebug::enabled()) {
        std::string label =
            name_.empty() ? convert_optype_to_string(type_) : name_;
        auto &dst = graph_cmd_ ? graph_cmd_ : m_cmd_;
        if (dst)
            dst->set_debug_label(std::move(label));
    }

    // Graph-submit mode: record into the Runtime-owned shared command
    // buffer. m_cmd_ is temporarily repointed at it so the op's own
    // helper methods and m_cmd_->get() calls (barriers, copies, binds,
    // dispatches) land there too; begin()/end()/submit() belong to the
    // Runtime in this mode. The replay cache is skipped: a recording shared
    // with other ops cannot be replayed per op. The op's own buffer (if it
    // has one from an earlier non-graph round) is restored afterwards so no
    // per-op buffer is ever allocated while the mode is on.
    if (graph_cmd_) {
        auto saved_cmd = m_cmd_;
        m_cmd_ = graph_cmd_;
        // objs_ is rebuilt by execute()'s bind helpers and read back by
        // fillWriteDescriptorSets, which indexes it from 0. It MUST start
        // empty for every recording, exactly as the per-op path guarantees
        // via the objs_.clear() further down. Without this reset the second
        // and later rounds kept the FIRST round's entries at [0, n) and
        // bound descriptors for stale resources — on the LLM decode path
        // that silently produced all-zero logits (prefill round 0 is
        // correct because objs_ is still empty at that point).
        objs_.clear();
        fp_passes_.clear();
        if (prof_query_pool_ != VK_NULL_HANDLE) {
            m_cmd_->writeTimestampBegin(prof_query_pool_, prof_query_base_);
        }
        execute(inputs, outputs);
        if (prof_query_pool_ != VK_NULL_HANDLE) {
            m_cmd_->writeTimestampEnd(prof_query_pool_, prof_query_base_ + 1);
        }
        m_cmd_ = saved_cmd;
        return;
    }

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
    // GPU timestamp: TOP_OF_PIPE = the moment this cmd reaches the front
    // of the queue (before its work starts). Written inside begin/end so
    // CACHED-replay recordings carry it forward.
    if (prof_query_pool_ != VK_NULL_HANDLE) {
        m_cmd_->writeTimestampBegin(prof_query_pool_, prof_query_base_);
    }
    execute(inputs, outputs);
    if (prof_query_pool_ != VK_NULL_HANDLE) {
        // BOTTOM_OF_PIPE = after all prior work in this cmd completes.
        m_cmd_->writeTimestampEnd(prof_query_pool_, prof_query_base_ + 1);
    }
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

} // namespace ops
} // namespace vkop
