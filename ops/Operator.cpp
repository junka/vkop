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
