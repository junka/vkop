// Copyright 2026 @junka
#ifndef OPS_WHERE_HPP_
#define OPS_WHERE_HPP_

#include "core/Tensor.hpp"
#include "ops/BufferBase.hpp"
#include "ops/Operator.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>

extern "C" {
extern unsigned char buffer_where_spv[];
extern unsigned int buffer_where_spv_len;
extern unsigned char buffer_where_int64_spv[];
extern unsigned int buffer_where_int64_spv_len;
}
namespace vkop {
namespace ops {

// Push-constant for the int64 Where GPU shader (where_int64.comp). Where's
// outputs are rank ≤ 4 in the LLM (rotary shape-meta — e.g. [5] = 1-D
// 5-elem, [4], [3], [2]), so ivec4 shapes suffice and keep the PC small. The
// int64 pipeline builds its own pc_size_ (Where's base pipeline has pc_size 0).
//
// LAYOUT NOTE: the GLSL push-constant block declares the dim fields as ivec4,
// which under std430 (Vulkan's default for push_constant) is 16-byte aligned.
// So outDims lands at byte offset 16 in the shader. The host struct MUST match:
// `rank` (offset 0) is followed by a 3-int pad so outDims starts at offset 16.
// (A single int _pad would put outDims at offset 8 in C++ — a silent mismatch
// that made the shader read garbage outDims → total=0 → only gid=0 wrote,
// producing [1,0,0] instead of [1,9,20].)
struct alignas(16) WhereInt64PC {
    int rank;    // offset 0 — common rank (== out rank, ≤ 4)
    int _pad[3]; // offset 4,8,12 — pad so outDims is 16-aligned (matches ivec4)
    int outDims[4]; // offset 16 — output shape (left-aligned, rank valid,
                    // rest=1)
    int in0Dims[4]; // offset 32 — cond shape (right-aligned to rank)
    int in1Dims[4]; // offset 48 — X shape (right-aligned to rank)
    int in2Dims[4]; // offset 64 — Y shape (right-aligned to rank)
};
static_assert(sizeof(WhereInt64PC) == 80, "WhereInt64PC size (alignas→80)");
static_assert(offsetof(WhereInt64PC, outDims) == 16,
              "outDims must be at offset 16 to match shader ivec4 std430");

// SSBO-only op: no image path. Selects elements from X or Y based on a
// boolean condition buffer.
class Where : public Operator {
  public:
    explicit Where()
        : Operator(OpType::WHERE, buffer_where_spv, buffer_where_spv_len,
                   {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                   0) {}

    // Build the int64-data pipeline lazily on first int64 execute. The int64
    // shader (where_int64.comp) has the same descriptor layout (4× STORAGE)
    // as the fp32 where, but a different (non-zero) PC layout (WhereInt64PC),
    // so the pipeline builds with sizeof(WhereInt64PC) instead of pc_size_.
    void ensure_int64_pipeline() {
        if (pipeline_int64_)
            return;
        bool use_uab = update_after_bind_ &&
                       m_dev_->is_support_descriptor_update_after_bind();
        pipeline_int64_ = std::make_unique<VulkanPipeline>(
            m_dev_->getLogicalDevice(), types_, sizeof(WhereInt64PC),
            reinterpret_cast<const uint32_t *>(buffer_where_int64_spv),
            buffer_where_int64_spv_len, use_uab, required_subgroup_size_);
        for (auto &ds : m_ds_int64_) {
            ds = pipeline_int64_->allocDescriptorSets();
        }
    }

    // When in int64 mode, bind the int64 pipeline + its descriptor sets.
    void submit(void *ptr, int width, int height, int layers) override {
        if (!int64_mode_) {
            Operator::submit(ptr, width, height, layers);
            return;
        }
        if (!m_ds_int64_[m_id_]) {
            m_ds_int64_[m_id_] = pipeline_int64_->allocDescriptorSets();
        }
        fillWriteDescriptorSets(m_ds_int64_[m_id_]);
        pipeline_int64_->updateDescriptorSets(writes_);
        m_cmd_->bind(*pipeline_int64_, m_ds_int64_[m_id_]);
        if (ptr) {
            m_cmd_->push_constants(*pipeline_int64_,
                                   static_cast<uint32_t>(sizeof(WhereInt64PC)),
                                   ptr);
        }
        m_cmd_->dispatch(width, height, layers);
        if (replay_enabled_) {
            record_fingerprint(ptr, width, height, layers);
        }
    }

    // Phase-boundary reset: kept as a no-op override (the int64 path is now
    // GPU-driven and reads no host data, so there is no readback cache to
    // reset). The runtime calls this at phase boundaries unconditionally.
    void invalidate_shape_cache() override {}

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {

        std::vector<int> out_shape = outputs[0]->getShape();
        // The graph's recorded output shape is often stale for the int64
        // shape-meta chain (symbolic dims resolved to a max, not the runtime
        // value). For int64 Where, recompute the broadcasted output shape from
        // the authoritative inputs (cond/X/Y) — the shader broadcasts against
        // this shape.
        if (inputs[0]->dtype() == typeid(int64_t)) {
            std::vector<int> bcast;
            for (const auto &in : inputs) {
                auto s = in->getShape();
                if (s.empty())
                    continue;
                if (bcast.empty()) {
                    bcast = s;
                } else {
                    size_t m = std::max(bcast.size(), s.size());
                    std::vector<int> nb(m, 1);
                    for (size_t i = 0; i < m; ++i) {
                        int a = (i < bcast.size()) ? bcast[bcast.size() - 1 - i]
                                                   : 1;
                        int b = (i < s.size()) ? s[s.size() - 1 - i] : 1;
                        nb[m - 1 - i] = std::max(a, b);
                    }
                    bcast = nb;
                }
            }
            if (!bcast.empty())
                out_shape = bcast;
        }
        if (out_shape.empty()) {
            auto inshape = inputs[0]->getShape();
            out_shape = inshape;
        }

        // int64 Where runs on the GPU (where_int64.comp). All 66 instances are
        // part of the shape meta-chain: cond = Equal int64 (GPU-resident now),
        // X = ConstantOfShape (host-only constant), Y = Concat/Shape int64
        // (GPU-produced). The shader reads int64 cond/X/Y as ivec2[] and
        // broadcasts-selects into the output ivec2[], verbatim 8-byte copies
        // (no 64-bit math — cond != 0 is a low-word check). The output stays
        // GPU-resident for downstream Expand/Unsqueeze, so the Equal->Where->
        // Expand chain never crosses GPU->CPU.
        if (inputs[0]->dtype() == typeid(int64_t)) {
            ensure_int64_pipeline();
            int64_mode_ = true;

            // Output SSBO (int64 as ivec2 — 8 bytes/elem, same stride).
            auto output = core::as_tensor<int64_t>(outputs[0]);
            output->resize(out_shape);
            auto out_buf = output->as_storage_buffer(m_dev_, m_cmd_);
            objs_.emplace_back(out_buf);

            // cond/X/Y input SSBOs. cond (Equal output) and Y (Concat output)
            // are GPU-produced: bind their existing vkobj_ (nullptr cmd — the
            // producer's shaderWriteBarrier / the level boundary makes the
            // writes visible). X is host-only (ConstantOfShape constant): its
            // vkobj_ is allocated by as_storage_buffer but UNINITIALIZED, so
            // upload data_ via copyToGPUDeferred (vkCmdUpdateBuffer into this
            // level's cmd), then an explicit transferBarrier(->SHADER_READ)
            // forces the TRANSFER_WRITE visible to the Where dispatch's
            // COMPUTE read in the SAME command buffer.
            auto cond = core::as_tensor<int64_t>(inputs[0]);
            auto cond_buf = cond->as_storage_buffer(m_dev_, nullptr);
            objs_.emplace_back(cond_buf);
            auto x = core::as_tensor<int64_t>(inputs[1]);
            auto x_buf = x->as_storage_buffer(m_dev_, nullptr);
            x->copyToGPUDeferred(m_cmd_);
            x_buf->transferBarrier(m_cmd_->get(), VK_ACCESS_SHADER_READ_BIT,
                                   x_buf->getSize(), 0);
            objs_.emplace_back(x_buf);
            auto y = core::as_tensor<int64_t>(inputs[2]);
            auto y_buf = y->as_storage_buffer(m_dev_, nullptr);
            objs_.emplace_back(y_buf);

            int total = total_elems(out_shape);
            WhereInt64PC pc{};
            pc.rank = static_cast<int>(out_shape.size());
            fill_dims4(pc.outDims, out_shape);
            fill_dims4_broadcast(pc.in0Dims, inputs[0]->getShape(), pc.rank);
            fill_dims4_broadcast(pc.in1Dims, inputs[1]->getShape(), pc.rank);
            fill_dims4_broadcast(pc.in2Dims, inputs[2]->getShape(), pc.rank);
            submit(&pc, UP_DIV(total, 256), 1, 1);
            // Flush the shader's writes to the output buffer so the next
            // level's GPU reader (Expand/Unsqueeze) sees them.
            if (out_buf) {
                out_buf->shaderWriteBarrier(m_cmd_->get());
            }
            int64_mode_ = false;
            return;
        }

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->num_elements() != total_elems(out_shape)) {
                output->resize(out_shape);
            }
            auto output_buffer = output->as_storage_buffer(m_dev_, m_cmd_);
            objs_.emplace_back(output_buffer);
        });

        for (const auto &in : inputs) {
            dispatch_by_dtype(in->dtype(), [&](auto dummy) {
                using T = decltype(dummy);
                auto input = core::as_tensor<T>(in);
                auto input_buffer = input->as_storage_buffer(m_dev_, m_cmd_);
                objs_.emplace_back(input_buffer);
            });
        }

        auto total_size = std::accumulate(out_shape.begin(), out_shape.end(), 1,
                                          std::multiplies<>());
        submit(nullptr, UP_DIV(total_size, 256), 1, 1);
    }

    std::unique_ptr<VulkanPipeline> pipeline_int64_;
    VkDescriptorSet m_ds_int64_[vkop::kInflight] = {nullptr};
    bool int64_mode_ = false;
};

} // namespace ops
} // namespace vkop
#endif // OPS_WHERE_HPP_
