// Copyright 2025 @junka
#ifndef OPS_MATMUL_HPP_
#define OPS_MATMUL_HPP_

#include "Operator.hpp"
#include "ops/BufferBase.hpp"
#include "ops/PimplFacade.hpp"
extern "C" {
extern unsigned char image_matmul_spv[];
extern unsigned int image_matmul_spv_len;
extern unsigned char image_matmul_nv_spv[];
extern unsigned int image_matmul_nv_spv_len;
extern unsigned char image_matmul_coop_spv[];
extern unsigned int image_matmul_coop_spv_len;
extern unsigned char buffer_matmul_spv[];
extern unsigned int buffer_matmul_spv_len;
extern unsigned char buffer_matmul_fp16_spv[];
extern unsigned int buffer_matmul_fp16_spv_len;
extern unsigned char buffer_matmul_pack_spv[];
extern unsigned int buffer_matmul_pack_spv_len;
}
namespace vkop {
namespace ops {

namespace matmul {

enum class Method {
    BASIC_ARITHMETIC = 0,
    VK_COOPERATE_MATRIX = 1,
    NV_TENSORCORE = 2,
};

struct alignas(16) GpuMatMulParam {
    int M;
    int N;
    int K;
    int C;    // image path: channel count; buffer path: batch count
    int fp32; // 1 = fp32, 0 = fp16
};
} // namespace matmul

// Image (image2DArray NCHW->RGBA) implementation. Uses cooperative matrix
// extensions on supported GPUs (KHR coop / NV tensorcore).
class MatMulImage : public Operator {
  public:
    MatMulImage(const MatMulImage &) = delete;
    MatMulImage &operator=(const MatMulImage &) = delete;
    MatMulImage(MatMulImage &&) = delete;
    MatMulImage &operator=(MatMulImage &&) = delete;

    explicit MatMulImage(int use_tensorcore = 0)
        : Operator(OpType::MATMUL,
                   use_tensorcore == 2
                       ? image_matmul_nv_spv
                       : (use_tensorcore == 1 ? image_matmul_coop_spv
                                              : image_matmul_spv),
                   use_tensorcore == 2
                       ? image_matmul_nv_spv_len
                       : (use_tensorcore == 1 ? image_matmul_coop_spv_len
                                              : image_matmul_spv_len),
                   {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER},
                   sizeof(matmul::GpuMatMulParam)) {
        method_ =
            use_tensorcore == 2
                ? matmul::Method::NV_TENSORCORE
                : (use_tensorcore == 1 ? matmul::Method::VK_COOPERATE_MATRIX
                                       : matmul::Method::BASIC_ARITHMETIC);
    };

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        int chan = inputs[0]->get_channel();
        int m = inputs[0]->get_height();
        int n = inputs[1]->get_width();
        int k = inputs[0]->get_width();
        int rank = inputs[0]->num_dims();
        auto shape = inputs[0]->getShape();
        shape[rank - 1] = n;
        shape[rank - 2] = m;
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto t) {
            using T = decltype(t);
            auto outputptr = core::as_tensor<T>(outputs[0]);
            if (outputptr->size() == 0) {
                outputptr->resize(shape);
            }
            auto output_image = outputptr->as_output_image(m_dev_, m_cmd_);
            objs_.emplace_back(output_image);
        });
        for (const auto &input : inputs) {
            dispatch_by_dtype(input->dtype(), [&](auto t) {
                using T = decltype(t);
                auto inputptr = core::as_tensor<T>(input);
                auto input_image = inputptr->as_input_image(m_dev_, m_cmd_);
                objs_.emplace_back(input_image);
                if (typeid(uint16_t) == typeid(T)) {
                    para_.fp32 = 0;
                } else if (typeid(float) == typeid(T)) {
                    para_.fp32 = 1;
                }
            });
        }
        para_.M = m;
        para_.N = n;
        para_.K = k;
        para_.C = chan;
        if (method_ == matmul::Method::VK_COOPERATE_MATRIX) {
            submit(&para_, UP_DIV(n, 32), UP_DIV(m, 16), UP_DIV(chan, 4));
        } else if (method_ == matmul::Method::NV_TENSORCORE) {
            submit(&para_, UP_DIV(n, 32), UP_DIV(m, 16), UP_DIV(chan, 4));
        } else {
            submit(&para_, UP_DIV(n, 16), UP_DIV(m, 16), UP_DIV(chan, 4));
        }
    }

    matmul::GpuMatMulParam para_;
    matmul::Method method_ = matmul::Method::BASIC_ARITHMETIC;
};

// Buffer (SSBO, compact row-major) implementation. fp32: one thread per
// output element. fp16: 2-pass (reduce to fp32 scratch + pack half2) to
// avoid cross-thread word races when N is odd. The pack pass uses a
// SEPARATE pipeline (buffer_matmul_pack_spv) to avoid Intel ANV
// push-constant interference between dispatches of the same pipeline.
class MatMulBuffer : public BufferFactory {
  public:
    explicit MatMulBuffer(int fp16)
        : BufferFactory(
              OpType::MATMUL, fp16 ? buffer_matmul_fp16_spv : buffer_matmul_spv,
              fp16 ? buffer_matmul_fp16_spv_len : buffer_matmul_spv_len,
              std::vector<VkDescriptorType>{
                  DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                  DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
              sizeof(matmul::GpuMatMulParam), fp16) {
        update_after_bind_ = true;
    }

    void set_runtime_device(
        const std::shared_ptr<VulkanDevice> &dev,
        const std::shared_ptr<VulkanCommandPool> &cmdpool) override {
        BufferFactory::set_runtime_device(dev, cmdpool);
        if (fp16_ != 0 && !pack_pipeline_) {
            bool use_uab = update_after_bind_ &&
                           dev->is_support_descriptor_update_after_bind();
            pack_pipeline_ = std::make_unique<VulkanPipeline>(
                dev->getLogicalDevice(),
                std::vector<VkDescriptorType>{
                    DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                    DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                sizeof(MatMulPackPC),
                reinterpret_cast<const uint32_t *>(buffer_matmul_pack_spv),
                static_cast<int>(buffer_matmul_pack_spv_len), use_uab, 0);
            for (auto &ds : pack_ds_) {
                ds = pack_pipeline_->allocDescriptorSets();
            }
        }
    }

  private:
    struct alignas(16) MatMulPackPC {
        int total;
        int _pad0;
        int _pad1;
        int _pad2;
    };

    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto shape_a = inputs[0]->getShape();
        auto shape_b = inputs[1]->getShape();
        int rank_a = static_cast<int>(shape_a.size());
        int rank_b = static_cast<int>(shape_b.size());

        // ONNX MatMul: A is [..., M, K], B is [..., K, N]. The leading
        // ("batch") dims are broadcast (right-aligned). The reduce shader works
        // on a flat [batch, M, K] / [batch, K, N] layout where `batch` is the
        // product of the leading broadcast dims, so we still compute that
        // scalar product for the dispatch/total. BUT the output *shape* must
        // preserve the un-collapsed leading dims: collapsing to {batch, m, n}
        // loses the rank and breaks downstream shape-meta ops that carry a
        // multi-D view (e.g. a Transpose with a 4-D perm reading a 3-D MatMul
        // output does an OOB shape read -> 0 -> corrupts the whole
        // rotary/attention chain). The output SSBO is still a flat batch*m*n
        // buffer — only the reported logical shape (rank) carries the leading
        // dims. The runtime guarantees both operands share the same
        // leading-broadcast shape by the time MatMul runs (graph shape
        // inference / the preceding Expand materializes it), so the per-dim max
        // is exact and product(batch) is exact.
        if (rank_a < 2 || rank_b < 2) {
            // Degenerate; nothing to do (should not happen for a valid model).
            return;
        }
        int m = shape_a[rank_a - 2];
        int k = shape_a[rank_a - 1];
        int n = shape_b[rank_b - 1];

        int batch = 1;
        int lead_a = rank_a - 2;
        int lead_b = rank_b - 2;
        int lead = std::max(lead_a, lead_b);
        // Leading broadcast dims in natural (left-to-right) order, taking the
        // per-axis max (ONNX broadcast: a dim of 1 matches the other's value).
        // Built left-to-right by indexing from the front of each operand's
        // leading dims.
        std::vector<int> lead_shape;
        lead_shape.reserve(lead);
        for (int i = 0; i < lead; ++i) {
            int da = (i < lead_a) ? shape_a[i] : 1;
            int db = (i < lead_b) ? shape_b[i] : 1;
            int dmax = std::max(da, db);
            batch *= dmax;
            lead_shape.push_back(dmax);
        }

        std::vector<int> out_shape = lead_shape;
        out_shape.push_back(m);
        out_shape.push_back(n);
        if (out_shape.empty()) {
            out_shape = {m, n}; // degenerate guard
        }

        int total = batch * m * n;

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->size() == 0) {
                // Fresh/empty output (recorded shape had 0 sentinels): allocate
                // at the concrete flat size.
                output->resize(out_shape);
            } else if (output->num_elements() == total) {
                // Recycled buffer with the right element count but possibly the
                // wrong rank (e.g. collapsed to {batch,m,n} on a prior round):
                // metadata-only reshape — no GPU realloc. The reshape_view
                // element-count guard is a backstop that silently skips on any
                // mismatch.
                output->reshape_view(out_shape);
            } else {
                // Element count changed (growing kv_len across decode rounds,
                // etc.): reallocate.
                output->resize(out_shape);
            }
            bind_ssbo<T>(outputs[0], /*is_output=*/true);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            bind_ssbo<T>(inputs[0], /*is_output=*/false);
        });
        dispatch_by_dtype(inputs[1]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            bind_ssbo<T>(inputs[1], /*is_output=*/false);
        });

        // fp16 needs a scratch fp32 buffer (binding 3) for the 2-pass
        // reduce->pack. fp32 binds dummy for the 4th slot.
        if (fp16_ != 0) {
            // total may be 0 for a dynamic-shape output that resolved empty
            // (a 0 dim). vkCreateBuffer rejects size 0 with
            // VK_ERROR_INITIALIZATION_FAILED on Intel, so clamp to a minimal
            // 16-byte dummy — the dispatch is 0 threads anyway.
            size_t scratch_bytes = static_cast<size_t>(total) * sizeof(float);
            if (scratch_bytes == 0) {
                scratch_bytes = 16;
            }
            scratch_ = std::make_shared<VulkanBuffer>(
                m_dev_, scratch_bytes,
                STORAGE | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            objs_.emplace_back(scratch_);
        } else {
            objs_.emplace_back(dummy_buffer_);
        }

        // Reduce pass: one thread per output element. Uses the main pipeline.
        para_.M = m;
        para_.N = n;
        para_.K = k;
        para_.C = batch;
        para_.fp32 = (fp16_ != 0) ? 0 : 1;
        submit(&para_, UP_DIV(n, 16), UP_DIV(batch * m, 16), 1);

        if (fp16_ != 0) {
            // Barrier: flush reduce's scratch writes for pack's reads.
            scratch_->shaderWriteBarrier(m_cmd_->get());

            // Pack pass: separate pipeline (no PC interference).
            int nwords = (total + 1) / 2;
            MatMulPackPC pack_pc{};
            pack_pc.total = total;

            // Fill the pack descriptor set with the same objs_.
            fillPackDescriptorSet(pack_ds_[m_id_]);
            pack_pipeline_->updateDescriptorSets(pack_writes_);
            m_cmd_->bind(*pack_pipeline_, pack_ds_[m_id_]);
            m_cmd_->push_constants(*pack_pipeline_, sizeof(MatMulPackPC),
                                   &pack_pc);
            m_cmd_->dispatch(UP_DIV(nwords, 256), 1, 1);
        }
    }

    void fillPackDescriptorSet(VkDescriptorSet ds) {
        pack_writes_.resize(objs_.size());
        for (size_t i = 0; i < objs_.size(); ++i) {
            pack_writes_[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            pack_writes_[i].dstSet = ds;
            pack_writes_[i].dstBinding = static_cast<uint32_t>(i);
            pack_writes_[i].dstArrayElement = 0;
            pack_writes_[i].descriptorCount = 1;
            pack_writes_[i].descriptorType = DESCRIPTOR_TYPE_STORAGE;
            switch (objs_[i]->getResourceType()) {
            case ResourceType::VK_BUFFER:
                pack_writes_[i].pBufferInfo =
                    std::get<VkDescriptorBufferInfo *>(
                        objs_[i]->getDescriptorInfo());
                break;
            default:
                break;
            }
        }
    }

    matmul::GpuMatMulParam para_;
    std::shared_ptr<VulkanBuffer> scratch_;
    std::unique_ptr<VulkanPipeline> pack_pipeline_;
    VkDescriptorSet pack_ds_[vkop::kInflight] = {nullptr};
    std::vector<VkWriteDescriptorSet> pack_writes_;
};

// PIMPL façade: buffer SSBO impl when backend_buffer is set, else image.
class MatMul : public PimplFacade {
  public:
    MatMul(int use_tensorcore, int fp16, bool backend_buffer)
        : PimplFacade(OpType::MATMUL) {
        if (backend_buffer) {
            impl_ = std::make_unique<MatMulBuffer>(fp16);
        } else {
            impl_ = std::make_unique<MatMulImage>(use_tensorcore);
        }
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_MATMUL_HPP_
