// Copyright 2025 @junka
#ifndef OPS_CONCAT_HPP_
#define OPS_CONCAT_HPP_

#include "ops/BufferBase.hpp"
#include "ops/Operator.hpp"
#include "ops/PimplFacade.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
extern "C" {
extern unsigned char image_concat_spv[];
extern unsigned int image_concat_spv_len;
extern unsigned char buffer_concat_spv[];
extern unsigned int buffer_concat_spv_len;
extern unsigned char buffer_concat_fp16_spv[];
extern unsigned int buffer_concat_fp16_spv_len;
extern unsigned char buffer_concat_int64_spv[];
extern unsigned int buffer_concat_int64_spv_len;
}
namespace vkop {
namespace ops {

namespace concat {

struct ConcatParam {
    ivec4 inShape;
    ivec4 outShape;
    ivec4 offset;
    int axis;
};

} // namespace concat

class ConcatImage : public Operator {
  public:
    explicit ConcatImage()
        : Operator(OpType::CONCAT, image_concat_spv, image_concat_spv_len,
                   {
                       VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                       VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                   },
                   sizeof(concat::ConcatParam)) {
        update_after_bind_ = true;
    }

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("axis") != attributes.end()) {
            axis_ = std::stol(attributes.at("axis"));
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        int rank = inputs[0]->num_dims();
        if (axis_ < 0) {
            axis_ = rank + axis_;
        }
        assert(rank >= 3);
        std::vector<int> out_shape = inputs[0]->getShape();
        for (size_t i = 1; i < inputs.size(); i++) {
            auto shape = inputs[i]->getShape();
            out_shape[axis_] += shape[axis_];
        }
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->num_elements() != total_elems(out_shape)) {
                output->resize(out_shape);
            }
            auto output_image = output->as_output_image(m_dev_, m_cmd_);
            objs_.emplace_back(output_image);
        });
        auto output_image = std::dynamic_pointer_cast<VulkanImage>(objs_[0]);
        int offset = 0;
        int submit_count = 0;
        // Count how many submit() calls we'll make to pre-allocate descriptor
        // sets
        for (const auto &in : inputs) {
            auto gpu_axis = axis_ + 4 - rank;
            if (gpu_axis == 2 || (gpu_axis == 1 && offset % 4 != 0)) {
                submit_count++;
            }
            // For other axes we use copyImageToImage, no submit
            offset += in->get_channel(); // approximate, just for counting
        }
        std::vector<VkDescriptorSet> pass_ds(submit_count > 0 ? submit_count
                                                              : 1);
        for (int i = 0; i < static_cast<int>(pass_ds.size()); i++) {
            pass_ds[i] = allocPassDescriptorSet();
        }
        offset = 0;
        int ds_idx = 0;
        for (const auto &in : inputs) {
            dispatch_by_dtype(in->dtype(), [&](auto dummy) {
                using T = decltype(dummy);
                auto input = core::as_tensor<T>(in);
                auto input_image = input->as_input_image(m_dev_, m_cmd_);
                auto in_gpu_shape = input->getGPUShape();
                if (axis_ + 4 - rank == 1 && (offset % 4 == 0)) {
                    // assume they can be divided by 4
                    input_image->transferReadBarrier(m_cmd_->get());
                    output_image->copyImageToImage(m_cmd_->get(), input_image,
                                                   {0, 0, 0}, offset / 4);
                    input_image->readBarrier(m_cmd_->get());
                    offset += in->get_channel();
                } else if (axis_ + 4 - rank == 0) {
                    input_image->transferReadBarrier(m_cmd_->get());
                    output_image->copyImageToImage(m_cmd_->get(), input_image,
                                                   {0, offset, 0}, 0);
                    input_image->readBarrier(m_cmd_->get());
                    offset += in_gpu_shape[1];
                } else if (axis_ + 4 - rank == 3) {
                    input_image->transferReadBarrier(m_cmd_->get());
                    output_image->copyImageToImage(m_cmd_->get(), input_image,
                                                   {offset, 0, 0}, 0);
                    input_image->readBarrier(m_cmd_->get());
                    offset += in_gpu_shape[0];
                } else if (axis_ + 4 - rank == 2) {
                    if (objs_.size() == 2) {
                        objs_.pop_back();
                    }
                    objs_.emplace_back(input_image);
                    concat::ConcatParam para = {};
                    auto input_shape = in->getShape();
                    in->get_shape(para.inShape);
                    outputs[0]->get_shape(para.outShape);
                    para.offset[0] = 0;
                    para.offset[1] = 0;
                    para.offset[2] = offset;
                    para.offset[3] = 0;
                    para.axis = 2;
                    offset += in->get_height();
                    submit_per_ds(pass_ds[ds_idx++], &para,
                                  UP_DIV(in_gpu_shape[0], 16),
                                  UP_DIV(in_gpu_shape[1], 16), in_gpu_shape[2]);
                } else if (axis_ + 4 - rank == 1) {
                    if (objs_.size() == 2) {
                        objs_.pop_back();
                    }
                    objs_.emplace_back(input_image);
                    auto input_shape = in->getShape();
                    concat::ConcatParam para = {};
                    in->get_shape(para.inShape);
                    outputs[0]->get_shape(para.outShape);
                    para.offset[0] = 0;
                    para.offset[1] = offset;
                    para.offset[2] = 0;
                    para.offset[3] = 0;
                    para.axis = 1;
                    offset += in->get_channel();
                    submit_per_ds(pass_ds[ds_idx++], &para,
                                  UP_DIV(in_gpu_shape[0], 16),
                                  UP_DIV(in_gpu_shape[1], 16), in_gpu_shape[2]);
                }
            });
        }
        for (int i = 0; i < static_cast<int>(pass_ds.size()); i++) {
            freePassDescriptorSet(pass_ds[i]);
        }
    }

    int axis_ = 1;
};
// Concat buffer op (fp32). Per-input submit: bind this input + the shared
// output, dispatch UP_DIV(in_total, 256).
class ConcatBuffer : public BufferFactory {
  public:
    explicit ConcatBuffer(int fp16)
        : BufferFactory(
              OpType::CONCAT, fp16 ? buffer_concat_fp16_spv : buffer_concat_spv,
              fp16 ? buffer_concat_fp16_spv_len : buffer_concat_spv_len,
              {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
              sizeof(ConcatPC), fp16) {
        update_after_bind_ = true;
    }

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("axis") != attributes.end()) {
            axis_ = std::stol(attributes.at("axis"));
        }
    }

  private:
    // Build the int64-data pipeline lazily on first int64 execute. The int64
    // shader (concat_int64.comp) has the same descriptor layout (2x STORAGE)
    // and PC layout (ConcatPC) as the fp32 concat, so it builds from the same
    // types_/pc_size_; only the spv differs.
    void ensure_int64_pipeline() {
        if (pipeline_int64_)
            return;
        bool use_uab = update_after_bind_ &&
                       m_dev_->is_support_descriptor_update_after_bind();
        pipeline_int64_ = std::make_unique<VulkanPipeline>(
            m_dev_->getLogicalDevice(), types_, pc_size_,
            reinterpret_cast<const uint32_t *>(buffer_concat_int64_spv),
            buffer_concat_int64_spv_len, use_uab, required_subgroup_size_);
    }

    // Route the int64 path through the int64 GPU shader (concat_int64.comp).
    // Same per-input submit pattern as the fp32 path (bind output + one input,
    // dispatch that input's element count), but data/output are int64 SSBOs
    // (bound as ivec2[] in-shader). Eliminates the N x copyToCPU + copyToGPU
    // sync stalls (~1940ms/round, the #1 decode bottleneck). The output is
    // GPU-resident only (data_ NOT populated); downstream int64 CPU readers
    // (Reshape/Slice/Range/Split) call copyToCPU themselves to read it back,
    // the same way they consume the int64 Gather shader's GPU output.
    void
    gpuConcatInt64(const std::vector<std::shared_ptr<core::ITensor>> &inputs,
                   const std::vector<std::shared_ptr<core::ITensor>> &outputs,
                   const std::vector<int> &out_shape) {
        ensure_int64_pipeline();
        int rank = static_cast<int>(out_shape.size());

        auto output = core::as_tensor<int64_t>(outputs[0]);
        if (output->num_elements() != total_elems(out_shape)) {
            output->resize(out_shape);
        }
        auto out_buf = output->as_storage_buffer(m_dev_, m_cmd_);

        int n_inputs = static_cast<int>(inputs.size());
        std::vector<VkDescriptorSet> pass_ds(n_inputs);
        for (int i = 0; i < n_inputs; ++i) {
            pass_ds[i] = pipeline_int64_->allocDescriptorSets();
        }

        int offset = 0;
        for (int i = 0; i < n_inputs; ++i) {
            auto in_shape = inputs[i]->getShape();
            int in_total = total_elems(in_shape);
            auto input = core::as_tensor<int64_t>(inputs[i]);
            auto in_buf = input->as_storage_buffer(m_dev_, nullptr);

            objs_.clear();
            objs_.emplace_back(out_buf);
            objs_.emplace_back(in_buf);

            ConcatPC pc{};
            pc.axis = axis_;
            pc.rank = rank;
            fill_dims(pc.inDims, in_shape);
            fill_dims(pc.outDims, out_shape);
            pc.offset = offset;

            fillWriteDescriptorSets(pass_ds[i]);
            pipeline_int64_->updateDescriptorSets(writes_);
            m_cmd_->bind(*pipeline_int64_, pass_ds[i]);
            m_cmd_->push_constants(*pipeline_int64_,
                                   static_cast<uint32_t>(pc_size_), &pc);
            m_cmd_->dispatch(UP_DIV(in_total, 256), 1, 1);
            if (replay_enabled_) {
                record_fingerprint(&pc, UP_DIV(in_total, 256), 1, 1);
            }
            offset += in_shape[axis_];
        }
        for (int i = 0; i < n_inputs; ++i) {
            pipeline_int64_->freeDescriptorSets(pass_ds[i]);
        }
    }

    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        int rank = static_cast<int>(inputs[0]->num_dims());
        if (axis_ < 0) {
            axis_ += rank;
        }
        std::vector<int> out_shape = inputs[0]->getShape();
        for (size_t i = 1; i < inputs.size(); ++i) {
            auto s = inputs[i]->getShape();
            out_shape[axis_] += s[axis_];
        }

        // int64 concat: GPU shader path (concat_int64.comp). Previously this
        // was a synchronous CPU path (copyToCPU per input + host strided-copy
        // loop + copyToGPU = N+1 sync stalls, ~1940ms/round — the #1 decode
        // bottleneck after Shape/Gather were GPU-ified). The GPU dispatch
        // records into the level command buffer with no stall.
        if (inputs[0]->dtype() == typeid(int64_t)) {
            gpuConcatInt64(inputs, outputs, out_shape);
            return;
        }

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->num_elements() != total_elems(out_shape)) {
                output->resize(out_shape);
            }
            bind_ssbo<T>(outputs[0], /*is_output=*/true);
        });
        // keep the output obj at objs_[0]; each input submit replaces objs_[1]
        auto out_buf = std::dynamic_pointer_cast<VulkanBuffer>(objs_[0]);

        int n_inputs = static_cast<int>(inputs.size());
        std::vector<VkDescriptorSet> pass_ds(n_inputs);
        for (int i = 0; i < n_inputs; ++i) {
            pass_ds[i] = allocPassDescriptorSet();
        }

        int offset = 0;
        for (int i = 0; i < n_inputs; ++i) {
            int in_total = total_elems(inputs[i]->getShape());
            dispatch_by_dtype(inputs[i]->dtype(), [&](auto dummy) {
                using T = decltype(dummy);
                // replace objs_[1] with this input's buffer
                if (objs_.size() > 1) {
                    objs_[1] = nullptr;
                }
                auto in_buf = bind_ssbo<T>(inputs[i], /*is_output=*/false);
                // re-seat objs_[1] (bind_ssbo appends; pop the output slot)
                (void)in_buf;
            });
            // bind_ssbo appended the input at objs_.back(); we want the
            // layout [output, input]. Trim back to exactly 2 in the right
            // order.
            std::vector<std::shared_ptr<VulkanResource>> ordered = {
                out_buf, objs_.back()};
            objs_ = ordered;

            ConcatPC pc{};
            pc.axis = axis_;
            pc.rank = rank;
            fill_dims(pc.inDims, inputs[i]->getShape());
            fill_dims(pc.outDims, out_shape);
            pc.offset = offset;
            // fp16 packs two elements per uint word; dispatch one thread per
            // input word (the fp16 shader writes each output word once).
            int nthreads = (fp16_ != 0) ? (in_total + 1) / 2 : in_total;
            submit_per_ds(pass_ds[i], &pc, UP_DIV(nthreads, 256), 1, 1);
            offset += static_cast<int>(inputs[i]->getShape()[axis_]);
        }
        for (int i = 0; i < n_inputs; ++i) {
            freePassDescriptorSet(pass_ds[i]);
        }
    }

    int axis_ = 1;

    // int64-data GPU path state. Lazily built on first int64 execute.
    std::unique_ptr<VulkanPipeline> pipeline_int64_;
};

// PIMPL façade: buffer SSBO impl when backend_buffer is set, else image.
class Concat : public PimplFacade {
  public:
    Concat(int fp16, bool backend_buffer) : PimplFacade(OpType::CONCAT) {
        impl_ = backend_buffer ? std::unique_ptr<Operator>(
                                     std::make_unique<ConcatBuffer>(fp16))
                               : std::make_unique<ConcatImage>();
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_CONCAT_HPP_
