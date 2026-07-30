// Copyright 2025 @junka
#ifndef OPS_CONCAT_HPP_
#define OPS_CONCAT_HPP_

#include "ops/BufferBase.hpp"
#include "ops/Operator.hpp"
#include "ops/PimplFacade.hpp"
extern "C" {
extern unsigned char image_concat_spv[];
extern unsigned int image_concat_spv_len;
extern unsigned char buffer_concat_spv[];
extern unsigned int buffer_concat_spv_len;
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
            if (output->size() == 0) {
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
            printf("concat %d: offset %d\n", axis_ + 4 - rank, offset);
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
    explicit ConcatBuffer(int /*fp16*/)
        : BufferFactory(OpType::CONCAT, buffer_concat_spv,
                        buffer_concat_spv_len,
                        {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                        sizeof(ConcatPC)) {
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
        int rank = static_cast<int>(inputs[0]->num_dims());
        if (axis_ < 0) {
            axis_ += rank;
        }
        std::vector<int> out_shape = inputs[0]->getShape();
        for (size_t i = 1; i < inputs.size(); ++i) {
            auto s = inputs[i]->getShape();
            out_shape[axis_] += s[axis_];
        }

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->size() == 0) {
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
            fill_dims(pc.outDims, out_shape);
            pc.offset = offset;
            submit_per_ds(pass_ds[i], &pc, UP_DIV(in_total, 256), 1, 1);
            offset += static_cast<int>(inputs[i]->getShape()[axis_]);
        }
        for (int i = 0; i < n_inputs; ++i) {
            freePassDescriptorSet(pass_ds[i]);
        }
    }

    int axis_ = 1;
};

// PIMPL façade: buffer SSBO impl when backend_buffer is set, else image.
class Concat : public PimplFacade {
  public:
    Concat(int /*fp16*/, bool backend_buffer) : PimplFacade(OpType::CONCAT) {
        impl_ =
            backend_buffer
                ? std::unique_ptr<Operator>(std::make_unique<ConcatBuffer>(0))
                : std::make_unique<ConcatImage>();
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_CONCAT_HPP_
