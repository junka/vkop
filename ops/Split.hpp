// Copyright 2025 @junka
#ifndef OPS_SPLIT_HPP_
#define OPS_SPLIT_HPP_

#include "ops/BufferBase.hpp"
#include "ops/Operator.hpp"
#include "ops/PimplFacade.hpp"
namespace vkop {
namespace ops {
namespace split {

struct GpuSplitParam {
    ivec4 inShape;
    ivec4 outShape;
    int split; // offset
    int axis;
};
} // namespace split

class SplitImage : public Operator {
  public:
    explicit SplitImage()
        : Operator(OpType::SPLIT,
                   {
                       VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                       VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                   },
                   sizeof(split::GpuSplitParam)) {
        update_after_bind_ = true;
        para_.split = 0;
    }

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("axis") != attributes.end()) {
            para_.axis = std::stoi(attributes.at("axis"));
        }
        if (attributes.find("num_outputs") != attributes.end()) {
            num_outputs_ = std::stoi(attributes.at("num_outputs"));
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        int rank = inputs[0]->num_dims();
        if (para_.axis < 0) {
            para_.axis += rank;
        }
        if (num_outputs_ == 0) {
            num_outputs_ = static_cast<int>(outputs.size());
        }
        int tosplit = inputs[0]->getShape()[para_.axis];
        std::vector<std::shared_ptr<VulkanResource>> output_images;
        std::vector<int64_t> split_vec;
        if (inputs.size() > 1) {
            auto split = core::as_tensor<int64_t>(inputs[1]);
            // The split input may be GPU-resident only (produced by the int64
            // Concat/Gather GPU shader in a prior level); read back before the
            // direct (*split)[i] access. No-op if host-only.
            split->copyToCPU(m_cmdpool_);
            split_vec.resize(split->num_elements());
            for (size_t i = 0; i < split_vec.size(); i++) {
                split_vec[i] = (*split)[i];
            }
        } else {
            split_vec.resize(num_outputs_);
            for (int i = 1; i < num_outputs_; i++) {
                split_vec[i] = tosplit / num_outputs_;
            }
        }
        for (int i = 0; i < num_outputs_; i++) {
            dispatch_by_dtype(outputs[i]->dtype(), [&](auto dummy) {
                using T = decltype(dummy);
                auto output = core::as_tensor<T>(outputs[i]);
                auto shape = inputs[0]->getShape();
                shape[para_.axis] = static_cast<int>(split_vec[i]);
                if (output->num_elements() != total_elems(shape)) {
                    output->resize(shape);
                }
                auto output_image = output->as_output_image(m_dev_, m_cmd_);
                if (i == 0) {
                    objs_.emplace_back(output_image);
                }
                output_images.emplace_back(output_image);
            });
        }
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto input = core::as_tensor<T>(inputs[0]);
            auto input_image = input->as_input_image(m_dev_, m_cmd_);
            objs_.emplace_back(input_image);
        });

        para_.axis = para_.axis + 4 - rank;
        if (rank == 4 || rank == 3) {
            inputs[0]->get_shape(para_.inShape);
            outputs[0]->get_shape(para_.outShape);
        }
        // Each submit needs its own descriptor set to avoid cross-pass
        // binding interference on drivers like Intel ANV.
        std::vector<VkDescriptorSet> pass_ds(num_outputs_);
        for (int i = 0; i < num_outputs_; i++) {
            pass_ds[i] = allocPassDescriptorSet();
        }
        para_.split = 0;
        for (int i = 0; i < num_outputs_; i++) {
            para_.outShape[para_.axis] =
                outputs[i]->getShape()[para_.axis - 4 + rank];

            auto gpushape = outputs[i]->getGPUShape();
            objs_[0] = output_images[i];
            submit_per_ds(pass_ds[i], &para_, UP_DIV(gpushape[0], 16),
                          UP_DIV(gpushape[1], 16), gpushape[2]);
            para_.split += outputs[i]->getShape()[para_.axis - 4 + rank];
        }
        for (int i = 0; i < num_outputs_; i++) {
            freePassDescriptorSet(pass_ds[i]);
        }
    }

    int num_outputs_ = 0;
    split::GpuSplitParam para_;
};
// Split buffer op (fp32). Per-output submit: bind the shared input + this
// output, dispatch UP_DIV(out_total, 256).
class SplitBuffer : public BufferFactory {
  public:
    explicit SplitBuffer(int fp16)
        : BufferFactory(OpType::SPLIT,
                        {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                        sizeof(SplitPC)) {
        update_after_bind_ = true;
        fp16_ = fp16;
    }

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("axis") != attributes.end()) {
            axis_ = std::stoi(attributes.at("axis"));
        }
        if (attributes.find("num_outputs") != attributes.end()) {
            num_outputs_ = std::stoi(attributes.at("num_outputs"));
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
        if (num_outputs_ == 0) {
            num_outputs_ = static_cast<int>(outputs.size());
        }
        int tosplit = inputs[0]->getShape()[axis_];
        std::vector<int64_t> split_vec;
        if (inputs.size() > 1) {
            auto split = core::as_tensor<int64_t>(inputs[1]);
            split->copyToCPU(m_cmdpool_);
            split_vec.resize(split->num_elements());
            for (size_t i = 0; i < split_vec.size(); ++i) {
                split_vec[i] = (*split)[i];
            }
        } else {
            split_vec.resize(num_outputs_);
            for (int i = 1; i < num_outputs_; ++i) {
                split_vec[i] = tosplit / num_outputs_;
            }
        }

        auto in_shape = inputs[0]->getShape();
        // Fail closed on a <= 1-byte payload: no byte build in this mover.
        core::require_word_movable_elem(inputs[0]->elem_kind(), "Split");
        for (const auto &o : outputs) {
            core::require_word_movable_elem(o->elem_kind(), "Split");
        }

        // Zero-copy fast path: a Split along axis 0 (the outermost dim) carves
        // the input into contiguous blocks, so each output can bind the input's
        // VkBuffer at its block's byte offset instead of dispatching
        // split.comp. (Any other axis interleaves the pieces -> a real gather.)
        // Requires a GPU input and every block offset to satisfy the device
        // alignment + fp16 word parity; otherwise fall through to the dispatch
        // below.
        if (axis_ == 0 && static_cast<int>(in_shape.size()) == rank) {
            size_t esz = (inputs[0]->dtype() == typeid(uint16_t))
                             ? sizeof(uint16_t)
                             : sizeof(float);
            VkDeviceSize inner = 1;
            for (int d = 1; d < rank; ++d) {
                inner *= static_cast<VkDeviceSize>(in_shape[d]);
            }
            VkDeviceSize align = m_dev_->minStorageBufferOffsetAlignment();
            bool offs_ok = true;
            VkDeviceSize off_elems = 0;
            for (int i = 0; i < num_outputs_; ++i) {
                if (!offset_view_aligned(off_elems * esz, align)) {
                    offs_ok = false;
                    break;
                }
                off_elems += static_cast<VkDeviceSize>(split_vec[i]) * inner;
            }
            if (offs_ok) {
                std::shared_ptr<VulkanBuffer> src_buff;
                dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
                    using T = decltype(dummy);
                    auto src = core::as_tensor<T>(inputs[0]);
                    if (!src->has_gpu_buffer()) {
                        return;
                    }
                    src_buff = std::dynamic_pointer_cast<VulkanBuffer>(
                        src->as_storage_buffer(m_dev_, m_cmd_));
                });
                if (src_buff) {
                    VkDeviceSize block_elems = 0;
                    for (int i = 0; i < num_outputs_; ++i) {
                        auto out_shape = in_shape;
                        out_shape[axis_] = static_cast<int>(split_vec[i]);
                        int out_total = total_elems(out_shape);
                        dispatch_by_dtype(outputs[i]->dtype(), [&](auto d2) {
                            using T = decltype(d2);
                            auto output = core::as_tensor<T>(outputs[i]);
                            if (output->num_elements() != out_total) {
                                output->resize(out_shape);
                            }
                            output->alias_storage_buffer_range(
                                src_buff, block_elems * esz,
                                static_cast<VkDeviceSize>(out_total) * esz,
                                m_cmd_);
                        });
                        block_elems +=
                            static_cast<VkDeviceSize>(split_vec[i]) * inner;
                    }
                    return;
                }
            }
        }

        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            bind_ssbo<T>(inputs[0], /*is_output=*/false);
        });
        auto in_buf = std::dynamic_pointer_cast<VulkanBuffer>(objs_[0]);

        std::vector<VkDescriptorSet> pass_ds(num_outputs_);
        for (int i = 0; i < num_outputs_; ++i) {
            pass_ds[i] = allocPassDescriptorSet();
        }

        int split_offset = 0;
        for (int i = 0; i < num_outputs_; ++i) {
            auto out_shape = in_shape;
            out_shape[axis_] = static_cast<int>(split_vec[i]);
            int out_total = total_elems(out_shape);

            dispatch_by_dtype(outputs[i]->dtype(), [&](auto dummy) {
                using T = decltype(dummy);
                auto output = core::as_tensor<T>(outputs[i]);
                if (output->num_elements() != total_elems(out_shape)) {
                    output->resize(out_shape);
                }
                bind_ssbo<T>(outputs[i], /*is_output=*/true);
            });
            // layout [output, input] — matches split.comp (binding 0 =
            // uOutput writeonly, binding 1 = uInput readonly). The previous
            // {input, output} order bound the input as the write target,
            // clobbering the producer's output (e.g. the qkv Transpose).
            std::vector<std::shared_ptr<VulkanResource>> ordered = {
                objs_.back(), in_buf};
            objs_ = ordered;

            SplitPC pc{};
            pc.axis = axis_;
            pc.rank = rank;
            fill_dims(pc.inDims, in_shape);
            fill_dims(pc.outDims, out_shape);
            pc.split = split_offset;
            // fp16 packs 2 half/uint word: one thread per output word.
            int nthreads = (fp16_ != 0) ? (out_total + 1) / 2 : out_total;
            submit_per_ds(pass_ds[i], &pc, UP_DIV(nthreads, 256), 1, 1);
            split_offset += static_cast<int>(split_vec[i]);
        }
        for (int i = 0; i < num_outputs_; ++i) {
            freePassDescriptorSet(pass_ds[i]);
        }
    }

    int axis_ = 1;
    int num_outputs_ = 0;
};

// PIMPL façade: buffer SSBO impl when backend_buffer is set, else image.
class Split : public PimplFacade {
  public:
    Split(int fp16, bool backend_buffer) : PimplFacade(OpType::SPLIT) {
        impl_ =
            backend_buffer
                ? std::unique_ptr<Operator>(std::make_unique<SplitBuffer>(fp16))
                : std::make_unique<SplitImage>();
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_SPLIT_HPP_
