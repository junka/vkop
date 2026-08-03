// Copyright 2025 @junka
#ifndef OPS_SOFTMAX_HPP_
#define OPS_SOFTMAX_HPP_

#include "Operator.hpp"
#include "ops/BufferBase.hpp"
#include "ops/PimplFacade.hpp"
extern "C" {
extern unsigned char image_softmax_spv[];
extern unsigned int image_softmax_spv_len;
extern unsigned char buffer_softmax_spv[];
extern unsigned int buffer_softmax_spv_len;
extern unsigned char buffer_softmax_fp16_spv[];
extern unsigned int buffer_softmax_fp16_spv_len;
}
namespace vkop {
namespace ops {
namespace softmax {

struct GpuSoftMaxParam {
    ivec4 outShape;
    int axis; // 0: N, 1: C, 2: H, 3: W
    int fp16;
    int nanwhere;
    float nanvalue;
};

} // namespace softmax

// Image (image2DArray NCHW->RGBA) implementation. The old softmax2_spv
// (2-D SSBO auto-path) was folded into the buffer backend's BufferSoftmax
// façade; the image path is image-only now.
class SoftmaxImage : public Operator {
  public:
    SoftmaxImage(const SoftmaxImage &) = delete;
    SoftmaxImage &operator=(const SoftmaxImage &) = delete;
    SoftmaxImage(SoftmaxImage &&) = delete;
    SoftmaxImage &operator=(SoftmaxImage &&) = delete;
    explicit SoftmaxImage()
        : Operator(OpType::SOFTMAX, image_softmax_spv, image_softmax_spv_len,
                   {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER},
                   sizeof(softmax::GpuSoftMaxParam)) {
        // should be safe to fail to set subgroup size
        required_subgroup_size_ = 32;
    }

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("axis") != attributes.end()) {
            para_.axis = std::stol(attributes.at("axis"));
        } else if (attributes.find("dim") != attributes.end()) {
            para_.axis = std::stol(attributes.at("dim"));
        }
        if (attributes.find("nan_optimization") != attributes.end()) {
            para_.nanwhere = 1;
        }
        if (attributes.find("nan_replacement_value") != attributes.end()) {
            para_.nanvalue = std::stof(attributes.at("nan_replacement_value"));
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {

        auto input_shape = inputs[0]->getShape();
        int rank = inputs[0]->num_dims();

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->size() == 0) {
                output->resize(inputs[0]->getShape());
            }
            auto output_image = output->as_output_image(m_dev_, m_cmd_);
            objs_.emplace_back(output_image);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto input = core::as_tensor<T>(inputs[0]);
            auto input_image = input->as_input_image(m_dev_, m_cmd_);
            objs_.emplace_back(input_image);
            if (typeid(T) == typeid(float)) {
                para_.fp16 = 0;
            } else if (typeid(T) == typeid(uint16_t)) {
                para_.fp16 = 1;
            }
        });

        int batch = input_shape[0];
        int depth = input_shape[1];
        int out_height = input_shape.size() > 2 ? input_shape[2] : 1;
        int out_width = input_shape.size() > 3 ? input_shape[3] : 1;

        int realheight = out_height * batch;

        // vkimage params
        para_.outShape[0] = batch;
        para_.outShape[1] = depth;
        para_.outShape[2] = out_height;
        para_.outShape[3] = out_width;
        if (para_.axis < 0) {
            para_.axis = rank + para_.axis;
        }

        if (para_.axis == 0) {
            submit(&para_, UP_DIV(out_width, 16), UP_DIV(out_height, 16),
                   UP_DIV(depth, 4));
        } else if (para_.axis == 1) {
            submit(&para_, UP_DIV(out_width, 16), UP_DIV(realheight, 16),
                   batch);
        } else if (para_.axis == 2) {
            submit(&para_, UP_DIV(out_width, 16), UP_DIV(batch, 16),
                   UP_DIV(depth, 4));
        } else if (para_.axis == 3) {
            submit(&para_, UP_DIV(out_height, 16), UP_DIV(batch, 16),
                   UP_DIV(depth, 4));
        }
    }

    softmax::GpuSoftMaxParam para_;
};

// Buffer (SSBO) softmax. Reduces over an arbitrary axis (one workgroup per
// non-axis slice, shared-mem tree reduce). Ships an fp16 dual build.
//
// fp32: single dispatch writes results directly to the output SSBO.
// fp16: two dispatches — (1) reduce + write fp32 results to a scratch SSBO,
// (2) a pack pass (axis_size==0 marker) packs fp32->packed-half2 into the
// output, one thread per output word (no cross-workgroup word race, which
// would otherwise occur when inner_size>1 scatters axis elements across
// words shared between workgroups).
class SoftmaxBuffer : public BufferFactory {
  public:
    explicit SoftmaxBuffer(int fp16)
        : BufferFactory(OpType::SOFTMAX,
                        fp16 ? buffer_softmax_fp16_spv : buffer_softmax_spv,
                        fp16 ? buffer_softmax_fp16_spv_len
                             : buffer_softmax_spv_len,
                        {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                         DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                        sizeof(SoftmaxPC), fp16) {}

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("axis") != attributes.end()) {
            axis_ = std::stol(attributes.at("axis"));
        } else if (attributes.find("dim") != attributes.end()) {
            axis_ = std::stol(attributes.at("dim"));
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto shape = inputs[0]->getShape();
        int rank = static_cast<int>(shape.size());
        int ax = axis_;
        if (ax < 0) {
            ax += rank;
        }
        int axis_size = shape[ax];
        int total = total_elems(shape);
        // inner_size: product of dims after the axis (stride between
        // consecutive axis elements of the same slice).
        int inner_size = 1;
        for (int i = ax + 1; i < rank; ++i) {
            inner_size *= shape[i];
        }
        // outer_size: number of independent slices = total / axis_size
        // (one per (outer, inner) position). The shader maps workgroup id
        // back to (outer, inner) via outer = wg / inner_size,
        // inner = wg % inner_size.
        int outer_size = total / axis_size;

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->size() == 0) {
                output->resize(shape);
            }
            bind_ssbo<T>(outputs[0], /*is_output=*/true);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            bind_ssbo<T>(inputs[0], /*is_output=*/false);
        });

        // fp16 needs a scratch fp32 buffer (binding 2) for the decoupled
        // reduce->pack write; binding 3 is unused by the shader but the
        // descriptor set declares 4 storage bindings, so bind the dummy
        // buffer there. fp32 leaves both binding 2/3 unused; bind dummy for
        // each so the descriptor set is valid.
        if (fp16_ != 0) {
            scratch_ = std::make_shared<VulkanBuffer>(
                m_dev_, static_cast<size_t>(total) * sizeof(float),
                STORAGE | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            objs_.emplace_back(scratch_);
            objs_.emplace_back(dummy_buffer_);
        } else {
            objs_.emplace_back(dummy_buffer_);
            objs_.emplace_back(dummy_buffer_);
        }

        SoftmaxPC pc{};
        pc.axis = ax;
        pc.axis_size = axis_size;
        pc.outer_size = outer_size;
        pc.inner_size = inner_size;

        if (fp16_ != 0) {
            // fp16 2-pass: use separate descriptor sets to avoid Intel ANV's
            // immediate-descriptor-update interference between passes.
            VkDescriptorSet reduce_ds = allocPassDescriptorSet();
            submit_per_ds(reduce_ds, &pc, outer_size, 1, 1);

            // Barrier: flush reduce's scratch writes for pack's reads.

            // Pack pass: axis_size==0 marks pack-only; outer_size carries
            // the total element count; one thread per output WORD.
            scratch_->shaderWriteBarrier(m_cmd_->get());
            SoftmaxPC pack_pc{};
            pack_pc.axis = 0;
            pack_pc.axis_size = 0;
            pack_pc.outer_size = total;
            pack_pc.inner_size = 0;
            VkDescriptorSet pack_ds = allocPassDescriptorSet();
            submit_per_ds(pack_ds, &pack_pc, UP_DIV(total, 512), 1, 1);
            freePassDescriptorSet(reduce_ds);
            freePassDescriptorSet(pack_ds);
        } else {
            submit(&pc, outer_size, 1, 1);
        }
    }

    int axis_ = 1;
    std::shared_ptr<VulkanBuffer> scratch_;
};

// PIMPL façade: buffer SSBO impl (SoftmaxBuffer) when backend_buffer is set,
// else the image impl. The old use_ssbo 2-D auto-path is gone — buffer mode
// is selected solely by backend_buffer.
class Softmax : public PimplFacade {
  public:
    Softmax(int fp16, bool backend_buffer) : PimplFacade(OpType::SOFTMAX) {
        impl_ = backend_buffer ? std::unique_ptr<Operator>(
                                     std::make_unique<SoftmaxBuffer>(fp16))
                               : std::make_unique<SoftmaxImage>();
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_SOFTMAX_HPP_
