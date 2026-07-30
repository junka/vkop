// Copyright 2025 @junka
#ifndef OPS_RESHAPE_HPP_
#define OPS_RESHAPE_HPP_

#include "ops/BufferBase.hpp"
#include "ops/PimplFacade.hpp"
#include <numeric>

#include "core/Tensor.hpp"
#include "ops/Operator.hpp"
extern "C" {
extern unsigned char image_reshape_spv[];
extern unsigned int image_reshape_spv_len;
extern unsigned char buffer_reshape_spv[];
extern unsigned int buffer_reshape_spv_len;
extern unsigned char buffer_reshape_fp16_spv[];
extern unsigned int buffer_reshape_fp16_spv_len;
}
namespace vkop {
namespace ops {

namespace reshape {
struct GpuReshapeParam {
    ivec4 inImgSize;
    ivec4 outImgSize;
    ivec4 inShape;
    ivec4 outShape;
};

} // namespace reshape

class ReshapeImage : public Operator {
  public:
    explicit ReshapeImage()
        : Operator(OpType::RESHAPE, image_reshape_spv, image_reshape_spv_len,
                   {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER},
                   sizeof(reshape::GpuReshapeParam)) {}
    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("allowzero") != attributes.end()) {
            allowzero_ = std::stol(attributes.at("allowzero"));
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto inshape = inputs[0]->getShape();
        auto shape = core::as_tensor<int64_t>(inputs[1]);
        assert(shape->num_dims() == 1);
        int n = shape->num_elements();

        std::vector<int> dim(n);
        for (int i = 0; i < n; i++) {
            dim[i] = static_cast<int>((*shape)[i]);
        }
        auto total = std::accumulate(inshape.begin(), inshape.end(), 1,
                                     std::multiplies<>());
        for (int i = 0; i < n; i++) {
            if (!allowzero_ && dim[i] == 0) {
                dim[i] = inshape[i];
            }
        }
        for (int i = 0; i < n; i++) {
            if (dim[i] != 0 && dim[i] != -1)
                total = total / dim[i];
        }
        for (int i = 0; i < n; i++) {
            if (dim[i] == -1)
                dim[i] = total;
        }

        bool noop = false;
        if (inshape.size() == dim.size()) {
            noop = true;
            for (size_t i = 0; i < inshape.size(); i++) {
                if (inshape[i] != dim[i]) {
                    noop = false;
                    break;
                }
            }
        }
        if (inshape.size() == 4 && dim.size() == 3) {
            if (inshape[0] == 1 && inshape[1] == dim[0] &&
                inshape[2] == dim[1] && inshape[3] == dim[2]) {
                noop = true;
            }
        } else if (inshape.size() == 3 && dim.size() == 4) {
            if (dim[0] == 1 && dim[1] == inshape[0] && dim[2] == inshape[1] &&
                dim[3] == inshape[2]) {
                noop = true;
            }
        } else if (inshape.size() == 4 && dim.size() == 2) {
            if (inshape[0] == 1 && inshape[1] == 1 && inshape[2] == dim[0] &&
                inshape[3] == dim[1]) {
                noop = true;
            }
        } else if (inshape.size() == 2 && dim.size() == 4) {
            if (dim[0] == 1 && dim[1] == 1 && inshape[0] == dim[2] &&
                inshape[1] == dim[3]) {
                noop = true;
            }
        }

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->size() == 0) {
                output->resize(dim);
            }
            if (dim.size() <= 2) {
                auto output_buff = output->as_storage_buffer(m_dev_);
                objs_.emplace_back(output_buff);
            } else {
                auto output_image = output->as_output_image(m_dev_, m_cmd_);
                objs_.emplace_back(output_image);
            }
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto input = core::as_tensor<T>(inputs[0]);
            if (inputs[0]->num_dims() <= 2) {
                auto input_buff = input->as_storage_buffer(m_dev_);
                objs_.emplace_back(input_buff);
            } else {
                auto input_image = input->as_input_image(m_dev_, m_cmd_);
                objs_.emplace_back(input_image);
            }
        });

        if (noop) {
            // copy directly, could be optimized by preprocess/compiler
            if (inshape.size() < 3) {
                auto output_buff =
                    std::dynamic_pointer_cast<VulkanBuffer>(objs_[0]);
                auto input_buff =
                    std::dynamic_pointer_cast<VulkanBuffer>(objs_[1]);
                input_buff->copyBufferToStageBuffer(m_cmd_->get(),
                                                    output_buff->getBuffer(), 0,
                                                    output_buff->getSize(), 0);
            } else {
                auto output_image =
                    std::dynamic_pointer_cast<VulkanImage>(objs_[0]);
                auto input_image =
                    std::dynamic_pointer_cast<VulkanImage>(objs_[1]);
                input_image->transferReadBarrier(m_cmd_->get());
                output_image->copyImageToImage(m_cmd_->get(), input_image,
                                               {0, 0, 0}, 0);
            }
            return;
        }

        if (dim.size() <= 2) {
            auto output_buff =
                std::dynamic_pointer_cast<VulkanBuffer>(objs_[0]);
            auto input_image = std::dynamic_pointer_cast<VulkanImage>(objs_[1]);
            input_image->copyImageToBuffer(m_cmd_->get(),
                                           output_buff->getBuffer(), 0);
            return;
        }

        auto out_gpu_shape = outputs[0]->getGPUShape();
        auto in_gpu_shape = inputs[0]->getGPUShape();
        reshape::GpuReshapeParam param;
        param.inImgSize[0] = in_gpu_shape[0];
        param.inImgSize[1] = in_gpu_shape[1];
        param.inImgSize[2] = in_gpu_shape[2];
        param.inImgSize[3] = 1;
        param.outImgSize[0] = out_gpu_shape[0];
        param.outImgSize[1] = out_gpu_shape[1];
        param.outImgSize[2] = out_gpu_shape[2];
        param.outImgSize[3] = 1;
        if (inshape.size() == 4) {
            param.inShape[0] = inshape[0];
            param.inShape[1] = inshape[1];
            param.inShape[2] = inshape[2];
            param.inShape[3] = inshape[3];
        } else if (inshape.size() == 3) {
            param.inShape[0] = 1;
            param.inShape[1] = inshape[0];
            param.inShape[2] = inshape[1];
            param.inShape[3] = inshape[2];
        }
        if (n == 4) {
            param.outShape[0] = dim[0];
            param.outShape[1] = dim[1];
            param.outShape[2] = dim[2];
            param.outShape[3] = dim[3];
        } else if (n == 3) {
            param.outShape[0] = 1;
            param.outShape[1] = dim[0];
            param.outShape[2] = dim[1];
            param.outShape[3] = dim[2];
        }
        submit(&param, UP_DIV(out_gpu_shape[0], 16),
               UP_DIV(out_gpu_shape[1], 16), out_gpu_shape[2]);
    }

    int allowzero_ = 0;
};

// Buffer (SSBO, compact row-major) reshape. Data is flat row-major so a
// reshape is a 1:1 copy of `total` contiguous scalars; the shape change is
// metadata-only. fp16 packs two elements per uint word (one thread/word).
class ReshapeBuffer : public BufferFactory {
  public:
    explicit ReshapeBuffer(int fp16)
        : BufferFactory(OpType::RESHAPE,
                        fp16 ? buffer_reshape_fp16_spv : buffer_reshape_spv,
                        fp16 ? buffer_reshape_fp16_spv_len
                             : buffer_reshape_spv_len,
                        {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                        sizeof(ReshapePC), fp16) {}

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto in_shape = inputs[0]->getShape();
        auto shape = core::as_tensor<int64_t>(inputs[1]);
        int n = shape->num_elements();
        std::vector<int> dim(n);
        for (int i = 0; i < n; ++i) {
            dim[i] = static_cast<int>((*shape)[i]);
        }
        int total = total_elems(in_shape);
        // resolve a 0 dim by copying from the input, and -1 from the remainder
        for (int i = 0; i < n; ++i) {
            if (dim[i] == 0 && i < static_cast<int>(in_shape.size())) {
                dim[i] = in_shape[i];
            }
        }
        int known = total;
        for (int i = 0; i < n; ++i) {
            if (dim[i] != 0 && dim[i] != -1) {
                known /= dim[i];
            }
        }
        for (int i = 0; i < n; ++i) {
            if (dim[i] == -1) {
                dim[i] = known;
            }
        }

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->size() == 0) {
                output->resize(dim);
            }
            bind_ssbo<T>(outputs[0], /*is_output=*/true);
        });
        // The shape input (inputs[1]) is int64_t and lives on the CPU; the
        // buffer shader only reads from inputs[0], so the shape tensor does
        // not need an SSBO binding (the host already consumed it above).
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            bind_ssbo<T>(inputs[0], /*is_output=*/false);
        });

        ReshapePC pc{};
        pc.rank_in = static_cast<int>(in_shape.size());
        fill_dims(pc.inDims, in_shape);
        pc.rank_out = static_cast<int>(dim.size());
        fill_dims(pc.outDims, dim);
        pc.total = total;
        int nthreads = (fp16_ != 0) ? (total + 1) / 2 : total;
        submit(&pc, UP_DIV(nthreads, 256), 1, 1);
    }
};

// PIMPL façade: buffer SSBO impl when backend_buffer is set, else image.
class Reshape : public PimplFacade {
  public:
    Reshape(int fp16, bool backend_buffer) : PimplFacade(OpType::RESHAPE) {
        impl_ = backend_buffer ? std::unique_ptr<Operator>(
                                     std::make_unique<ReshapeBuffer>(fp16))
                               : std::make_unique<ReshapeImage>();
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_RESHAPE_HPP_
