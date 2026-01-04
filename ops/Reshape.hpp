// Copyright 2025 @junka
#ifndef OPS_RESHAPE_HPP_
#define OPS_RESHAPE_HPP_

#include <numeric>

#include "core/Tensor.hpp"
#include "ops/Operator.hpp"

extern unsigned char reshape_spv[];
extern unsigned int reshape_spv_len;
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

class Reshape : public Operator {
  public:
    explicit Reshape()
        : Operator(OpType::RESHAPE, reshape_spv, reshape_spv_len,
                   sizeof(reshape::GpuReshapeParam)) {
        n_imgs_ = 2;
        types_ = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER};
        objs_.reserve(2);
    }

  private:
    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("allowzero") != attributes.end()) {
            allowzero_ = std::stol(attributes.at("allowzero"));
        }
    }

    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto inshape = inputs[0]->getShape();
        auto shape = core::as_tensor<int64_t>(inputs[1]);
        assert(shape->num_dims() == 1);
        int n = shape->num_elements();

        std::vector<int> dim(n);
        for (int i = 0; i < n; i++) {
            dim[i] = (*shape)[i];
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
                input_buff->copyBufferToStageBuffer(
                    m_cmd_->get(), output_buff->getBuffer(), 0);
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

} // namespace ops
} // namespace vkop
#endif // OPS_RESHAPE_HPP_
