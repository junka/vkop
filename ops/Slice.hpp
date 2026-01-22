// Copyright 2025 @junka
#ifndef OPS_SLICE_HPP_
#define OPS_SLICE_HPP_

#include "core/Tensor.hpp"
#include "ops/Operator.hpp"
#include <numeric>
extern "C" {
extern unsigned char slice_spv[];
extern unsigned int slice_spv_len;
}
namespace vkop {
namespace ops {

namespace slice {
struct GpuSliceParam {
    ivec4 inImgSize;
    ivec4 outImgSize;
    ivec4 inShape;
    ivec4 outShape;
    ivec4 start; // Start indices for slicing
    ivec4 end;   // End indices for slicing
    ivec4 step;  // Step sizes for slicing
};

} // namespace slice

class Slice : public Operator {
  public:
    explicit Slice()
        : Operator(OpType::SLICE, slice_spv, slice_spv_len,
                   {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER},
                   sizeof(slice::GpuSliceParam)) {}

    template <typename T>
    static std::vector<std::vector<int>>
    CalculateOutputShape(const std::vector<int> &input_shape,
                         const std::vector<T> &starts,
                         const std::vector<T> &ends, const std::vector<T> &axes,
                         const std::vector<T> &steps) {
        assert(input_shape.size() >= 3);
        const int dims = static_cast<int>(input_shape.size());
        std::vector<std::vector<int>> ret;

        std::vector<T> norm_axes = axes;
        if (norm_axes.empty()) {
            for (int i = 0; i < dims; ++i) {
                norm_axes.push_back(static_cast<T>(i));
            }
        }

        for (auto &ax : norm_axes) {
            if (ax < 0)
                ax += dims;
            if (ax < 0 || ax >= dims) {
                throw std::out_of_range("axis out of range");
            }
        }
        if (starts.size() != norm_axes.size() ||
            ends.size() != norm_axes.size()) {
            printf("%zd %zd %zd\n", norm_axes.size(), starts.size(),
                   ends.size());
            throw std::invalid_argument("starts/ends length must match axes");
        }

        std::vector<int> full_starts(dims);
        std::vector<int> full_ends(dims);
        std::vector<int> full_steps(dims, 1);

        for (int i = 0; i < dims; ++i) {
            full_starts[i] = 0;
            full_ends[i] = input_shape[i];
        }

        for (size_t i = 0; i < norm_axes.size(); ++i) {
            auto axis = norm_axes[i];
            auto dim_size = input_shape[axis];

            int start = static_cast<int>(starts[i]);
            int end = static_cast<int>(ends[i]);
            int step = (steps.size() > i) ? static_cast<int>(steps[i]) : 1;

            if (step == 0)
                step = 1;

            if (start < 0)
                start += dim_size;
            if (end < 0)
                end += dim_size;

            start = std::max(0, std::min(start, dim_size));
            end = std::max(0, std::min(end, dim_size));

            full_starts[axis] = start;
            full_ends[axis] = end;
            full_steps[axis] = step;
        }

        std::vector<int> output_shape(dims);
        for (int i = 0; i < dims; ++i) {
            auto start = full_starts[i];
            auto end = full_ends[i];
            auto step = full_steps[i];

            if (step > 0) {
                if (start >= end) {
                    output_shape[i] = 0;
                } else {
                    output_shape[i] = (end - start + step - 1) / step;
                }
            } else {
                if (start <= end) {
                    output_shape[i] = 0;
                } else {
                    output_shape[i] = (start - end - step - 1) / (-step);
                }
            }
            output_shape[i] = std::max(0, output_shape[i]);
        }
        ret.emplace_back(output_shape);
        ret.emplace_back(full_starts);
        ret.emplace_back(full_ends);
        ret.emplace_back(full_steps);

        return ret;
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {

        auto inshape = inputs[0]->getShape();
        auto rank = inputs[0]->num_dims();
        std::vector<std::vector<int>> out_size;

        dispatch_by_dtype(inputs[1]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto starts = core::as_tensor<T>(inputs[1]);
            auto ends = core::as_tensor<T>(inputs[2]);
            std::shared_ptr<core::Tensor<T>> axes;
            std::shared_ptr<core::Tensor<T>> steps;

            if (inputs.size() > 3) {
                axes = core::as_tensor<T>(inputs[3]);
            } else {
                axes = std::make_shared<core::Tensor<T>>(rank);
                std::vector<T> axes_data(rank);
                std::iota(axes_data.begin(), axes_data.end(), 0);
                axes->fillToCPU(axes_data);
            }
            if (inputs.size() > 4) {
                steps = core::as_tensor<T>(inputs[4]);
            } else {
                steps = std::make_shared<core::Tensor<T>>(rank);
                std::vector<T> step_data(rank);
                for (int i = 0; i < rank; i++) {
                    step_data[i] = 1;
                }
                steps->fillToCPU(step_data);
            }

            out_size =
                CalculateOutputShape(inshape, starts->data(), ends->data(),
                                     axes->data(), steps->data());

            for (auto i = 0; i < static_cast<int>(out_size[0].size()); i++) {
                printf("outSize[%d] = %d\n", i, out_size[0][i]);
            }
        });
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->size() == 0) {
                output->resize(out_size[0]);
            }
            auto output_image = output->as_output_image(m_dev_, m_cmd_);
            objs_.emplace_back(output_image);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto input = core::as_tensor<T>(inputs[0]);
            auto input_image = input->as_input_image(m_dev_, m_cmd_);
            objs_.emplace_back(input_image);
        });

        auto outshape = outputs[0]->getShape();
        auto out_gpu_shape = outputs[0]->getGPUShape();
        auto in_gpu_shape = inputs[0]->getGPUShape();
        slice::GpuSliceParam param;
        param.inImgSize[0] = in_gpu_shape[0];
        param.inImgSize[1] = in_gpu_shape[1];
        param.inImgSize[2] = in_gpu_shape[2];
        param.inImgSize[3] = 1;
        param.outImgSize[0] = out_gpu_shape[0];
        param.outImgSize[1] = out_gpu_shape[1];
        param.outImgSize[2] = out_gpu_shape[2];
        param.outImgSize[3] = 1;
        if (rank == 4) {
            for (int i = 0; i < 4; i++) {
                param.inShape[i] = inshape[i];
                param.outShape[i] = out_size[0][i];
                param.start[i] = out_size[1][i];
                param.end[i] = out_size[2][i];
                param.step[i] = out_size[3][i];
            }
        } else if (rank == 3) {
            param.inShape[0] = 1;
            param.outShape[0] = 1;
            param.start[0] = 0;
            param.end[0] = 1;
            param.step[0] = 1;
            for (int i = 0; i < 3; i++) {
                param.inShape[i + 1] = inshape[i];
                param.outShape[i + 1] = out_size[0][i];
                param.start[i + 1] = out_size[1][i];
                param.end[i + 1] = out_size[2][i];
                param.step[i + 1] = out_size[3][i];
            }
        }
        submit(&param, UP_DIV(out_gpu_shape[0], 16),
               UP_DIV(out_gpu_shape[1], 16), out_gpu_shape[2]);
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_SLICE_HPP_
