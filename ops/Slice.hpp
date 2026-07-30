// Copyright 2025 @junka
#ifndef OPS_SLICE_HPP_
#define OPS_SLICE_HPP_

#include "core/Tensor.hpp"
#include "ops/BufferBase.hpp"
#include "ops/Operator.hpp"
#include "ops/PimplFacade.hpp"
#include "ops/SliceCalc.hpp"
#include <numeric>
extern "C" {
extern unsigned char image_slice_spv[];
extern unsigned int image_slice_spv_len;
extern unsigned char buffer_slice_spv[];
extern unsigned int buffer_slice_spv_len;
}
namespace vkop {
namespace ops {

namespace slice {
struct GpuSliceParam {
    // ivec4 inImgSize;
    // ivec4 outImgSize;
    ivec4 inShape;
    ivec4 outShape;
    ivec4 start; // Start indices for slicing
    ivec4 end;   // End indices for slicing
    ivec4 step;  // Step sizes for slicing
};

} // namespace slice

class SliceImage : public Operator {
  public:
    explicit SliceImage()
        : Operator(OpType::SLICE, image_slice_spv, image_slice_spv_len,
                   {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER},
                   sizeof(slice::GpuSliceParam)) {}

    template <typename T>
    static std::vector<std::vector<int>>
    CalculateOutputShape(const std::vector<int> &input_shape,
                         const std::vector<T> &starts,
                         const std::vector<T> &ends, const std::vector<T> &axes,
                         const std::vector<T> &steps) {
        return slice_calc::calculate_output_shape<T>(input_shape, starts, ends,
                                                     axes, steps);
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

            // for (auto i = 0; i < static_cast<int>(out_size[0].size()); i++) {
            //     printf("outSize[%d] = %d\n", i, out_size[0][i]);
            // }
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
// Slice buffer op (fp32). Inputs: data, starts, ends, [axes], [steps].
class SliceBuffer : public BufferFactory {
  public:
    explicit SliceBuffer(int /*fp16*/)
        : BufferFactory(OpType::SLICE, buffer_slice_spv, buffer_slice_spv_len,
                        {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                        sizeof(SlicePC)) {}

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto inshape = inputs[0]->getShape();
        int rank = static_cast<int>(inshape.size());
        std::vector<std::vector<int>> out_size =
            slice_calc::calculate_output_shape<int64_t>(
                inshape, core::as_tensor<int64_t>(inputs[1])->data(),
                core::as_tensor<int64_t>(inputs[2])->data(),
                inputs.size() > 3 ? core::as_tensor<int64_t>(inputs[3])->data()
                                  : std::vector<int64_t>{},
                inputs.size() > 4 ? core::as_tensor<int64_t>(inputs[4])->data()
                                  : std::vector<int64_t>{});

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->size() == 0) {
                output->resize(out_size[0]);
            }
            bind_ssbo<T>(outputs[0], /*is_output=*/true);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            bind_ssbo<T>(inputs[0], /*is_output=*/false);
        });

        SlicePC pc{};
        pc.rank = rank;
        for (int i = 0; i < 6; ++i) {
            pc.inDims[i] = (i < rank) ? inshape[i] : 1;
            pc.outDims[i] = (i < rank) ? out_size[0][i] : 1;
            pc.starts[i] = (i < rank) ? out_size[1][i] : 0;
            pc.ends[i] = (i < rank) ? out_size[2][i] : 1;
            pc.steps[i] = (i < rank) ? out_size[3][i] : 1;
        }
        int total = total_elems(out_size[0]);
        submit(&pc, UP_DIV(total, 256), 1, 1);
    }
};

// PIMPL façade: buffer SSBO impl when backend_buffer is set, else image.
class Slice : public PimplFacade {
  public:
    Slice(int /*fp16*/, bool backend_buffer) : PimplFacade(OpType::SLICE) {
        impl_ =
            backend_buffer
                ? std::unique_ptr<Operator>(std::make_unique<SliceBuffer>(0))
                : std::make_unique<SliceImage>();
    }

    // ONNX slice output-shape helper (used by tests), delegating to the
    // shared slice_calc free function.
    template <typename T>
    static std::vector<std::vector<int>>
    CalculateOutputShape(const std::vector<int> &input_shape,
                         const std::vector<T> &starts,
                         const std::vector<T> &ends, const std::vector<T> &axes,
                         const std::vector<T> &steps) {
        return slice_calc::calculate_output_shape<T>(input_shape, starts, ends,
                                                     axes, steps);
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_SLICE_HPP_
