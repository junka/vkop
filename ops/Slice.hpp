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
            if (output->num_elements() != total_elems(out_size[0])) {
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

        // int64 data: CPU slice (part of the shape meta-chain). Walk the
        // output linearly; each output coordinate maps to an input coordinate
        // via in_coord[d] = full_starts[d] + out_coord[d] * full_steps[d].
        if (inputs[0]->dtype() == typeid(int64_t)) {
            auto out_shape = out_size[0];
            auto &full_starts = out_size[1];
            auto &full_steps = out_size[3];
            int total = total_elems(out_shape);
            std::vector<int64_t> out(static_cast<size_t>(total));
            auto src = core::as_tensor<int64_t>(inputs[0]);

            std::vector<int> in_stride(rank, 1);
            for (int d = rank - 2; d >= 0; --d) {
                in_stride[d] = in_stride[d + 1] * inshape[d + 1];
            }
            std::vector<int> out_stride(rank, 1);
            for (int d = rank - 2; d >= 0; --d) {
                out_stride[d] = out_stride[d + 1] * out_shape[d + 1];
            }
            std::vector<int> in_coord(rank, 0);
            std::vector<int> out_coord(rank, 0);
            for (int o = 0; o < total; ++o) {
                int r = o;
                for (int d = 0; d < rank; ++d) {
                    out_coord[d] = (r / out_stride[d]) % out_shape[d];
                }
                int in_lin = 0;
                for (int d = 0; d < rank; ++d) {
                    in_coord[d] = full_starts[d] + out_coord[d] * full_steps[d];
                    in_lin += in_coord[d] * in_stride[d];
                }
                out[static_cast<size_t>(o)] = (*src)[in_lin];
            }

            auto output = core::as_tensor<int64_t>(outputs[0]);
            // Always resize to the computed out_shape so output->size() matches
            // out.size()*sizeof(int64_t); the runtime may have pre-created the
            // output with the model's symbolic dims (e.g. [1]), which would
            // otherwise mismatch an empty slice result (total==0) and cause
            // fillToCPU to overread `out`.
            output->resize(out_shape);
            output->fillToCPU(out);
            objs_.emplace_back(output->as_storage_buffer(m_dev_, m_cmd_));
            output->copyToGPU(m_cmdpool_, out.data());
            return;
        }

        // fp16 data: CPU slice. The fp32 GPU slice shader indexes the SSBO in
        // uint words (1 word = 2 packed fp16), but inDims/outDims are in
        // elements — so for fp16 the shader's linear index runs to 2× the
        // word count and reads OOB. The rotary cos/sin Slice ([1,16,1,128]
        // fp16 -> [1,16,1,64]) is the canonical case. Small + rare -> host
        // slice is correct and cheap. Same coord-map math as the int64 path.
        if (inputs[0]->dtype() == typeid(uint16_t)) {
            auto out_shape = out_size[0];
            auto &full_starts = out_size[1];
            auto &full_steps = out_size[3];
            int total = total_elems(out_shape);
            std::vector<uint16_t> out(static_cast<size_t>(total));
            auto src = core::as_tensor<uint16_t>(inputs[0]);
            if (!src->has_cpu_data()) {
                src->copyToCPU(m_cmdpool_);
            }

            std::vector<int> in_stride(rank, 1);
            for (int d = rank - 2; d >= 0; --d) {
                in_stride[d] = in_stride[d + 1] * inshape[d + 1];
            }
            std::vector<int> out_stride(rank, 1);
            for (int d = rank - 2; d >= 0; --d) {
                out_stride[d] = out_stride[d + 1] * out_shape[d + 1];
            }
            std::vector<int> in_coord(rank, 0);
            std::vector<int> out_coord(rank, 0);
            for (int o = 0; o < total; ++o) {
                int r = o;
                for (int d = 0; d < rank; ++d) {
                    out_coord[d] = (r / out_stride[d]) % out_shape[d];
                }
                int in_lin = 0;
                for (int d = 0; d < rank; ++d) {
                    in_coord[d] = full_starts[d] + out_coord[d] * full_steps[d];
                    in_lin += in_coord[d] * in_stride[d];
                }
                out[static_cast<size_t>(o)] = (*src)[in_lin];
            }

            auto output = core::as_tensor<uint16_t>(outputs[0]);
            output->resize(out_shape);
            output->fillToCPU(out);
            objs_.emplace_back(output->as_storage_buffer(m_dev_, m_cmd_));
            output->copyToGPU(m_cmdpool_, out.data());
            return;
        }

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->num_elements() != total_elems(out_size[0])) {
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
        // SlicePC uses int[6] (not the int[8]/IArr8 convention of the other
        // buffer ops) so the push constant fits in 128 bytes — the shader's
        // slice.comp matches with int[6] fields + 6-D nd<->linear helpers.
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
