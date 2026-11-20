// Copyright 2025 @junka
#ifndef OPS_GRIDSAMPLE_HPP_
#define OPS_GRIDSAMPLE_HPP_

#include "ops/BufferBase.hpp"
#include "ops/PimplFacade.hpp"
#include <vector>

#include "Operator.hpp"
namespace vkop {
namespace ops {

namespace gridsample {
enum class InterpolationMode { BILINEAR = 0, NEAREST = 1 };

enum class PaddingMode { ZEROS = 0, BORDER = 1, REFLECTION = 2 };

struct alignas(32) GpuGridSampleParam {
    ivec4 outImgSize;
    ivec2 inShape;
    ivec2 outShape;
    int padding_mode;
    int interpolation_mode;
    int align_corners;
};

// Buffer PC (mirrors shaders/buffer/gridsample.comp). Input/grid/output are
// compact (N, C, H, W) / (N, Ho, Wo, 2) / (N, C, Ho, Wo) row-major planes.
// `fp16` says both the data and grid edges pack two halfs per word (they share
// the graph dtype), and one module serves both layouts.
struct alignas(16) GridSamplePC {
    int N;
    int C;
    int H_in;
    int W_in;
    int H_out;
    int W_out;
    int padding_mode;
    int interpolation_mode;
    int align_corners;
    int fp16;      // input/output packing (two halfs per word)
    int grid_fp16; // grid packing — ONNX does not tie it to the data dtype
    int _pad1;
};
static_assert(sizeof(GridSamplePC) == 48, "GridSamplePC drift");
static_assert(sizeof(GridSamplePC) <= 128, "GridSamplePC PC overflow");
} // namespace gridsample

class GridSampleImage : public Operator {
  public:
    GridSampleImage()
        : Operator(OpType::GRIDSAMPLE,
                   {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER},
                   sizeof(gridsample::GpuGridSampleParam)) {}

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        attributes.find("align_corners") != attributes.end()
            ? align_corners_ = (attributes.at("align_corners") == "1" ||
                                attributes.at("align_corners") == "true")
            : align_corners_ = false;
        if (attributes.find("interpolation_mode") != attributes.end()) {
            std::string mode = attributes.at("interpolation_mode");
            if (mode == "linear" || mode == "bilinear") {
                interpolation_mode_ = gridsample::InterpolationMode::BILINEAR;
            } else if (mode == "nearest") {
                interpolation_mode_ = gridsample::InterpolationMode::NEAREST;
            } else {
                throw std::invalid_argument("Unsupported interpolation_mode: " +
                                            mode);
            }
        } else {
            interpolation_mode_ = gridsample::InterpolationMode::BILINEAR;
        }
        if (attributes.find("padding_mode") != attributes.end()) {
            std::string mode = attributes.at("padding_mode");
            if (mode == "zeros") {
                padding_mode_ = gridsample::PaddingMode::ZEROS;
            } else if (mode == "border") {
                padding_mode_ = gridsample::PaddingMode::BORDER;
            } else if (mode == "reflection") {
                padding_mode_ = gridsample::PaddingMode::REFLECTION;
            } else {
                throw std::invalid_argument("Unsupported padding_mode: " +
                                            mode);
            }
        } else {
            padding_mode_ = gridsample::PaddingMode::ZEROS;
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {

        auto input_shape = inputs[0]->getShape();
        auto grid_shape = inputs[1]->getShape();
        if (input_shape.size() != 4 || grid_shape.size() != 4) {
            throw std::invalid_argument(
                "Input and grid must have 4 dimensions.");
        }

        int batch = input_shape[0];
        int depth = input_shape[1];
        int out_height = grid_shape[1];
        int out_width = grid_shape[2];

        int in_height = input_shape[2];
        int in_width = input_shape[3];
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto t) {
            using T = decltype(t);
            auto outputptr = core::as_tensor<T>(outputs[0]);
            if (outputptr->size() == 0) {
                outputptr->resize(batch, depth, out_height, out_width);
            }
            auto output_image = outputptr->as_output_image(m_dev_, m_cmd_);
            objs_.emplace_back(output_image);
        });

        for (size_t i = 0; i <= 1; ++i) {
            dispatch_by_dtype(inputs[i]->dtype(), [&](auto t) {
                using T = decltype(t);
                auto inputptr = core::as_tensor<T>(inputs[i]);
                auto input_image = inputptr->as_input_image(m_dev_, m_cmd_);
                objs_.emplace_back(input_image);
            });
        }

        int realheight = out_height * batch;

        gridsample::GpuGridSampleParam para;
        para.outImgSize[0] = out_width;
        para.outImgSize[1] = realheight;
        para.outImgSize[2] = UP_DIV(depth, 4);
        para.outImgSize[3] = 0;
        para.inShape[0] = in_width;
        para.inShape[1] = in_height;
        para.outShape[0] = out_width;
        para.outShape[1] = out_height;
        para.padding_mode = static_cast<int>(padding_mode_);
        para.interpolation_mode = static_cast<int>(interpolation_mode_);
        para.align_corners = align_corners_ ? 1 : 0;

        submit(&para, UP_DIV(out_width, 16), UP_DIV(realheight, 16),
               UP_DIV(depth, 4));
    }

    gridsample::InterpolationMode interpolation_mode_ =
        gridsample::InterpolationMode::BILINEAR;
    gridsample::PaddingMode padding_mode_ = gridsample::PaddingMode::ZEROS;
    bool align_corners_ = false;
};
// Buffer factory — SSBO-backed 2-D grid sampler, the buffer twin of
// GridSampleImage (same coordinate/interpolation/padding math, see
// shaders/buffer/gridsample.comp). Both interpolation modes (bilinear,
// nearest) and all three padding modes (zeros, border, reflection) are
// implemented; bicubic/3-D grids are not part of this op's contract and the
// attribute parser rejects unknown spellings by name.
class GridSampleBuffer : public BufferFactory {
  public:
    explicit GridSampleBuffer(int fp16)
        : BufferFactory(OpType::GRIDSAMPLE,
                        {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                         DESCRIPTOR_TYPE_STORAGE},
                        sizeof(gridsample::GridSamplePC), fp16) {}

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        auto it = attributes.find("align_corners");
        align_corners_ = it != attributes.end() &&
                         (it->second == "1" || it->second == "true");
        if (const auto *s = [&]() -> const std::string * {
                auto f = attributes.find("interpolation_mode");
                return f == attributes.end() ? nullptr : &f->second;
            }()) {
            if (*s == "linear" || *s == "bilinear") {
                interpolation_mode_ = gridsample::InterpolationMode::BILINEAR;
            } else if (*s == "nearest") {
                interpolation_mode_ = gridsample::InterpolationMode::NEAREST;
            } else {
                throw std::invalid_argument(
                    "GridSample (buffer backend): unsupported "
                    "interpolation_mode '" +
                    *s + "' (expected linear/bilinear or nearest)");
            }
        }
        if (const auto *s = [&]() -> const std::string * {
                auto f = attributes.find("padding_mode");
                return f == attributes.end() ? nullptr : &f->second;
            }()) {
            if (*s == "zeros") {
                padding_mode_ = gridsample::PaddingMode::ZEROS;
            } else if (*s == "border") {
                padding_mode_ = gridsample::PaddingMode::BORDER;
            } else if (*s == "reflection") {
                padding_mode_ = gridsample::PaddingMode::REFLECTION;
            } else {
                throw std::invalid_argument(
                    "GridSample (buffer backend): unsupported padding_mode '" +
                    *s + "' (expected zeros, border or reflection)");
            }
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto in_shape = inputs[0]->getShape();
        auto grid_shape = inputs[1]->getShape();
        if (in_shape.size() != 4 || grid_shape.size() != 4) {
            throw std::invalid_argument(
                "GridSample (buffer backend): input and grid must both be "
                "rank 4 (NCHW and NHW2)");
        }
        if (grid_shape[3] != 2) {
            throw std::invalid_argument(
                "GridSample (buffer backend): grid's last dim must be 2");
        }
        int N = in_shape[0];
        int C = in_shape[1];
        int H_in = in_shape[2];
        int W_in = in_shape[3];
        int H_out = grid_shape[1];
        int W_out = grid_shape[2];
        std::vector<int> out_shape = {N, C, H_out, W_out};

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->num_elements() != total_elems(out_shape)) {
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

        gridsample::GridSamplePC pc{};
        pc.N = N;
        pc.C = C;
        pc.H_in = H_in;
        pc.W_in = W_in;
        pc.H_out = H_out;
        pc.W_out = W_out;
        pc.padding_mode = static_cast<int>(padding_mode_);
        pc.interpolation_mode = static_cast<int>(interpolation_mode_);
        pc.align_corners = align_corners_ ? 1 : 0;
        // Single module for both layouts: the packed-half flags follow the
        // tensors actually bound, not the constructor's precision hint. The
        // grid is tracked separately because ONNX does not tie its dtype to
        // the data's.
        const bool data_fp16 = inputs[0]->dtype() == typeid(uint16_t);
        pc.fp16 = data_fp16 ? 1 : 0;
        pc.grid_fp16 = (inputs[1]->dtype() == typeid(uint16_t)) ? 1 : 0;

        int total = total_elems(out_shape);
        int nthreads = data_fp16 ? (total + 1) / 2 : total;
        submit(&pc, UP_DIV(nthreads, 256), 1, 1);
    }

    gridsample::InterpolationMode interpolation_mode_ =
        gridsample::InterpolationMode::BILINEAR;
    gridsample::PaddingMode padding_mode_ = gridsample::PaddingMode::ZEROS;
    bool align_corners_ = false;
};

// PIMPL façade: buffer SSBO impl when backend_buffer is set, else image.
class GridSample : public PimplFacade {
  public:
    GridSample(int fp16, bool backend_buffer)
        : PimplFacade(OpType::GRIDSAMPLE) {
        impl_ = backend_buffer
                    ? std::unique_ptr<Operator>(
                          std::make_unique<GridSampleBuffer>(fp16))
                    : std::make_unique<GridSampleImage>();
    }
};

} // namespace ops
} // namespace vkop

#endif // OPS_GRIDSAMPLE_HPP_
