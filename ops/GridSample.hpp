// Copyright 2025 @junka
#ifndef OPS_GRIDSAMPLE_HPP_
#define OPS_GRIDSAMPLE_HPP_

#include <unistd.h>
#include <vector>

#include "Operator.hpp"

extern unsigned char gridsample_spv[];
extern unsigned int gridsample_spv_len;

namespace vkop {
namespace ops {

namespace gridsample {
enum class InterpolationMode { BILINEAR = 0, NEAREST = 1 };

enum class PaddingMode { ZEROS = 0, BORDER = 1, REFLECTION = 2 };

using ivec4 = int[4];
using ivec2 = int[2];

struct alignas(32) GpuGridSampleParam {
    ivec4 outImgSize;
    ivec2 inShape;
    ivec2 outShape;
    int padding_mode;
    int interpolation_mode;
    int align_corners;
};
} // namespace gridsample

class GridSample : public Operator {
  public:
    GridSample()
        : Operator(OpType::GRIDSAMPLE, gridsample_spv, gridsample_spv_len,
                   sizeof(gridsample::GpuGridSampleParam)) {
        n_imgs_ = 3;
        types_ = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER};
        objs_.reserve(types_.size());
    }

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

} // namespace ops
} // namespace vkop

#endif // OPS_GRIDSAMPLE_HPP_
