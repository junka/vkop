// Copyright 2025 @junka
#ifndef OPS_RESIZE_HPP_
#define OPS_RESIZE_HPP_
#include "UnaryFactory.hpp"
#include <climits>
#include <cmath>
#include <numeric>

extern unsigned char resize_spv[];
extern unsigned int resize_spv_len;

namespace vkop {
namespace ops {
namespace resize {

using ivec4 = int[4];
using ivec2 = int[2];

struct GpuResizeParam {
    ivec4 outImgSize;
    ivec4 inShape;
    ivec4 outShape; // N C H W
    int mode;
    int nearest_mode;
    int antialias;
    int coordinate_transformation_mode;
    float cubic_coeff_a;
};

enum class NearestMode { ROUND_PREFER_FLOOR, ROUND_PREFER_CEIL, FLOOR, CEIL };

enum class CoordinateTransformationMode {
    HALF_PIXEL,
    HALF_PIXEL_SYMMETRIC,
    PYTORCH_HALF_PIXEL,
    ALIGN_CORNERS,
    ASYMMETRIC,
    TF_CROP_AND_RESIZE,
};

enum class ResizeMode {
    NEAREST,
    LINEAR,
    CUBIC,
};

enum class KeepAspectRatioPolicy { STRETCH, NOT_LARGER, NOT_SMALLER };

} // namespace resize

class Resize : public Operator {
  public:
    Resize()
        : Operator(OpType::RESIZE, resize_spv, resize_spv_len,
                   sizeof(resize::GpuResizeParam)) {
        n_imgs_ = 2;
        types_ = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER};
        objs_.reserve(types_.size());
    }

    void setAttribute(
        const std::unordered_map<std::string, std::string> &attrs) override {
        Operator::setAttribute(attrs);
        if (attrs.find("antialias") != attrs.end()) {
            if (attrs.at("antialias") == "False") {
                antialias_ = 0;
            } else if (attrs.at("antialias") == "True") {
                antialias_ = 1;
            } else {
                antialias_ = std::stol(attrs.at("antialias"));
            }
        }
        if (attrs.find("axes") != attrs.end()) {
            axes_ = parse_attr_list(attrs.at("axes"));
        }
        if (attrs.find("coordinate_transformation_mode") != attrs.end()) {

            static const std::unordered_map<
                std::string, resize::CoordinateTransformationMode>
                kTmodeMap = {
                    {"half_pixel",
                     resize::CoordinateTransformationMode::HALF_PIXEL},
                    {"half_pixel_symmetric",
                     resize::CoordinateTransformationMode::
                         HALF_PIXEL_SYMMETRIC},
                    {"pytorch_half_pixel",
                     resize::CoordinateTransformationMode::PYTORCH_HALF_PIXEL},
                    {"align_corners",
                     resize::CoordinateTransformationMode::ALIGN_CORNERS},
                    {"asymmetric",
                     resize::CoordinateTransformationMode::ASYMMETRIC},
                    {"tf_crop_and_resize",
                     resize::CoordinateTransformationMode::TF_CROP_AND_RESIZE}};
            auto it =
                kTmodeMap.find(attrs.at("coordinate_transformation_mode"));
            if (it != kTmodeMap.end()) {
                coordinate_transformation_mode_ = static_cast<int>(it->second);
            }
        }

        if (attrs.find("cubic_coeff_a") != attrs.end()) {
            cubic_coeff_a_ = std::stof(attrs.at("cubic_coeff_a"));
        }

        if (attrs.find("exclude_outside") != attrs.end()) {
            exclude_outside_ = std::stol(attrs.at("exclude_outside"));
        }
        if (attrs.find("extrapolation_value") != attrs.end()) {
            extrapolation_value_ = std::stof(attrs.at("extrapolation_value"));
        }
        if (attrs.find("keep_aspect_ratio") != attrs.end()) {

            static const std::unordered_map<std::string,
                                            resize::KeepAspectRatioPolicy>
                kPolicyMap = {
                    {"stretch", resize::KeepAspectRatioPolicy::STRETCH},
                    {"not_larger", resize::KeepAspectRatioPolicy::NOT_LARGER},
                    {"not_smaller",
                     resize::KeepAspectRatioPolicy::NOT_SMALLER}};
            const auto &policy_str = attrs.at("keep_aspect_ratio_policy");
            auto it = kPolicyMap.find(policy_str);
            if (it != kPolicyMap.end()) {
                keep_aspect_ratio_policy_ = static_cast<int>(it->second);
            }
        }
        if (attrs.find("mode") != attrs.end()) {
            const std::string &mode_value = attrs.at("mode");
            if (mode_value == "nearest") {
                mode_ = 0;
            } else if (mode_value == "linear" || mode_value == "bilinear") {
                mode_ = 1;
            } else if (mode_value == "cubic" || mode_value == "bicubic") {
                mode_ = 2;
            }
        }
        if (attrs.find("coordinate_transformation_mode") != attrs.end()) {

            static std::unordered_map<std::string, resize::NearestMode>
                mode_map = {{"round_prefer_floor",
                             resize::NearestMode::ROUND_PREFER_FLOOR},
                            {"round_prefer_ceil",
                             resize::NearestMode::ROUND_PREFER_CEIL},
                            {"floor", resize::NearestMode::FLOOR},
                            {"ceil", resize::NearestMode::CEIL}};
            auto itr = mode_map.find(attrs.at("nearest_mode"));
            if (itr != mode_map.end()) {
                nearest_mode_ = static_cast<int>(itr->second);
            }
        }
        // only for torch test, it should be the fourth input
        if (attrs.find("size") != attrs.end()) {
            sizes_ = parse_attr_list(attrs.at("size"));
            // need to prefix with sptial
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        std::shared_ptr<core::Tensor<int64_t>> sizes = nullptr;
        std::shared_ptr<core::Tensor<float>> scales = nullptr;

        auto input_shape = inputs[0]->getShape();
        int rank = inputs[0]->num_dims();
        if (!sizes_.empty() && static_cast<int>(sizes_.size()) < rank) {
            // only for torch test case
            int off = rank - sizes_.size();
            sizes_.resize(rank);
            for (int i = rank - 1; i >= off; i--) {
                sizes_[i] = sizes_[i - off];
            }
            for (int i = 0; i < off; i++) {
                sizes_[i] = input_shape[i];
            }
        }

        if (inputs[1]) {
            // doube, float, fp16
            dispatch_by_dtype(inputs[1]->dtype(), [&](auto t) {
                using T = decltype(t);
                auto roi = core::as_tensor<T>(inputs[1]);
                roi_.resize(rank * 2);
                for (int i = 0; i < rank; i++) {
                    roi_[i * 2] = (*roi)[i * 2];
                    roi_[(i * 2) + 1] = (*roi)[(i * 2) + 1];
                }
            });
        }
        if (inputs.size() > 3 && inputs[3]) {
            sizes = core::as_tensor<int64_t>(inputs[3]);
            sizes_.resize(rank);
            for (int i = 0; i < sizes->num_elements(); i++) {
                sizes_[i] = (*sizes)[i];
            }
        }
        if (inputs.size() > 2 && inputs[2]) {
            scales = core::as_tensor<float>(inputs[2]);
            scales_.resize(rank);
            for (int i = 0; i < scales->num_elements(); i++) {
                scales_[i] = (*scales)[i];
            }
        }
        if (sizes && scales) {
            throw std::runtime_error("Resize: both sizes and scales are set");
        }
        if (sizes && !scales) {
            scales_.resize(rank);
            for (int i = 0; i < rank; i++) {
                scales_[i] = static_cast<float>(sizes_[i]) /
                             static_cast<float>(input_shape[i]);
            }
        } else if (!sizes && scales) {
            sizes_.resize(rank);
            for (int i = 0; i < rank; i++) {
                sizes_[i] = static_cast<int>(input_shape[i] * scales_[i]);
            }
        }

        if (axes_.size() == 0) {
            axes_ = std::vector<int>(rank);
            std::iota(axes_.begin(), axes_.end(), 0);
        } else {
            for (int &axe : axes_) {
                if (axe < 0) {
                    axe += rank;
                }
            }
        }

        std::vector<int> out_shape = input_shape;
        if (!scales) {
            // keep_aspect_ratio_policy valid when scales is null
            if (keep_aspect_ratio_policy_ ==
                static_cast<int>(resize::KeepAspectRatioPolicy::STRETCH)) {
                for (int i = 0; i < rank; i++) {
                    out_shape[axes_[i]] = sizes_[i];
                }
            } else if (keep_aspect_ratio_policy_ ==
                       static_cast<int>(
                           resize::KeepAspectRatioPolicy::NOT_LARGER)) {
                int scale = INT_MAX;
                for (int i = 0; i < rank; i++) {
                    scale = std::min(scale, sizes_[i] / input_shape[axes_[i]]);
                }
                for (int i = 0; i < rank; i++) {
                    out_shape[axes_[i]] =
                        std::round(scale * input_shape[axes_[i]]);
                }
            } else if (keep_aspect_ratio_policy_ ==
                       static_cast<int>(
                           resize::KeepAspectRatioPolicy::NOT_SMALLER)) {
                int scale = INT_MIN;
                for (int i = 0; i < rank; i++) {
                    scale = std::max(scale, sizes_[i] / input_shape[axes_[i]]);
                }
                for (int i = 0; i < rank; i++) {
                    out_shape[axes_[i]] =
                        std::round(scale * input_shape[axes_[i]]);
                }
            }
        } else if (!sizes && !roi_.empty() &&
                   coordinate_transformation_mode_ ==
                       static_cast<int>(resize::CoordinateTransformationMode::
                                            TF_CROP_AND_RESIZE)) {
            // valid when input sizes null
            for (int i = 0; i < rank; i++) {
                out_shape[i] = std::floor(
                    input_shape[i] * (roi_[rank + i] - roi_[i]) * scales_[i]);
            }
        } else if (!sizes) {
            for (int i = 0; i < rank; i++) {
                out_shape[i] = std::floor(input_shape[i] * scales_[i]);
            }
        }

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto t) {
            using T = decltype(t);
            auto outputptr = core::as_tensor<T>(outputs[0]);
            if (outputptr->size() == 0) {
                outputptr->resize(out_shape);
            }
            auto output_image = outputptr->as_output_image(m_dev_, m_cmd_);
            objs_.emplace_back(output_image);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto t) {
            using T = decltype(t);
            auto inputptr = core::as_tensor<T>(inputs[0]);
            auto input_image = inputptr->as_input_image(m_dev_, m_cmd_);
            objs_.emplace_back(input_image);
        });

        auto outGPUshape = outputs[0]->getGPUShape();
        resize::GpuResizeParam para;
        para.outImgSize[0] = outGPUshape[0];
        para.outImgSize[1] = outGPUshape[1];
        para.outImgSize[2] = outGPUshape[2];
        para.outImgSize[3] = 0;
        para.inShape[0] = inputs[0]->get_batch();
        para.inShape[1] = inputs[0]->get_channel();
        para.inShape[2] = inputs[0]->get_height();
        para.inShape[3] = inputs[0]->get_width();
        para.outShape[0] = outputs[0]->get_batch();
        para.outShape[1] = outputs[0]->get_channel();
        para.outShape[2] = outputs[0]->get_height();
        para.outShape[3] = outputs[0]->get_width();
        para.mode = mode_;
        para.nearest_mode = nearest_mode_;
        para.antialias = antialias_;
        para.coordinate_transformation_mode = coordinate_transformation_mode_;
        para.cubic_coeff_a = cubic_coeff_a_;

        submit(&para, UP_DIV(outGPUshape[0], 16), UP_DIV(outGPUshape[1], 16),
               outGPUshape[2]);
    }

  private:
    int antialias_ = 0;
    std::vector<int> axes_;
    int coordinate_transformation_mode_ = 0;
    float cubic_coeff_a_ = -0.75;
    int exclude_outside_ = 0;
    float extrapolation_value_ = 0.0F;
    int keep_aspect_ratio_policy_ = 0;
    int mode_ = 0;
    int nearest_mode_ = 0;
    std::vector<int> sizes_;
    std::vector<float> scales_;
    std::vector<float> roi_;
};

} // namespace ops
} // namespace vkop
#endif // OPS_RESIZE_HPP_
