// Copyright 2025 @junka
#ifndef OPS_RESIZE_HPP_
#define OPS_RESIZE_HPP_

#include "UnaryFactory.hpp"

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

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("antialias") != attributes.end()) {
            if (attributes.at("antialias") == "False") {
                antialias_ = 0;
            } else if (attributes.at("antialias") == "True") {
                antialias_ = 1;
            } else {
                antialias_ = std::stol(attributes.at("antialias"));
            }
        }
        if (attributes.find("axes") != attributes.end()) {
            axes_ = parse_attr_list(attributes.at("axes"));
        } else {
            axes_ = {2, 3};
        }

        if (attributes.find("coordinate_transformation_mode") !=
            attributes.end()) {
            std::string valid_coord_mode[] = {
                "half_pixel",    "half_pixel_symmetric", "pytorch_half_pixel",
                "align_corners", "asymmetric",           "tf_crop_and_resize",
            };
            auto str = attributes.at("coordinate_transformation_mode");
            for (int i = 0; i < 6; i++) {
                if (valid_coord_mode[i] == str) {
                    coordinate_transformation_mode_ = i;
                    break;
                }
            }
        } else if (attributes.find("align_corners") != attributes.end()) {
            if (attributes.at("align_corners") == "True" ||
                attributes.at("align_corners") == "true") {
                coordinate_transformation_mode_ = 3;
            }
        }

        if (attributes.find("cubic_coeff_a") != attributes.end()) {
            cubic_coeff_a_ = std::stof(attributes.at("cubic_coeff_a"));
        }

        if (attributes.find("exclude_outside") != attributes.end()) {
            exclude_outside_ = std::stol(attributes.at("exclude_outside"));
        }
        if (attributes.find("extrapolation_value") != attributes.end()) {
            extrapolation_value_ =
                std::stof(attributes.at("extrapolation_value"));
        }
        if (attributes.find("keep_aspect_ratio_policy") != attributes.end()) {
            std::string valid_keep_str[] = {
                "stretch",
                "not_larger",
                "not_smaller",
            };
            for (int i = 0; i < 3; i++) {
                if (valid_keep_str[i] ==
                    attributes.at("keep_aspect_ratio_policy")) {
                    keep_aspect_ratio_policy_ = i;
                    break;
                }
            }
        }
        if (attributes.find("mode") != attributes.end()) {
            std::string valid_mode_str[] = {"nearest", "linear", "cubic"};
            if (attributes.at("mode") == "bilinear") {
                mode_ = 1;
            } else if (attributes.at("mode") == "bicubic") {
                mode_ = 2;
            } else {
                for (int i = 0; i < 3; i++) {
                    if (valid_mode_str[i] == attributes.at("mode")) {
                        mode_ = i;
                        break;
                    }
                }
            }
        }
        if (attributes.find("nearest_mode") != attributes.end()) {
            std::string valid_mode_str[] = {
                "round_prefer_floor", "round_prefer_ceil", "floor", "ceil"};
            for (int i = 0; i < 3; i++) {
                if (valid_mode_str[i] == attributes.at("mode")) {
                    nearest_mode_ = i;
                    break;
                }
            }
        }
        if (attributes.find("sizes") != attributes.end()) {
            size_ = parse_attr_list(attributes.at("sizes"));
        } else if (attributes.find("size") != attributes.end()) {
            size_ = parse_attr_list(attributes.at("size"));
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto sizes = core::as_tensor<int64_t>(inputs[3]);

        auto input_shape = inputs[0]->getShape();
        if (input_shape.size() != 4) {
            throw std::invalid_argument("Input must have 4 dimensions.");
        }
        int batch = input_shape[0];
        int depth = input_shape[1];
        int out_height = input_shape[2];
        int out_width = input_shape[3];

        if (size_.size() == 4) {
            size_[0] = size_[2];
            size_[1] = size_[3];
            size_.resize(2);
        } else if (size_.empty()) {
            size_.resize(2);
            size_[0] = static_cast<int>((*sizes)[static_cast<std::size_t>(2)]);
            size_[1] = static_cast<int>((*sizes)[static_cast<std::size_t>(3)]);
        }

        assert(axes_[0] == 2 && axes_[1] == 3);
        assert(axes_.size() == size_.size());

        if (keep_aspect_ratio_policy_ == 0) {
            out_height = size_[0];
            out_width = size_[1];
        } else {
            float h_scale = size_[0] / static_cast<float>(input_shape[2]);
            float w_scale = size_[1] / static_cast<float>(input_shape[3]);
            auto scale = keep_aspect_ratio_policy_ == 1
                             ? std::min(h_scale, w_scale)
                             : std::max(h_scale, w_scale);
            out_height = input_shape[2] * scale;
            out_width = input_shape[3] * scale;
        }
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto t) {
            using T = decltype(t);
            auto outputptr = core::as_tensor<T>(outputs[0]);
            if (outputptr->size() == 0) {
                outputptr->resize(batch, depth, out_height, out_width);
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
        auto roi = core::as_tensor<float>(inputs[1]);
        auto scales = core::as_tensor<float>(inputs[2]);

        int realwidth = out_width * UP_DIV(depth, 4);
        int realheight = out_height * batch;

        resize::GpuResizeParam para;
        para.outImgSize[0] = realwidth;
        para.outImgSize[1] = realheight;
        para.outImgSize[2] = 1;
        para.outImgSize[3] = 0;
        para.inShape[0] = input_shape[0];
        para.inShape[1] = input_shape[1];
        para.inShape[2] = input_shape[2];
        para.inShape[3] = input_shape[3];
        para.outShape[0] = batch;
        para.outShape[1] = depth;
        para.outShape[2] = out_height;
        para.outShape[3] = out_width;
        para.mode = mode_;
        para.nearest_mode = nearest_mode_;
        para.antialias = antialias_;
        para.coordinate_transformation_mode = coordinate_transformation_mode_;
        para.cubic_coeff_a = cubic_coeff_a_;

        submit(&para, out_width, realheight, UP_DIV(depth, 4));
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
    std::vector<int> size_;
    std::vector<float> scale_factor_;
};

} // namespace ops
} // namespace vkop
#endif // OPS_RESIZE_HPP_
