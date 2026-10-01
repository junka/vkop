// Copyright 2025 @junka
#ifndef OPS_RESIZE_HPP_
#define OPS_RESIZE_HPP_

#include "core/Tensor.hpp"
#include "ops/BufferBase.hpp"
#include "ops/Operator.hpp"
#include "ops/PimplFacade.hpp"

#include <climits>
#include <cmath>
#include <numeric>
extern "C" {
extern unsigned char image_resize_spv[];
extern unsigned int image_resize_spv_len;
extern unsigned char buffer_resize_spv[];
extern unsigned int buffer_resize_spv_len;
extern unsigned char buffer_resize_fp16_spv[];
extern unsigned int buffer_resize_fp16_spv_len;
}
namespace vkop {
namespace ops {
namespace resize {

struct GpuResizeParam {
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

class ResizeImage : public Operator {
  public:
    ResizeImage()
        : Operator(OpType::RESIZE, image_resize_spv, image_resize_spv_len,
                   {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER},
                   sizeof(resize::GpuResizeParam)) {}

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
            axes_ = parse_attr_list<int>(attrs.at("axes"));
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
        if (attrs.find("size") != attrs.end()) {
            sizes_ = parse_attr_list<int>(attrs.at("size"));
            // need to prefix with sptial
        }
        // move inputs to attr by compiler
        if (attrs.find("scales") != attrs.end()) {
            scales_ = parse_attr_list<float>(attrs.at("scales"));
            scale_valid_ = true;
        }
        if (attrs.find("sizes") != attrs.end()) {
            sizes_ = parse_attr_list<int>(attrs.at("sizes"));
            size_valid_ = true;
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
            int off = rank - static_cast<int>(sizes_.size());
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
                    roi_[i * 2] = static_cast<float>((*roi)[i * 2]);
                    roi_[(i * 2) + 1] = static_cast<float>((*roi)[(i * 2) + 1]);
                }
            });
        }
        if (inputs.size() > 3 && inputs[3]) {
            sizes = core::as_tensor<int64_t>(inputs[3]);
            sizes_.resize(rank);
            for (int i = 0; i < sizes->num_elements(); i++) {
                sizes_[i] = static_cast<int>((*sizes)[i]);
            }
            size_valid_ = true;
        }
        if (inputs.size() > 2 && inputs[2]) {
            scales = core::as_tensor<float>(inputs[2]);
            scales_.resize(rank);
            for (int i = 0; i < scales->num_elements(); i++) {
                scales_[i] = (*scales)[i];
            }
            scale_valid_ = true;
        }
        if (scale_valid_ && size_valid_) {
            throw std::runtime_error("Resize: both sizes and scales are set");
        }
        if (!scale_valid_ && size_valid_) {
            scales_.resize(rank);
            for (int i = 0; i < rank; i++) {
                scales_[i] = static_cast<float>(sizes_[i]) /
                             static_cast<float>(input_shape[i]);
            }
        } else if (!size_valid_ && scale_valid_) {
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
        if (!scale_valid_) {
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
                    out_shape[axes_[i]] = static_cast<int>(
                        std::round(scale * input_shape[axes_[i]]));
                }
            } else if (keep_aspect_ratio_policy_ ==
                       static_cast<int>(
                           resize::KeepAspectRatioPolicy::NOT_SMALLER)) {
                int scale = INT_MIN;
                for (int i = 0; i < rank; i++) {
                    scale = std::max(scale, sizes_[i] / input_shape[axes_[i]]);
                }
                for (int i = 0; i < rank; i++) {
                    out_shape[axes_[i]] = static_cast<int>(
                        std::round(scale * input_shape[axes_[i]]));
                }
            }
        } else if (!size_valid_ && !roi_.empty() &&
                   coordinate_transformation_mode_ ==
                       static_cast<int>(resize::CoordinateTransformationMode::
                                            TF_CROP_AND_RESIZE)) {
            // valid when input sizes null
            for (int i = 0; i < rank; i++) {
                out_shape[i] = static_cast<int>(std::floor(
                    input_shape[i] * (roi_[rank + i] - roi_[i]) * scales_[i]));
            }
        } else if (!size_valid_) {
            for (int i = 0; i < rank; i++) {
                out_shape[i] =
                    static_cast<int>(std::floor(input_shape[i] * scales_[i]));
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

        auto out_gpu_shape = outputs[0]->getGPUShape();
        resize::GpuResizeParam para;
        inputs[0]->get_shape(para.inShape);
        outputs[0]->get_shape(para.outShape);
        para.mode = mode_;
        para.nearest_mode = nearest_mode_;
        para.antialias = antialias_;
        para.coordinate_transformation_mode = coordinate_transformation_mode_;
        para.cubic_coeff_a = cubic_coeff_a_;

        submit(&para, UP_DIV(out_gpu_shape[0], 16),
               UP_DIV(out_gpu_shape[1], 16), out_gpu_shape[2]);
    }

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
    bool size_valid_ = false;
    bool scale_valid_ = false;
    std::vector<float> scales_;
    std::vector<float> roi_;
};
// Buffer (SSBO) Resize. Nearest-neighbour with asymmetric/floor coordinate
// mapping: out element (n,c,y,x...) reads in coord floor(out*i*in_dim/out_dim)
// per axis, so the whole op is an indexed copy — the index math is exact in
// integer arithmetic (no scale floats in the push constant).
struct ResizeBufferPC {
    int rank;
    int fp16;
    int inDims[8];
    int outDims[8];
    int _pad0;
    int _pad1;
};

class ResizeBuffer : public BufferFactory {
  public:
    explicit ResizeBuffer(int fp16)
        : BufferFactory(
              OpType::RESIZE, fp16 ? buffer_resize_fp16_spv : buffer_resize_spv,
              fp16 ? buffer_resize_fp16_spv_len : buffer_resize_spv_len,
              {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
              sizeof(ResizeBufferPC), fp16) {}

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        auto get = [&](const char *key) -> const std::string * {
            auto it = attributes.find(key);
            return it == attributes.end() ? nullptr : &it->second;
        };
        if (const auto *s = get("mode")) {
            if (*s == "nearest") {
                mode_ = 0;
            } else if (*s == "linear" || *s == "bilinear") {
                mode_ = 1;
            } else {
                mode_ = 2;
            }
        }
        if (const auto *s = get("coordinate_transformation_mode")) {
            coordinate_transformation_mode_ =
                (*s == "asymmetric")
                    ? static_cast<int>(
                          resize::CoordinateTransformationMode::ASYMMETRIC)
                    : -1;
        }
        if (const auto *s = get("axes"); s && !s->empty()) {
            axes_ = parse_attr_list<int>(*s);
        }
        if (const auto *s = get("scales"); s && !s->empty()) {
            scales_ = parse_attr_list<float>(*s);
        }
        if (const auto *s = get("sizes"); s && !s->empty()) {
            sizes_ = parse_attr_list<int>(*s);
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        if (mode_ != 0 ||
            coordinate_transformation_mode_ !=
                static_cast<int>(
                    resize::CoordinateTransformationMode::ASYMMETRIC)) {
            throw std::runtime_error(
                "Resize (buffer backend) supports only mode=nearest with "
                "coordinate_transformation_mode=asymmetric");
        }

        auto inshape = inputs[0]->getShape();
        int rank = static_cast<int>(inshape.size());

        // The converter folds the ONNX scales/sizes inputs into attributes;
        // either way the output dims come from the host (the shader does not
        // read shape buffers). scales are per-axis and may cover only the
        // resized axes (the converter emits `axes` in that case).
        std::vector<int> outshape = inshape;
        if (!sizes_.empty()) {
            int off = rank - static_cast<int>(sizes_.size());
            for (int i = 0; i < static_cast<int>(sizes_.size()); ++i) {
                int d = sizes_[i];
                outshape[off + i] = (d > 0) ? d : inshape[off + i];
            }
        } else if (!scales_.empty()) {
            int off = rank - static_cast<int>(scales_.size());
            if (!axes_.empty() && static_cast<int>(axes_.size()) ==
                                      static_cast<int>(scales_.size())) {
                for (size_t i = 0; i < axes_.size(); ++i) {
                    int a = axes_[i] < 0 ? axes_[i] + rank : axes_[i];
                    outshape[a] = static_cast<int>(inshape[a] * scales_[i]);
                }
            } else {
                for (int i = 0; i < static_cast<int>(scales_.size()); ++i) {
                    outshape[off + i] =
                        static_cast<int>(inshape[off + i] * scales_[i]);
                }
            }
        } else {
            auto recorded = outputs[0]->getShape();
            if (recorded.size() != static_cast<size_t>(rank)) {
                throw std::runtime_error(
                    "Resize (buffer backend): no scales/sizes attribute and "
                    "no recorded output shape");
            }
            outshape = recorded;
        }

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->num_elements() != total_elems(outshape)) {
                output->resize(outshape);
            }
            bind_ssbo<T>(outputs[0], /*is_output=*/true);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            bind_ssbo<T>(inputs[0], /*is_output=*/false);
        });

        ResizeBufferPC pc{};
        pc.rank = rank;
        pc.fp16 = (fp16_ != 0) ? 1 : 0;
        fill_dims(pc.inDims, inshape);
        fill_dims(pc.outDims, outshape);
        // fp16 packs two elements per uint word; dispatch one thread per word
        // (the fp16 shader writes each word once — no RMW race).
        int total = total_elems(outshape);
        int nthreads = (fp16_ != 0) ? (total + 1) / 2 : total;
        submit(&pc, UP_DIV(nthreads, 256), 1, 1);
    }

    int mode_ = 0;
    int coordinate_transformation_mode_ = 4; // ASYMMETRIC
    std::vector<int> axes_;
    std::vector<float> scales_;
    std::vector<int> sizes_;
};

// PIMPL façade: buffer SSBO impl when backend_buffer is set, else image.
class Resize : public PimplFacade {
  public:
    Resize(int fp16, bool backend_buffer) : PimplFacade(OpType::RESIZE) {
        impl_ = backend_buffer ? std::unique_ptr<Operator>(
                                     std::make_unique<ResizeBuffer>(fp16))
                               : std::make_unique<ResizeImage>();
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_RESIZE_HPP_
