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
    Resize() : Operator(OpType::RESIZE){};

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("antialias") != attributes.end()) {
            if (attributes.at("antialias") == "False") {
                antialias_ = 0;
            } else if (attributes.at("antialias") == "True") {
                antialias_ = 1;
            } else {
                antialias_ = std::stoi(attributes.at("antialias"));
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
            exclude_outside_ = std::stoi(attributes.at("exclude_outside"));
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
    template <typename T>
    void prepare(std::vector<std::shared_ptr<core::ITensor>> inputs,
                 std::vector<std::shared_ptr<core::ITensor>> outputs) {
        auto input = core::as_tensor<T>(inputs[0]);
        auto output = core::as_tensor<T>(outputs[0]);

        auto input_shape = input->getTensorShape();

        VkDevice device = m_dev_->getLogicalDevice();
        int exflags = 0;
        if (m_dev_->is_support_host_image_copy()) {
#ifdef VK_EXT_host_image_copy
            exflags |= VK_IMAGE_USAGE_HOST_TRANSFER_BIT;
#endif
        }

        outputImage_ = output->make_vkimg(
            m_dev_, VK_IMAGE_USAGE_STORAGE_BIT |
                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | exflags);

        inputImage_ = input->make_vkimg(
            m_dev_, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT | exflags);

        paramBuffer_ = std::make_shared<VulkanBuffer>(
            m_dev_, sizeof(resize::GpuResizeParam),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
#ifdef VK_EXT_host_image_copy
        if (m_dev_->is_support_host_image_copy()) {
            if (m_dev_->checkHostImageCopyDstLayoutSupport(
                    VK_IMAGE_LAYOUT_GENERAL)) {
                outputImage_->hostImaggeTransition(VK_IMAGE_LAYOUT_GENERAL);
            } else {
                outputImage_->hostImaggeTransition(
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            }
            inputImage_->hostImaggeTransition(
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        } else
#endif
        {
            VulkanCommandBuffer cmd(device, m_cmdpool_->getCommandPool());
            cmd.begin();
            outputImage_->writeBarrier(cmd.get());
            inputImage_->readBarrier(cmd.get());
            cmd.end();
            cmd.submit(m_dev_->getComputeQueue());
        }
    }

    template <typename T>
    void apply(std::vector<std::shared_ptr<core::ITensor>> inputs,
               std::vector<std::shared_ptr<core::ITensor>> outputs) {
        auto input = core::as_tensor<T>(inputs[0]);
        auto output = core::as_tensor<T>(outputs[0]);
        auto roi = core::as_tensor<T>(inputs[1]);
        auto scales = core::as_tensor<T>(inputs[2]);
        auto sizes = core::as_tensor<int64_t>(inputs[3]);

        auto input_shape = input->getTensorShape();
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

        int realwidth = out_width * UP_DIV(depth, 4);
        int realheight = out_height * batch;

        if (output->size() == 0) {
            output->resize(batch, depth, out_height, out_width);
        }
        prepare<T>(inputs, outputs);

        auto *para = static_cast<resize::GpuResizeParam *>(
            paramBuffer_->getMappedMemory());
        // vkimage params
        para->outImgSize[0] = realwidth;
        para->outImgSize[1] = realheight;
        para->outImgSize[2] = 1;
        para->outImgSize[3] = 0;
        para->inShape[0] = input_shape[0];
        para->inShape[1] = input_shape[1];
        para->inShape[2] = input_shape[2];
        para->inShape[3] = input_shape[3];
        para->outShape[0] = batch;
        para->outShape[1] = depth;
        para->outShape[2] = out_height;
        para->outShape[3] = out_width;
        para->mode = mode_;
        para->nearest_mode = nearest_mode_;
        para->antialias = antialias_;
        para->coordinate_transformation_mode = coordinate_transformation_mode_;
        para->cubic_coeff_a = cubic_coeff_a_;

        paramBuffer_->unmapMemory();

        VkDevice device = m_dev_->getLogicalDevice();

        // auto input_rgba = inputImage_->convertNCHWToRGBA(input);
        auto input_rgba = input->convertTensorToRGBA();
#ifdef VK_EXT_host_image_copy
        if (m_dev_->is_support_host_image_copy()) {
            inputImage_->hostImageCopyToDevice(input_rgba.data());
        } else
#endif
        {
            VulkanCommandBuffer cmdstg(device, m_cmdpool_->getCommandPool());
            cmdstg.begin();
            inputImage_->stagingBufferCopyToImage(cmdstg.get(),
                                                  input_rgba.data());
            cmdstg.end();
            cmdstg.submit(m_dev_->getComputeQueue());
        }

        VulkanCommandBuffer cmd(device, m_cmdpool_->getCommandPool());
        cmd.begin();
        inputImage_->readBarrier(cmd.get());
        cmd.end();
        cmd.submit(m_dev_->getComputeQueue());

        submit(resize_spv, resize_spv_len, realwidth, realheight);

        std::vector<T> tmp(realheight * realwidth * 4);
        T *ptr = tmp.data();
#ifdef VK_EXT_host_image_copy
        if (m_dev_->is_support_host_image_copy()) {
            outputImage_->hostImageCopyToHost(ptr);
        } else
#endif
        {
            VulkanCommandBuffer cmdstg1(device, m_cmdpool_->getCommandPool());
            cmdstg1.begin();
            outputImage_->stagingBufferCopyToHost(cmdstg1.get());
            cmdstg1.end();
            cmdstg1.submit(m_dev_->getComputeQueue());
            outputImage_->readStaingBuffer(ptr);
        }

        output->convertRGBAToTensor(ptr);
    }

    void execute(std::vector<std::shared_ptr<core::ITensor>> inputs,
                 std::vector<std::shared_ptr<core::ITensor>> outputs) override {
        apply<float>(inputs, outputs);
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

    std::shared_ptr<VulkanImage> outputImage_;
    std::shared_ptr<VulkanImage> inputImage_;

    std::shared_ptr<VulkanBuffer> paramBuffer_;

    void submit(const unsigned char *spv, unsigned int spv_len, int out_width,
                int out_height) override;
};

} // namespace ops
} // namespace vkop
#endif // OPS_RESIZE_HPP_
