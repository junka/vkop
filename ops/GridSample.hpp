// Copyright 2025 @junka
#ifndef OPS_GRIDSAMPLE_HPP_
#define OPS_GRIDSAMPLE_HPP_

#include <unistd.h>
#include <vector>

#include "Operator.hpp"

#include "core/Tensor.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanCommandBuffer.hpp"
#include "vulkan/VulkanImage.hpp"
#include "vulkan/VulkanPipeline.hpp"
#include "vulkan/VulkanQueryPool.hpp"

#include "include/logger.hpp"

extern unsigned char grid_sample_spv[];
extern unsigned int grid_sample_spv_len;

namespace vkop {
namespace ops {

namespace gridsample {
enum class InterpolationMode { BILINEAR, NEAREST };

enum class PaddingMode { ZEROS, BORDER, REFLECTION };

using ivec4 = int[4];
using ivec2 = int[2];

struct GpuGridSampleParam {
    ivec4 outImgSize;
    ivec2 inShape;
    ivec2 outShape;
    bool align_corners;
    int padding_mode;
    int interpolation_mode;
};
} // namespace gridsample

class GridSample : public Operator {
  public:
    GridSample() : Operator(OpType::GRIDSAMPLE) {}

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
                LOG_ERROR("Unsupported interpolation_mode: " + mode);
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
                LOG_ERROR("Unsupported padding_mode: " + mode);
                throw std::invalid_argument("Unsupported padding_mode: " +
                                            mode);
            }
        } else {
            padding_mode_ = gridsample::PaddingMode::ZEROS;
        }
    }

    template <typename T>
    void prepare(std::vector<std::shared_ptr<core::ITensor>> inputs,
                 std::vector<std::shared_ptr<core::ITensor>> outputs) {
        auto input = core::as_tensor<T>(inputs[0]);
        auto output = core::as_tensor<T>(outputs[0]);
        auto grid = core::as_tensor<T>(inputs[1]);

        auto input_shape = input->getTensorShape();
        auto grid_shape = grid->getTensorShape();

        int batch = input_shape[0];
        int depth = input_shape[1];
        int out_height = grid_shape[1];
        int out_width = grid_shape[2];

        if (output->size() == 0) {
            output->resize(batch, depth, out_height, out_width);
        }

        auto input_image = input->as_input_image(m_dev_, m_cmdpool_);
        auto output_image = output->as_output_image(m_dev_, m_cmdpool_);

        auto weight_image = grid->as_input_image(m_dev_, m_cmdpool_);

        paramBuffer_ = std::make_shared<VulkanBuffer>(
            m_dev_, sizeof(gridsample::GpuGridSampleParam),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        types_ = {output_image->getDescriptorType(),
                  input_image->getDescriptorType(),
                  weight_image->getDescriptorType(),
                  paramBuffer_->getDescriptorType()};
        objs_ = {output_image, input_image, weight_image, paramBuffer_};
    }

    void
    apply(const std::vector<std::shared_ptr<core::ITensor>> &inputs,
          const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        if (inputs[0]->dtype() == typeid(float)) {
            prepare<float>(inputs, outputs);
        } else if (inputs[0]->dtype() == typeid(uint16_t)) {
            prepare<uint16_t>(inputs, outputs);
        } else {
            LOG_ERROR("Unsupported data type");
        }
    }

    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        if (inputs[0]->dtype() == typeid(float)) {
            auto input = core::as_tensor<float>(inputs[0]);
            auto output = core::as_tensor<float>(outputs[0]);
            auto grid = core::as_tensor<float>(inputs[1]);

            auto input_shape = input->getTensorShape();
            auto grid_shape = grid->getTensorShape();

            if (input_shape.size() != 4 || grid_shape.size() != 4) {
                throw std::invalid_argument(
                    "Input and grid must have 4 dimensions.");
            }
            int batch = input_shape[0];
            int depth = input_shape[1];
            int in_height = input_shape[2];
            int in_width = input_shape[3];
            int out_height = grid_shape[1];
            int out_width = grid_shape[2];

            int realwidth = out_width * UP_DIV(depth, 4);
            int realheight = out_height * batch;

            auto *para = static_cast<gridsample::GpuGridSampleParam *>(
                paramBuffer_->getMappedMemory());
            // vkimage params
            para->outImgSize[0] = realwidth;
            para->outImgSize[1] = realheight;
            para->outImgSize[2] = 1;
            para->outImgSize[3] = 0;
            // original params
            para->inShape[0] = in_width;
            para->inShape[1] = in_height;
            para->outShape[0] = out_width;
            para->outShape[1] = out_height;
            para->align_corners = align_corners_;
            para->padding_mode = static_cast<int>(padding_mode_);
            para->interpolation_mode = static_cast<int>(interpolation_mode_);
            paramBuffer_->unmapMemory();

            submit(grid_sample_spv, grid_sample_spv_len, UP_DIV(out_width, 16),
                   UP_DIV(out_height, 16));
        }
    }

  private:
    gridsample::InterpolationMode interpolation_mode_;
    gridsample::PaddingMode padding_mode_;
    bool align_corners_;
    std::shared_ptr<VulkanBuffer> paramBuffer_;
};

} // namespace ops
} // namespace vkop

#endif // OPS_GRIDSAMPLE_HPP_
