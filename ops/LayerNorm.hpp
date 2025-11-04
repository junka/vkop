// Copyright 2025 @junka
#ifndef OPS_LAYERNORM_HPP_
#define OPS_LAYERNORM_HPP_

#include "Operator.hpp"

#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanCommandBuffer.hpp"
#include "vulkan/VulkanImage.hpp"
#include "vulkan/VulkanPipeline.hpp"
#include "vulkan/VulkanQueryPool.hpp"

#include <memory>

extern unsigned char layernorm_spv[];
extern unsigned int layernorm_spv_len;

namespace vkop {
namespace ops {
namespace layernorm {

using ivec4 = int[4];

// torch.nn.functional.layer_norm(input, normalized_shape, weight=None,
// bias=None, eps=1e-05)

struct GpuLayerNormParam {
    ivec4 outShape;
    ivec4 normalizedShape;
    float eps; // default 1e-5
    int normalizedDim;
    int innerSize;
};
} // namespace layernorm

class LayerNorm : public Operator {
  public:
    LayerNorm() = default;
    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("eps") != attributes.end()) {
            eps_ = std::stof(attributes.at("eps"));
        }
        if (attributes.find("normalized_shape") != attributes.end()) {
            std::string norm_shape_str = attributes.at("normalized_shape");
            normalized_shape_ = parse_attr_list(norm_shape_str);
        }
    }
    template <typename T>
    void prepare(std::vector<std::shared_ptr<core::Tensor<T>>> inputs,
                 std::vector<std::shared_ptr<core::Tensor<T>>> outputs) {
        auto input = inputs[0];

        std::shared_ptr<core::Tensor<T>> weight;
        std::shared_ptr<core::Tensor<T>> bias;
        if (inputs.size() > 1) {
            weight = inputs[1];
        }
        if (inputs.size() > 2) {
            bias = inputs[2];
        }

        auto output = outputs[0];

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

        tensorBuffer_ = std::make_shared<VulkanBuffer>(
            m_dev_, weight->size() * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        paramBuffer_ = std::make_shared<VulkanBuffer>(
            m_dev_, sizeof(layernorm::GpuLayerNormParam),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
#ifdef VK_EXT_host_image_copy
        if (m_dev->is_support_host_image_copy()) {
            if (m_dev->checkHostImageCopyDstLayoutSupport(
                    VK_IMAGE_LAYOUT_GENERAL)) {
                outputImage->hostImaggeTransition(VK_IMAGE_LAYOUT_GENERAL);
            } else {
                outputImage->hostImaggeTransition(
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            }
            inputImage->hostImaggeTransition(
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
    void apply(std::vector<std::shared_ptr<core::Tensor<T>>> inputs,
               std::vector<std::shared_ptr<core::Tensor<T>>> outputs) {
        auto input = inputs[0];
        auto output = outputs[0];

        std::shared_ptr<core::Tensor<T>> weight;
        std::shared_ptr<core::Tensor<T>> bias;
        if (inputs.size() > 1) {
            weight = inputs[1];
        }
        if (inputs.size() > 2) {
            bias = inputs[2];
        }

        auto input_shape = input->getTensorShape();
        int batch = input_shape[0];
        int depth = input_shape[1];
        int out_height = input_shape[2];
        int out_width = input_shape[3];

        int realwidth = out_width * UP_DIV(depth, 4);
        int realheight = out_height * batch;
        if (output->size() == 0) {
            output->resize(input->getTensorShape());
        }
        prepare(inputs, outputs);

        VkDevice device = m_dev_->getLogicalDevice();

        auto *para = static_cast<layernorm::GpuLayerNormParam *>(
            paramBuffer_->getMappedMemory());
        para->eps = eps_;
        para->outShape[0] = batch;
        para->outShape[1] = depth;
        para->outShape[2] = out_height;
        para->outShape[3] = out_width;
        para->normalizedDim = normalized_shape_.size();
        para->innerSize = 1;
        for (size_t i = 0; i < normalized_shape_.size(); i++) {
            para->normalizedShape[i] = normalized_shape_[i];
            para->innerSize *= normalized_shape_[i];
        }

        auto *var_buffer =
            static_cast<float *>(tensorBuffer_->getMappedMemory());
        for (int i = 0; i < weight->num_elements(); i++) {
            if (inputs.size() > 1) {
                *(var_buffer + 2 * i) = weight->data()[i];
            } else {
                *(var_buffer + 2 * i) = 1.0F;
            }
            if (inputs.size() > 2) {
                *(var_buffer + 2 * i + 1) = bias->data()[i];
            } else {
                *(var_buffer + 2 * i + 1) = 0.0F;
            }
        }

        auto input_rgba = input->convertTensorToRGBA();
#ifdef VK_EXT_host_image_copy
        if (m_dev->is_support_host_image_copy()) {
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
        if (normalized_shape_.size() == 1) { // 归一化最后一个维度 W
            submit(layernorm_spv, layernorm_spv_len, batch * UP_DIV(depth, 4),
                   out_height);
        } else if (normalized_shape_.size() == 2) { // 归一化最后两个维度 HW
            submit(layernorm_spv, layernorm_spv_len, batch, UP_DIV(depth, 4));
        } else { // 归一化所有维度 CHW
            submit(layernorm_spv, layernorm_spv_len, realwidth, realheight);
        }
        std::vector<T> tmp(realheight * realwidth * 4);
        T *ptr = tmp.data();
#ifdef VK_EXT_host_image_copy
        if (m_dev->is_support_host_image_copy()) {
            outputImage->hostImageCopyToHost(ptr);
        } else
#endif
        {
            VulkanCommandBuffer cmd(device, m_cmdpool_->getCommandPool());
            cmd.begin();
            VulkanCommandBuffer cmdstg1(device, m_cmdpool_->getCommandPool());
            cmdstg1.begin();
            outputImage_->stagingBufferCopyToHost(cmdstg1.get());
            cmdstg1.end();
            cmdstg1.submit(m_dev_->getComputeQueue());
            outputImage_->readStaingBuffer(ptr);
        }

        output->convertRGBAToTensor(ptr);
    }

    void execute(
        std::vector<std::shared_ptr<core::Tensor<float>>> inputs,
        std::vector<std::shared_ptr<core::Tensor<float>>> outputs) override {
        apply<float>(inputs, outputs);
    }

    void
    execute(std::vector<std::shared_ptr<core::Tensor<int>>> inputs,
            std::vector<std::shared_ptr<core::Tensor<int>>> outputs) override {
        apply<int>(inputs, outputs);
    }

    void execute(
        std::vector<std::shared_ptr<core::Tensor<uint16_t>>> inputs,
        std::vector<std::shared_ptr<core::Tensor<uint16_t>>> outputs) override {
        apply<uint16_t>(inputs, outputs);
    }

  private:
    float eps_ = 1e-5;
    std::vector<int> normalized_shape_;
    std::shared_ptr<VulkanImage> outputImage_;
    std::shared_ptr<VulkanImage> inputImage_;
    std::shared_ptr<VulkanBuffer> tensorBuffer_;
    std::shared_ptr<VulkanBuffer> paramBuffer_;

    void submit(const unsigned char *spv, unsigned int spv_len, int out_width,
                int out_height) override {
        std::vector<VkDescriptorType> types = {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
        std::vector<std::shared_ptr<VulkanResource>> objs = {
            outputImage_, inputImage_, tensorBuffer_, paramBuffer_};
        VkDevice device = m_dev_->getLogicalDevice();
        VulkanPipeline pipeline(device, types, objs,
                                reinterpret_cast<const uint32_t *>(spv),
                                spv_len);

        VulkanCommandBuffer cmd2(device, m_cmdpool_->getCommandPool());
        VulkanQueryPool query_pool(device, 2, VK_QUERY_TYPE_TIMESTAMP);
        cmd2.begin();
        cmd2.bind(pipeline);
        query_pool.begin(cmd2.get());
        cmd2.dispatch(out_width, out_height);
        query_pool.end(cmd2.get());
        cmd2.end();
        cmd2.submit(m_dev_->getComputeQueue());
        auto r = query_pool.getResults();
        double ts = static_cast<double>(r[1] - r[0]) * (1e-9) *
                    m_dev_->getTimestampPeriod();
        LOG_INFO("Time: %f s", ts);
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_LAYERNORM_HPP_
