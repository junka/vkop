// Copyright 2025 @junka
#ifndef OPS_BATCHNORM2D_HPP_
#define OPS_BATCHNORM2D_HPP_

#include "Operator.hpp"

#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanCommandBuffer.hpp"
#include "vulkan/VulkanImage.hpp"
#include "vulkan/VulkanPipeline.hpp"
#include "vulkan/VulkanQueryPool.hpp"

#include <memory>

extern unsigned char batchnorm2d_spv[];
extern unsigned int batchnorm2d_spv_len;

namespace vkop {
namespace ops {
namespace batchnorm {

using ivec4 = int[4];

// torch.nn.functional.batch_norm(input, running_mean, running_var, weight=None,
//                                bias=None, training=False, momentum=0.1,
//                                eps=1e-05)
struct GpuBatchNormParam {
    ivec4 outShape;
    float eps;      // default 1e-5
    float momentum; // default 0.1
};
} // namespace batchnorm

class BatchNorm2d : public Operator {
  public:
    BatchNorm2d() : Operator(OpType::BATCHNORM) {}
    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        attributes.find("training") != attributes.end()
            ? training_ = (attributes.at("align_corners") == "1" ||
                           attributes.at("align_corners") == "true")
            : training_ = false;
        if (attributes.find("eps") != attributes.end()) {
            eps_ = std::stof(attributes.at("eps"));
        }
        if (attributes.find("momentum") != attributes.end()) {
            momentum_ = std::stof(attributes.at("momentum"));
        }
    }

    template <typename T>
    void prepare(std::vector<std::shared_ptr<core::ITensor>> inputs,
                 std::vector<std::shared_ptr<core::ITensor>> outputs) {
        auto input = core::as_tensor<T>(inputs[0]);
        auto output = core::as_tensor<T>(outputs[0]);
        auto running_mean = core::as_tensor<T>(inputs[1]);
        auto running_var = core::as_tensor<T>(inputs[2]);

        auto weight =
            (inputs.size() > 3) ? core::as_tensor<T>(inputs[3]) : nullptr;
        auto bias =
            (inputs.size() > 4) ? core::as_tensor<T>(inputs[4]) : nullptr;

        auto input_shape = input->getTensorShape();

        if (output->size() == 0) {
            output->resize(input->getTensorShape());
        }

        auto input_image = input->as_input_image(m_dev_, m_cmdpool_);
        auto output_image = output->as_output_image(m_dev_, m_cmdpool_);

        tensorBuffer_ = std::make_shared<VulkanBuffer>(
            m_dev_, running_mean->size() * 4,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        paramBuffer_ = std::make_shared<VulkanBuffer>(
            m_dev_, sizeof(batchnorm::GpuBatchNormParam),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        inputImages_ = {input_image};
        outputImage_ = output_image;
    }

    void
    apply(const std::vector<std::shared_ptr<core::ITensor>> &inputs,
          const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        if (inputs[0]->dtype() == typeid(float)) {
            prepare<float>(inputs, outputs);
        }
    }

    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        if (inputs[0]->dtype() == typeid(float)) {
            auto input = core::as_tensor<float>(inputs[0]);
            auto output = core::as_tensor<float>(outputs[0]);
            auto running_mean = core::as_tensor<float>(inputs[1]);
            auto running_var = core::as_tensor<float>(inputs[2]);

            auto weight = (inputs.size() > 3)
                              ? core::as_tensor<float>(inputs[3])
                              : nullptr;
            auto bias = (inputs.size() > 4) ? core::as_tensor<float>(inputs[4])
                                            : nullptr;

            auto input_shape = input->getTensorShape();
            int batch = input_shape[0];
            int depth = input_shape[1];
            int out_height = input_shape[2];
            int out_width = input_shape[3];

            int realwidth = out_width * UP_DIV(depth, 4);
            int realheight = out_height * batch;

            auto *para = static_cast<batchnorm::GpuBatchNormParam *>(
                paramBuffer_->getMappedMemory());
            para->eps = eps_;
            para->momentum = momentum_;
            para->outShape[0] = batch;
            para->outShape[1] = depth;
            para->outShape[2] = out_height;
            para->outShape[3] = out_width;

            auto *var_buffer =
                static_cast<float *>(tensorBuffer_->getMappedMemory());
            for (int i = 0; i < running_mean->num_elements(); i++) {
                *(var_buffer + 4 * i) = running_mean->data()[i];
                *(var_buffer + 4 * i + 1) = running_var->data()[i];
                if (inputs.size() > 3) {
                    *(var_buffer + 4 * i + 2) = weight->data()[i];
                } else {
                    *(var_buffer + 4 * i + 2) = 1.0F;
                }
                if (inputs.size() > 4) {
                    *(var_buffer + 4 * i + 3) = bias->data()[i];
                } else {
                    *(var_buffer + 4 * i + 3) = 0.0F;
                }
            }

            // do copy before submit
            submit(batchnorm2d_spv, batchnorm2d_spv_len, realwidth, realheight);
        }
    }

  private:
    bool training_ = false;
    float momentum_ = 0.1;
    float eps_ = 1e-5;

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
            outputImage_, inputImages_[0], tensorBuffer_, paramBuffer_};
        VkDevice device = m_dev_->getLogicalDevice();
        VulkanPipeline pipeline(device, types, objs,
                                reinterpret_cast<const uint32_t *>(spv),
                                spv_len);

        VulkanCommandBuffer cmd2(device, m_cmdpool_->getCommandPool());
#ifdef USE_MEASURE_TIME
        VulkanQueryPool query_pool(device, 2, VK_QUERY_TYPE_TIMESTAMP);
#endif
        cmd2.begin();
        cmd2.bind(pipeline);
#ifdef USE_MEASURE_TIME
        query_pool.begin(cmd2.get());
#endif
        cmd2.dispatch(UP_DIV(out_width, 16), UP_DIV(out_height, 16));
#ifdef USE_MEASURE_TIME
        query_pool.end(cmd2.get());
#endif
        cmd2.end();
        cmd2.submit(m_dev_->getComputeQueue());
#ifdef USE_MEASURE_TIME
        auto r = query_pool.getResults();
        LOG_INFO("Time: %f s", static_cast<double>(r[1] - r[0]) * (1e-9) *
                                   m_dev_->getTimestampPeriod());
#endif
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_BATCHNORM2D_HPP_
