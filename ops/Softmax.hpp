// Copyright 2025 @junka
#ifndef OPS_SOFTMAX_HPP_
#define OPS_SOFTMAX_HPP_

#include "UnaryFactory.hpp"

extern unsigned char softmax_spv[];
extern unsigned int softmax_spv_len;

namespace vkop {
namespace ops {
namespace softmax {

using ivec4 = int[4];
using ivec2 = int[2];

struct GpuSoftMaxParam {
    ivec4 outImgSize;
    ivec4 outShape;
    int axis; // 0: N, 1: C, 2: H, 3: W
};

} // namespace softmax

class Softmax : public Operator {
  public:
    Softmax() : Operator(OpType::SOFTMAX) {}

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("axis") != attributes.end()) {
            auto axis = std::stoi(attributes.at("axis"));
            axis_ = axis;
        } else if (attributes.find("dim") != attributes.end()) {
            auto axis = std::stoi(attributes.at("dim"));
            axis_ = axis;
        }
    }

    template <typename T>
    void prepare(std::vector<std::shared_ptr<core::ITensor>> inputs,
                 std::vector<std::shared_ptr<core::ITensor>> outputs) {
        auto input = core::as_tensor<T>(inputs[0]);
        auto output = core::as_tensor<T>(outputs[0]);

        auto input_shape = input->getTensorShape();

        if (input_shape.size() != 4) {
            throw std::invalid_argument("Input must have 4 dimensions.");
        }

        if (output->size() == 0) {
            output->resize(input_shape);
        }
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

        auto input_image = input->make_vkimg(
            m_dev_, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT | exflags);

        paramBuffer_ = std::make_shared<VulkanBuffer>(
            m_dev_, sizeof(softmax::GpuSoftMaxParam),
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
            input_image->readBarrier(cmd.get());
            cmd.end();
            cmd.submit(m_dev_->getComputeQueue());
        }
        inputImages_.push_back(input_image);
    }

    void apply(std::vector<std::shared_ptr<core::ITensor>> inputs,
               std::vector<std::shared_ptr<core::ITensor>> outputs) override {
        if (inputs[0]->dtype() == typeid(float)) {
            prepare<float>(inputs, outputs);
        } else if (inputs[0]->dtype() == typeid(uint16_t)) {
            prepare<uint16_t>(inputs, outputs);
        } else {
            LOG_ERROR("Unsupported data type");
        }
    }

    void execute(std::vector<std::shared_ptr<core::ITensor>> inputs,
                 std::vector<std::shared_ptr<core::ITensor>> outputs) override {
        if (inputs[0]->dtype() == typeid(float)) {
            auto input = core::as_tensor<float>(inputs[0]);
            auto output = core::as_tensor<float>(outputs[0]);
            auto input_shape = input->getTensorShape();

            int batch = input_shape[0];
            int depth = input_shape[1];
            int out_height = input_shape[2];
            int out_width = input_shape[3];

            int realwidth = out_width * UP_DIV(depth, 4);
            int realheight = out_height * batch;

            auto *para = static_cast<softmax::GpuSoftMaxParam *>(
                paramBuffer_->getMappedMemory());
            // vkimage params
            para->outImgSize[0] = realwidth;
            para->outImgSize[1] = realheight;
            para->outImgSize[2] = 1;
            para->outImgSize[3] = 0;
            para->outShape[0] = batch;
            para->outShape[1] = out_height;
            para->outShape[2] = out_width;
            para->outShape[3] = depth;
            para->axis = axis_;
            paramBuffer_->unmapMemory();

            // do copy before submit
            if (axis_ == 0) {
                submit(softmax_spv, softmax_spv_len, out_width,
                       out_height * UP_DIV(depth, 4));
            } else if (axis_ == 1) {
                submit(softmax_spv, softmax_spv_len, out_width,
                       out_height * batch);
            } else if (axis_ == 2) {
                submit(softmax_spv, softmax_spv_len, out_width,
                       UP_DIV(depth, 4) * batch);
            } else if (axis_ == 3) {
                submit(softmax_spv, softmax_spv_len, out_height,
                       UP_DIV(depth, 4) * batch);
            }
        }
    }

  private:
    int axis_;

    std::shared_ptr<VulkanBuffer> paramBuffer_;

    void submit(const unsigned char *spv, unsigned int spv_len, int out_width,
                int out_height) override {
        std::vector<VkDescriptorType> types = {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
        std::vector<std::shared_ptr<VulkanResource>> objs = {
            outputImage_, inputImages_[0], paramBuffer_};
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
#endif // OPS_SOFTMAX_HPP_
