// Copyright 2025 @junka
#ifndef OPS_COL2IM_HPP_
#define OPS_COL2IM_HPP_

#include "UnaryFactory.hpp"

extern unsigned char col2im_spv[];
extern unsigned int col2im_spv_len;

namespace vkop {
namespace ops {
namespace col2im {

using ivec4 = int[4];
using ivec2 = int[2];

struct GpuCol2ImParam {
    ivec4 outImgSize;
    ivec4 outShape;
    int groupSize;
    int totalGroups;
};

} // namespace col2im

class Col2im : public Operator {
  public:
    Col2im() : Operator(OpType::COL2IM) {}

    template <typename T>
    void prepare(std::vector<std::shared_ptr<core::ITensor>> inputs,
                 std::vector<std::shared_ptr<core::ITensor>> outputs) {
        auto input = core::as_tensor<T>(inputs[0]);
        auto output = core::as_tensor<T>(outputs[0]);

        auto input_shape = input->getTensorShape();

        if (output->size() == 0) {
            output->resize(input_shape);
        }

        auto input_image = input->as_input_image(m_dev_, m_cmdpool_);
        auto output_image = output->as_output_image(m_dev_, m_cmdpool_);

        paramBuffer_ = std::make_shared<VulkanBuffer>(
            m_dev_, sizeof(col2im::GpuCol2ImParam),
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
            auto input_shape = input->getTensorShape();

            if (input_shape.size() != 4) {
                throw std::invalid_argument("Input must have 4 dimensions.");
            }
            int batch = input_shape[0];
            int depth = input_shape[1];
            int out_height = input_shape[2];
            int out_width = input_shape[3];

            int realwidth = out_width * UP_DIV(depth, 4);
            int realheight = out_height * batch;

            auto *para = static_cast<col2im::GpuCol2ImParam *>(
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

            paramBuffer_->unmapMemory();

            // do copy before submit
            submit(col2im_spv, col2im_spv_len, out_width, out_height);
        } else {
            LOG_ERROR("Unsupported data type");
        }
    }

  private:
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
#endif // OPS_COL2IM_HPP_
