// Copyright 2025 @junka
#ifndef OPS_GLOBALAVERAGEPOOL_HPP_
#define OPS_GLOBALAVERAGEPOOL_HPP_

#include "ops/Operator.hpp"

extern unsigned char globalaveragepool_spv[];
extern unsigned int globalaveragepool_spv_len;

namespace vkop {
namespace ops {
namespace globalaveragepool {
using ivec4 = int[4];
struct alignas(16) GpuGAPParam {
    ivec4 inShape; // NCHW
};
} // namespace globalaveragepool

class GlobalAveragePool : public Operator {
  public:
    GlobalAveragePool()
        : Operator(OpType::GLOBALAVERAGEPOOL, globalaveragepool_spv,
                   globalaveragepool_spv_len,
                   sizeof(globalaveragepool::GpuGAPParam)) {
        n_imgs_ = 1;
        types_ = {DESCRIPTOR_TYPE_STORAGE,
                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER};
        objs_.reserve(types_.size());
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto input_shape = inputs[0]->getShape();

        int batch = input_shape[0];
        int depth = input_shape[1];

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->size() == 0) {
                output->resize(std::vector<int>{batch, depth});
            }
            auto output_buffer = output->as_storage_buffer(m_dev_);
            objs_.emplace_back(output_buffer);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto input = core::as_tensor<T>(inputs[0]);
            auto input_image = input->as_input_image(m_dev_, m_cmd_);

            objs_.emplace_back(input_image);
        });

        globalaveragepool::GpuGAPParam para;
        para.inShape[0] = inputs[0]->get_batch();
        para.inShape[1] = inputs[0]->get_channel();
        para.inShape[2] = inputs[0]->get_height();
        para.inShape[3] = inputs[0]->get_width();
        submit(&para, UP_DIV(batch, 16), 1, UP_DIV(depth, 4));
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_GLOBALAVERAGEPOOL_HPP_
