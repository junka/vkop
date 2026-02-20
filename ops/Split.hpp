// Copyright 2025 @junka
#ifndef OPS_SPLIT_HPP_
#define OPS_SPLIT_HPP_

#include "ops/Operator.hpp"
extern "C" {
extern unsigned char split_spv[];
extern unsigned int split_spv_len;
}
namespace vkop {
namespace ops {
namespace split {

struct GpuSplitParam {
    ivec4 inShape;
    ivec4 outShape;
    int split; // offset
    int axis;
};
} // namespace split

class Split : public Operator {
  public:
    explicit Split()
        : Operator(OpType::SPLIT, split_spv, split_spv_len,
                   {
                       VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                       VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                   },
                   sizeof(split::GpuSplitParam)) {
        update_after_bind_ = true;
        para_.split = 0;
    }

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("axis") != attributes.end()) {
            para_.axis = std::stoi(attributes.at("axis"));
        }
        if (attributes.find("num_outputs") != attributes.end()) {
            num_outputs_ = std::stoi(attributes.at("num_outputs"));
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        int rank = inputs[0]->num_dims();
        if (para_.axis < 0) {
            para_.axis += rank;
        }
        if (num_outputs_ == 0) {
            num_outputs_ = static_cast<int>(outputs.size());
        }
        int tosplit = inputs[0]->getShape()[para_.axis];
        std::vector<std::shared_ptr<VulkanResource>> output_images;
        std::vector<int64_t> split_vec;
        if (inputs.size() > 1) {
            auto split = core::as_tensor<int64_t>(inputs[1]);
            split_vec.resize(split->num_elements());
            for (size_t i = 0; i < split_vec.size(); i++) {
                split_vec[i] = (*split)[i];
            }
        } else {
            split_vec.resize(num_outputs_);
            for (int i = 1; i < num_outputs_; i++) {
                split_vec[i] = tosplit / num_outputs_;
            }
        }
        for (int i = 0; i < num_outputs_; i++) {
            dispatch_by_dtype(outputs[i]->dtype(), [&](auto dummy) {
                using T = decltype(dummy);
                auto output = core::as_tensor<T>(outputs[i]);
                auto shape = inputs[0]->getShape();
                shape[para_.axis] = static_cast<int>(split_vec[i]);
                if (output->size() == 0) {
                    output->resize(shape);
                }
                auto output_image = output->as_output_image(m_dev_, m_cmd_);
                if (i == 0) {
                    objs_.emplace_back(output_image);
                }
                output_images.emplace_back(output_image);
            });
        }
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto input = core::as_tensor<T>(inputs[0]);
            auto input_image = input->as_input_image(m_dev_, m_cmd_);
            objs_.emplace_back(input_image);
        });

        para_.axis = para_.axis + 4 - rank;
        if (rank == 4 || rank == 3) {
            inputs[0]->get_shape(para_.inShape);
            outputs[0]->get_shape(para_.outShape);
        }
        para_.split = 0;
        for (int i = 0; i < num_outputs_; i++) {
            para_.outShape[para_.axis] =
                outputs[i]->getShape()[para_.axis - 4 + rank];

            auto gpushape = outputs[i]->getGPUShape();
            objs_[0] = output_images[i];
            submit(&para_, UP_DIV(gpushape[0], 16), UP_DIV(gpushape[1], 16),
                   gpushape[2]);
            para_.split += outputs[i]->getShape()[para_.axis - 4 + rank];
        }
    }

    int num_outputs_ = 0;
    split::GpuSplitParam para_;
};

} // namespace ops
} // namespace vkop
#endif // OPS_SPLIT_HPP_
