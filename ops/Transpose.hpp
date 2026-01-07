// Copyright 2025 @junka
#ifndef OPS_TRANSPOSE_HPP_
#define OPS_TRANSPOSE_HPP_

#include "core/Tensor.hpp"
#include "ops/Operator.hpp"

extern unsigned char transpose_spv[];
extern unsigned int transpose_spv_len;
namespace vkop {
namespace ops {

namespace transpose {
struct GpuTransposeParam {
    ivec4 inShape;
    ivec4 outShape;
    ivec4 perms;
    ivec4 reverse_perms;
};

} // namespace transpose

class Transpose : public Operator {
  public:
    explicit Transpose()
        : Operator(OpType::TRANSPOSE, transpose_spv, transpose_spv_len,
                   {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER},
                   sizeof(transpose::GpuTransposeParam)) {}
    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("perm") != attributes.end()) {
            perm_ = parse_attr_list<int>(attributes.at("perm"));
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {

        auto inshape = inputs[0]->getShape();
        std::vector<int> outshape(4);
        for (size_t i = 0; i < 4; ++i) {
            outshape[i] = inshape[perm_[i]];
        }

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->size() == 0) {
                output->resize(outshape);
            }
            auto output_image = output->as_output_image(m_dev_, m_cmd_);
            objs_.emplace_back(output_image);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto input = core::as_tensor<T>(inputs[0]);
            auto input_image = input->as_input_image(m_dev_, m_cmd_);
            objs_.emplace_back(input_image);
        });

        std::vector<int> reverse_perms(4);
        for (size_t i = 0; i < 4; ++i) {
            reverse_perms[perm_[i]] = i;
        }

        transpose::GpuTransposeParam param;
        for (int i = 0; i < 4; i++) {
            param.inShape[i] = inshape[i];
            param.outShape[i] = outshape[i];
            param.perms[i] = perm_[i];
            param.reverse_perms[i] = reverse_perms[i];
        }
        submit(&param, UP_DIV(outshape[3], 16),
               UP_DIV(outshape[2] * outshape[0], 16), UP_DIV(outshape[1], 4));
    }

    std::vector<int> perm_ = {3, 2, 1, 0};
};

} // namespace ops
} // namespace vkop
#endif // OPS_TRANSPOSE_HPP_
