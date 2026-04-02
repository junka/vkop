// Copyright 2026 @junka
#ifndef OPS_GATHER_HPP_
#define OPS_GATHER_HPP_

#include "core/Tensor.hpp"
#include "ops/Operator.hpp"
#include <numeric>

extern "C" {
extern unsigned char gather_spv[];
extern unsigned int gather_spv_len;
}
namespace vkop {
namespace ops {

namespace gather {
struct GpuGatherParam {
    ivec4 inShape;
    ivec4 indicesShape;
    ivec4 outShape;
    int axis;
    int idims;
    int odims;
    int nindex;
};

} // namespace gather

class Gather : public Operator {
  public:
    explicit Gather()
        : Operator(OpType::GATHER, gather_spv, gather_spv_len,
                   {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                   sizeof(gather::GpuGatherParam)) {}

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("axis") != attributes.end()) {
            auto axis = std::stol(attributes.at("axis"));
            param_.axis = axis;
        }
    }

  private:
    static std::vector<int>
    calculateGatherOutputShape(const std::vector<int> &data_shape,
                               const std::vector<int> &indices_shape,
                               int axis) {

        if (data_shape.empty()) {
            return indices_shape;
        }

        int rank = static_cast<int>(data_shape.size());
        int normalized_axis = axis;
        if (normalized_axis < 0) {
            normalized_axis += rank;
        }

        if (normalized_axis < 0 || normalized_axis >= rank) {
            throw std::out_of_range("Axis is out of range for data shape");
        }

        std::vector<int> output_shape;
        output_shape.reserve(rank - 1 + indices_shape.size());

        for (int i = 0; i < rank; ++i) {
            if (i == normalized_axis) {
                output_shape.insert(output_shape.end(), indices_shape.begin(),
                                    indices_shape.end());
            } else {
                output_shape.push_back(data_shape[i]);
            }
        }

        return output_shape;
    }
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {

        auto inshape = inputs[0]->getShape();
        auto indshape = inputs[1]->getShape();
        auto rank = inputs[0]->num_dims();
        std::vector<std::vector<int>> out_size;
        if (param_.axis < 0) {
            param_.axis += rank;
        }
        std::vector<int> out_shape =
            calculateGatherOutputShape(inshape, indshape, param_.axis);

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->size() == 0) {
                output->resize(out_shape);
            }
            auto output_buffer = output->as_storage_buffer(m_dev_, m_cmd_);
            objs_.emplace_back(output_buffer);
        });

        // input weight [L, N]
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto input = core::as_tensor<T>(inputs[0]);
            auto input_buffer = input->as_storage_buffer(m_dev_, m_cmd_);
            objs_.emplace_back(input_buffer);
        });

        // indexes, [batch, length]
        dispatch_by_dtype(inputs[1]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto input = core::as_tensor<T>(inputs[1]);
            auto input_buffer = input->as_storage_buffer(m_dev_, m_cmd_);
            objs_.emplace_back(input_buffer);
        });

        param_.idims = rank;
        param_.nindex = inputs[1]->getShape().size();
        for (int i = 0; i < rank; ++i) {
            param_.inShape[i] = inshape[i];
        }
        for (size_t i = 0; i < indshape.size(); ++i) {
            param_.indicesShape[i] = indshape[i];
        }
        param_.odims = out_shape.size();
        for (size_t i = 0; i < out_shape.size(); ++i) {
            param_.outShape[i] = out_shape[i];
        }
        auto total_size = std::accumulate(out_shape.begin(), out_shape.end(), 1,
                                          std::multiplies<>());
        submit(&param_, UP_DIV(total_size, 256), 1, 1);
    }

    gather::GpuGatherParam param_;
};

} // namespace ops
} // namespace vkop
#endif // OPS_GATHER_HPP_
