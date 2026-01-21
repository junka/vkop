// Copyright 2025 @junka
#ifndef OPS_REDUCE_HPP_
#define OPS_REDUCE_HPP_

#include "Operator.hpp"

extern unsigned char reduce_spv[];
extern unsigned int reduce_spv_len;

namespace vkop {
namespace ops {
namespace reduce {
enum class ReduceType {
    L1 = 0,
    L2,
    LOGSUM,
    LOGSUMEXP,
    MAX,
    MEAN,
    MIN,
    PROD,
    SUM,
    SUMSQUARE,
};

struct GpuReduceParam {
    ivec4 shape;
    int reduce_op;
    int axes_mask;
};
} // namespace reduce

class Reduce : public Operator {
  public:
    Reduce()
        : Operator(OpType::REDUCE, reduce_spv, reduce_spv_len,
                   {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER},
                   sizeof(reduce::GpuReduceParam)) {}

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("reduce_op") != attributes.end()) {
            auto op = attributes.at("reduce_op");
            if (op == "l1_norm") {
                reduce_op_ = static_cast<int>(reduce::ReduceType::L1);
            } else if (op == "l2_norm") {
                reduce_op_ = static_cast<int>(reduce::ReduceType::L2);
            } else if (op == "log_sum") {
                reduce_op_ = static_cast<int>(reduce::ReduceType::LOGSUM);
            } else if (op == "log_sum_exp") {
                reduce_op_ = static_cast<int>(reduce::ReduceType::LOGSUMEXP);
            } else if (op == "max") {
                reduce_op_ = static_cast<int>(reduce::ReduceType::MAX);
            } else if (op == "mean") {
                reduce_op_ = static_cast<int>(reduce::ReduceType::MEAN);
            } else if (op == "min") {
                reduce_op_ = static_cast<int>(reduce::ReduceType::MIN);
            } else if (op == "prod") {
                reduce_op_ = static_cast<int>(reduce::ReduceType::PROD);
            } else if (op == "sum") {
                reduce_op_ = static_cast<int>(reduce::ReduceType::SUM);
            } else if (op == "sum_square") {
                reduce_op_ = static_cast<int>(reduce::ReduceType::SUMSQUARE);
            }
        }
        // opset ver 18 will move this to inputs
        if (attributes.find("axes") != attributes.end()) {
            axes_ = parse_attr_list<int>(attributes.at("axes"));
        }
        if (attributes.find("keepdims") != attributes.end()) {
            keepdims_ = std::stol(attributes.at("keepdims"));
        }
        // for opset over 18, when axes is empty, reduction will be performed
        // over all dimensions, if Ture then, reduction will be performed over
        // empty axes. noop_with_empty_axes processed in compiler
    }
    static std::vector<int>
    calculateOutputShape(const std::vector<int> &input_shape,
                         const std::vector<int> &axes_input, bool keepdims) {
        // Normalize axes (handle negative indices)
        std::vector<int> normalized_axes = axes_input;
        for (int &ax : normalized_axes) {
            if (ax < 0)
                ax += static_cast<int>(input_shape.size());
        }

        std::vector<bool> is_reduced(input_shape.size(), false);
        for (int ax : normalized_axes) {
            if (ax >= 0 && ax < static_cast<int>(input_shape.size())) {
                is_reduced[ax] = true;
            }
        }

        std::vector<int> output_shape;
        for (size_t i = 0; i < input_shape.size(); ++i) {
            if (is_reduced[i]) {
                if (keepdims) {
                    output_shape.push_back(1);
                }
            } else {
                output_shape.push_back(input_shape[i]);
            }
        }

        return output_shape;
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto input_shape = inputs[0]->getShape();
        int rank = input_shape.size();

        auto output_shape =
            calculateOutputShape(input_shape, axes_, keepdims_ == 1);
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto t) {
            using T = decltype(t);
            auto outputptr = core::as_tensor<T>(outputs[0]);
            if (outputptr->size() == 0) {
                outputptr->resize(output_shape);
            }
            auto output_image = outputptr->as_output_image(m_dev_, m_cmd_);
            objs_.emplace_back(output_image);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto t) {
            using T = decltype(t);
            auto inputptr = core::as_tensor<T>(inputs[0]);
            auto input_image = inputptr->as_input_image(m_dev_, m_cmd_);
            objs_.emplace_back(input_image);
        });

        reduce::GpuReduceParam para;
        for (size_t i = 0; i < input_shape.size(); i++) {
            para.shape[i] = input_shape[i];
        }
        para.reduce_op = reduce_op_;
        para.axes_mask = 0;
        for (auto &a : axes_) {
            int ax = (a < 0) ? static_cast<int>(a + rank) : static_cast<int>(a);
            if (ax >= 0 && ax < rank) {
                para.axes_mask |= (1 << ax);
            }
        }
        auto out_gpu_shape = outputs[0]->getGPUShape();
        submit(&para, UP_DIV(out_gpu_shape[0], 16),
               UP_DIV(out_gpu_shape[1], 16), out_gpu_shape[2]);
    }

    int reduce_op_ = 0;
    std::vector<int> axes_;
    int keepdims_ = 1;
};

} // namespace ops
} // namespace vkop
#endif // OPS_REDUCE_HPP_
