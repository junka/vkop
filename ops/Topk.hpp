// Copyright 2025 @junka
#ifndef OPS_TOPK_HPP_
#define OPS_TOPK_HPP_

#include "core/Tensor.hpp"
#include "ops/Operator.hpp"
#include <cassert>
#include <memory>
#include <vector>

extern unsigned char topk_spv[];
extern unsigned int topk_spv_len;

namespace vkop {
namespace ops {

namespace topk {
struct alignas(16) GpuTopkParam {
    ivec4 inShape;
    int k;
    int axis;
    int largest;
    int sorted;
};

} // namespace topk

class Topk : public Operator {
  public:
    explicit Topk()
        : Operator(OpType::TOPK, topk_spv, topk_spv_len,
                   sizeof(topk::GpuTopkParam)) {
        n_imgs_ = 0;
        types_ = {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                  DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE};
        objs_.reserve(4);
    }

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("K") != attributes.end()) {
            k_ = std::stol(attributes.at("K"));
        } else if (attributes.find("k") != attributes.end()) {
            k_ = std::stol(attributes.at("k"));
        }

        if (attributes.find("largest") != attributes.end()) {
            largest_ = std::stol(attributes.at("largest"));
        }
        if (attributes.find("sorted") != attributes.end()) {
            sorted_ = std::stol(attributes.at("sorted"));
        }
        if (attributes.find("axis") != attributes.end()) {
            axis_ = std::stol(attributes.at("axis"));
        }
    }

  private:
    static int compute_passes(int data_size, int K) {
        if (data_size <= 256) {
            return 1;
        }
        int passes = 1;

        while (data_size > 256) {
            data_size = K * UP_DIV(data_size, 256);
            passes++;
        }
        return passes;
    }

    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {

        auto inshape = inputs[0]->getShape();
        int rank = inputs[0]->num_dims();
        int axis = axis_ < 0 ? axis_ + rank : axis_;
        assert(axis <= 2 && axis >= 0);
        assert(k_ < 256);

        std::vector<int> indice;
        if (axis == 0) {
            indice.resize(inshape[axis]);
            for (int i = 0; i < inshape[axis]; i++) {
                indice[i] = i;
            }
        } else {
            indice.resize(inshape[axis] * inshape[0]);
            for (int i = 0; i < inshape[0]; i++) {
                for (int j = 0; j < inshape[axis]; j++) {
                    indice[(i * inshape[axis]) + j] = (i * inshape[axis]) + j;
                }
            }
        }

        auto outshape = inshape;
        outshape[axis] = k_;

        auto outputvalue = core::as_tensor<float>(outputs[0]);
        if (outputvalue->size() == 0) {
            outputvalue->resize(outshape);
        }
        auto output_value = outputvalue->as_storage_buffer(m_dev_);

        auto outputindex = core::as_tensor<int>(outputs[1]);
        if (outputindex->size() == 0) {
            outputindex->resize(outshape);
        }
        auto output_index = outputindex->as_storage_buffer(m_dev_);

        int data_size = inshape[axis];
        int round = compute_passes(data_size, k_);
        topk::GpuTopkParam param;
        param.k = k_;
        param.axis = axis;
        param.largest = largest_;
        param.sorted = sorted_;
        for (int i = 0; i < rank; i++) {
            param.inShape[i] = inshape[i];
        }

        // FIXME, temp tensor should be reserved and release outside of this op
        outshape[axis] = k_ * UP_DIV(data_size, 256);
        auto tempvalue1 = std::make_shared<core::Tensor<float>>(outshape);
        auto tempvalue2 = std::make_shared<core::Tensor<float>>(outshape);
        auto tempindex1 = std::make_shared<core::Tensor<int>>(outshape);
        auto tempindex2 = std::make_shared<core::Tensor<int>>(inshape);

        auto tmpvalue1 = tempvalue1->as_storage_buffer(m_dev_);
        auto tmpvalue2 = tempvalue2->as_storage_buffer(m_dev_);
        auto tmpindex1 = tempindex1->as_storage_buffer(m_dev_);
        auto tmpindex2 = tempindex2->as_storage_buffer(m_dev_, m_cmd_);

        tempindex2->copyToGPU(m_cmdpool_, indice.data());

        auto input = core::as_tensor<float>(inputs[0]);
        auto input_buffer = input->as_storage_buffer(m_dev_, m_cmd_);

        std::shared_ptr<VulkanResource> output_index_cur;
        std::shared_ptr<VulkanResource> output_value_cur;
        std::shared_ptr<VulkanResource> input_index_cur;
        std::shared_ptr<VulkanResource> input_value_cur;

        input_index_cur = tmpindex2;
        input_value_cur = input_buffer;
        output_index_cur = tmpindex1;
        output_value_cur = tmpvalue1;

        int width = inshape[axis];
        objs_.emplace_back(output_index_cur);
        objs_.emplace_back(output_value_cur);
        objs_.emplace_back(input_index_cur);
        objs_.emplace_back(input_value_cur);
        for (int i = 0; i < round; i++) {
            // output
            if (i == round - 1) {
                objs_[0] = (output_index);
                objs_[1] = (output_value);
            } else {
                objs_[0] = (output_index_cur);
                objs_[1] = (output_value_cur);
            }
            objs_[2] = (input_index_cur);
            objs_[3] = (input_value_cur);
            param.inShape[axis] = width;
            int dispatch_width = UP_DIV(width, 256);
            if (axis == 0) {
                submit(&param, dispatch_width, 1, 1);
            } else {
                submit(&param, dispatch_width, inshape[0], 1);
            }

            if (i == 0) {
                input_value_cur = tmpvalue2;
            }
            if (i < round - 1) {
                std::swap(input_index_cur, output_index_cur);
                std::swap(input_value_cur, output_value_cur);

                width = k_ * dispatch_width;
                auto o1 = std::dynamic_pointer_cast<VulkanBuffer>(objs_[0]);
                auto o2 = std::dynamic_pointer_cast<VulkanBuffer>(objs_[1]);
                o1->readBarrier(m_cmd_->get());
                o2->readBarrier(m_cmd_->get());
            }
        }
    }

    int64_t k_ = 1;
    int axis_ = -1;
    int largest_ = 1;
    int sorted_ = 1;
};

} // namespace ops
} // namespace vkop
#endif // OPS_TOPK_HPP_
