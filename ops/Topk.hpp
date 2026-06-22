// Copyright 2025 @junka
#ifndef OPS_TOPK_HPP_
#define OPS_TOPK_HPP_

#include "core/Tensor.hpp"
#include "ops/Operator.hpp"
#include <cassert>
#include <memory>
#include <vector>

extern "C" {
extern unsigned char topk_spv[];
extern unsigned int topk_spv_len;
}
namespace vkop {
namespace ops {

namespace topk {
struct alignas(16) GpuTopkParam {
    ivec4 inShape;
    int k;
    int axis;
    int largest;
    int sorted;
    int init;
    int fp16;
};

} // namespace topk

class Topk : public Operator {
  public:
    explicit Topk(int fp16 = 0)
        : Operator(OpType::TOPK, topk_spv, topk_spv_len,
                   {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                    DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                   sizeof(topk::GpuTopkParam), fp16) {
        update_after_bind_ = true;

        para_.k = 1;
        para_.axis = -1;
        para_.largest = 1;
        para_.sorted = 1;
        para_.fp16 = fp16_;

        if (fp16_ == 1) {
            tempvalue1_ = std::make_shared<core::Tensor<uint16_t>>();
            tempvalue2_ = std::make_shared<core::Tensor<uint16_t>>();
        } else {
            tempvalue1_ = std::make_shared<core::Tensor<float>>();
            tempvalue2_ = std::make_shared<core::Tensor<float>>();
        }
        tempindex1_ = std::make_unique<core::Tensor<int>>();
        tempindex2_ = std::make_unique<core::Tensor<int>>(true);
    }

    ~Topk() override {
        tempvalue1_.reset();
        tempvalue2_.reset();
    };

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("K") != attributes.end()) {
            para_.k = std::stol(attributes.at("K"));
        } else if (attributes.find("k") != attributes.end()) {
            para_.k = std::stol(attributes.at("k"));
        }

        if (attributes.find("largest") != attributes.end()) {
            para_.largest = std::stol(attributes.at("largest"));
        }
        if (attributes.find("sorted") != attributes.end()) {
            para_.sorted = std::stol(attributes.at("sorted"));
        }
        if (attributes.find("axis") != attributes.end()) {
            para_.axis = std::stol(attributes.at("axis"));
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
        int axis = para_.axis < 0 ? para_.axis + rank : para_.axis;
        assert(axis <= 2 && axis >= 0);
        assert(para_.k < 256);
        // fp16 packed writes (packHalf2x16) need an even out_base, so k is
        // rounded up to even. The shader then lays out output rows with the
        // bumped k as stride, while the output tensor keeps the original k.
        // row_pad records the extra trailing element(s) per row so that
        // copyToCPU can compact the rows back to the logical k.
        int original_k = para_.k;
        if (fp16_ == 1 && para_.k % 2 != 0) {
            para_.k = UP_DIV(para_.k, 2) * 2;
        }
        int row_pad = para_.k - original_k;

        auto outshape = inshape;
        outshape[axis] = para_.k;

        std::shared_ptr<VulkanBuffer> output_value = nullptr;
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto t) {
            using T = decltype(t);
            auto outputvalue = core::as_tensor<T>(outputs[0]);
            if (outputvalue->size() == 0) {
                outputvalue->resize(outshape);
            }
            output_value = outputvalue->as_storage_buffer(m_dev_);
        });

        auto outputindex = core::as_tensor<int>(outputs[1]);
        if (outputindex->size() == 0) {
            outputindex->resize(outshape);
        }
        auto output_index = outputindex->as_storage_buffer(m_dev_);

        int data_size = inshape[axis];
        int round = compute_passes(data_size, para_.k);
        for (int i = 0; i < rank; i++) {
            para_.inShape[i] = inshape[i];
        }
        para_.axis = axis;

        outshape[axis] = para_.k * UP_DIV(data_size, 256);

        std::shared_ptr<VulkanBuffer> tmpvalue1 = nullptr;
        std::shared_ptr<VulkanBuffer> tmpvalue2 = nullptr;
        dispatch_by_dtype(tempvalue1_->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto temp1 = core::as_tensor<T>(tempvalue1_);
            auto temp2 = core::as_tensor<T>(tempvalue2_);
            if (temp1->size() == 0) {
                temp1->resize(outshape);
                temp2->resize(outshape);
                tempindex1_->resize(outshape);
                tempindex2_->resize(inshape);
            }
            tmpvalue1 = temp1->as_storage_buffer(m_dev_);
            tmpvalue2 = temp2->as_storage_buffer(m_dev_);
        });
        auto tmpindex1 = tempindex1_->as_storage_buffer(m_dev_);
        auto tmpindex2 = tempindex2_->as_storage_buffer(m_dev_, m_cmd_);

        std::shared_ptr<VulkanBuffer> input_buffer = nullptr;
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto t) {
            using T = decltype(t);
            auto input = core::as_tensor<T>(inputs[0]);
            input_buffer = input->as_storage_buffer(m_dev_, m_cmd_);
        });

        std::shared_ptr<VulkanResource> output_index_cur;
        std::shared_ptr<VulkanResource> output_value_cur;
        std::shared_ptr<VulkanResource> input_index_cur;
        std::shared_ptr<VulkanResource> input_value_cur;

        assert(tmpvalue1 != nullptr);
        assert(tmpvalue2 != nullptr);

        input_index_cur = tmpindex2;
        input_value_cur = input_buffer;
        output_index_cur = tmpindex1;
        output_value_cur = tmpvalue1;

        int width = inshape[axis];
        objs_.emplace_back(output_index_cur);
        objs_.emplace_back(output_value_cur);
        objs_.emplace_back(input_index_cur);
        objs_.emplace_back(input_value_cur);

        // Each pass must use its own descriptor set. vkUpdateDescriptorSets
        // modifies the set immediately, so all dispatches referencing the same
        // set see the LAST update's bindings — earlier passes would read
        // wrong buffers on drivers like Intel ANV.
        if (round > 16) {
            std::cerr << "too many rounds" << std::endl;
            assert(0);
        }
        std::vector<VkDescriptorSet> pass_ds(round);
        for (int i = 0; i < round; i++) {
            pass_ds[i] = allocPassDescriptorSet();
        }

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
            para_.inShape[axis] = width;
            if (i > 0) {
                para_.init = 1;
            } else {
                para_.init = 0;
            }
            int dispatch_width =
                fp16_ ? UP_DIV(width, 512) : UP_DIV(width, 256);
            if (axis == 0) {
                submit_per_ds(pass_ds[i], &para_, dispatch_width, 1, 1);
            } else {
                submit_per_ds(pass_ds[i], &para_, dispatch_width, inshape[0],
                              1);
            }

            if (i == 0) {
                input_value_cur = tmpvalue2;
            }
            if (i < round - 1) {
                std::swap(input_index_cur, output_index_cur);
                std::swap(input_value_cur, output_value_cur);

                width = para_.k * dispatch_width;
                auto o1 = std::dynamic_pointer_cast<VulkanBuffer>(objs_[0]);
                auto o2 = std::dynamic_pointer_cast<VulkanBuffer>(objs_[1]);
                o1->readBarrier(m_cmd_->get());
                o2->readBarrier(m_cmd_->get());
            }
        }

        for (int i = 0; i < round; i++) {
            freePassDescriptorSet(pass_ds[i]);
        }

        // Mark outputs whose GPU buffer carries per-row padding so the host
        // readback compacts rows back to the logical k. The fp16 odd-k path
        // bumps k to even, and the shader lays out output rows with the
        // bumped k as the stride along the last dim. copyBufferToCPU's
        // compact path assumes the padded dim is the last (cols) dim, so this
        // only applies to axis==1 (2D, k along cols) — the tested layout.
        if (row_pad > 0 && axis == 1) {
            outputs[0]->set_gpu_row_pad(row_pad);
            outputs[1]->set_gpu_row_pad(row_pad);
        }
    }

    topk::GpuTopkParam para_;
    std::shared_ptr<core::ITensor> tempvalue1_;
    std::shared_ptr<core::ITensor> tempvalue2_;
    std::unique_ptr<core::Tensor<int>> tempindex1_;
    std::unique_ptr<core::Tensor<int>> tempindex2_;
};

} // namespace ops
} // namespace vkop
#endif // OPS_TOPK_HPP_
