// Copyright 2025 @junka
#ifndef OPS_COL2IM_HPP_
#define OPS_COL2IM_HPP_

#include "Operator.hpp"
extern "C" {
extern unsigned char col2im_spv[];
extern unsigned int col2im_spv_len;
}
namespace vkop {
namespace ops {
namespace col2im {

struct alignas(16) GpuCol2ImParam {
    ivec4 outImgSize;
    ivec4 outShape;
    int groupSize;
    int totalGroups;
};

} // namespace col2im

class Col2Im : public Operator {
  public:
    Col2Im()
        : Operator(OpType::COL2IM, col2im_spv, col2im_spv_len,
                   {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER},
                   sizeof(col2im::GpuCol2ImParam)) {}

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("dilations") != attributes.end()) {
            std::string dila_str = attributes.at("dilations");
            if (dila_str.find(',') != std::string::npos) {
                dilations_ = parse_attr_list<int>(dila_str);
            } else {
                int d = std::stol(dila_str);
                dilations_ = {d, d};
            }
        }
        if (attributes.find("pads") != attributes.end()) {
            std::string pads_str = attributes.at("pads");
            if (pads_str.find(',') != std::string::npos) {
                pads_ = parse_attr_list<int>(pads_str);
            } else {
                int d = std::stol(pads_str);
                pads_ = {d, d};
            }
        }
        if (attributes.find("strides") != attributes.end()) {
            std::string strides_str = attributes.at("strides");
            if (strides_str.find(',') != std::string::npos) {
                strides_ = parse_attr_list<int>(strides_str);
            } else {
                int d = std::stol(strides_str);
                strides_ = {d, d};
            }
        }
        if (attributes.find("image_shape") != attributes.end()) {
            std::string shape_str = attributes.at("image_shape");
            if (shape_str.find(',') != std::string::npos) {
                image_shape_ = parse_attr_list<int>(shape_str);
            }
        }
        if (attributes.find("block_shape") != attributes.end()) {
            std::string block_str = attributes.at("block_shape");
            if (block_str.find(',') != std::string::npos) {
                block_shape_ = parse_attr_list<int>(block_str);
            }
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {

        std::vector<int> input_shape = inputs[0]->getShape();

        if (input_shape.size() != 4) {
            throw std::invalid_argument("Input must have 4 dimensions.");
        }
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto t) {
            using T = decltype(t);
            auto outputptr = core::as_tensor<T>(outputs[0]);
            if (outputptr->size() == 0) {
                outputptr->resize(input_shape);
            }
            auto output_image = outputptr->as_output_image(m_dev_, m_cmd_);
            objs_.emplace_back(output_image);
        });

        for (const auto &input : inputs) {
            dispatch_by_dtype(input->dtype(), [&](auto t) {
                using T = decltype(t);
                auto inputptr = core::as_tensor<T>(input);
                auto input_image = inputptr->as_input_image(m_dev_, m_cmd_);
                objs_.emplace_back(input_image);
            });
        }

        int batch = input_shape[0];
        int depth = input_shape[1];
        int out_height = input_shape[2];
        int out_width = input_shape[3];

        int realwidth = out_width * UP_DIV(depth, 4);
        int realheight = out_height * batch;

        col2im::GpuCol2ImParam para;
        // vkimage params
        para.outImgSize[0] = realwidth;
        para.outImgSize[1] = realheight;
        para.outImgSize[2] = 1;
        para.outImgSize[3] = 0;
        para.outShape[0] = batch;
        para.outShape[1] = out_height;
        para.outShape[2] = out_width;
        para.outShape[3] = depth;

        submit(&para, UP_DIV(out_width, 16), UP_DIV(out_height, 16),
               UP_DIV(depth, 4));
    }

    std::vector<int> dilations_;
    std::vector<int> strides_;
    std::vector<int> pads_;
    std::vector<int> image_shape_;
    std::vector<int> block_shape_;
};

} // namespace ops
} // namespace vkop
#endif // OPS_COL2IM_HPP_
