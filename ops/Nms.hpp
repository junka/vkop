// Copyright 2025 @junka
#ifndef OPS_NMS_HPP_
#define OPS_NMS_HPP_

#include "core/Tensor.hpp"
#include "ops/Operator.hpp"

extern "C" {
extern unsigned char nms_spv[];
extern unsigned int nms_spv_len;
}
namespace vkop {
namespace ops {
namespace nms {

struct alignas(16) GpuNMSParam {
    int num_batch;
    int num_class;
    int num_spatial;
    int center_point_box;
    int max_output_boxes_per_class;
    float iou_threshold;
    float score_threshold;
};

} // namespace nms

class Nms : public Operator {
  public:
    Nms()
        : Operator(OpType::NMS, nms_spv, nms_spv_len,
                   {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                   sizeof(nms::GpuNMSParam)) {
        // default values for attributes
        para_.center_point_box = 0;
        para_.max_output_boxes_per_class = 0;
        para_.iou_threshold = 0.0F;
        para_.score_threshold = 0.0F;
    }

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("center_point_box") != attributes.end()) {
            std::string cpb_str = attributes.at("center_point_box");
            para_.center_point_box = std::stol(cpb_str);
        }
        // below scalar will be moved to attributes
        if (attributes.find("iou_threshold") != attributes.end()) {
            std::string iou_str = attributes.at("iou_threshold");
            para_.iou_threshold = std::stof(iou_str);
        }
        if (attributes.find("score_threshold") != attributes.end()) {
            std::string thre_str = attributes.at("score_threshold");
            para_.score_threshold = std::stof(thre_str);
        }
        if (attributes.find("max_output_boxes_per_class") != attributes.end()) {
            std::string max_str = attributes.at("max_output_boxes_per_class");
            para_.max_output_boxes_per_class = std::stol(max_str);
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto box_shape = inputs[0]->getShape();
        auto score_shape = inputs[1]->getShape();
        int batch = box_shape[0];
        int num_boxes = box_shape[1];
        int num_class = score_shape[1];

        assert(box_shape[2] == 4);
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->size() == 0) {
                // assume 1024 indices at most
                int output_num = 100;
                if (para_.max_output_boxes_per_class > 0) {
                    output_num = para_.max_output_boxes_per_class * num_class;
                }
                output->resize(std::vector<int>{output_num, 3});
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

        dispatch_by_dtype(inputs[1]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto input = core::as_tensor<T>(inputs[1]);
            auto input_image = input->as_input_image(m_dev_, m_cmd_);

            objs_.emplace_back(input_image);
        });

        // use dummy buffer as for output count
        objs_.emplace_back(dummy_buffer_);
        para_.num_batch = batch;
        para_.num_class = num_class;
        para_.num_spatial = num_boxes;

        submit(&para_, 1, batch * num_class, 1);
    }

    nms::GpuNMSParam para_;
};

} // namespace ops
} // namespace vkop
#endif // OPS_NMS_HPP_
