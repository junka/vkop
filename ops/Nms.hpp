// Copyright 2025 @junka
#ifndef OPS_NMS_HPP_
#define OPS_NMS_HPP_

#include "core/Tensor.hpp"
#include "ops/BufferBase.hpp"
#include "ops/Operator.hpp"
#include "ops/PimplFacade.hpp"

namespace vkop {
namespace ops {
namespace nms {

// Push-constant block shared by both backends: shaders/image/nms.comp and
// shaders/buffer/nms.comp declare the same seven fields in this order.
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

class NmsImage : public Operator {
  public:
    NmsImage()
        : Operator(OpType::NMS,
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

        // binding 3: the selected-box counter the shader increments via
        // atomicAdd(total_count, ...). Reuses the process-wide dummy_buffer_
        // (only 4 bytes needed, well within its 16). Zeroed each run with
        // vkCmdFillBuffer so the offset base is 0 regardless of prior runs.
        dummy_buffer_->fillBuffer(m_cmd_->get(), 0u);
        objs_.emplace_back(dummy_buffer_);

        para_.num_batch = batch;
        para_.num_class = num_class;
        para_.num_spatial = num_boxes;

        submit(&para_, 1, batch * num_class, 1);
    }

    // The buffer holding the selected-box count written by the shader's
    // atomicAdd(total_count, ...). Read back on the host to know how many
    // output rows are valid (the output tensor is padded to max rows).
  public:
    std::shared_ptr<VulkanBuffer> get_count_buffer() const {
        return dummy_buffer_;
    }

  private:
    nms::GpuNMSParam para_;
};

// SSBO backend: same algorithm and dispatch geometry as NmsImage, with both
// inputs read as the compact row-major arrays of the buffer backend instead
// of texelFetch-packed images. One workgroup per (batch, class), dispatch
// (1, B*C, 1); the group's selected count is claimed with one atomicAdd on a
// dedicated counter buffer, which the host reads back through
// get_count_buffer() to know how many output rows are valid.
class NmsBuffer : public BufferFactory {
  public:
    explicit NmsBuffer(int fp16)
        : BufferFactory(OpType::NMS,
                        {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                         DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                        sizeof(nms::GpuNMSParam), fp16) {
        para_.center_point_box = 0;
        para_.max_output_boxes_per_class = 0;
        para_.iou_threshold = 0.0F;
        para_.score_threshold = 0.0F;
    }

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("center_point_box") != attributes.end()) {
            para_.center_point_box = std::stol(attributes.at("center_point_box"));
        }
        if (attributes.find("iou_threshold") != attributes.end()) {
            para_.iou_threshold = std::stof(attributes.at("iou_threshold"));
        }
        if (attributes.find("score_threshold") != attributes.end()) {
            para_.score_threshold = std::stof(attributes.at("score_threshold"));
        }
        if (attributes.find("max_output_boxes_per_class") != attributes.end()) {
            para_.max_output_boxes_per_class =
                std::stol(attributes.at("max_output_boxes_per_class"));
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto box_shape = inputs[0]->getShape();
        auto score_shape = inputs[1]->getShape();
        if (box_shape.size() != 3 || box_shape[2] != 4) {
            throw std::invalid_argument(
                "NMS: boxes must be [B][N][4], got rank " +
                std::to_string(box_shape.size()));
        }
        if (score_shape.size() != 3 || score_shape[2] != box_shape[1] ||
            score_shape[0] != box_shape[0]) {
            throw std::invalid_argument(
                "NMS: scores must be [B][C][N] with the same B and N as the "
                "boxes");
        }
        const int batch = box_shape[0];
        const int num_boxes = box_shape[1];
        const int num_class = score_shape[1];

        // The kernel reads one dtype for both tensors (fp32 words or packed
        // half2), selected at pipeline build time from fp16_. Two inputs of
        // different widths would need two builds, which this op never asks
        // for -- say so instead of reading half bytes as floats.
        const bool inputs_fp16 =
            inputs[0]->dtype() == typeid(uint16_t);
        if ((inputs[1]->dtype() == typeid(uint16_t)) != inputs_fp16) {
            throw std::invalid_argument(
                "NMS: boxes and scores must share a dtype");
        }
        if (inputs_fp16 != (fp16_ != 0)) {
            throw std::runtime_error(
                "NMS: the buffer kernel was built for " +
                std::string(fp16_ != 0 ? "fp16" : "fp32") +
                " but the input tensors are " +
                std::string(inputs_fp16 ? "fp16" : "fp32"));
        }

        // Rows this dispatch can produce at most, across every (batch,
        // class) workgroup. Only used when the caller left the output
        // unsized: a pre-sized output (the tests build it from the expected
        // shape) keeps its own capacity, and shrinking it here would either
        // truncate the result or move the tensor under the caller.
        const int rows_per_class =
            para_.max_output_boxes_per_class > 0
                ? std::min(para_.max_output_boxes_per_class, num_boxes)
                : num_boxes;
        const int max_rows = batch * num_class * rows_per_class;

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->size() == 0) {
                output->resize(std::vector<int>{max_rows, 3});
            }
            bind_ssbo<T>(outputs[0], /*is_output=*/true);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            bind_ssbo<T>(inputs[0], /*is_output=*/false);
        });
        dispatch_by_dtype(inputs[1]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            bind_ssbo<T>(inputs[1], /*is_output=*/false);
        });

        // The atomic counter, zeroed on the GPU every run so the offset base
        // is 0 no matter what the previous dispatch left behind.
        if (!count_buf_) {
            count_buf_ = std::make_shared<VulkanBuffer>(
                m_dev_, 16, STORAGE | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        }
        count_buf_->fillBuffer(m_cmd_->get(), 0u);
        // fillBuffer ends at TRANSFER access and resets the tracked access to
        // "none", which is not enough for the shader's atomicAdd (a
        // read-modify-write) to observe the zeroed word. Re-establish the
        // write as the source of a dependency, then publish it to shader
        // reads and writes. Without this a second dispatch in the same
        // recording could start from the previous count.
        count_buf_->transferWriteBarrier(m_cmd_->get());
        count_buf_->transferBarrier(
            m_cmd_->get(),
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        objs_.emplace_back(count_buf_);

        para_.num_batch = batch;
        para_.num_class = num_class;
        para_.num_spatial = num_boxes;

        submit(&para_, 1, batch * num_class, 1);
    }

  public:
    // The selected-box count the shader's atomicAdd produced. Read back on
    // the host after the dispatch to know how many output rows are valid.
    std::shared_ptr<VulkanBuffer> get_count_buffer() const {
        return count_buf_;
    }

  private:
    nms::GpuNMSParam para_;
    // Allocated on the first dispatch (needs m_dev_). 16 B is the smallest
    // size every driver accepts for an SSBO binding.
    std::shared_ptr<VulkanBuffer> count_buf_;
};

// PIMPL façade: buffer SSBO impl when backend_buffer is set, else image.
class Nms : public PimplFacade {
  public:
    Nms(int fp16, bool backend_buffer) : PimplFacade(OpType::NMS) {
        if (backend_buffer) {
            impl_ = std::make_unique<NmsBuffer>(fp16);
        } else {
            impl_ = std::make_unique<NmsImage>();
        }
    }

    // Forwarded to whichever impl is live; both publish the counter buffer
    // the shader's atomicAdd writes (used by the NMS test for count
    // readback). Null until the impl has dispatched once.
    std::shared_ptr<VulkanBuffer> get_count_buffer() const {
        if (auto *buf = dynamic_cast<NmsBuffer *>(impl_.get())) {
            return buf->get_count_buffer();
        }
        if (auto *img = dynamic_cast<NmsImage *>(impl_.get())) {
            return img->get_count_buffer();
        }
        return nullptr;
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_NMS_HPP_
