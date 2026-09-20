// Copyright 2026 @junka
#ifndef OPS_SCATTER_ELEMENTS_HPP_
#define OPS_SCATTER_ELEMENTS_HPP_

#include "ops/BufferBase.hpp"
#include <numeric>

extern "C" {
extern unsigned char buffer_scatter_elements_spv[];
extern unsigned int buffer_scatter_elements_spv_len;
extern unsigned char buffer_scatter_elements_fp16_spv[];
extern unsigned int buffer_scatter_elements_fp16_spv_len;
}

namespace vkop {
namespace ops {

namespace scatter {
struct alignas(16) ScatterPC {
    int n_threads; // n_idx * cols
    int cols;      // row width
    int reduction; // 0 = none (overwrite), 1 = add
    int _pad;
};
} // namespace scatter

// SSBO-only op: ONNX ScatterElements. Writes (or adds) updates to data
// at the given indices along axis=0. The LLM uses axis=0 with 'add'
// reduction (deepstack visual feature injection).
//
// For axis=0, indices[i] gives the row index in data where updates[i] is
// scattered. Since the data is flat row-major [rows, cols], the linear
// offset for (idx, col) = idx * cols + col. Each thread handles one
// (index, update_col) pair.
//
// Actually, for the LLM's use case, the indices are 1-D and updates have
// the same shape as indices (each update is a full row). So we dispatch
// one thread per (index, col) pair where col ranges over the row width.
class ScatterElements : public BufferFactory {
  public:
    explicit ScatterElements(int fp16 = 0)
        : BufferFactory(OpType::SCATTER_ELEMENTS,
                        fp16 ? buffer_scatter_elements_fp16_spv
                             : buffer_scatter_elements_spv,
                        fp16 ? buffer_scatter_elements_fp16_spv_len
                             : buffer_scatter_elements_spv_len,
                        {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                         DESCRIPTOR_TYPE_STORAGE},
                        sizeof(scatter::ScatterPC), fp16) {
        update_after_bind_ = true;
    }

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("axis") != attributes.end()) {
            axis_ = std::stol(attributes.at("axis"));
        }
        if (attributes.find("reduction") != attributes.end()) {
            std::string r = attributes.at("reduction");
            if (r == "add")
                reduction_ = 1;
            else
                reduction_ = 0; // none
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        // inputs: [0]=data, [1]=indices, [2]=updates
        // The test passes (data, indices, updates, data) — inputs[0] and
        // outputs[0] are the SAME tensor (in-place scatter). The shader
        // writes to binding 0 (uData = output = data).
        auto data_shape = inputs[0]->getShape();

        int cols = 1;
        for (size_t i = 1; i < data_shape.size(); ++i) {
            cols *= data_shape[i];
        }
        // indices is int64 in the model (bound as ivec2[] in the shader);
        // read the element count on the correct dtype.
        int n_idx = 0;
        dispatch_by_dtype(inputs[1]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            n_idx =
                static_cast<int>(core::as_tensor<T>(inputs[1])->num_elements());
        });
        int n_threads = n_idx * cols;

        // GPU dispatch path for BOTH fp32 and fp16. The fp16 shader variant
        // (buffer_scatter_elements_fp16_spv, built with -DFP16) uses a
        // word-level CAS: two adjacent fp16 elements share a uint word, so a
        // thread updating one half CAS-swaps the whole word to avoid racing a
        // sibling thread owning the other half. This is far cheaper than the
        // host-side scatter it replaced (~16ms/call — the single biggest
        // per-op readback cost in decode).
        // Bind: [0]=data/output (read-write), [1]=indices, [2]=updates
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->num_elements() != total_elems(data_shape)) {
                output->resize(data_shape);
            }
            // ScatterElements is read-modify-write: the shader reads the
            // CURRENT data value at each target, then adds/overwrites. When
            // output is a distinct tensor from data (the runtime's usual case —
            // it allocates a fresh output per node), the output buffer starts
            // empty, so the shader would read zeros and lose the original data
            // at non-scattered positions. Seed the output with a device→device
            // copy of the data buffer first. The copy has to happen on the GPU
            // rather than via a host round-trip.
            auto out_buf = std::dynamic_pointer_cast<VulkanBuffer>(
                output->as_storage_buffer(m_dev_, m_cmd_));
            if (outputs[0].get() != inputs[0].get()) {
                auto data = core::as_tensor<T>(inputs[0]);
                auto src_buf = std::dynamic_pointer_cast<VulkanBuffer>(
                    data->as_storage_buffer(m_dev_, m_cmd_));
                VkBufferCopy region{};
                region.size =
                    static_cast<VkDeviceSize>(std::min(output->num_elements(),
                                                       data->num_elements())) *
                    sizeof(T);
                if (region.size > 0 && src_buf && out_buf) {
                    // src is in SHADER_READ/WRITE from a prior op; transition
                    // to TRANSFER_READ for the copy. out_buf was just created
                    // and readBarrier'd by as_storage_buffer; transition it to
                    // TRANSFER_WRITE as the copy destination.
                    src_buf->transferReadBarrier(m_cmd_->get(), region.size);
                    out_buf->transferWriteBarrier(m_cmd_->get(), region.size);
                    vkCmdCopyBuffer(m_cmd_->get(), src_buf->getBuffer(),
                                    out_buf->getBuffer(), 1, &region);
                    // Leave out_buf in SHADER_READ for the scatter dispatch
                    // (read-modify-write: the shader reads then CAS-writes).
                    // Restore src to SHADER_READ for any downstream consumer.
                    out_buf->readBarrier(m_cmd_->get(), region.size);
                    src_buf->readBarrier(m_cmd_->get(), region.size);
                }
            }
            objs_.emplace_back(out_buf);
        });
        // indices (binding 1): int64 data is byte-packed; bind as int64_t so
        // the ivec2[] shader view reads the true stride.
        dispatch_by_dtype(inputs[1]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            bind_ssbo<T>(inputs[1], false);
        });
        // updates (binding 2): may be fp16 or fp32 — bind on its own dtype so
        // as_tensor<T> doesn't dynamic_cast to null.
        dispatch_by_dtype(inputs[2]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            bind_ssbo<T>(inputs[2], false);
        });

        scatter::ScatterPC pc{};
        pc.n_threads = n_threads;
        pc.cols = cols;
        pc.reduction = reduction_;
        submit(&pc, UP_DIV(n_threads, 256), 1, 1);
    }

    int axis_ = 0;
    int reduction_ = 0; // 0=none, 1=add
};

} // namespace ops
} // namespace vkop
#endif // OPS_SCATTER_ELEMENTS_HPP_
