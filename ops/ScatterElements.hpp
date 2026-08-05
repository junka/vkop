// Copyright 2026 @junka
#ifndef OPS_SCATTER_ELEMENTS_HPP_
#define OPS_SCATTER_ELEMENTS_HPP_

#include "ops/BufferBase.hpp"
#include <numeric>

extern "C" {
extern unsigned char buffer_scatter_elements_spv[];
extern unsigned int buffer_scatter_elements_spv_len;
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
    explicit ScatterElements()
        : BufferFactory(OpType::SCATTER_ELEMENTS, buffer_scatter_elements_spv,
                        buffer_scatter_elements_spv_len,
                        {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                         DESCRIPTOR_TYPE_STORAGE},
                        sizeof(scatter::ScatterPC)) {
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
        int n_idx = core::as_tensor<int>(inputs[1])->num_elements();
        int n_threads = n_idx * cols;

        // Bind: [0]=data/output (read-write), [1]=indices, [2]=updates
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->size() == 0) {
                output->resize(data_shape);
            }
            bind_ssbo<T>(outputs[0], true);
        });
        bind_ssbo<int>(inputs[1], false);   // indices (binding 1)
        bind_ssbo<float>(inputs[2], false); // updates (binding 2)

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
