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
        // indices is int64 in the model (bound as ivec2[] in the shader);
        // read the element count on the correct dtype.
        int n_idx = 0;
        dispatch_by_dtype(inputs[1]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            n_idx =
                static_cast<int>(core::as_tensor<T>(inputs[1])->num_elements());
        });
        int n_threads = n_idx * cols;

        // Host-compute path for fp16 data/updates. The fp32 GPU shader can't
        // be trivially reused for fp16 (the float atomic-add CAS loop and the
        // uintBitsToFloat reads would corrupt fp16 bits), and a packed fp16
        // shader would need a word-level CAS to avoid RMW races between
        // adjacent columns. ScatterElements in the LLM is tiny (1x2048 per
        // layer, 28 layers) so a host scatter + re-upload is correct and cheap.
        if (outputs[0]->dtype() == typeid(uint16_t)) {
            hostScatter<uint16_t>(inputs, outputs, data_shape, cols, n_idx);
            return;
        }

        // Bind: [0]=data/output (read-write), [1]=indices, [2]=updates
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->num_elements() != total_elems(data_shape)) {
                output->resize(data_shape);
            }
            bind_ssbo<T>(outputs[0], true);
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

    // Host scatter for fp16 (uint16_t bits) data. Reads data + updates to the
    // host, applies the scatter (overwrite or add) along axis=0, and uploads
    // the result. indices is int64.
    //
    // Shape contract (ONNX ScatterElements, axis=0, the LLM's deepstack use):
    //   data    [rows, cols]
    //   indices [n_idx]          (1-D; one row index per update row)
    //   updates [n_idx, cols]    (one full row per index)
    // n_idx is taken from the indices tensor's element count. Because runtime
    // tensor recycling can leave a stale shape on a reused output, we read the
    // true count from the indices data itself after pulling it to the host.
    template <typename T>
    void hostScatter(const std::vector<std::shared_ptr<core::ITensor>> &inputs,
                     const std::vector<std::shared_ptr<core::ITensor>> &outputs,
                     const std::vector<int> &data_shape, int cols,
                     int /*n_idx*/) {
        auto data = core::as_tensor<T>(inputs[0]);
        auto updates = core::as_tensor<T>(inputs[2]);
        // Unconditional GPU readback: a GPU-resident input may have a stale
        // CPU data_ from a prior round (the KV-cache/ScatterElements data
        // tensor is recycled across decode rounds — its GPU buffer is
        // rewritten each round but data_ still holds the previous round's
        // values, tail zero-filled by reserveOnCPU). has_cpu_data() only
        // checks data_ is non-empty, so the old guard skipped the readback
        // and hostScatter seeded the output with stale data. copyToCPU reads
        // back whenever a vkobj_ exists, so it is the authoritative check.
        data->copyToCPU(m_cmdpool_);
        updates->copyToCPU(m_cmdpool_);
        auto indices = core::as_tensor<int64_t>(inputs[1]);
        indices->copyToCPU(m_cmdpool_);
        // True index count from the host data (recycled shapes can lie).
        int n_idx = static_cast<int>(indices->num_elements());

        auto output = core::as_tensor<T>(outputs[0]);
        // Resize if the output doesn't match data's element count. The output
        // may be a recycled tensor with a stale shape/size from a prior round
        // (e.g. a different kv_len / n_img); the old `size()==0` guard only
        // caught the first-ever run, leaving later rounds with an undersized
        // output whose `(*output)[i]` read (below, for i<total) went OOB →
        // segfault on q_len>1 prefill. Mirror the fp32 path's num_elements
        // check (line ~101).
        if (output->num_elements() != total_elems(data_shape)) {
            output->resize(data_shape);
        }
        // If output is a distinct tensor from data, seed it with the data.
        // (In-place scatter: output == data, already loaded.)
        if (outputs[0].get() != inputs[0].get()) {
            std::vector<T> seed(data->num_elements());
            for (int i = 0; i < data->num_elements(); ++i) {
                seed[i] = (*data)[i];
            }
            output->fillToCPU(seed);
        }

        int rows = (data_shape.empty()) ? 0 : data_shape[0];
        int total = rows * cols;
        std::vector<T> out(total);
        for (int i = 0; i < total; ++i) {
            out[i] = (*output)[i];
        }
        // ONNX ScatterElements (axis=0): indices and updates are broadcast to
        // data's shape with the axis dimension removed. Each flat index `i`
        // maps to a coordinate in that reduced shape; the axis-0 slot is then
        // replaced by the gathered index value. For 2-D data [rows, cols],
        // flat i -> coord [i/cols, i%cols] -> target [indices[i], i%cols]
        // -> linear indices[i]*cols + (i%cols). n_idx is the flat count of
        // indices (== flat count of updates).
        for (int i = 0; i < n_idx; ++i) {
            int row = static_cast<int>((*indices)[i]);
            if (row < 0) {
                row += rows;
            }
            int col = i % cols;
            int target = row * cols + col;
            if (reduction_ == 1) {
                // fp16 add via fp32 rounding (matches the GPU fp32 path).
                float a = core::ITensor::fp16_to_fp32(
                    reinterpret_cast<const uint16_t &>(out[target]));
                float b = core::ITensor::fp16_to_fp32(
                    reinterpret_cast<const uint16_t &>((*updates)[i]));
                uint16_t r = core::ITensor::fp32_to_fp16(a + b);
                out[target] = reinterpret_cast<const T &>(r);
            } else {
                out[target] = (*updates)[i];
            }
        }
        output->fillToCPU(out);
        objs_.emplace_back(output->as_storage_buffer(m_dev_, m_cmd_));
        output->copyToGPU(m_cmdpool_, out.data());
    }

    int axis_ = 0;
    int reduction_ = 0; // 0=none, 1=add
};

} // namespace ops
} // namespace vkop
#endif // OPS_SCATTER_ELEMENTS_HPP_
