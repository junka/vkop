// Copyright 2026 @junka
#ifndef OPS_SCATTER_ELEMENTS_HPP_
#define OPS_SCATTER_ELEMENTS_HPP_

#include "ops/BufferBase.hpp"
#include <numeric>

namespace vkop {
namespace ops {

namespace scatter {
struct alignas(16) ScatterPC {
    int n_threads;  // indices/updates element count (== updates.numel())
    int inner;      // product of dims after `axis` (trailing width)
    int idx_axis;   // indices.shape[axis]
    int data_axis;  // data.shape[axis] — the coordinate the index selects
    int reduction;  // 0=none 1=add 2=mul 3=max 4=min
    int data_total; // elements in data/output; bounds-checks the target
    int _pad0;
    int _pad1;
};
} // namespace scatter

// ScatterElements, SSBO/GPU only. data, indices and updates share rank with
// indices.shape == updates.shape (ONNX), e.g. the LLM's deepstack
// visual-feature injection, where torch exports the 1-D nonzero row indices
// already expanded to the update width.
//
// General axis: for index element i, every coordinate of its data destination
// equals i's own coordinate except along `axis`, where it is indices[i]. So
// with `inner` = product(shape[axis+1:]), one thread handles one index/update
// element and computes
//   dst = (outer * data_axis + indices[i]) * inner + inner_i
// where inner_i = i % inner and outer = (i / inner) / idx_axis.
class ScatterElements : public BufferFactory {
  public:
    explicit ScatterElements(int fp16 = 0)
        : BufferFactory(OpType::SCATTER_ELEMENTS,
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
            const std::string r = attributes.at("reduction");
            if (r == "none")
                reduction_ = 0;
            else if (r == "add")
                reduction_ = 1;
            else if (r == "mul")
                reduction_ = 2;
            else if (r == "max")
                reduction_ = 3;
            else if (r == "min")
                reduction_ = 4;
            else
                throw std::runtime_error(
                    "ScatterElements: unsupported reduction '" + r +
                    "' (want none|add|mul|max|min)");
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
        auto idx_shape = inputs[1]->getShape();
        auto upd_shape = inputs[2]->getShape();
        const int rank = static_cast<int>(idx_shape.size());

        // axis: ONNX allows a negative value; normalize into [0, rank).
        int axis = axis_;
        if (axis < 0)
            axis += rank;
        if (rank == 0 || axis < 0 || axis >= rank) {
            throw std::runtime_error("ScatterElements " + get_name() +
                                     ": axis " + std::to_string(axis_) +
                                     " out of range for rank " +
                                     std::to_string(rank));
        }

        // A shape chain that lost a rank-increasing view breaks the address
        // math below, so validate the shapes before using them. Skip when a dim
        // is still dynamic (-1), where the comparison would be meaningless.
        auto concrete = [](const std::vector<int> &s) {
            for (int d : s)
                if (d < 0)
                    return false;
            return true;
        };
        auto shp = [](const std::vector<int> &s) {
            std::string r = "[";
            for (size_t i = 0; i < s.size(); ++i)
                r += std::to_string(s[i]) + (i + 1 == s.size() ? "]" : ",");
            return r;
        };
        // ONNX: indices.shape == updates.shape (same rank, same dims).
        if (concrete(idx_shape) && concrete(upd_shape) &&
            idx_shape != upd_shape) {
            throw std::runtime_error(
                "ScatterElements " + get_name() + ": indices " +
                shp(idx_shape) + " and updates " + shp(upd_shape) +
                " must have the same shape (ONNX); a view node was folded out "
                "of the shape chain");
        }
        if (concrete(data_shape) &&
            static_cast<int>(data_shape.size()) != rank) {
            throw std::runtime_error(
                "ScatterElements " + get_name() + ": data rank " +
                std::to_string(data_shape.size()) + " != indices rank " +
                std::to_string(rank));
        }

        const int idx_axis = idx_shape[axis];
        const int data_axis =
            (concrete(data_shape) && axis < static_cast<int>(data_shape.size()))
                ? data_shape[axis]
                : idx_axis;
        int inner = 1;
        for (int i = axis + 1; i < rank; ++i)
            inner *= idx_shape[i];

        // indices is int64 in the model (bound as ivec2[] in the shader);
        // read the element count on the correct dtype.
        int n_idx = 0;
        dispatch_by_dtype(inputs[1]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            n_idx =
                static_cast<int>(core::as_tensor<T>(inputs[1])->num_elements());
        });
        const int n_threads = n_idx;

        // GPU dispatch path for BOTH fp32 and fp16. The fp16 shader variant
        // (buffer_scatter_elements_fp16, built with -DFP16) uses a
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
        pc.inner = inner;
        pc.idx_axis = idx_axis;
        pc.data_axis = data_axis;
        pc.reduction = reduction_;
        pc.data_total = concrete(data_shape)
                            ? static_cast<int>(total_elems(data_shape))
                            : 0; // shader skips the bounds check on 0
        submit(&pc, UP_DIV(n_threads, 256), 1, 1);
    }

    int axis_ = 0;
    int reduction_ = 0; // 0=none 1=add 2=mul 3=max 4=min
};

} // namespace ops
} // namespace vkop
#endif // OPS_SCATTER_ELEMENTS_HPP_
