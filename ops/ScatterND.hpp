// Copyright 2026 @junka
#ifndef OPS_SCATTER_ND_HPP_
#define OPS_SCATTER_ND_HPP_

#include "ops/BufferBase.hpp"
#include <numeric>

extern "C" {
extern unsigned char buffer_scatternd_spv[];
extern unsigned int buffer_scatternd_spv_len;
}

namespace vkop {
namespace ops {

namespace scatternd {
struct alignas(16) ScatterNDPC {
    int mode;          // 0 = copy data->out, 1 = scatter updates
    int data_total;    // elements in data / output
    int index_rank;    // indices.shape[-1]
    int num_tuples;    // product(indices.shape[:-1])
    int data_rank;     // rank of data
    int dataDims[8];   // data shape, left-aligned (matches GLSL IArr8 layout)
    int slice_size;    // product(data.shape[index_rank:])
    int updates_total; // elements in updates
};
static_assert(sizeof(ScatterNDPC) <= 128, "ScatterNDPC PC overflow");
} // namespace scatternd

// SSBO-only op: ONNX ScatterND (fp32 data + updates, int64 indices).
// llm.vkopbin has 2 instances: data=/rotary_emb/Gather_1_output_0
// [1,1,64] float (GPU-produced), indices=/rotary_emb/Concat_1_output_0
// [1,1,1,3] int64, updates=/rotary_emb/Reshape_1_output_0 [1,1,1] float.
// Both data and updates are GPU-produced, so this is a GPU shader op (a
// read-back of a GPU-produced input during execute() would see stale data).
//
// ONNX defines output as a fresh copy: output = data, then scatter updates.
// Two dispatches in one command buffer (in-order on the same queue): mode 0
// copies data->out, mode 1 scatters updates. Each mode gets its own
// descriptor set (submit_per_ds) so the second dispatch's updated bindings
// don't clobber the first's (vkUpdateDescriptorSets is immediate).
class ScatterND : public BufferFactory {
  public:
    explicit ScatterND()
        : BufferFactory(OpType::SCATTER_ND, buffer_scatternd_spv,
                        buffer_scatternd_spv_len,
                        {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                         DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                        sizeof(scatternd::ScatterNDPC)) {
        update_after_bind_ = true;
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto data_shape = inputs[0]->getShape();
        auto indices_shape = inputs[1]->getShape();
        auto updates_shape = inputs[2]->getShape();
        int data_rank = static_cast<int>(data_shape.size());
        int index_rank = indices_shape.empty()
                             ? 1
                             : indices_shape.back(); // last dim = tuple size
        int num_tuples =
            indices_shape.empty()
                ? 1
                : total_elems(std::vector<int>(indices_shape.begin(),
                                               indices_shape.end() - 1));
        int slice_size = 1;
        for (int i = index_rank; i < data_rank; ++i) {
            slice_size *= data_shape[i];
        }
        int data_total = total_elems(data_shape);
        int updates_total = total_elems(updates_shape);

        // Bind [0]=output, [1]=data, [2]=indices (int64), [3]=updates.
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->size() == 0) {
                output->resize(data_shape);
            }
            bind_ssbo<T>(outputs[0], /*is_output=*/true);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            bind_ssbo<T>(inputs[0], /*is_output=*/false);
        });
        bind_ssbo<int64_t>(inputs[1], /*is_output=*/false);
        dispatch_by_dtype(inputs[2]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            bind_ssbo<T>(inputs[2], /*is_output=*/false);
        });

        scatternd::ScatterNDPC pc{};
        pc.data_total = data_total;
        pc.index_rank = index_rank;
        pc.num_tuples = num_tuples;
        pc.data_rank = data_rank;
        pc.slice_size = slice_size;
        pc.updates_total = updates_total;
        fill_dims(pc.dataDims, data_shape);

        // Pass 0: copy data -> output.
        pc.mode = 0;
        VkDescriptorSet ds0 = allocPassDescriptorSet();
        submit_per_ds(ds0, &pc, UP_DIV(data_total, 256), 1, 1);
        // Barrier: flush pass 0's shader writes so pass 1's read-modify-write
        // of the same output buffer sees them (same-command-buffer compute
        // dispatches have no implicit ordering guarantee).
        auto out_buf = std::dynamic_pointer_cast<VulkanBuffer>(objs_[0]);
        out_buf->shaderWriteBarrier(m_cmd_->get());
        // Pass 1: scatter updates into output.
        pc.mode = 1;
        VkDescriptorSet ds1 = allocPassDescriptorSet();
        submit_per_ds(ds1, &pc, UP_DIV(updates_total, 256), 1, 1);
        freePassDescriptorSet(ds0);
        freePassDescriptorSet(ds1);
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_SCATTER_ND_HPP_
