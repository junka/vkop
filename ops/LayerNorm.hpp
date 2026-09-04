// Copyright 2025 @junka
#ifndef OPS_LAYERNORM_HPP_
#define OPS_LAYERNORM_HPP_

#include "Operator.hpp"
#include "ops/BufferBase.hpp"
#include "ops/PimplFacade.hpp"

#include <memory>
extern "C" {
extern unsigned char image_layernorm_spv[];
extern unsigned int image_layernorm_spv_len;
extern unsigned char buffer_layernorm_spv[];
extern unsigned int buffer_layernorm_spv_len;
// Reused from MatMul: packs fp32 scratch → half2 output (bindings 0=out,
// 3=scratch).
extern unsigned char buffer_matmul_pack_spv[];
extern unsigned int buffer_matmul_pack_spv_len;
}
namespace vkop {
namespace ops {
namespace layernorm {

// torch.nn.functional.layer_norm(input, normalized_shape, weight=None,
// bias=None, eps=1e-05)

struct alignas(16) GpuLayerNormParam {
    ivec4 outShape;
    ivec4 normalizedShape;
    float eps; // default 1e-5
    int normalizedDim;
    int innerSize;
};
} // namespace layernorm

class LayerNormImage : public Operator {
  public:
    LayerNormImage()
        : Operator(OpType::LAYERNORM, image_layernorm_spv,
                   image_layernorm_spv_len,
                   {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                   sizeof(layernorm::GpuLayerNormParam)) {}

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("eps") != attributes.end()) {
            eps_ = std::stof(attributes.at("eps"));
        }
        if (attributes.find("normalized_shape") != attributes.end()) {
            std::string norm_shape_str = attributes.at("normalized_shape");
            normalized_shape_ = parse_attr_list<int>(norm_shape_str);
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto input_shape = inputs[0]->getShape();
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto t) {
            using T = decltype(t);
            auto outputptr = core::as_tensor<T>(outputs[0]);
            if (outputptr->num_elements() != total_elems(input_shape)) {
                outputptr->resize(input_shape);
            }
            auto output_image = outputptr->as_output_image(m_dev_, m_cmd_);
            objs_.emplace_back(output_image);
        });

        dispatch_by_dtype(inputs[0]->dtype(), [&](auto t) {
            using T = decltype(t);
            auto inputptr = core::as_tensor<T>(inputs[0]);
            auto input_image = inputptr->as_input_image(m_dev_, m_cmd_);
            objs_.emplace_back(input_image);
        });
        for (size_t i = 1; i <= 2; ++i) {
            dispatch_by_dtype(inputs[i]->dtype(), [&](auto t) {
                using T = decltype(t);
                auto tensor = core::as_tensor<T>(inputs[i]);
                auto buffer = tensor->as_storage_buffer(m_dev_);
                objs_.emplace_back(buffer);
            });
        }
        int batch = input_shape[0];
        int depth = input_shape[1];
        int out_height = input_shape[2];
        int out_width = input_shape[3];

        layernorm::GpuLayerNormParam para;
        para.eps = eps_;
        para.outShape[0] = batch;
        para.outShape[1] = depth;
        para.outShape[2] = out_height;
        para.outShape[3] = out_width;
        para.normalizedDim = static_cast<int>(normalized_shape_.size());
        para.innerSize = 1;
        for (size_t i = 0; i < normalized_shape_.size(); i++) {
            para.normalizedShape[i] = normalized_shape_[i];
            para.innerSize *= normalized_shape_[i];
        }

        if (normalized_shape_.size() == 1) {
            submit(&para, batch, out_height, UP_DIV(depth, 4));
        } else if (normalized_shape_.size() == 2) {
            submit(&para, batch, 1, UP_DIV(depth, 4));
        } else {
            submit(&para, batch, 1, 1);
        }
    }

    float eps_ = 1e-5;
    std::vector<int> normalized_shape_;
};

// Buffer (SSBO) LayerNorm. Normalizes over the trailing normalized_shape
// (inner_size elements); weight & bias are SSBOs. fp32-only.
class LayerNormBuffer : public BufferFactory {
  public:
    explicit LayerNormBuffer(int /*fp16*/)
        : BufferFactory(OpType::LAYERNORM, buffer_layernorm_spv,
                        buffer_layernorm_spv_len,
                        {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                         DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                        sizeof(LayerNormPC)) {}

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("eps") != attributes.end()) {
            eps_ = std::stof(attributes.at("eps"));
        }
        if (attributes.find("epsilon") != attributes.end()) {
            eps_ = std::stof(attributes.at("epsilon"));
        }
        if (attributes.find("normalized_shape") != attributes.end()) {
            normalized_shape_ =
                parse_attr_list<int>(attributes.at("normalized_shape"));
        }
        if (attributes.find("axis") != attributes.end()) {
            axis_ = std::stol(attributes.at("axis"));
        }
    }

  private:
    // Upcast a fp16 (uint16_t) tensor to a fresh fp32 Tensor (CPU → GPU).
    // Used to route fp16 graph data through the fp32-only layernorm shader.
    std::shared_ptr<core::ITensor>
    upcast_fp16(const std::shared_ptr<core::ITensor> &t) {
        auto src = core::as_tensor<uint16_t>(t);
        src->copyToCPU(m_cmdpool_);
        auto out =
            std::make_shared<core::Tensor<float>>(src->getShape(), false);
        // Tensor(shape,false) ctor sets dims_/size_ but does NOT allocate the
        // CPU data_ vector (data() returns *data_ — null deref if touched).
        // reserveOnCPU() allocates data_ to num_elements() and is the safe
        // entry point for fresh-CPU tensors.
        out->reserveOnCPU();
        for (int i = 0; i < src->num_elements(); ++i) {
            out->data()[i] = core::ITensor::fp16_to_fp32(src->data()[i]);
        }
        out->as_storage_buffer(m_dev_, m_cmd_);
        out->copyToGPU(m_cmdpool_, out->data().data());
        return out;
    }

    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto shape = inputs[0]->getShape();
        int total = total_elems(shape);
        int inner_size = 1;
        if (!normalized_shape_.empty()) {
            // Explicit normalized_shape (ONNX opset<17 style): product of the
            // trailing dims to normalize over.
            for (int d : normalized_shape_) {
                inner_size *= d;
            }
        } else {
            // ONNX LayerNormalization (opset 17+) with axis instead of
            // normalized_shape: normalize over dims [axis, rank). The scale
            // (inputs[1]) carries the normalized shape, so its element count
            // is the inner size. This is the form Qwen3-VL's visual encoder
            // uses (axis=-1, scale [1024]).
            int rank = static_cast<int>(shape.size());
            int ax = axis_;
            if (ax < 0)
                ax += rank;
            inner_size = 1;
            for (int d = ax; d < rank; ++d) {
                inner_size *= shape[d];
            }
        }
        int outer_size = total / inner_size;

        // The buffer layernorm shader is fp32-only (its shared-memory tree
        // reduce races on packed half2 words). When the graph data is fp16
        // (Qwen3-VL visual is fully fp16), upcast input/weight/bias to fp32
        // scratch tensors, run the fp32 shader, then downcast the output back
        // to fp16. The visual encoder runs once per image (not per decode
        // step), so the host-side conversion cost is negligible.
        bool in_fp16 = (inputs[0]->dtype() == typeid(uint16_t));
        if (in_fp16) {
            // Two-pass, all on the GPU (no host readback — that would race
            // the not-yet-submitted m_cmd_):
            //   pass 1 (this shader, fp32): upcast in/w/b to fp32, reduce to a
            //          fp32 scratch buffer (binding 0 of the layernorm shader).
            //   pass 2 (buffer_matmul_pack_spv): pack fp32 scratch → half2 into
            //          the real fp16 output (binding 0=out, 3=scratch).
            // Mirrors MatMul's fp16 reduce→pack pattern. The visual encoder
            // runs once per image, so the upcast cost is negligible.
            auto out32 = std::make_shared<core::Tensor<float>>(shape, false);
            out32->reserveOnCPU(); // alloc CPU backing (size_ only); GPU buf
                                   // made below
            auto in32 = upcast_fp16(inputs[0]);
            auto w32 = upcast_fp16(inputs[1]);
            auto b32 = upcast_fp16(inputs[2]);

            // layernorm shader bindings: 0=out(scratch), 1=in, 2=w, 3=b.
            // scratch uses nullptr cmd (no readBarrier — it's a fresh write
            // target, like MatMul's scratch_; a pre-write readBarrier would
            // mark it READ and conflict with the shader's storage write).
            auto out32_buf = out32->as_storage_buffer(m_dev_, nullptr);
            auto in32_buf = core::as_tensor<float>(in32)->as_storage_buffer(
                m_dev_, nullptr);
            auto w32_buf =
                core::as_tensor<float>(w32)->as_storage_buffer(m_dev_, nullptr);
            auto b32_buf =
                core::as_tensor<float>(b32)->as_storage_buffer(m_dev_, nullptr);
            objs_.emplace_back(out32_buf);
            objs_.emplace_back(in32_buf);
            objs_.emplace_back(w32_buf);
            objs_.emplace_back(b32_buf);

            LayerNormPC pc{};
            pc.eps = eps_;
            pc.inner_size = inner_size;
            pc.outer_size = outer_size;
            submit(&pc, outer_size, 1, 1);

            // Barrier: flush pass-1 scratch writes for pass-2 reads.
            out32_buf->shaderWriteBarrier(m_cmd_->get());

            // Pack pass: separate pipeline (matmul_pack: binding 0=out_fp16,
            // binding 3=scratch). Lazy-build on first fp16 use.
            if (!pack_pipeline_) {
                pack_pipeline_ = std::make_unique<VulkanPipeline>(
                    m_dev_->getLogicalDevice(),
                    std::vector<VkDescriptorType>{
                        DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                        DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                    sizeof(LayerNormPackPC),
                    reinterpret_cast<const uint32_t *>(buffer_matmul_pack_spv),
                    static_cast<int>(buffer_matmul_pack_spv_len), false, 0);
                for (auto &ds : pack_ds_) {
                    ds = pack_pipeline_->allocDescriptorSets();
                }
            }
            // Ensure the real fp16 output is sized + has a GPU buffer.
            auto output = core::as_tensor<uint16_t>(outputs[0]);
            if (output->num_elements() != total) {
                output->resize(shape);
            }
            auto out_buf = output->as_storage_buffer(m_dev_, m_cmd_);

            // Fill pack descriptor set: binding 0=fp16 output, 3=fp32 scratch.
            // (bindings 1,2 bound to inert buffers — matmul_pack never reads
            // them, but the set declares 4 storage slots.)
            fillPackDescriptorSet(pack_ds_[m_id_], out_buf, out32_buf);
            pack_pipeline_->updateDescriptorSets(pack_writes_);
            m_cmd_->bind(*pack_pipeline_, pack_ds_[m_id_]);
            LayerNormPackPC pack_pc{};
            pack_pc.total = total;
            m_cmd_->push_constants(*pack_pipeline_, sizeof(LayerNormPackPC),
                                   &pack_pc);
            int nwords = (total + 1) / 2;
            m_cmd_->dispatch(UP_DIV(nwords, 256), 1, 1);
            return;
        }

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->num_elements() != total_elems(shape)) {
                output->resize(shape);
            }
            bind_ssbo<T>(outputs[0], /*is_output=*/true);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            bind_ssbo<T>(inputs[0], /*is_output=*/false);
        });
        for (size_t i = 1; i <= 2; ++i) {
            dispatch_by_dtype(inputs[i]->dtype(), [&](auto dummy) {
                using T = decltype(dummy);
                bind_ssbo<T>(inputs[i], /*is_output=*/false);
            });
        }

        LayerNormPC pc{};
        pc.eps = eps_;
        pc.inner_size = inner_size;
        pc.outer_size = outer_size;
        submit(&pc, outer_size, 1, 1);
    }

    // Fill the pack-pass descriptor set: binding 0 = fp16 output, binding 3 =
    // fp32 scratch. bindings 1,2 are bound to the scratch too (inert —
    // matmul_pack only reads binding 3) so all 4 declared storage slots are
    // valid and validation layers stay quiet.
    void fillPackDescriptorSet(VkDescriptorSet ds,
                               std::shared_ptr<VulkanBuffer> &out_buf,
                               std::shared_ptr<VulkanBuffer> &scratch_buf) {
        pack_writes_.resize(4);
        auto *out_info =
            std::get<VkDescriptorBufferInfo *>(out_buf->getDescriptorInfo());
        auto *scratch_info = std::get<VkDescriptorBufferInfo *>(
            scratch_buf->getDescriptorInfo());
        for (int i = 0; i < 4; ++i) {
            pack_writes_[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            pack_writes_[i].dstSet = ds;
            pack_writes_[i].dstBinding = static_cast<uint32_t>(i);
            pack_writes_[i].dstArrayElement = 0;
            pack_writes_[i].descriptorCount = 1;
            pack_writes_[i].descriptorType = DESCRIPTOR_TYPE_STORAGE;
            pack_writes_[i].pBufferInfo = (i == 0) ? out_info : scratch_info;
        }
    }

    float eps_ = 1e-5f;
    std::vector<int> normalized_shape_;
    int axis_ = -1;

    // fp16 two-pass pack pipeline (lazy-built on first fp16 input).
    struct alignas(16) LayerNormPackPC {
        int total;
        int _pad0;
        int _pad1;
        int _pad2;
    };
    std::unique_ptr<VulkanPipeline> pack_pipeline_;
    VkDescriptorSet pack_ds_[vkop::kInflight] = {nullptr};
    std::vector<VkWriteDescriptorSet> pack_writes_;
};

// PIMPL façade: buffer SSBO impl when backend_buffer is set, else image.
class LayerNorm : public PimplFacade {
  public:
    LayerNorm(int /*fp16*/, bool backend_buffer)
        : PimplFacade(OpType::LAYERNORM) {
        impl_ = backend_buffer ? std::unique_ptr<Operator>(
                                     std::make_unique<LayerNormBuffer>(0))
                               : std::make_unique<LayerNormImage>();
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_LAYERNORM_HPP_
