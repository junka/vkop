// Copyright 2025 @junka
#ifndef OPS_TRANSPOSE_HPP_
#define OPS_TRANSPOSE_HPP_

#include "core/Tensor.hpp"
#include "ops/BufferBase.hpp"
#include "ops/Operator.hpp"
#include "ops/PimplFacade.hpp"
extern "C" {
extern unsigned char image_transpose_spv[];
extern unsigned int image_transpose_spv_len;
extern unsigned char buffer_transpose_spv[];
extern unsigned int buffer_transpose_spv_len;
}
namespace vkop {
namespace ops {

namespace transpose {
struct GpuTransposeParam {
    ivec4 inShape;
    ivec4 outShape;
    ivec4 perms;
    ivec4 reverse_perms;
};

} // namespace transpose

class TransposeImage : public Operator {
  public:
    explicit TransposeImage()
        : Operator(OpType::TRANSPOSE, image_transpose_spv,
                   image_transpose_spv_len,
                   {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER},
                   sizeof(transpose::GpuTransposeParam)) {}
    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("perm") != attributes.end()) {
            perm_ = parse_attr_list<int>(attributes.at("perm"));
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {

        auto inshape = inputs[0]->getShape();
        std::vector<int> outshape(4);
        for (size_t i = 0; i < 4; ++i) {
            outshape[i] = inshape[perm_[i]];
        }

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->size() == 0) {
                output->resize(outshape);
            }
            auto output_image = output->as_output_image(m_dev_, m_cmd_);
            objs_.emplace_back(output_image);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto input = core::as_tensor<T>(inputs[0]);
            auto input_image = input->as_input_image(m_dev_, m_cmd_);
            objs_.emplace_back(input_image);
        });

        std::vector<int> reverse_perms(4);
        for (int i = 0; i < 4; ++i) {
            reverse_perms[perm_[i]] = i;
        }

        transpose::GpuTransposeParam param;
        for (int i = 0; i < 4; i++) {
            param.inShape[i] = inshape[i];
            param.outShape[i] = outshape[i];
            param.perms[i] = perm_[i];
            param.reverse_perms[i] = reverse_perms[i];
        }
        submit(&param, UP_DIV(outshape[3], 16),
               UP_DIV(outshape[2] * outshape[0], 16), UP_DIV(outshape[1], 4));
    }

    std::vector<int> perm_ = {3, 2, 1, 0};
};
// Transpose buffer op (fp32). Arbitrary perm up to 8-D.
class TransposeBuffer : public BufferFactory {
  public:
    explicit TransposeBuffer(int /*fp16*/)
        : BufferFactory(OpType::TRANSPOSE, buffer_transpose_spv,
                        buffer_transpose_spv_len,
                        {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                        sizeof(TransposePC)) {}

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("perm") != attributes.end()) {
            perm_ = parse_attr_list<int>(attributes.at("perm"));
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto inshape = inputs[0]->getShape();
        int rank = static_cast<int>(inshape.size());
        std::vector<int> outshape(rank);
        for (int i = 0; i < rank; ++i) {
            outshape[i] = inshape[perm_[i]];
        }

        // int64 transpose: CPU permute (the single instance feeds NonZero,
        // part of the shape meta-chain).
        if (inputs[0]->dtype() == typeid(int64_t)) {
            int total = total_elems(outshape);
            std::vector<int64_t> out(static_cast<size_t>(total));
            auto src = core::as_tensor<int64_t>(inputs[0]);
            // The input may be GPU-resident only (e.g. NonZero's output is
            // computed by a shader, leaving data_ empty). Pull it back to the
            // host before the CPU permute reads (*src)[i].
            if (!src->has_cpu_data()) {
                src->copyToCPU(m_cmdpool_);
            }
            std::vector<int> in_stride(rank, 1);
            for (int d = rank - 2; d >= 0; --d) {
                in_stride[d] = in_stride[d + 1] * inshape[d + 1];
            }
            std::vector<int> out_stride(rank, 1);
            for (int d = rank - 2; d >= 0; --d) {
                out_stride[d] = out_stride[d + 1] * outshape[d + 1];
            }
            for (int o = 0; o < total; ++o) {
                int r = o;
                int in_lin = 0;
                for (int d = 0; d < rank; ++d) {
                    int coord = (r / out_stride[d]) % outshape[d];
                    in_lin += coord * in_stride[perm_[d]];
                }
                out[static_cast<size_t>(o)] = (*src)[in_lin];
            }
            auto output = core::as_tensor<int64_t>(outputs[0]);
            output->resize(outshape);
            output->fillToCPU(out);
            objs_.emplace_back(output->as_storage_buffer(m_dev_, m_cmd_));
            output->copyToGPU(m_cmdpool_, out.data());
            return;
        }

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->size() == 0) {
                output->resize(outshape);
            }
            bind_ssbo<T>(outputs[0], /*is_output=*/true);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            bind_ssbo<T>(inputs[0], /*is_output=*/false);
        });

        TransposePC pc{};
        pc.rank = rank;
        fill_dims(pc.inDims, inshape);
        fill_dims(pc.outDims, outshape);
        for (int i = 0; i < 8; ++i) {
            pc.perm[i] = (i < rank) ? perm_[i] : i;
        }
        int total = total_elems(outshape);
        submit(&pc, UP_DIV(total, 256), 1, 1);
    }

    std::vector<int> perm_ = {3, 2, 1, 0};
};

// PIMPL façade: buffer SSBO impl when backend_buffer is set, else image.
class Transpose : public PimplFacade {
  public:
    Transpose(int /*fp16*/, bool backend_buffer)
        : PimplFacade(OpType::TRANSPOSE) {
        impl_ = backend_buffer ? std::unique_ptr<Operator>(
                                     std::make_unique<TransposeBuffer>(0))
                               : std::make_unique<TransposeImage>();
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_TRANSPOSE_HPP_
