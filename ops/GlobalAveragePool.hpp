// Copyright 2025 @junka
#ifndef OPS_GLOBALAVERAGEPOOL_HPP_
#define OPS_GLOBALAVERAGEPOOL_HPP_

#include "ops/BufferBase.hpp"
#include "ops/Operator.hpp"
#include "ops/PimplFacade.hpp"
namespace vkop {
namespace ops {
namespace globalaveragepool {
struct alignas(16) GpuGAPParam {
    ivec4 inShape; // NCHW
    int accuracy;
};

// Buffer PC (mirrors shaders/buffer/globalaveragepool.comp). GAP reduces
// every dim past the first two, so the host folds the spatial tail into one
// count and the shader needs no rank/dims: input (n, c, s) is at linear
// (n*C + c)*spatial + s, output is the compact (N, C) plane.
struct alignas(16) GapBufferPC {
    int batch;
    int channels;
    int spatial;
    int _pad;
};

// Device-address variant: 16 B scalar prefix + out/in pointers.
struct alignas(16) GapBdaPC {
    GapBufferPC base;
    VkDeviceAddress out_addr;
    VkDeviceAddress in_addr;
};
static_assert(offsetof(GapBdaPC, out_addr) == 16, "GapBdaPC drift");
static_assert(offsetof(GapBdaPC, in_addr) == 24, "GapBdaPC drift");
static_assert(sizeof(GapBdaPC) == 32, "GapBdaPC drift");
} // namespace globalaveragepool

class GlobalAveragePoolImage : public Operator {
  public:
    GlobalAveragePoolImage()
        : Operator(OpType::GLOBALAVERAGEPOOL,
                   {DESCRIPTOR_TYPE_STORAGE,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER},
                   sizeof(globalaveragepool::GpuGAPParam)) {}

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto input_shape = inputs[0]->getShape();
        assert(input_shape.size() > 2);

        int batch = input_shape[0];
        int depth = input_shape[1];

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->size() == 0) {
                output->resize(std::vector<int>{batch, depth});
            }
            auto output_buffer = output->as_storage_buffer(m_dev_);
            objs_.emplace_back(output_buffer);
        });

        // globalaveragepool.comp only stores for accuracy 0 (fp32) and 1
        // (fp16); it has no int8 path, so an int8_t-storage input (a mask, or
        // any weight-only payload riding the same container) must throw instead
        // of silently leaving the output unwritten.
        const int accuracy = core::image_accuracy_elem(
            inputs[0]->elem_kind(), "GlobalAveragePool", "input 0",
            /*allow_int8=*/false);
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto input = core::as_tensor<T>(inputs[0]);
            auto input_image = input->as_input_image(m_dev_, m_cmd_);

            objs_.emplace_back(input_image);
        });

        globalaveragepool::GpuGAPParam para{};
        for (size_t i = 0; i < input_shape.size(); i++) {
            para.inShape[i] = input_shape[i];
        }
        para.accuracy = accuracy;
        submit(&para, UP_DIV(batch, 16), 1, UP_DIV(depth, 4));
    }
};
// Buffer factory — SSBO-backed global average pool, fp32 and fp16. Unlike
// the image path's NCHW->RGBA packing (rank 3/4 only), the compact layout
// carries any rank >= 3; the spatial tail folds into one push-constant count.
class GlobalAveragePoolBuffer : public BufferFactory {
  public:
    explicit GlobalAveragePoolBuffer(int fp16)
        : BufferFactory(OpType::GLOBALAVERAGEPOOL,
                        {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                        sizeof(globalaveragepool::GapBufferPC), fp16) {}

    bool supports_bda_variant() const override { return true; }
    size_t bda_pc_size() const override {
        return sizeof(globalaveragepool::GapBdaPC);
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto input_shape = inputs[0]->getShape();
        if (input_shape.size() < 3) {
            throw std::invalid_argument(
                "GlobalAveragePool input must have at least 3 dimensions");
        }
        int batch = input_shape[0];
        int channels = input_shape[1];
        int spatial = 1;
        for (size_t i = 2; i < input_shape.size(); ++i) {
            spatial *= input_shape[i];
        }

        std::shared_ptr<VulkanBuffer> out_buf;
        std::shared_ptr<VulkanBuffer> in_buf;
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->num_elements() != batch * channels) {
                output->resize(std::vector<int>{batch, channels});
            }
            out_buf = bind_ssbo<T>(outputs[0], /*is_output=*/true);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            in_buf = bind_ssbo<T>(inputs[0], /*is_output=*/false);
        });

        globalaveragepool::GapBufferPC pc{};
        pc.batch = batch;
        pc.channels = channels;
        pc.spatial = spatial;
        int out_total = batch * channels;
        int nthreads = (fp16_ != 0) ? (out_total + 1) / 2 : out_total;
        const int groups = UP_DIV(nthreads, 256);
        if (bda_ready() && out_buf && in_buf && out_buf->hasDeviceAddress() &&
            in_buf->hasDeviceAddress()) {
            const VkDeviceAddress out_addr = out_buf->getDeviceAddress();
            const VkDeviceAddress in_addr = in_buf->getDeviceAddress();
            if ((out_addr % 4) == 0 && (in_addr % 4) == 0) {
                globalaveragepool::GapBdaPC bpc{};
                bpc.base = pc;
                bpc.out_addr = out_addr;
                bpc.in_addr = in_addr;
                submit_bda(&bpc, sizeof(bpc), groups, 1, 1);
                return;
            }
        }
        submit(&pc, groups, 1, 1);
    }
};

// PIMPL façade: buffer SSBO impl when backend_buffer is set, else image.
class GlobalAveragePool : public PimplFacade {
  public:
    GlobalAveragePool(int fp16, bool backend_buffer)
        : PimplFacade(OpType::GLOBALAVERAGEPOOL) {
        if (backend_buffer) {
            impl_ = std::make_unique<GlobalAveragePoolBuffer>(fp16);
        } else {
            impl_ = std::make_unique<GlobalAveragePoolImage>();
        }
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_GLOBALAVERAGEPOOL_HPP_
