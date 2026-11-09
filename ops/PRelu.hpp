// Copyright 2025 @junka
#ifndef OPS_PRELU_HPP_
#define OPS_PRELU_HPP_

#include "BinaryFactory.hpp"
#include "ops/BufferBase.hpp"
#include "ops/PimplFacade.hpp"
namespace vkop {
namespace ops {

// Compact scalar prefix used by the BDA build. The descriptor build keeps the
// historical BinaryElemPC layout (and its 116-byte shader block) so models
// converted before this change keep working; changing that layout would make
// an old archive's prelu shader read the wrong offsets. The BDA shader block
// declares exactly these fields, then the three pointers.
struct PReluPC {
    int rank;
    int outDims[8];
    int in1Dims[8];
    int broadcast;
    int total;
};
static_assert(sizeof(PReluPC) == 76, "PReluPC drift");

// Device-address variant. The shader's in0Dims/activation are unused, so the
// compact prefix leaves room for the three pointers at 80/88/96 (104 B).
struct alignas(8) PReluBdaPC {
    PReluPC base;
    VkDeviceAddress out_addr;
    VkDeviceAddress in_addr;
    VkDeviceAddress slope_addr;
};
static_assert(offsetof(PReluBdaPC, out_addr) == 80, "PReluBdaPC drift");
static_assert(sizeof(PReluBdaPC) == 104, "PReluBdaPC drift");

class PReluImage : public BinaryFactory {
  public:
    PReluImage() : BinaryFactory(OpType::PRELU) {}
};

// PReLU buffer op: f(x) = max(0,x) + slope*min(0,x). Slope is broadcast
// right-aligned against the input/output shape. 3 SSBOs (out, input, slope).
// Uses the BinaryElemPC push constant (activation field unused).
class PReluBuffer : public BufferFactory {
  public:
    explicit PReluBuffer(int fp16)
        : BufferFactory(OpType::PRELU,
                        {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                         DESCRIPTOR_TYPE_STORAGE},
                        sizeof(PReluPC), fp16) {}

    bool supports_bda_variant() const override { return true; }
    size_t bda_pc_size() const override { return sizeof(PReluBdaPC); }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto shape = inputs[0]->getShape();
        auto slope_shape = inputs[1]->getShape();
        int total = total_elems(shape);

        std::shared_ptr<VulkanBuffer> out_buf;
        std::shared_ptr<VulkanBuffer> in_buf;
        std::shared_ptr<VulkanBuffer> slope_buf;
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->num_elements() != total_elems(shape)) {
                output->resize(shape);
            }
            out_buf = bind_ssbo<T>(outputs[0], /*is_output=*/true);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            in_buf = bind_ssbo<T>(inputs[0], /*is_output=*/false);
        });
        dispatch_by_dtype(inputs[1]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            slope_buf = bind_ssbo<T>(inputs[1], /*is_output=*/false);
        });

        PReluPC pc{};
        pc.rank = static_cast<int>(shape.size());
        fill_dims(pc.outDims, shape);
        fill_dims_broadcast(pc.in1Dims, slope_shape, pc.rank);
        pc.broadcast = (slope_shape == shape) ? 0 : 1;
        pc.total = total;
        int nthreads = (fp16_ != 0) ? (total + 1) / 2 : total;
        const int groups = UP_DIV(nthreads, 256);
        if (bda_ready() && out_buf && in_buf && slope_buf &&
            out_buf->hasDeviceAddress() && in_buf->hasDeviceAddress() &&
            slope_buf->hasDeviceAddress()) {
            const VkDeviceAddress out_addr = out_buf->getDeviceAddress();
            const VkDeviceAddress in_addr = in_buf->getDeviceAddress();
            const VkDeviceAddress slope_addr = slope_buf->getDeviceAddress();
            if ((out_addr % 4) == 0 && (in_addr % 4) == 0 &&
                (slope_addr % 4) == 0) {
                PReluBdaPC bpc{};
                bpc.base = pc;
                bpc.out_addr = out_addr;
                bpc.in_addr = in_addr;
                bpc.slope_addr = slope_addr;
                submit_bda(&bpc, sizeof(bpc), groups, 1, 1);
                return;
            }
        }
        submit(&pc, groups, 1, 1);
    }
};

class PRelu : public PimplFacade {
  public:
    PRelu(int fp16, bool backend_buffer) : PimplFacade(OpType::PRELU) {
        impl_ =
            backend_buffer
                ? std::unique_ptr<Operator>(std::make_unique<PReluBuffer>(fp16))
                : std::make_unique<PReluImage>();
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_PRELU_HPP_
