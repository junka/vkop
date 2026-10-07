// Copyright 2025 @junka
#ifndef OPS_GLOBALAVERAGEPOOL_HPP_
#define OPS_GLOBALAVERAGEPOOL_HPP_

#include "ops/Operator.hpp"
#include "ops/PimplFacade.hpp"
extern "C" {
extern unsigned char image_globalaveragepool_spv[];
extern unsigned int image_globalaveragepool_spv_len;
}
namespace vkop {
namespace ops {
namespace globalaveragepool {
struct alignas(16) GpuGAPParam {
    ivec4 inShape; // NCHW
    int accuracy;
};
} // namespace globalaveragepool

class GlobalAveragePoolImage : public Operator {
  public:
    GlobalAveragePoolImage()
        : Operator(OpType::GLOBALAVERAGEPOOL, image_globalaveragepool_spv,
                   image_globalaveragepool_spv_len,
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
// PIMPL façade: buffer SSBO impl when backend_buffer is set, else image.
class GlobalAveragePool : public PimplFacade {
  public:
    GlobalAveragePool(int /*fp16*/, bool backend_buffer)
        : PimplFacade(OpType::GLOBALAVERAGEPOOL) {
        (void)backend_buffer;
        // buffer port not yet available; using image impl.
        impl_ = std::make_unique<GlobalAveragePoolImage>();
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_GLOBALAVERAGEPOOL_HPP_
