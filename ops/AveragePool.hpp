// Copyright 2025 @junka
#ifndef OPS_AVERAGEPOOL_HPP_
#define OPS_AVERAGEPOOL_HPP_

#include "ops/BufferBase.hpp"
#include "ops/Operator.hpp"
#include "ops/PimplFacade.hpp"
namespace vkop {
namespace ops {
namespace averagepool {
struct alignas(16) GpuAveragePoolParam {
    ivec4 inShape; // NCHW
    ivec4 outShape;
    ivec4 pads;
    ivec2 kernel_shape;
    ivec2 strides;
    int count_include_pad;
};

// Buffer PC (mirrors shaders/buffer/averagepool.comp). One module serves fp32
// and fp16 storage: `fp16` says the data edges are packed two halfs per word,
// and the host sets it from the tensor it bound.
struct alignas(16) AveragePoolPC {
    int N;
    int C;
    int H_in;
    int W_in;
    int H_out;
    int W_out;
    int pad_top;
    int pad_left;
    int pad_bottom;
    int pad_right;
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;
    int dilation_h;
    int dilation_w;
    int count_include_pad;
    int fp16;
};
static_assert(sizeof(AveragePoolPC) == 80, "AveragePoolPC drift");
static_assert(sizeof(AveragePoolPC) <= 128, "AveragePoolPC PC overflow");

// Output shape for the buffer impl: ONNX AveragePool, dilation-aware. The
// image impl's own loop is the same rule with d=1 (its VALID branch differs,
// but no graph uses VALID), so the two backends allocate the same output for
// every configuration the tests drive.
inline std::vector<int>
out_shape(const std::vector<int> &in_shape, const std::vector<int> &kernel,
          const std::vector<int> &strides, const std::vector<int> &pads,
          int ceil_mode, int auto_pad,
          const std::vector<int> &dilations = {1, 1}) {
    std::vector<int> out = in_shape;
    int rank = static_cast<int>(in_shape.size());
    auto extent = [&](int a) {
        int d = (a < static_cast<int>(dilations.size())) ? dilations[a] : 1;
        return d * (kernel[a] - 1) + 1;
    };
    if (auto_pad == 1) { // VALID
        for (int i = 2; i < rank; ++i) {
            out[i] = (in_shape[i] - extent(i - 2)) / strides[i - 2] + 1;
        }
    } else if (auto_pad == 2) { // SAME_UPPER / SAME_LOWER
        for (int i = 2; i < rank; ++i) {
            out[i] = (in_shape[i] + strides[i - 2] - 1) / strides[i - 2];
        }
    } else {
        for (int i = 2; i < rank; ++i) {
            auto padding = pads[i - 2] + pads[i];
            if (ceil_mode == 0) {
                out[i] =
                    (in_shape[i] + padding - extent(i - 2)) / strides[i - 2] +
                    1;
            } else {
                out[i] = (in_shape[i] + padding - extent(i - 2) +
                          strides[i - 2] - 1) /
                             strides[i - 2] +
                         1;
            }
        }
    }
    return out;
}
} // namespace averagepool

class AveragePoolImage : public Operator {
  public:
    AveragePoolImage()
        : Operator(OpType::AVERAGEPOOL,
                   {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER},
                   sizeof(averagepool::GpuAveragePoolParam)) {}

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("ceil_mode") != attributes.end()) {
            ceil_mode_ = std::stol(attributes.at("ceil_mode"));
        }
        if (attributes.find("count_include_pad") != attributes.end()) {
            count_include_pad_ = std::stol(attributes.at("count_include_pad"));
        }
        if (attributes.find("dilations") != attributes.end()) {
            std::string dila_str = attributes.at("dilations");
            if (dila_str.find(',') != std::string::npos) {
                dilations_ = parse_attr_list<int>(dila_str);
            } else {
                int d = std::stol(dila_str);
                dilations_ = {d, d};
            }
        }

        if (attributes.find("kernel_shape") != attributes.end()) {
            std::string kernel_str = attributes.at("kernel_shape");
            if (kernel_str.find(',') != std::string::npos) {
                kernel_shape_ = parse_attr_list<int>(kernel_str);
            } else {
                int k = std::stol(kernel_str);
                kernel_shape_ = {k, k};
            }
        } else if (attributes.find("kernel_size") != attributes.end()) {
            std::string kernel_str = attributes.at("kernel_size");
            if (kernel_str.find(',') != std::string::npos) {
                kernel_shape_ = parse_attr_list<int>(kernel_str);
            } else {
                int k = std::stol(kernel_str);
                kernel_shape_ = {k, k};
            }
        }

        if (attributes.find("pads") != attributes.end()) {
            std::string pad_str = attributes.at("pads");
            if (pad_str.find(',') != std::string::npos) {
                pads_ = parse_attr_list<int>(pad_str);
            } else {
                int p = std::stol(pad_str);
                pads_ = {p, p};
            }
        }

        if (attributes.find("strides") != attributes.end()) {
            std::string stride_str = attributes.at("strides");
            if (stride_str.find(',') != std::string::npos) {
                strides_ = parse_attr_list<int>(stride_str);
            } else {
                int s = std::stol(stride_str);
                strides_ = {s, s};
            }
        }
        if (attributes.find("auto_pad") != attributes.end()) {
            std::string auto_pad = attributes.at("auto_pad");
            if (auto_pad == "VALID") {
                auto_pad_ = 1;
            } else if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
                auto_pad_ = 2;
            } else if (auto_pad == "NOTSET") {
                // do nothing
            } else {
                throw std::invalid_argument("Unsupported auto_pad: " +
                                            auto_pad);
            }
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto input_shape = inputs[0]->getShape();

        int rank = inputs[0]->num_dims();
        auto outshape = input_shape;

        if (auto_pad_ != 0) {
            if (auto_pad_ == 1) {
                for (int i = 2; i < rank; i++) {
                    outshape[i] = (input_shape[i] + kernel_shape_[i - 2]) /
                                      strides_[i - 2] +
                                  1;
                }
            } else {
                for (int i = 2; i < rank; i++) {
                    outshape[i] = (input_shape[i] + strides_[i - 2] - 1) /
                                  strides_[i - 2];
                }
            }
        } else {
            for (int i = 2; i < rank; i++) {
                auto padding = pads_[i - 2] + pads_[i];
                if (ceil_mode_ == 0) {
                    outshape[i] =
                        (input_shape[i] + padding - kernel_shape_[i - 2]) /
                            strides_[i - 2] +
                        1;
                } else {
                    outshape[i] = (input_shape[i] + padding -
                                   kernel_shape_[i - 2] + strides_[i - 2] - 1) /
                                      strides_[i - 2] +
                                  1;
                }
            }
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

        averagepool::GpuAveragePoolParam para;
        for (int i = 0; i < rank; i++) {
            para.inShape[i] = input_shape[i];
            para.outShape[i] = outshape[i];
            para.pads[i] = pads_[i];
        }
        for (int i = 0; i < 2; i++) {
            para.kernel_shape[i] = kernel_shape_[i];
            para.strides[i] = strides_[i];
        }
        para.count_include_pad = count_include_pad_;
        auto out_gpu_shape = outputs[0]->getGPUShape();

        submit(&para, UP_DIV(out_gpu_shape[0], 16),
               UP_DIV(out_gpu_shape[1], 16), out_gpu_shape[2]);
    }

    std::vector<int> kernel_shape_;
    std::vector<int> strides_ = {1, 1};
    std::vector<int> pads_ = {0, 0, 0, 0};
    std::vector<int> dilations_ = {1, 1};
    int ceil_mode_ = 0;
    int count_include_pad_ = 1;
    int auto_pad_ = 0; // notset
};
// Buffer factory — SSBO-backed average pool, rank-4 NCHW. One kernel serves
// fp32 and fp16 storage (its `fp16` PC field picks the packed-half layout, set
// from the tensor dtype), so it is NOT a dual-fp16 shader: the fp16 request
// resolves to the same archive key.
class AveragePoolBuffer : public BufferFactory {
  public:
    explicit AveragePoolBuffer(int fp16)
        : BufferFactory(OpType::AVERAGEPOOL,
                        {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                        sizeof(averagepool::AveragePoolPC), fp16) {}

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        auto get = [&](const char *key) -> const std::string * {
            auto it = attributes.find(key);
            return it == attributes.end() ? nullptr : &it->second;
        };
        auto pair = [&](const std::string &s) {
            if (!s.empty() && s.front() == '[') {
                return parse_attr_list<int>(s);
            }
            int v = std::stol(s);
            return std::vector<int>{v, v};
        };
        if (const auto *s = get("kernel_shape")) {
            kernel_shape_ = pair(*s);
        } else if (const auto *s = get("kernel_size")) {
            kernel_shape_ = pair(*s);
        } else {
            throw std::invalid_argument(
                "AveragePool (buffer backend): kernel_shape is required");
        }
        if (const auto *s = get("strides")) {
            strides_ = pair(*s);
        }
        if (const auto *s = get("pads")) {
            pads_ = parse_attr_list<int>(*s);
        }
        if (const auto *s = get("dilations")) {
            dilations_ = pair(*s);
        }
        if (const auto *s = get("ceil_mode")) {
            ceil_mode_ = std::stol(*s);
        }
        if (const auto *s = get("count_include_pad")) {
            count_include_pad_ = std::stol(*s);
        }
        if (const auto *s = get("auto_pad")) {
            if (*s == "VALID") {
                auto_pad_ = 1;
            } else if (*s == "SAME_UPPER" || *s == "SAME_LOWER") {
                auto_pad_ = 2;
            } else if (*s == "NOTSET") {
                auto_pad_ = 0;
            } else {
                throw std::invalid_argument("Unsupported auto_pad: " + *s);
            }
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto in_shape = inputs[0]->getShape();
        if (in_shape.size() != 4) {
            throw std::invalid_argument(
                "AveragePool (buffer backend): input must be rank 4 (NCHW), "
                "got rank " +
                std::to_string(in_shape.size()));
        }
        if (pads_.size() < 4) {
            pads_.resize(4, 0);
        }
        if (kernel_shape_.size() < 2 || strides_.size() < 2 ||
            dilations_.size() < 2) {
            throw std::invalid_argument(
                "AveragePool (buffer backend): kernel_shape/strides/dilations "
                "must each have 2 entries");
        }
        std::vector<int> out_shape = averagepool::out_shape(
            in_shape, kernel_shape_, strides_, pads_, ceil_mode_, auto_pad_);

        std::shared_ptr<VulkanBuffer> out_buf;
        std::shared_ptr<VulkanBuffer> in_buf;
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->num_elements() != total_elems(out_shape)) {
                output->resize(out_shape);
            }
            out_buf = bind_ssbo<T>(outputs[0], /*is_output=*/true);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            in_buf = bind_ssbo<T>(inputs[0], /*is_output=*/false);
        });
        (void)out_buf;
        (void)in_buf;

        averagepool::AveragePoolPC pc{};
        pc.N = in_shape[0];
        pc.C = in_shape[1];
        pc.H_in = in_shape[2];
        pc.W_in = in_shape[3];
        pc.H_out = out_shape[2];
        pc.W_out = out_shape[3];
        pc.pad_top = pads_[0];
        pc.pad_left = pads_[1];
        pc.pad_bottom = pads_[2];
        pc.pad_right = pads_[3];
        pc.kernel_h = kernel_shape_[0];
        pc.kernel_w = kernel_shape_[1];
        pc.stride_h = strides_[0];
        pc.stride_w = strides_[1];
        pc.dilation_h = dilations_[0];
        pc.dilation_w = dilations_[1];
        pc.count_include_pad = count_include_pad_;
        // The op is a single module for both layouts, so the packed-half flag
        // follows the tensor actually bound, not the constructor's precision
        // hint (which for this op type comes from the graph precision).
        const bool data_fp16 = inputs[0]->dtype() == typeid(uint16_t);
        pc.fp16 = data_fp16 ? 1 : 0;

        // fp16 packs two elements per word: one thread per output word.
        int total = total_elems(out_shape);
        int nthreads = data_fp16 ? (total + 1) / 2 : total;
        submit(&pc, UP_DIV(nthreads, 256), 1, 1);
    }

    std::vector<int> kernel_shape_;
    std::vector<int> strides_ = {1, 1};
    std::vector<int> pads_ = {0, 0, 0, 0};
    std::vector<int> dilations_ = {1, 1};
    int ceil_mode_ = 0;
    int count_include_pad_ = 1;
    int auto_pad_ = 0; // NOTSET
};

// PIMPL façade: buffer SSBO impl when backend_buffer is set, else image.
class AveragePool : public PimplFacade {
  public:
    AveragePool(int fp16, bool backend_buffer)
        : PimplFacade(OpType::AVERAGEPOOL) {
        if (backend_buffer) {
            impl_ = std::make_unique<AveragePoolBuffer>(fp16);
        } else {
            impl_ = std::make_unique<AveragePoolImage>();
        }
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_AVERAGEPOOL_HPP_
