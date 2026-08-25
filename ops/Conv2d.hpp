// Copyright 2025 @junka
#ifndef OPS_OCONV2D_HPP_
#define OPS_OCONV2D_HPP_

#include "ops/PimplFacade.hpp"
#include <memory>
#include <string>
#include <unordered_map>

#include "Operator.hpp"
#include "ops/BufferBase.hpp"
extern "C" {
extern unsigned char image_conv2d_spv[];
extern unsigned int image_conv2d_spv_len;
extern unsigned char buffer_conv2d_spv[];
extern unsigned int buffer_conv2d_spv_len;
extern unsigned char buffer_conv2d_fp16_spv[];
extern unsigned int buffer_conv2d_fp16_spv_len;
// Reuse the MatMul pack pass (read two adjacent fp32 -> pack half2) for the
// fp16 conv two-pass reduce+pack.
extern unsigned char buffer_matmul_pack_spv[];
extern unsigned int buffer_matmul_pack_spv_len;
}
namespace vkop {
namespace ops {
namespace conv2d {

enum class PaddingMode { ZEROS, REFLECT, REPLICATE, CIRCULAR };
enum class ActivationMode {
    NONE,
    RELU,
    SIGMOID,
    TANH,
    HARDSWISH,
    MISH,
    RELU6,
    SWISH,
};

struct alignas(16) GPUConv2dParam {
    ivec4 inputSize;
    ivec4 outputSize;
    ivec2 kernel_shape;
    ivec2 stride;
    ivec2 padding;
    ivec2 dilation;

    int groups;
    int bias;
    int transpose;
    int pack;
    int activation;

    int accuracy; // only for conv para, 0 : fp32, 1 : fp16, 2: int8
};

// Buffer-backend (SSBO) push constant. Naive direct convolution: one thread
// per output element [n,oc,oh,ow]. 21 ints = 84 bytes (well under the 128B
// push-constant limit).
struct alignas(16) Conv2dBufferPC {
    int N, IC, IH, IW;
    int OC, OH, OW;
    int KH, KW;
    int stride_h, stride_w;
    int pad_h, pad_w;
    int dil_h, dil_w;
    int groups;
    int has_bias;
    int fp32; // 1 = fp32 single-pass (write output); 0 = fp16 reduce (write
              // scratch)
    int activation;  // conv2d::ActivationMode (0 = NONE)
    int weight_int8; // 1 = weights are packed int8 bytes (dequant via scale);
                     // 0 = fp32/fp16 weights
};

} // namespace conv2d

class Conv2dImage : public Operator {
  public:
    Conv2dImage()
        : Operator(OpType::CONV2D, image_conv2d_spv, image_conv2d_spv_len,
                   {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    DESCRIPTOR_TYPE_UNIFORM, DESCRIPTOR_TYPE_UNIFORM},
                   sizeof(conv2d::GPUConv2dParam)) {
        activation_ = conv2d::ActivationMode::NONE;
    }

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("dilations") != attributes.end()) {
            std::string dila_str = attributes.at("dilations");
            if (dila_str.find(',') != std::string::npos) {
                dilations_ = parse_attr_list<int>(dila_str);
            } else {
                int d = std::stol(dila_str);
                dilations_ = {d, d};
            }
        }

        if (attributes.find("group") != attributes.end()) {
            groups_ = std::stol(attributes.at("group"));
        }

        if (attributes.find("kernel_shape") != attributes.end()) {
            std::string kernel_str = attributes.at("kernel_shape");
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
                pads_ = {0, 0};
            } else if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
                // SAME would let out_h = ceil(in_h/stride_h)
                // so padding_h = ((out_h-1)*stride_h + (kernel_h-1)*dilations_h
                // + 1 - in_h)/2 here we just set padding to kernel_size/2, and
                // only support stride=1,dilation=1 case
                if (strides_[0] != 1 || strides_[1] != 1 ||
                    dilations_[0] != 1 || dilations_[1] != 1) {
                    throw std::invalid_argument("Only support stride=1 and "
                                                "dilation=1 for SAME auto_pad");
                }
                pads_ = {kernel_shape_[0] / 2, kernel_shape_[1] / 2};
            } else if (auto_pad == "NOTSET") {
                // do nothing
            } else {
                throw std::invalid_argument("Unsupported auto_pad: " +
                                            auto_pad);
            }
        }

        if (attributes.find("activation") != attributes.end()) {
            std::string activation = attributes.at("activation");
            if (activation == "Relu") {
                activation_ = conv2d::ActivationMode::RELU;
            } else if (activation == "Sigmoid") {
                activation_ = conv2d::ActivationMode::SIGMOID;
            } else if (activation == "Tanh") {
                activation_ = conv2d::ActivationMode::TANH;
            } else if (activation == "HardSwish") {
                activation_ = conv2d::ActivationMode::HARDSWISH;
            } else if (activation == "Mish") {
                activation_ = conv2d::ActivationMode::MISH;
            } else if (activation == "Relu6") {
                activation_ = conv2d::ActivationMode::RELU6;
            } else if (activation == "Swish") {
                activation_ = conv2d::ActivationMode::SWISH;
            } else {
                throw std::invalid_argument("Unsupported activation: " +
                                            activation);
            }
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        std::vector<int> input_shape = inputs[0]->getShape();
        std::vector<int> weight_shape = inputs[1]->getShape();

        int batch = input_shape[0];
        int depth = input_shape[1];
        int in_height = input_shape[2];
        int in_width = input_shape[3];
        int out_batch = batch;
        int out_depth = weight_shape[0];
        int kernel_h =
            kernel_shape_[0] == 0 ? weight_shape[2] : kernel_shape_[0];
        int kernel_w =
            kernel_shape_[1] == 0 ? weight_shape[3] : kernel_shape_[1];
        int out_height =
            ((in_height + 2 * pads_[0] - dilations_[0] * (kernel_h - 1) - 1) /
             strides_[0]) +
            1;
        int out_width =
            ((in_width + 2 * pads_[1] - dilations_[1] * (kernel_w - 1) - 1) /
             strides_[1]) +
            1;
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto type_tag) {
            using T = decltype(type_tag);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->size() == 0) {
                output->resize(out_batch, out_depth, out_height, out_width);
            }

            auto output_image = output->as_output_image(m_dev_, m_cmd_);
            objs_.emplace_back(output_image);
        });

        dispatch_by_dtype(inputs[0]->dtype(), [&](auto type_tag) {
            using T = decltype(type_tag);
            auto input = core::as_tensor<T>(inputs[0]);
            auto input_image = input->as_input_image(m_dev_, m_cmd_);
            objs_.emplace_back(input_image);
        });
        int accuracy = 0;
        dispatch_by_dtype(inputs[1]->dtype(), [&](auto type_tag) {
            using T = decltype(type_tag);
            auto weight = core::as_tensor<T>(inputs[1]);
            auto weight_image =
                weight->as_input_image(m_dev_, m_cmd_, false, true);
            objs_.emplace_back(weight_image);
            if (typeid(T) == typeid(uint16_t)) {
                accuracy = 1;
            } else if (typeid(T) == typeid(int8_t)) {
                accuracy = 2;
            }
        });
        size_t scale_index = 2;
        if ((inputs.size() == 3 && accuracy != 2) ||
            (inputs.size() == 4 && accuracy == 2)) {
            dispatch_by_dtype(inputs[2]->dtype(), [&](auto type_tag) {
                using T = decltype(type_tag);
                auto bias = core::as_tensor<T>(inputs[2]);
                auto bias_buffer = bias->as_uniform_bufferview(m_dev_);
                objs_.emplace_back(bias_buffer);
            });
            scale_index++;
        } else {
            objs_.emplace_back(dummy_bufferview_);
        }

        if (accuracy == 2) {
            // has bias it is 3; no bias it is 2;
            dispatch_by_dtype(inputs[scale_index]->dtype(), [&](auto type_tag) {
                using T = decltype(type_tag);
                auto scale = core::as_tensor<T>(inputs[scale_index]);
                auto scale_buffer = scale->as_uniform_bufferview(m_dev_);
                objs_.emplace_back(scale_buffer);
            });
        } else {
            objs_.emplace_back(dummy_bufferview_);
        }

        auto out_gpu_shape = outputs[0]->getGPUShape();
        conv2d::GPUConv2dParam para;
        para.inputSize[0] = in_width;
        para.inputSize[1] = in_height;
        para.inputSize[2] = depth;
        para.inputSize[3] = batch;
        para.outputSize[0] = out_width;
        para.outputSize[1] = out_height;
        para.outputSize[2] = out_depth;
        para.outputSize[3] = out_batch;
        para.kernel_shape[0] = weight_shape[3];
        para.kernel_shape[1] = weight_shape[2];
        para.stride[0] = strides_[0];
        para.stride[1] = strides_[1];
        para.padding[0] = pads_[0];
        para.padding[1] = pads_[1];
        para.dilation[0] = dilations_[0];
        para.dilation[1] = dilations_[1];

        para.groups = groups_;
        para.bias = ((inputs.size() > 2)) ? 1 : 0;
        para.transpose = inputs[1]->get_transpose() ? 1 : 0;
        para.pack = inputs[1]->get_pack() ? 1 : 0;
        para.activation = static_cast<int>(activation_);
        para.accuracy = accuracy;

        submit(&para, UP_DIV(out_gpu_shape[0], 16),
               UP_DIV(out_gpu_shape[1], 16), out_gpu_shape[2]);
    }

    std::vector<int> kernel_shape_ = {0, 0};
    std::vector<int> strides_ = {1, 1};
    std::vector<int> pads_ = {0, 0};
    std::vector<int> dilations_ = {1, 1};
    int groups_ = 1;

    conv2d::ActivationMode activation_ = conv2d::ActivationMode::NONE;
};

// Conv2d buffer-backend (SSBO) op. Naive direct convolution: one thread per
// output element [n,oc,oh,ow], accumulating IC/groups * KH * KW MACs in fp32.
// Mirrors the MatMulBuffer two-pass fp16 pattern: the reduce pass (this
// shader, fp16 build) writes fp32 results to a scratch buffer (one uint per
// output element — no half2 word race); the pack pass (buffer_matmul_pack_spv,
// reused) packs two adjacent fp32 results into a half2 output word.
class Conv2dBuffer : public BufferFactory {
  public:
    explicit Conv2dBuffer(int fp16)
        : BufferFactory(
              OpType::CONV2D, fp16 ? buffer_conv2d_fp16_spv : buffer_conv2d_spv,
              fp16 ? buffer_conv2d_fp16_spv_len : buffer_conv2d_spv_len,
              std::vector<VkDescriptorType>{
                  DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                  DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                  DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
              sizeof(conv2d::Conv2dBufferPC), fp16) {
        update_after_bind_ = true;
    }

    void set_runtime_device(
        const std::shared_ptr<VulkanDevice> &dev,
        const std::shared_ptr<VulkanCommandPool> &cmdpool) override {
        BufferFactory::set_runtime_device(dev, cmdpool);
        if (fp16_ != 0 && !pack_pipeline_) {
            bool use_uab = update_after_bind_ &&
                           dev->is_support_descriptor_update_after_bind();
            // The pack shader (buffer_matmul_pack_spv) declares 4 bindings
            // (Out=0, Scratch=3); we bind 6 (matching the reduce pipeline) so
            // the same objs_ vector feeds both passes — the extra bias/scale
            // slots are inert in the pack shader.
            pack_pipeline_ = std::make_unique<VulkanPipeline>(
                dev->getLogicalDevice(),
                std::vector<VkDescriptorType>{
                    DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                    DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                    DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                sizeof(MatMulPackPC),
                reinterpret_cast<const uint32_t *>(buffer_matmul_pack_spv),
                static_cast<int>(buffer_matmul_pack_spv_len), use_uab, 0);
            for (auto &ds : pack_ds_) {
                ds = pack_pipeline_->allocDescriptorSets();
            }
        }
    }

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        // Same attribute parsing as Conv2dImage (strides/pads/dilations/group/
        // kernel_shape/auto_pad) plus fused activation (applied in the fp32
        // reduce domain before the pack pass).
        if (attributes.find("dilations") != attributes.end()) {
            std::string dila_str = attributes.at("dilations");
            if (dila_str.find(',') != std::string::npos) {
                dilations_ = parse_attr_list<int>(dila_str);
            } else {
                int d = std::stol(dila_str);
                dilations_ = {d, d};
            }
        }
        if (attributes.find("group") != attributes.end()) {
            groups_ = std::stol(attributes.at("group"));
        }
        if (attributes.find("kernel_shape") != attributes.end()) {
            std::string kernel_str = attributes.at("kernel_shape");
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
                pads_ = {0, 0};
            } else if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
                if (strides_[0] != 1 || strides_[1] != 1 ||
                    dilations_[0] != 1 || dilations_[1] != 1) {
                    throw std::invalid_argument("Only support stride=1 and "
                                                "dilation=1 for SAME auto_pad");
                }
                pads_ = {kernel_shape_[0] / 2, kernel_shape_[1] / 2};
            }
        }
        if (attributes.find("activation") != attributes.end()) {
            std::string activation = attributes.at("activation");
            if (activation == "Relu") {
                activation_ = conv2d::ActivationMode::RELU;
            } else if (activation == "Sigmoid") {
                activation_ = conv2d::ActivationMode::SIGMOID;
            } else if (activation == "Tanh") {
                activation_ = conv2d::ActivationMode::TANH;
            } else if (activation == "HardSwish") {
                activation_ = conv2d::ActivationMode::HARDSWISH;
            } else if (activation == "Mish") {
                activation_ = conv2d::ActivationMode::MISH;
            } else if (activation == "Relu6") {
                activation_ = conv2d::ActivationMode::RELU6;
            } else if (activation == "Swish") {
                activation_ = conv2d::ActivationMode::SWISH;
            } else {
                throw std::invalid_argument("Unsupported activation: " +
                                            activation);
            }
        }
    }

  private:
    struct alignas(16) MatMulPackPC {
        int total;
        int _pad0;
        int _pad1;
        int _pad2;
    };

    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        std::vector<int> input_shape = inputs[0]->getShape();
        std::vector<int> weight_shape = inputs[1]->getShape();

        int batch = input_shape[0];
        int depth = input_shape[1];
        int in_height = input_shape[2];
        int in_width = input_shape[3];
        int out_depth = weight_shape[0];
        int kernel_h =
            kernel_shape_[0] == 0 ? weight_shape[2] : kernel_shape_[0];
        int kernel_w =
            kernel_shape_[1] == 0 ? weight_shape[3] : kernel_shape_[1];
        int out_height =
            ((in_height + 2 * pads_[0] - dilations_[0] * (kernel_h - 1) - 1) /
             strides_[0]) +
            1;
        int out_width =
            ((in_width + 2 * pads_[1] - dilations_[1] * (kernel_w - 1) - 1) /
             strides_[1]) +
            1;
        std::vector<int> out_shape = {batch, out_depth, out_height, out_width};
        int total = batch * out_depth * out_height * out_width;

        // int8 weight-only quantization: weight (inputs[1]) is int8 and a
        // per-output-channel scale is appended as the last input. Layout:
        //   fp32/fp16, no bias : [X, W]
        //   fp32/fp16, w/ bias : [X, W, bias]
        //   int8,     no bias : [X, W_int8, scale]
        //   int8,     w/ bias : [X, W_int8, bias, scale]
        bool weight_int8 = inputs[1]->dtype() == typeid(int8_t);
        bool has_bias =
            weight_int8 ? (inputs.size() == 4) : (inputs.size() > 2);
        size_t bias_index = 2;
        size_t scale_index = weight_int8 ? (has_bias ? 3 : 2) : 0;
        // Reduce pipeline bindings: [out, X, W, scratch/dummy, bias/dummy,
        // scale/dummy].
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto type_tag) {
            using T = decltype(type_tag);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->num_elements() != total) {
                output->resize(out_shape);
            }
            bind_ssbo<T>(outputs[0], /*is_output=*/true);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto type_tag) {
            using T = decltype(type_tag);
            bind_ssbo<T>(inputs[0], /*is_output=*/false); // X
        });
        dispatch_by_dtype(inputs[1]->dtype(), [&](auto type_tag) {
            using T = decltype(type_tag);
            bind_ssbo<T>(inputs[1], /*is_output=*/false); // W
        });

        if (fp16_ != 0) {
            // fp16 two-pass: scratch holds fp32 reduce results (one uint per
            // output element). Clamp empty (total==0) to a 16-byte dummy —
            // vkCreateBuffer rejects size 0 on Intel, and the dispatch is 0
            // threads anyway.
            size_t scratch_bytes = static_cast<size_t>(total) * sizeof(float);
            if (scratch_bytes == 0) {
                scratch_bytes = 16;
            }
            scratch_ = std::make_shared<VulkanBuffer>(
                m_dev_, scratch_bytes,
                STORAGE | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            objs_.emplace_back(scratch_);
        } else {
            objs_.emplace_back(dummy_buffer_);
        }

        if (has_bias) {
            dispatch_by_dtype(inputs[bias_index]->dtype(), [&](auto type_tag) {
                using T = decltype(type_tag);
                bind_ssbo<T>(inputs[bias_index], /*is_output=*/false); // bias
            });
        } else {
            objs_.emplace_back(dummy_buffer_);
        }

        if (weight_int8) {
            // Scale is fp32 in the optimizer, but the runtime may upload it as
            // fp16 when the graph is fp16; bind by its own dtype. The shader
            // reads it via BUF_LOAD (dtype-aware).
            dispatch_by_dtype(inputs[scale_index]->dtype(), [&](auto type_tag) {
                using T = decltype(type_tag);
                bind_ssbo<T>(inputs[scale_index], /*is_output=*/false); // scale
            });
        } else {
            objs_.emplace_back(dummy_buffer_);
        }

        conv2d::Conv2dBufferPC pc{};
        pc.N = batch;
        pc.IC = depth;
        pc.IH = in_height;
        pc.IW = in_width;
        pc.OC = out_depth;
        pc.OH = out_height;
        pc.OW = out_width;
        pc.KH = kernel_h;
        pc.KW = kernel_w;
        pc.stride_h = strides_[0];
        pc.stride_w = strides_[1];
        pc.pad_h = pads_[0];
        pc.pad_w = pads_[1];
        pc.dil_h = dilations_[0];
        pc.dil_w = dilations_[1];
        pc.groups = groups_;
        pc.has_bias = has_bias ? 1 : 0;
        pc.fp32 = (fp16_ != 0) ? 0 : 1;
        pc.activation = static_cast<int>(activation_);
        pc.weight_int8 = weight_int8 ? 1 : 0;

        submit(&pc, UP_DIV(total, 256), 1, 1);

        if (fp16_ != 0) {
            // Barrier: flush reduce's scratch writes for pack's reads.
            scratch_->shaderWriteBarrier(m_cmd_->get());

            int nwords = (total + 1) / 2;
            MatMulPackPC pack_pc{};
            pack_pc.total = total;

            fillPackDescriptorSet(pack_ds_[m_id_]);
            pack_pipeline_->updateDescriptorSets(pack_writes_);
            m_cmd_->bind(*pack_pipeline_, pack_ds_[m_id_]);
            m_cmd_->push_constants(*pack_pipeline_, sizeof(MatMulPackPC),
                                   &pack_pc);
            m_cmd_->dispatch(UP_DIV(nwords, 256), 1, 1);
        }
    }

    void fillPackDescriptorSet(VkDescriptorSet ds) {
        pack_writes_.resize(objs_.size());
        for (size_t i = 0; i < objs_.size(); ++i) {
            pack_writes_[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            pack_writes_[i].dstSet = ds;
            pack_writes_[i].dstBinding = static_cast<uint32_t>(i);
            pack_writes_[i].dstArrayElement = 0;
            pack_writes_[i].descriptorCount = 1;
            pack_writes_[i].descriptorType = DESCRIPTOR_TYPE_STORAGE;
            switch (objs_[i]->getResourceType()) {
            case ResourceType::VK_BUFFER:
                pack_writes_[i].pBufferInfo =
                    std::get<VkDescriptorBufferInfo *>(
                        objs_[i]->getDescriptorInfo());
                break;
            default:
                break;
            }
        }
    }

    std::shared_ptr<VulkanBuffer> scratch_;
    std::unique_ptr<VulkanPipeline> pack_pipeline_;
    VkDescriptorSet pack_ds_[vkop::kInflight] = {nullptr};
    std::vector<VkWriteDescriptorSet> pack_writes_;

    std::vector<int> kernel_shape_ = {0, 0};
    std::vector<int> strides_ = {1, 1};
    std::vector<int> pads_ = {0, 0};
    std::vector<int> dilations_ = {1, 1};
    int groups_ = 1;
    conv2d::ActivationMode activation_ = conv2d::ActivationMode::NONE;
};
// PIMPL façade: buffer SSBO impl when backend_buffer is set, else image.
class Conv2d : public PimplFacade {
  public:
    Conv2d(int fp16, bool backend_buffer) : PimplFacade(OpType::CONV2D) {
        if (backend_buffer) {
            impl_ =
                std::unique_ptr<Operator>(std::make_unique<Conv2dBuffer>(fp16));
        } else {
            impl_ = std::make_unique<Conv2dImage>();
        }
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_OCONV2D_HPP_