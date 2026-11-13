// Copyright 2025 @junka
#ifndef OPS_CONV1D_HPP_
#define OPS_CONV1D_HPP_

#include "Operator.hpp"
#include "ops/BufferBase.hpp"
#include "ops/PimplFacade.hpp"
#include <cstring>

namespace vkop {
namespace ops {

// Causal depthwise Conv1d, the convolution inside Qwen3.5's gated delta net.
//
// HF builds it as nn.Conv1d(groups=conv_dim, kernel_size=K, padding=K-1) and
// slices off the K-1 trailing samples, which is exactly a causal depthwise
// convolution: output[c,t] = sum_k weight[c,k] * x[c, t-(K-1)+k], with the
// samples before t=0 supplied by the conv state instead of zero padding.
//
// Interface (set by the converter, not ONNX's Conv):
//   inputs[0]  x        [C, S] fp32, channel-major
//   inputs[1]  weight   [C, K] fp32 (groups == C, so no channel mixing)
//   inputs[2]  state_in  [C, K-1] fp32 - the previous round's last K-1 inputs,
//                        or zeros on a fresh prefill. This is a graph input, so
//                        the driver preallocates it once.
//   outputs[0] y        [C, S] fp32
//   outputs[1] state_out [C, K-1] fp32 - the tail after this round. It may BE
//                        inputs[2] (present aliases past); the kernel's second
//                        dispatch loads the old tail before storing, so the
//                        aliased case is race-free.
//
// Buffer-only. Data (x, weight, y) follows the tensor dtype: the fp16 builds
// read and write half storage -- either natively (float16_t) or as unpacked
// half2 words -- and the window is evaluated in fp32 either way. The conv STATE
// stays fp32 in every build: it is carried across rounds and rounding it to
// half each step would drift.
class Conv1dBuffer : public BufferFactory {
  public:
    explicit Conv1dBuffer(int fp16)
        : BufferFactory(OpType::CONV1D,
                        std::vector<VkDescriptorType>{
                            DESCRIPTOR_TYPE_STORAGE, // output
                            DESCRIPTOR_TYPE_STORAGE, // input x
                            DESCRIPTOR_TYPE_STORAGE, // weight
                            DESCRIPTOR_TYPE_STORAGE, // state in
                            DESCRIPTOR_TYPE_STORAGE, // state out
                        },
                        sizeof(Conv1dPC), fp16) {}

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("kernel") != attributes.end()) {
            kernel_ = std::stol(attributes.at("kernel"));
        }
    }

  private:
    bool supports_bda_variant() const override { return true; }
    size_t bda_pc_size() const override { return sizeof(Conv1dBdaPC); }

    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        if (inputs.size() < 2 || !inputs[0] || !inputs[1]) {
            throw std::runtime_error("Conv1d: needs x and weight");
        }
        if (outputs.empty() || !outputs[0]) {
            throw std::runtime_error("Conv1d: no output bound");
        }
        const auto x_shape = inputs[0]->getShape();
        if (x_shape.size() != 2) {
            throw std::runtime_error("Conv1d: x must be [C, S]");
        }
        const int channels = x_shape[0];
        const int seq = x_shape[1];
        if (channels <= 0 || seq <= 0) {
            throw std::runtime_error("Conv1d: empty input");
        }
        const auto w_shape = inputs[1]->getShape();
        const int kernel = kernel_ > 0 ? kernel_
                                       : (w_shape.size() == 2 ? w_shape[1] : 0);
        if (kernel <= 0) {
            throw std::runtime_error("Conv1d: kernel size unknown");
        }
        if (w_shape.size() != 2 || w_shape[0] != channels ||
            w_shape[1] != kernel) {
            throw std::runtime_error(
                "Conv1d: weight must be [C, K] matching x's channels");
        }
        const int state_len = kernel - 1;

        // Data buffers follow each tensor's own dtype; the state is always
        // fp32 (see the class comment), so it binds as float unconditionally.
        const std::vector<int> y_shape{channels, seq};
        std::shared_ptr<VulkanBuffer> y_buf, x_buf, w_buf;
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto tag) {
            using T = decltype(tag);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->num_elements() != channels * seq) {
                output->resize(y_shape);
            } else if (output->getShape() != y_shape) {
                output->reshape_view(y_shape);
            }
            y_buf = bind_ssbo<T>(outputs[0], /*is_output=*/true); // 0
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto tag) {
            using T = decltype(tag);
            x_buf = bind_ssbo<T>(inputs[0], /*is_output=*/false); // 1
        });
        dispatch_by_dtype(inputs[1]->dtype(), [&](auto tag) {
            using T = decltype(tag);
            w_buf = bind_ssbo<T>(inputs[1], /*is_output=*/false); // 2
        });

        // The state is optional: a graph with no cached form binds the dummy
        // to all three state slots and the shader reads zeros past t=0.
        const bool has_state =
            state_len > 0 && inputs.size() > 2 && inputs[2];
        std::shared_ptr<VulkanBuffer> state_in_buf, state_out_buf;
        if (has_state) {
            auto st = core::as_tensor<float>(inputs[2]);
            if (!st) {
                throw std::runtime_error("Conv1d: state is not an fp32 tensor");
            }
            const std::vector<int> s_shape{channels, state_len};
            if (st->num_elements() != channels * state_len) {
                st->resize(s_shape);
            }
            state_in_buf = bind_ssbo<float>(inputs[2], /*is_output=*/false);
            // state_out may alias state_in (present == past). Binding it as an
            // output is what tells the storage helper it is written; the
            // kernel's mode-1 dispatch loads the old tail before storing.
            if (outputs.size() > 1 && outputs[1]) {
                auto so = core::as_tensor<float>(outputs[1]);
                if (!so) {
                    throw std::runtime_error(
                        std::string("Conv1d: state_out is not an fp32 tensor "
                                    "(got dtype ") +
                        outputs[1]->dtype().name() + ", kind " +
                        std::to_string(static_cast<int>(
                            outputs[1]->elem_kind())) +
                        "); the graph must declare the state output FLOAT");
                }
                if (so->num_elements() != channels * state_len) {
                    so->resize(s_shape);
                }
                state_out_buf = bind_ssbo<float>(outputs[1], /*is_output=*/true);
            } else {
                state_out_buf = state_in_buf;
            }
        } else {
            objs_.emplace_back(dummy_buffer_); // state in
            objs_.emplace_back(dummy_buffer_); // state out
        }

        Conv1dPC pc{};
        pc.channels = channels;
        pc.seq = seq;
        pc.kernel = kernel;
        pc.state_len = state_len;

        // Dispatch 1: the convolution over every (channel, position). The
        // packed fp16 build handles two elements per thread, so its thread
        // count is half the element count; the native and fp32 builds are one
        // element per thread.
        pc.mode = 0;
        pc.total = channels * seq;
        const bool fp16_data = outputs[0]->dtype() == typeid(uint16_t);
        const int conv_threads = (fp16_data && fp16_packed_)
                                     ? (pc.total + 1) / 2
                                     : pc.total;
        const int conv_groups = UP_DIV(conv_threads, 256);
        dispatch(pc, conv_groups, y_buf, x_buf, w_buf, state_in_buf,
                 state_out_buf);

        // Dispatch 2: roll the state tail forward, one thread per channel.
        if (has_state && state_len > 0) {
            pc.mode = 1;
            pc.total = channels;
            const int st_groups = UP_DIV(channels, 256);
            dispatch(pc, st_groups, y_buf, x_buf, w_buf, state_in_buf,
                     state_out_buf);
        }
    }

    // One dispatch with the four-bound layout. Kept as one helper so the two
    // modes cannot drift apart: the only thing that differs is the scalar
    // prefix (mode / total), and a mismatch there is a silent wrong answer.
    void dispatch(const Conv1dPC &pc_in, int groups,
                  const std::shared_ptr<VulkanBuffer> &y_buf,
                  const std::shared_ptr<VulkanBuffer> &x_buf,
                  const std::shared_ptr<VulkanBuffer> &w_buf,
                  const std::shared_ptr<VulkanBuffer> &state_in_buf,
                  const std::shared_ptr<VulkanBuffer> &state_out_buf) {
        Conv1dPC pc = pc_in;
        // BDA variant: five addresses follow the 32-byte scalar prefix. Taken
        // per dispatch, so a buffer without a device address (host import)
        // keeps the descriptor path.
        if (bda_ready() && y_buf && x_buf && w_buf && state_in_buf &&
            state_out_buf && y_buf->hasDeviceAddress() &&
            x_buf->hasDeviceAddress() && w_buf->hasDeviceAddress() &&
            state_in_buf->hasDeviceAddress() &&
            state_out_buf->hasDeviceAddress()) {
            Conv1dBdaPC bpc{};
            std::memcpy(&bpc, &pc, offsetof(Conv1dBdaPC, out_addr));
            bpc.out_addr = y_buf->getDeviceAddress();
            bpc.x_addr = x_buf->getDeviceAddress();
            bpc.w_addr = w_buf->getDeviceAddress();
            bpc.state_in_addr = state_in_buf->getDeviceAddress();
            bpc.state_out_addr = state_out_buf->getDeviceAddress();
            submit_bda(&bpc, sizeof(bpc), groups, 1, 1);
            return;
        }
        submit(&pc, groups, 1, 1);
    }

    int kernel_ = 0;
};

// PIMPL façade: buffer-only, so both branches build the buffer leaf.
class Conv1d : public PimplFacade {
  public:
    Conv1d(int fp16, bool /*backend_buffer*/) : PimplFacade(OpType::CONV1D) {
        impl_ = std::make_unique<Conv1dBuffer>(fp16);
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_CONV1D_HPP_
