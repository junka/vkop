// Copyright 2026 @junka
#ifndef OPS_PAD_HPP_
#define OPS_PAD_HPP_

#include "core/Tensor.hpp"
#include "ops/BufferBase.hpp"
#include "ops/Operator.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace vkop {
namespace ops {

namespace pad {

// ONNX pad modes, numerically shared with shaders/buffer/pad.comp's
// PAD_MODE_* defines.
enum class PadMode {
    CONSTANT = 0,
    EDGE = 1,
    REFLECT = 2,
    SYMMETRIC = 3,
};

inline const char *pad_mode_name(PadMode m) {
    switch (m) {
    case PadMode::CONSTANT:
        return "constant";
    case PadMode::EDGE:
        return "edge";
    case PadMode::REFLECT:
        return "reflect";
    case PadMode::SYMMETRIC:
        return "symmetric";
    }
    return "?";
}

// Push constant for shaders/buffer/pad.comp, which declares the same fields in
// this order. The dims arrays are LEFT-aligned (fill_dims) like every other
// buffer op, because the shader walks 0..rank-1.
//
// There is deliberately no padEnd array. The end pad on an axis is
// out_len - in_dim - pad_begin, so it is derivable on the shader and the
// block stays at 112 B (rank + 3x IArr8 + mode + the encoded constant word).
// Adding a fourth IArr8 would overflow the 128 B push-constant budget at
// rank 8.
//
// `constant_word` is the already-encoded element bits, not a float: the kernel
// moves uint words, so an fp32 pad writes float bits, an int32 pad writes
// integer bits, and an fp16 pad writes the packed half in the low 16 bits.
// Only the host knows which of those the tensor is.
struct alignas(16) PadPC {
    int rank;
    int inDims[8];
    int outDims[8];
    int padBegin[8];
    int mode;             // pad::PadMode
    uint32_t constant_word;  // element bits used only by mode == CONSTANT
};
static_assert(sizeof(PadPC) == 112, "PadPC size drift (must match pad.comp)");
static_assert(offsetof(PadPC, inDims) == 4, "PadPC layout drift");
static_assert(offsetof(PadPC, outDims) == 36, "PadPC layout drift");
static_assert(offsetof(PadPC, padBegin) == 68, "PadPC layout drift");
static_assert(offsetof(PadPC, mode) == 100, "PadPC layout drift");
static_assert(offsetof(PadPC, constant_word) == 104, "PadPC layout drift");
static_assert(sizeof(PadPC) <= 128, "PadPC push-constant overflow");

// Normalize the ONNX `pads`/`axes` attribute pair into full-rank per-axis
// begin/end vectors. ONNX allows `pads` to name only the axes it carries, in
// which case `axes` (default 0..n-1) says which ones; every unnamed axis gets
// a 0/0. Throws on anything that cannot be applied rather than padding the
// wrong axes.
//
// The negative-pad case (a crop) is legal ONNX for mode=constant and is kept:
// the shader maps the coordinate through unchanged and the region is simply
// dropped. A negative pad with a mirror mode has no ONNX meaning, so it is
// rejected here instead of producing a clamped, wrong-but-plausible tensor.
inline void build_per_axis_pads(int rank, const std::vector<int> &pads,
                                const std::vector<int> &axes, PadMode mode,
                                std::vector<int> &begins,
                                std::vector<int> &ends) {
    begins.assign(static_cast<size_t>(rank), 0);
    ends.assign(static_cast<size_t>(rank), 0);
    if (pads.empty()) {
        return;
    }
    std::vector<int> norm_axes;
    if (axes.empty()) {
        if (static_cast<int>(pads.size()) != 2 * rank) {
            throw std::runtime_error(
                "Pad: pads has " + std::to_string(pads.size()) +
                " entries but the input is rank " + std::to_string(rank) +
                " (expected 2*rank = " + std::to_string(2 * rank) +
                " when no axes attribute is given)");
        }
        for (int i = 0; i < rank; ++i) norm_axes.push_back(i);
    } else {
        if (static_cast<int>(pads.size()) != 2 * static_cast<int>(axes.size())) {
            throw std::runtime_error(
                "Pad: pads has " + std::to_string(pads.size()) +
                " entries but axes has " + std::to_string(axes.size()) +
                " (expected 2*pads per axis)");
        }
        norm_axes = axes;
        for (int &ax : norm_axes) {
            if (ax < 0) ax += rank;
            if (ax < 0 || ax >= rank) {
                throw std::runtime_error(
                    "Pad: axis " + std::to_string(ax) +
                    " is outside the input's " + std::to_string(rank) +
                    "-D range");
            }
        }
    }
    const int n = static_cast<int>(norm_axes.size());
    for (int i = 0; i < n; ++i) {
        const int ax = norm_axes[static_cast<size_t>(i)];
        const int b = pads[static_cast<size_t>(i)];
        const int e = pads[static_cast<size_t>(i + n)];
        if ((b < 0 || e < 0) && mode != PadMode::CONSTANT) {
            throw std::runtime_error(
                std::string("Pad: negative pads are only defined for mode="
                            "constant, got mode=") +
                pad_mode_name(mode));
        }
        begins[static_cast<size_t>(ax)] = b;
        ends[static_cast<size_t>(ax)] = e;
    }
}

} // namespace pad

// SSBO-only op: no image path (the target device has none).
//
// `pads` reach here as an attribute: the converter's
// InitializerMerger::move_input_tensor_to_attr folds Pad's pads initializer
// into a `pads` attribute, so both spellings are accepted — the attribute when
// present, otherwise a 2nd input tensor read on the host.
//
// Single-impl (like Expand/Gather): the class IS the buffer impl, so the
// runtime's `uses_buffer_backend()` check and the fp16_ the archive-key
// resolution reads both land on the object that dispatches.
class PadBuffer : public BufferFactory {
  public:
    bool uses_buffer_backend() const override { return true; }

    explicit PadBuffer(int fp16)
        : BufferFactory(OpType::PAD,
                        {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                        sizeof(pad::PadPC), fp16) {}

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        auto get = [&](const char *key) -> const std::string * {
            auto it = attributes.find(key);
            return it == attributes.end() ? nullptr : &it->second;
        };
        if (const auto *s = get("mode")) {
            if (*s == "constant") {
                mode_ = pad::PadMode::CONSTANT;
            } else if (*s == "edge") {
                mode_ = pad::PadMode::EDGE;
            } else if (*s == "reflect") {
                mode_ = pad::PadMode::REFLECT;
            } else if (*s == "symmetric") {
                mode_ = pad::PadMode::SYMMETRIC;
            } else {
                throw std::runtime_error("Pad: unknown mode '" + *s +
                                         "' (expected constant, edge, reflect "
                                         "or symmetric)");
            }
        }
        if (const auto *s = get("pads")) {
            if (!s->empty()) {
                pads_ = parse_attr_list<int>(*s);
                pads_set_ = true;
            }
        }
        if (const auto *s = get("axes")) {
            if (!s->empty()) {
                axes_ = parse_attr_list<int>(*s);
            }
        }
        if (const auto *s = get("constant_value")) {
            try {
                constant_value_ = std::stof(*s);
            } catch (const std::exception &) {
                throw std::runtime_error("Pad: constant_value '" + *s +
                                         "' is not a number");
            }
        }
    }

  private:
    // Read the pads vector from the 2nd graph input when no `pads` attribute
    // was recorded. Handles both int64 (ONNX canonical) and the int32 form the
    // runtime's initializer path can produce.
    std::vector<int> read_int_input(
        const std::shared_ptr<core::ITensor> &input) {
        std::vector<int> out;
        dispatch_by_dtype(input->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto t = core::as_tensor<T>(input);
            if (!t) return;
            t->copyToCPU(m_cmdpool_);
            out.reserve(t->data().size());
            for (const auto &v : t->data()) {
                out.push_back(static_cast<int>(v));
            }
        });
        return out;
    }

    // Read a scalar float from an input tensor (ONNX opset 18 Pad's
    // constant_value input). The tensor's own dtype decides how its element
    // reads: a uint16_t storage is fp16, not an integer 16-bit value.
    float read_scalar_float(const std::shared_ptr<core::ITensor> &input) {
        float v = 0.0F;
        const bool is_16 = input->dtype() == typeid(uint16_t);
        // A uint16_t container is fp16 unless its elem_kind says bf16.
        const bool is_bf16 =
            input->elem_kind() == core::ElemKind::kBFloat16;
        dispatch_by_dtype(input->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto t = core::as_tensor<T>(input);
            if (!t) return;
            // copyToCPU before reading data(): a GPU-resident tensor has no
            // host vector until the readback fills one.
            t->copyToCPU(m_cmdpool_);
            if (t->data().empty()) return;
            const auto h = static_cast<uint16_t>(t->data()[0]);
            v = is_bf16  ? core::bf16_to_float(h)
                : is_16  ? core::ITensor::fp16_to_fp32(h)
                         : static_cast<float>(t->data()[0]);
        });
        return v;
    }

    // Encode the constant pad value into the exact element bits the kernel
    // writes. Pad is a word mover: an fp32 output word is float bits, an int32
    // word is integer bits, and an fp16 output word holds two halves, so the
    // constant goes in the low 16 bits (in_half masks it there).
    // A bf16 output (the _bf16 module, fp16_ == 2) takes the bf16 bits of the
    // constant: the kernel only moves words, so this is the one place the
    // 16-bit format matters.
    static uint32_t encode_constant(float value, const std::type_info &dtype,
                                    bool fp16, bool bf16 = false) {
        if (bf16) {
            return static_cast<uint32_t>(core::float_to_bf16(value));
        }
        if (fp16 || dtype == typeid(uint16_t)) {
            return static_cast<uint32_t>(
                core::ITensor::fp32_to_fp16(value));
        }
        if (dtype == typeid(int)) {
            return static_cast<uint32_t>(static_cast<int32_t>(value));
        }
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }

    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        // The pads are resolved on the host. With the `pads` attribute (what
        // the converter emits) that is attribute parsing only, and the padding
        // is a plain GPU gather — a segment may replay it. Reading pads out of
        // an input tensor is a real per-round readback, so report that.
        set_did_host_work_last_round(!pads_set_);

        auto inshape = inputs[0]->getShape();
        const int rank = static_cast<int>(inshape.size());
        if (rank == 0) {
            throw std::runtime_error("Pad: rank-0 input has no axis to pad");
        }
        if (rank > 8) {
            throw std::runtime_error(
                "Pad (buffer backend) supports at most rank 8, got rank " +
                std::to_string(rank));
        }

        // Fail closed on a <= 1-byte payload: this mover has no byte build (see
        // core::require_word_movable_elem), so an int8/bool/fp8 pad would copy
        // 4-per-word and mis-pad silently.
        core::require_word_movable_elem(inputs[0]->elem_kind(), "Pad");
        // This kernel indexes one element per uint word (fp32/int32) or two
        // (fp16). A 64-bit element is two words, so the shader's linear index
        // would address the wrong half — refuse rather than pad garbage.
        if (inputs[0]->dtype() == typeid(int64_t) ||
            inputs[0]->dtype() == typeid(uint64_t)) {
            throw std::runtime_error(
                "Padding an int64 tensor is not supported on the buffer "
                "backend (pad.comp indexes 32-bit words)");
        }

        // The shader build is fixed at construction, so the live dtype has to
        // match it: packed half2 read as float bits (or the reverse) is silent
        // garbage.
        const bool data_fp16 = inputs[0]->dtype() == typeid(uint16_t);
        if (data_fp16 != (fp16_ != 0)) {
            throw std::runtime_error(
                std::string("Pad: the buffer kernel was built for ") +
                (fp16_ != 0 ? "fp16" : "fp32") + " but the input tensor is " +
                (data_fp16 ? "fp16" : "fp32"));
        }

        std::vector<int> pads = pads_;
        if (!pads_set_) {
            if (inputs.size() < 2 || !inputs[1]) {
                throw std::runtime_error(
                    "Pad: no `pads` attribute and no pads input to read");
            }
            pads = read_int_input(inputs[1]);
        }
        // Opset-18 optional trailing inputs: ONNX Pad's input order is
        // (data, pads, constant_value, axes). The attribute spelling wins when
        // both are present, matching the converter's move_input_tensor_to_attr
        // fold.
        std::vector<int> axes = axes_;
        if (axes.empty() && inputs.size() > 3 && inputs[3]) {
            axes = read_int_input(inputs[3]);
        }
        float constant_value = constant_value_;
        if (inputs.size() > 2 && inputs[2]) {
            constant_value = read_scalar_float(inputs[2]);
        }

        std::vector<int> begins, ends;
        pad::build_per_axis_pads(rank, pads, axes, mode_, begins, ends);

        // Build the output shape from the normalized per-axis pads. ONNX Pad
        // forbids REFLECT padding that reaches the input's size (it would need
        // to mirror an element that does not exist); reject it rather than let
        // the shader clamp to a silently wrong tensor. `edge` and `symmetric`
        // are well defined for any pad size (the boundary element repeats), so
        // they only clamp.
        std::vector<int> out_shape(static_cast<size_t>(rank), 0);
        for (int d = 0; d < rank; ++d) {
            const int in_d = inshape[static_cast<size_t>(d)];
            const int b = begins[static_cast<size_t>(d)];
            const int e = ends[static_cast<size_t>(d)];
            const int out_d = in_d + b + e;
            if (out_d < 0) {
                throw std::runtime_error(
                    "Pad: axis " + std::to_string(d) + " of size " +
                    std::to_string(in_d) + " cannot be cropped by " +
                    std::to_string(b) + "+" + std::to_string(e));
            }
            if (mode_ == pad::PadMode::REFLECT && (b >= in_d || e >= in_d)) {
                throw std::runtime_error(
                    std::string("Pad: mode=") + pad::pad_mode_name(mode_) +
                    " cannot pad axis " + std::to_string(d) + " of size " +
                    std::to_string(in_d) + " by (" + std::to_string(b) + ", " +
                    std::to_string(e) + "): the reflect pad must be smaller "
                    "than the input (ONNX requirement)");
            }
            out_shape[static_cast<size_t>(d)] = out_d;
        }

        pad::PadPC pc{};
        pc.rank = rank;
        fill_dims(pc.inDims, inshape);
        fill_dims(pc.outDims, out_shape);
        for (int d = 0; d < 8; ++d) {
            pc.padBegin[d] = (d < rank) ? begins[static_cast<size_t>(d)] : 0;
        }
        pc.mode = static_cast<int>(mode_);
        pc.constant_word = encode_constant(
            constant_value, inputs[0]->dtype(), fp16_ != 0,
            inputs[0]->elem_kind() == core::ElemKind::kBFloat16);

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->num_elements() != total_elems(out_shape)) {
                output->resize(out_shape);
            }
            bind_ssbo<T>(outputs[0], /*is_output=*/true);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            bind_ssbo<T>(inputs[0], /*is_output=*/false);
        });

        const int total = total_elems(out_shape);
        // fp16 packs two halfs per uint word: one thread per output WORD.
        const int nthreads = (fp16_ != 0) ? (total + 1) / 2 : total;
        submit(&pc, UP_DIV(nthreads, 256), 1, 1);
    }

    pad::PadMode mode_ = pad::PadMode::CONSTANT;
    std::vector<int> pads_;
    std::vector<int> axes_;
    bool pads_set_ = false;
    float constant_value_ = 0.0F;
};

// Public op name. There is no image twin to select between, so this is a
// typedef rather than a PIMPL façade: PadBuffer IS the implementation and the
// factory constructs it directly. (A façade would also have to forward
// uses_buffer_backend()/fp16_, and getting one of those wrong is exactly the
// silent "asked for the fp32 shader" class of bug.)
using Pad = PadBuffer;

} // namespace ops
} // namespace vkop
#endif // OPS_PAD_HPP_
