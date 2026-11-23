// Copyright 2026 @junka
#ifndef OPS_CONVTRANSPOSE_HPP_
#define OPS_CONVTRANSPOSE_HPP_

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/Tensor.hpp"
#include "ops/BufferBase.hpp"
#include "ops/Operator.hpp"
#include "ops/PimplFacade.hpp"

namespace vkop {
namespace ops {
namespace convtranspose {

// Push constant block for shaders/buffer/convtranspose.comp. Field order and
// types must match the GLSL block exactly (it is declared field-for-field).
//
// 24 ints = 96 bytes, well inside the 128-byte push-constant limit.
//
// The D fields carry a 1-D or 2-D convolution the same way Conv2dBuffer does:
// a 1-D ConvTranspose ([N,C,L] / [C,Cout/g,K]) is ID = IH = KD = KH = 1 with
// unit strides/pads/dilations there, and a 2-D one is ID = KD = 1, so the
// shader's outer loops run once and the same kernel serves all three ranks.
struct alignas(16) ConvTransposePC {
    int N, IC, ID, IH, IW; // input  [N, IC, ID, IH, IW]
    int OC, OD, OH, OW;    // output [N, OC, OD, OH, OW]
    int KD, KH, KW;        // kernel (from the weight, or kernel_shape)
    int stride_d, stride_h, stride_w;
    int pad_d, pad_h, pad_w; // BEGIN pads only; the read side never needs ends
    int dil_d, dil_h, dil_w;
    int groups;
    int has_bias;
    int ocg; // OC / groups — output channels per group
    int icg; // IC / groups — input  channels per group
};
// 25 ints = 100 B of fields, rounded to 112 by alignas(16) (Conv2dBufferPC
// carries the same tail padding). The push-constant range covers the fields;
// the padding bytes are never read by the shader.
static_assert(sizeof(ConvTransposePC) == 112,
              "ConvTransposePC is 25 ints + alignas(16) tail padding");
static_assert(offsetof(ConvTransposePC, ocg) == 92,
              "ConvTransposePC field order drifted from the GLSL block");
static_assert(offsetof(ConvTransposePC, icg) == 96,
              "ConvTransposePC field order drifted from the GLSL block");

} // namespace convtranspose

// ONNX ConvTranspose, buffer (SSBO) backend only.
//
// The graph's ONNX node is `ConvTranspose`; the weight is
//   [IC, OC/group, kD, kH, kW]     (note the order — Conv's is the transpose
//                                   of this, which is the usual source of a
//                                   silently wrong result)
// and the op maps each input position to the outputs its kernel taps reach,
// which the kernel evaluates output-side (see the shader header comment).
//
// Semantics implemented: `group`, `strides`, `pads`, `dilations`,
// `output_padding`, `output_shape`, `kernel_shape`, and the `auto_pad`
// SAME_UPPER / SAME_LOWER / VALID forms (SAME is resolved into explicit pads
// and an exact output size here, because ONNX defines the SAME output size as
// `in * stride` and the generic formula does not land on it).
//
// fp32 and fp16 both run one thread per output element; the fp16 build stores
// `float16_t` directly, so no pack pass is needed (unlike Conv2d's half2
// convention) and accumulation stays fp32.
class ConvTransposeBuffer : public BufferFactory {
  public:
    explicit ConvTransposeBuffer(int fp16)
        : BufferFactory(OpType::CONVTRANSPOSE,
                        std::vector<VkDescriptorType>{
                            DESCRIPTOR_TYPE_STORAGE, // output
                            DESCRIPTOR_TYPE_STORAGE, // input x
                            DESCRIPTOR_TYPE_STORAGE, // weight
                            DESCRIPTOR_TYPE_STORAGE, // bias (or placeholder)
                        },
                        sizeof(convtranspose::ConvTransposePC), fp16) {}

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        auto has = [&attributes](const char *k) {
            return attributes.find(k) != attributes.end();
        };
        if (has("strides")) {
            strides_ = parseAttrList(attributes.at("strides"));
        }
        if (has("pads")) {
            pads_ = parseAttrList(attributes.at("pads"));
            pads_set_ = true;
        }
        if (has("dilations")) {
            dilations_ = parseAttrList(attributes.at("dilations"));
        }
        if (has("output_padding")) {
            output_padding_ = parseAttrList(attributes.at("output_padding"));
        }
        if (has("kernel_shape")) {
            kernel_shape_ = parseAttrList(attributes.at("kernel_shape"));
        }
        if (has("output_shape")) {
            // ONNX forbids combining `output_shape` with `pads`; that is
            // checked in execute() (this map is unordered, so the two flags
            // can arrive in either order).
            output_shape_ = parseAttrList(attributes.at("output_shape"));
            output_shape_set_ = true;
        }
        if (has("group")) {
            groups_ = static_cast<int>(std::stol(attributes.at("group")));
        }
        if (has("auto_pad")) {
            const std::string ap = attributes.at("auto_pad");
            if (ap == "NOTSET") {
                // explicit pads (or zero) already in hand
            } else if (ap == "VALID") {
                auto_pad_ = AutoPad::VALID;
            } else if (ap == "SAME_UPPER" || ap == "SAME_LOWER") {
                auto_pad_ = ap == "SAME_UPPER" ? AutoPad::SAME_UPPER
                                               : AutoPad::SAME_LOWER;
            } else {
                throw std::invalid_argument(
                    "ConvTranspose: unsupported auto_pad '" + ap + "'");
            }
        }
    }

  private:
    enum class AutoPad { NOTSET, VALID, SAME_UPPER, SAME_LOWER };

    // ONNX list attributes arrive as "[1, 2]" from the model loader and as a
    // bare "2" from a hand-written attribute map (the Conv2d tests do both),
    // so accept either and never return an empty vector for a bare scalar.
    std::vector<int> parseAttrList(const std::string &raw) {
        if (raw.empty()) {
            return {};
        }
        if (raw.front() == '[') {
            // Same "[a, b, c]" form Operator::parse_attr_list handles.
            auto v = parse_attr_list<int>(raw);
            return v;
        }
        return {static_cast<int>(std::stol(raw))};
    }

    // nth element of a list, with ONNX's "the last value repeats" rule for
    // lists shorter than the rank (a single stride applies to every axis).
    static int pick(const std::vector<int> &v, size_t i, int dflt) {
        if (v.empty()) {
            return dflt;
        }
        return v[std::min(i, v.size() - 1)];
    }

    // Number of spatial axes; the kernel supports 1-D and 2-D only.
    static constexpr int kMaxSpatial = 2;

    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        if (inputs.size() < 2 || !inputs[0] || !inputs[1]) {
            throw std::runtime_error("ConvTranspose: needs input and weight");
        }
        if (outputs.empty() || !outputs[0]) {
            throw std::runtime_error("ConvTranspose: no output bound");
        }
        // This kernel decodes float storage only (there is no int8/fp8
        // weight-only route here), so a byte payload would be read as half
        // exponents and produce plausible garbage.
        core::require_float_elem(inputs[0]->elem_kind(), "ConvTranspose",
                                 "input 0");
        core::require_float_elem(inputs[1]->elem_kind(), "ConvTranspose",
                                 "input 1");

        const std::vector<int> in_shape = inputs[0]->getShape();
        const std::vector<int> w_shape = inputs[1]->getShape();
        const int rank = static_cast<int>(in_shape.size());

        // 1-D ([N,C,L]) and 2-D ([N,C,H,W]) both run through the same 3-D
        // kernel (the extra spatial axes are 1). 3-D input is refused by name
        // rather than half-tested.
        if (rank != 3 && rank != 4) {
            throw std::runtime_error(
                "ConvTranspose: input must be 3-D [N,C,L] or 4-D [N,C,H,W], "
                "got rank " + std::to_string(rank));
        }
        if (static_cast<int>(w_shape.size()) != rank) {
            throw std::runtime_error(
                "ConvTranspose: weight rank " +
                std::to_string(w_shape.size()) + " does not match input rank " +
                std::to_string(rank));
        }
        const int spatial = rank - 2; // 1 or 2
        // ONNX does NOT forbid `output_shape` together with `pads`; it says
        // "If output_shape is specified pads values are ignored" and the pads
        // are then derived from the requested size. Rejecting the combination
        // would refuse graphs that are legal, so it is handled in the
        // output_shape branch below instead.
        if (output_shape_set_) {
            pads_set_ = false;
        }

        const int N = in_shape[0];
        const int IC = in_shape[1];
        // Spatial input sizes in ONNX attribute order: axis 0 is H (or L for
        // 1-D), axis 1 is W. Every spatial attribute list below is indexed
        // this way, so the pairing stays consistent.
        const int in_sp[kMaxSpatial] = {in_shape[2],
                                        rank == 4 ? in_shape[3] : 1};

        // Weight [IC, OC/group, k...]. The first axis is the INPUT channels —
        // the pair of axes ConvTranspose has swapped relative to Conv, which
        // is exactly what this check catches when a Conv weight is fed here.
        const int w_ic = w_shape[0];
        const int w_ocg = w_shape[1];
        if (w_ic != IC) {
            throw std::runtime_error(
                "ConvTranspose: weight's first dim " + std::to_string(w_ic) +
                " must equal the input channel count " +
                std::to_string(IC) +
                " (weight is [IC, OC/group, ...], unlike Conv's "
                "[OC, IC/group, ...])");
        }
        if (groups_ <= 0) {
            throw std::runtime_error(
                "ConvTranspose: group must be >= 1, got " +
                std::to_string(groups_));
        }
        if (IC % groups_ != 0) {
            throw std::runtime_error(
                "ConvTranspose: input channels " + std::to_string(IC) +
                " are not divisible by group " + std::to_string(groups_));
        }
        // IC/group is the number of INPUT channels per group (what the kernel
        // strides the weight's first axis by); the weight's second axis is
        // OUT/group, which is free — ONNX only ties it to the output count,
        // `OC = dim1 * group`. (Requiring dim1 == IC/group here was wrong:
        // it rejects every ordinary group-1 ConvTranspose whose input and
        // output channel counts differ.)
        const int ICG = IC / groups_;
        const int OCG = w_ocg;
        const int OC = OCG * groups_; // ONNX: num_output_channels

        const int K[kMaxSpatial] = {w_shape[2], rank == 4 ? w_shape[3] : 1};
        // kernel_shape is optional; when present ONNX requires it to agree
        // with the weight, and a disagreement means a pass reshaped the
        // weight without updating the attribute.
        if (!kernel_shape_.empty()) {
            for (int i = 0; i < spatial; ++i) {
                const int declared = pick(kernel_shape_, i, 0);
                if (declared > 0 && declared != K[i]) {
                    throw std::runtime_error(
                        "ConvTranspose: kernel_shape[" + std::to_string(i) +
                        "]=" + std::to_string(declared) +
                        " disagrees with the weight's kernel dim " +
                        std::to_string(K[i]));
                }
            }
        }

        // ONNX spatial attribute lists follow the tensor's axis order (H then
        // W for a 2-D convolution, L for a 1-D one), so `pick(i)` reads the
        // i-th spatial axis directly and a short list repeats its last value.
        int stride[kMaxSpatial];
        int dil[kMaxSpatial];
        int pad_begin[kMaxSpatial];
        int pad_end[kMaxSpatial];
        int out_pad[kMaxSpatial];
        for (int i = 0; i < spatial; ++i) {
            stride[i] = pick(strides_, i, 1);
            dil[i] = pick(dilations_, i, 1);
            // ONNX pads are [x1_begin, ..., xN_begin, x1_end, ..., xN_end].
            pad_begin[i] = pick(pads_, i, 0);
            pad_end[i] = pick(pads_, spatial + i, 0);
            out_pad[i] = pick(output_padding_, i, 0);
            if (stride[i] <= 0 || dil[i] <= 0) {
                throw std::runtime_error(
                    "ConvTranspose: stride and dilation must be >= 1");
            }
            if (out_pad[i] < 0 || out_pad[i] >= stride[i]) {
                throw std::runtime_error(
                    "ConvTranspose: output_padding " +
                    std::to_string(out_pad[i]) + " must be in [0, stride " +
                    std::to_string(stride[i]) + ")");
            }
        }

        int out_sp[kMaxSpatial];
        // out = stride*(in-1) + output_padding + dilation*(k-1) + 1
        //       - pad_begin - pad_end
        for (int i = 0; i < spatial; ++i) {
            out_sp[i] = stride[i] * (in_sp[i] - 1) + out_pad[i] +
                        dil[i] * (K[i] - 1) + 1 - pad_begin[i] - pad_end[i];
        }

        if (auto_pad_ == AutoPad::SAME_UPPER ||
            auto_pad_ == AutoPad::SAME_LOWER) {
            // ONNX pins the SAME size to in*stride (the formula above does not
            // land on it), and the pads absorb the difference. Resolve to
            // explicit pads so the kernel keeps a single code path.
            for (int i = 0; i < spatial; ++i) {
                out_sp[i] = in_sp[i] * stride[i];
                const int total = stride[i] * (in_sp[i] - 1) + out_pad[i] +
                                  dil[i] * (K[i] - 1) + 1 - out_sp[i];
                if (total < 0) {
                    throw std::runtime_error(
                        "ConvTranspose: SAME auto_pad needs negative padding "
                        "for kernel " + std::to_string(K[i]) + " stride " +
                        std::to_string(stride[i]));
                }
                pad_begin[i] = auto_pad_ == AutoPad::SAME_UPPER
                                   ? total / 2
                                   : total - total / 2;
                pad_end[i] = total - pad_begin[i];
            }
        } else if (output_shape_set_) {
            // `output_shape` fixes the spatial size and the pads are derived
            // from it; an explicit `pads` is ignored (ONNX: "If output_shape
            // is specified pads values are ignored"). The split follows the
            // spec's else branch --
            //   total = stride*(in-1) + output_padding + dil*(k-1) + 1 - out
            //   pad_begin = total - total/2,  pad_end = total/2
            // -- which is also what onnxruntime does: for the 13 -> 12 case it
            // drops the FIRST element (pads [0,0] and [1,0] give identical
            // output). Splitting it the other way would crop from the far side
            // and shift every value.
            for (int i = 0; i < spatial; ++i) {
                const int want = pick(output_shape_, i, -1);
                if (want <= 0) {
                    continue;
                }
                out_sp[i] = want;
                const int total = stride[i] * (in_sp[i] - 1) + out_pad[i] +
                                  dil[i] * (K[i] - 1) + 1 - want;
                if (total < 0) {
                    throw std::runtime_error(
                        "ConvTranspose: output_shape " + std::to_string(want) +
                        " is larger than the natural size " +
                        std::to_string(stride[i] * (in_sp[i] - 1) + out_pad[i] +
                                       dil[i] * (K[i] - 1) + 1) +
                        "; that needs negative padding");
                }
                pad_begin[i] = total - total / 2;
                pad_end[i] = total / 2;
            }
        }
        for (int i = 0; i < spatial; ++i) {
            if (out_sp[i] <= 0) {
                throw std::runtime_error(
                    "ConvTranspose: computed output size " +
                    std::to_string(out_sp[i]) + " on spatial axis " +
                    std::to_string(i) + " is not positive");
            }
        }

        const bool has_bias = inputs.size() > 2 && inputs[2] != nullptr;
        if (has_bias) {
            const std::vector<int> b_shape = inputs[2]->getShape();
            if (b_shape.size() != 1 || b_shape[0] != OC) {
                throw std::runtime_error(
                    "ConvTranspose: bias must be 1-D with OC=" +
                    std::to_string(OC) + " entries");
            }
            core::require_float_elem(inputs[2]->elem_kind(), "ConvTranspose",
                                     "input 2 (bias)");
        }

        // Output [N, OC, (H), (W)] — kept as an explicit list per rank so a
        // refactor cannot silently swap the two spatial axes.
        std::vector<int> out_shape =
            rank == 4 ? std::vector<int>{N, OC, out_sp[0], out_sp[1]}
                      : std::vector<int>{N, OC, out_sp[0]};
        const int total = total_elems(out_shape);

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto tag) {
            using T = decltype(tag);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->num_elements() != total) {
                output->resize(out_shape);
            }
            bind_ssbo<T>(outputs[0], /*is_output=*/true);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto tag) {
            using T = decltype(tag);
            bind_ssbo<T>(inputs[0], /*is_output=*/false);
        });
        dispatch_by_dtype(inputs[1]->dtype(), [&](auto tag) {
            using T = decltype(tag);
            bind_ssbo<T>(inputs[1], /*is_output=*/false);
        });
        if (has_bias) {
            dispatch_by_dtype(inputs[2]->dtype(), [&](auto tag) {
                using T = decltype(tag);
                bind_ssbo<T>(inputs[2], /*is_output=*/false);
            });
        } else {
            // Binding 3 is declared in the shader whether or not a bias is
            // bound, so a placeholder keeps the descriptor set complete; the
            // shader never dereferences it when has_bias == 0.
            objs_.emplace_back(dummy_buffer_);
        }

        // The kernel is 3-D; a 1-D/2-D conv compresses into it by putting the
        // spatial axes W (always) and H (2-D only) with the D axis at 1.
        convtranspose::ConvTransposePC pc{};
        pc.N = N;
        pc.IC = IC;
        pc.ID = 1;
        pc.IH = rank == 4 ? in_sp[0] : 1;
        pc.IW = rank == 4 ? in_sp[1] : in_sp[0];
        pc.OC = OC;
        pc.OD = 1;
        pc.OH = rank == 4 ? out_sp[0] : 1;
        pc.OW = rank == 4 ? out_sp[1] : out_sp[0];
        pc.KD = 1;
        pc.KH = rank == 4 ? K[0] : 1;
        pc.KW = rank == 4 ? K[1] : K[0];
        pc.stride_d = 1;
        pc.stride_h = rank == 4 ? stride[0] : 1;
        pc.stride_w = rank == 4 ? stride[1] : stride[0];
        pc.pad_d = 0;
        pc.pad_h = rank == 4 ? pad_begin[0] : 0;
        pc.pad_w = rank == 4 ? pad_begin[1] : pad_begin[0];
        pc.dil_d = 1;
        pc.dil_h = rank == 4 ? dil[0] : 1;
        pc.dil_w = rank == 4 ? dil[1] : dil[0];
        pc.groups = groups_;
        pc.has_bias = has_bias ? 1 : 0;
        pc.ocg = OCG;
        pc.icg = ICG;

        submit(&pc, UP_DIV(total, 256), 1, 1);
    }

    std::vector<int> strides_;
    std::vector<int> pads_;
    std::vector<int> dilations_;
    std::vector<int> output_padding_;
    std::vector<int> kernel_shape_;
    std::vector<int> output_shape_;
    bool pads_set_ = false;
    bool output_shape_set_ = false;
    int groups_ = 1;
    AutoPad auto_pad_ = AutoPad::NOTSET;
};

// PIMPL façade. ConvTranspose has no image twin (there is no image
// transposed-convolution kernel), so the image request is refused by name
// instead of silently building the SSBO op.
class ConvTranspose : public PimplFacade {
  public:
    ConvTranspose(int fp16, bool backend_buffer) : PimplFacade(OpType::CONVTRANSPOSE) {
        if (!backend_buffer) {
            throw std::runtime_error(
                "ConvTranspose is buffer-backend only (no image impl).");
        }
        impl_ = std::make_unique<ConvTransposeBuffer>(fp16);
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_CONVTRANSPOSE_HPP_
