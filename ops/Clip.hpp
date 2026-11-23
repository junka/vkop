// Copyright 2026 @junka
#ifndef OPS_CLIP_HPP_
#define OPS_CLIP_HPP_

#include "core/Tensor.hpp"
#include "ops/BufferBase.hpp"
#include "ops/Operator.hpp"
#include "ops/PimplFacade.hpp"

#include <string>

namespace vkop {
namespace ops {

// Push constant for shaders/buffer/clip.comp. Field order must match the
// shader's block exactly (24 words = 96 B, which is also what alignas(16)
// gives the struct).
//
// The two "has" flags exist because ONNX Clip may leave either bound ABSENT
// and an absent bound is an INFINITE one. A sentinel value cannot express
// that: a legitimate bound may itself be +-inf or +-FLT_MAX, and treating
// those as "unbounded" would silently turn a real clamp into an identity.
// Keeping each value and its presence apart is also what makes "only min" and
// "only max" -- the two forms torch.clamp(x, min=..) / clamp(x, max=..) export
// today -- expressible without guessing a value for the other side.
struct alignas(16) ClipPC {
    int rank;       // tensor rank
    int outDims[8]; // output shape (== input shape: Clip is elementwise)
    int in0Dims[8]; // input shape
    int total;      // element count
    float vmin;     // lower bound, meaningful only when has_min != 0
    float vmax;     // upper bound, meaningful only when has_max != 0
    int has_min;    // 1 = lower bound present
    int has_max;    // 1 = upper bound present
    int _pad0;
    int _pad1;
};
static_assert(sizeof(ClipPC) == 96, "ClipPC layout drift (shader block is 96)");
static_assert(sizeof(ClipPC) <= 128, "ClipPC PC overflow");

// Buffer (SSBO) implementation: y = clamp(x, min, max), either bound optional.
//
// Deliberately NOT derived from BufferUnaryFactory. That base fixes the push
// constant to UnaryElemPC (shape + element count only) and its execute()
// fills exactly those fields; Clip needs two floats and two presence flags on
// top. Extending UnaryElemPC would change the layout every other unary shader
// reads, and a second layout is the smaller, safer change. Everything else of
// the unary convention is kept identical: one thread per element (fp32), one
// per uint word (fp16 packed half2), 256 threads per group.
class ClipBuffer : public BufferFactory {
  public:
    explicit ClipBuffer(int fp16)
        : BufferFactory(OpType::CLIP,
                        {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                        sizeof(ClipPC), fp16) {}

    // Two ONNX spellings carry the bounds and both are accepted:
    //   * opset <= 6: the `min` / `max` attributes;
    //   * opset >= 11: optional `min` / `max` input tensors -- what
    //     torch.clamp exports, and with a single bound the exporter leaves the
    //     missing side as an EMPTY input name, so inputs[1] or inputs[2]
    //     arrives null (the runtime maps that to a null shared_ptr).
    // Attributes are parsed here; input bounds are read in execute(), where the
    // tensors are reachable and readable.
    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("min") != attributes.end()) {
            vmin_ = std::stof(attributes.at("min"));
            has_min_attr_ = true;
        }
        if (attributes.find("max") != attributes.end()) {
            vmax_ = std::stof(attributes.at("max"));
            has_max_attr_ = true;
        }
    }

    // Reading a bound INPUT is host work a replayed segment would skip. It is
    // only ever done once per distinct read-only leaf (an initializer or
    // Constant, whose bytes cannot change for the model's lifetime), so the
    // steady-state answer is "no host work" and the segment stays replayable.
    // A non-leaf bound is re-read every round and reported as host work.
    bool did_host_work_last_round() const override {
        return host_work_last_round_;
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        float vmin = vmin_;
        float vmax = vmax_;
        bool has_min = has_min_attr_;
        bool has_max = has_max_attr_;

        // Input bounds override the attributes (an ONNX graph carries one form
        // or the other, never both). A null entry is the ABSENT side: the
        // exporter writes "" for it, and absence means unbounded there.
        bool host_work = false;
        if (inputs.size() > 1 && inputs[1]) {
            has_min = true;
            if (!bound_cached(inputs[1], min_src_, min_valid_, min_cached_)) {
                min_cached_ = read_bound(inputs[1]);
                min_valid_ = inputs[1]->is_read_only_leaf();
                min_src_ = min_valid_ ? inputs[1].get() : nullptr;
                host_work = true;
            }
            vmin = min_cached_;
        }
        if (inputs.size() > 2 && inputs[2]) {
            has_max = true;
            if (!bound_cached(inputs[2], max_src_, max_valid_, max_cached_)) {
                max_cached_ = read_bound(inputs[2]);
                max_valid_ = inputs[2]->is_read_only_leaf();
                max_src_ = max_valid_ ? inputs[2].get() : nullptr;
                host_work = true;
            }
            vmax = max_cached_;
        }
        host_work_last_round_ = host_work;

        auto shape = inputs[0]->getShape();
        const int total = total_elems(shape);

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->num_elements() != total) {
                output->resize(shape);
            }
            bind_ssbo<T>(outputs[0], /*is_output=*/true);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            bind_ssbo<T>(inputs[0], /*is_output=*/false);
        });

        ClipPC pc{};
        pc.rank = static_cast<int>(shape.size());
        fill_dims(pc.outDims, shape);
        fill_dims(pc.in0Dims, shape);
        pc.total = total;
        pc.vmin = vmin;
        pc.vmax = vmax;
        pc.has_min = has_min ? 1 : 0;
        pc.has_max = has_max ? 1 : 0;

        const int nthreads = (fp16_ != 0) ? (total + 1) / 2 : total;
        submit(&pc, UP_DIV(nthreads, 256), 1, 1);
    }

    // Read a scalar bound. ONNX says min/max are scalars; the first element is
    // used, and a zero-element tensor is refused rather than silently clamping
    // against a value nobody supplied.
    float read_bound(const std::shared_ptr<core::ITensor> &t) {
        if (t->dtype() == typeid(float)) {
            auto f = core::as_tensor<float>(t);
            f->copyToCPU(m_cmdpool_);
            if (f->num_elements() >= 1) return (*f)[0];
        } else if (t->dtype() == typeid(uint16_t)) {
            auto h = core::as_tensor<uint16_t>(t);
            h->copyToCPU(m_cmdpool_);
            if (h->num_elements() >= 1) {
                return core::ITensor::fp16_to_fp32((*h)[0]);
            }
        } else if (t->dtype() == typeid(double)) {
            auto d = core::as_tensor<double>(t);
            d->copyToCPU(m_cmdpool_);
            if (d->num_elements() >= 1) return static_cast<float>((*d)[0]);
        } else if (t->dtype() == typeid(int64_t)) {
            auto i = core::as_tensor<int64_t>(t);
            i->copyToCPU(m_cmdpool_);
            if (i->num_elements() >= 1) return static_cast<float>((*i)[0]);
        }
        throw std::runtime_error(
            "vkop: Clip bound must be a non-empty float/fp16/double/int64 "
            "scalar, got a tensor of dtype " + std::string(t->dtype().name()));
    }

    // Leaf cache: a read-only leaf's bytes are fixed for the model's lifetime,
    // so re-reading it every round would be a GPU->CPU stall for a value that
    // cannot change -- and host work that would keep the segment out of replay
    // forever. Pointer identity is enough: a recycled tensor gets a new
    // object, and a leaf is never rewritten in place.
    static bool bound_cached(const std::shared_ptr<core::ITensor> &t,
                             const core::ITensor *src, bool valid,
                             float value) {
        (void)value;
        return valid && src == t.get();
    }

    // Bounds from setAttribute.
    float vmin_ = 0.0F;
    float vmax_ = 0.0F;
    bool has_min_attr_ = false;
    bool has_max_attr_ = false;
    // Leaf cache: identity + last value of each bound input.
    const core::ITensor *min_src_ = nullptr;
    const core::ITensor *max_src_ = nullptr;
    bool min_valid_ = false;
    bool max_valid_ = false;
    float min_cached_ = 0.0F;
    float max_cached_ = 0.0F;
    bool host_work_last_round_ = false;
};

// PIMPL façade, buffer-backend only. There is no image kernel, so an
// image-backend construction is refused by name instead of silently binding
// tensors that were laid out for the image path.
class Clip : public PimplFacade {
  public:
    Clip(int fp16, bool backend_buffer) : PimplFacade(OpType::CLIP) {
        if (!backend_buffer) {
            throw std::runtime_error(
                "Clip is buffer-backend only (no image impl).");
        }
        impl_ = std::make_unique<ClipBuffer>(fp16);
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_CLIP_HPP_
