// junka @ 2026
// KVCache —— 图里那组 past/present KV 张量的生命周期句柄（host 侧壳，不新增
// GPU 结构）。Runtime 只把 KV 当普通输入/输出暴露，「一条序列的 KV 到哪一步了」
// 这件会话级的事原本散在驱动里（预分配、每轮 reset、present→past 回填），这里
// 收拢成一个对象，让驱动只表达「本轮 prefill 从空 KV 开始」「本轮结束把 KV 接上」。
//
// 三件事：
//  1. 预分配：LoadModel 之后立刻把每层 past 的 buffer 一次性按 max_kv 开好，
//     present 直接 alias 到同一块（Tensor::alias_gpu_buffer）—— 每层一块，不是
//     两块。之后每轮 ResizeInput 只改逻辑形状，make_vkbuff 走 prealloc_keep_
//     复用同一个 VkBuffer，不再每轮重分配几十 MB，KV 显存也就减半。
//  2. reset_for_prefill()：把 past 的逻辑 kv_len 归零（整段重新 prefill 的
//     语义），随后 upload() 把「空 past」真正传到 GPU。
//  3. feedback()：present→past —— 现在只剩改一个逻辑形状。两块张量本来就是
//     同一块 buffer，「present 的字节就是下一轮 past 的字节」这句话是字面成立
//     的，连交换身份都不需要。于是每轮 KV 的开销与上下文长度完全无关：不是
//     「拷整段历史」O(kv)，也不是「换 buffer 身份」，而是 O(1) 的 host 改形状。
//     （历史：拷贝路径实测每步 0.35~0.45ms 且随 past_len 基本不涨 —— 被
//     submit/wait 的固定开销主导；换身份后 ~20µs。同一二进制两条路径、596 步
//     逐位相同。）
//
// 共享一块 buffer 对图侧也是对的，包括不走 cache 形式的路径：图里唯一的读者是
// Gather(past)，唯一的写者是写 present 的那个算子，两者之间隔着依赖序的 barrier，
// 而且写 present 的从来不是读 past 的那一条（矩阵乘/Concat 各自落在自己的临时
// 张量上）。所以「本轮写 present == 写下一轮的 past」不会与「本轮读 past」打架。
// attention 形式下更是如此：cache_in/cache_out 是同一块，kernel 直接跳过历史拷贝
// （inplace 由算子从绑定的两块的指针自己判定，见 ops/Attention.hpp）。
//
// 存储类型：cache_kind 决定每层 past/present 的字节容器语义。fp16 是历史默认
// （uint16_t 容器，2B/elem）；fp8（E4M3/E5M2）用 int8_t 容器（1B/elem），把
// KV cache 显存再减半。两者走同一套预分配/alias/reset 机制 —— preallocate_buffer
// 和 alias_gpu_buffer 都是 sizeof(T) 的纯字节操作，dtype 无关。attention
// 计算仍走 fp16：图里在 cache 边界插 QuantizeLinear/DequantizeLinear，所以
// 这里看到的 past/present 永远是「原始 payload 字节」，不参与反量化。
//
// 4-bit（kInt4/kNF4）沿用 int8_t 容器，但一个字节装两个值：逻辑元素数不变
// （2 × nkv × max_kv × hd），payload 字节只有它的一半。preallocate_buffer 拿的
// 是「容器槽位」数（= kv_elems/2），逻辑形状仍由 getShape() 表达 —— 4-bit 的
// K/N 本来就只能从形状读，不能靠 num_elements()。同理 num_kv_elems_ 要把字节数
// 乘回 2 才是「一段 KV 有几个值」，否则 feedback() 算出的 kv_len 会少一半。
// 图侧同样在 cache 边界插 Q/DQ（fmt 4/5），scale 是每张量一个 fp32 标量。
//
// 前缀续用（跨轮跳过已算 KV）的接口留在这里：需要 feedback() 之外再加一个
// 「保留前缀、只补后面」的入口，并先有 token 前缀校验（Conversation::prefixMatch）。
// 目前图是单段连续 buffer、无 block table，所以还没接。

#pragma once

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/DType.hpp"
#include "core/Tensor.hpp"
#include "core/runtime.hpp"
#include "vulkan/VulkanCommandPool.hpp"

namespace vkop {
namespace export_ {

class KVCache {
public:
    // cache_kind 选择 KV cache 的存储语义。fp16 是历史默认；传 kFloat8E4M3FN /
    // kFloat8E5M2 则每层 past/present 用 int8_t 容器承载 fp8 字节（图侧需在
    // cache 边界插 QuantizeLinear/DequantizeLinear，见 qwen3_export_onnx.py
    // 的 --kv-fp8 分支）。kInt4 / kNF4 同样走 int8_t 容器，但一字节两个值
    // （图侧 Q/DQ 用 fmt 4/5），显存再减半。
    KVCache(const std::shared_ptr<core::Runtime>& rt,
            const std::shared_ptr<VulkanCommandPool>& cmdpool, int nlayers,
            int nkv, int hd, int max_kv,
            core::ElemKind cache_kind = core::ElemKind::kFloat16,
            std::vector<int> layers = {})
        : rt_(rt), cmdpool_(cmdpool), nlayers_(nlayers), nkv_(nkv), hd_(hd),
          max_kv_(max_kv), cache_kind_(cache_kind) {
        if (nlayers <= 0 || nkv <= 0 || hd <= 0 || max_kv <= 0) {
            throw std::runtime_error("KVCache: invalid arch parameters");
        }
        // Which layer number each cache slot belongs to. A dense decoder's
        // inputs are past_key_values_0 .. _n-1, so the default numbering is
        // indistinguishable from the real one. A hybrid decoder caches only
        // its full-attention layers and names them by their position in the
        // whole stack (Qwen3.5-27B: 3, 7, ... 63), which no arithmetic
        // recovers -- it has to be passed in.
        if (layers.empty()) {
            for (int i = 0; i < nlayers; ++i) layers.push_back(i);
        }
        if (static_cast<int>(layers.size()) != nlayers) {
            throw std::runtime_error(
                "KVCache: " + std::to_string(layers.size()) +
                " layer number(s) for " + std::to_string(nlayers) +
                " cache slot(s)");
        }
        layers_ = std::move(layers);
        if (cache_kind != core::ElemKind::kFloat16 &&
            cache_kind != core::ElemKind::kFloat8E4M3FN &&
            cache_kind != core::ElemKind::kFloat8E5M2 &&
            cache_kind != core::ElemKind::kInt4 &&
            cache_kind != core::ElemKind::kNF4) {
            throw std::runtime_error(
                "KVCache: unsupported cache element format " +
                std::string(core::elem_name(cache_kind)));
        }
        // past 按 max_kv 开好，present 共享同一块（每层一块 buffer，长度增长时
        // 不再重分配）。元素总数与 dtype 无关（2 × nkv × max_kv × hd，K/V 在
        // dim 1 叠在一起）；字节宽由容器类型决定，preallocate_buffer 内部按
        // sizeof(T) 对齐。4-bit 的两个值挤一字节，所以给它的「容器槽位数」是
        // 逻辑元素数的一半（见 preallocate_kv_）。
        const std::size_t kv_elems = static_cast<std::size_t>(2) * nkv * max_kv * hd;
        auto dev = cmdpool->getVulkanDevice();
        for (int i = 0; i < nlayers; ++i) {
            auto pin = rt_->GetInput(past_name(i));
            auto pout = rt_->GetOutput(present_name(i));
            if (!pin || !pout) {
                throw std::runtime_error("KVCache: missing KV tensor for layer " +
                                         std::to_string(layers_[i]));
            }
            preallocate_kv_(pin, dev, kv_elems);
            share_kv_(pout, pin);
        }
    }

    // Graph names for cache slot `i` (0-based SLOT index, not the layer
    // number). Named rather than inlined so the three places that build these
    // strings cannot drift apart.
    std::string past_name(int i) const {
        return "past_key_values_" + std::to_string(layers_[i]);
    }
    std::string present_name(int i) const {
        return "present_key_values_" + std::to_string(layers_[i]);
    }

    // 每层 past 的逻辑形状成 kv_len=0（buffer 复用，不重新分配）。必须在填其它
    // 输入之前调用，且之后要 upload()。
    void reset_for_prefill() {
        for (int i = 0; i < nlayers_; ++i) {
            rt_->ResizeInput(past_name(i),
                             {1u, 2u, static_cast<uint32_t>(nkv_), 0u,
                              static_cast<uint32_t>(hd_)});
        }
    }

    // reset_for_prefill() 之后把空 past 传上 GPU。
    void upload() {
        for (int i = 0; i < nlayers_; ++i) {
            auto t = rt_->GetInput(past_name(i));
            upload_kv_(t);
        }
    }

    // 从已有的 KV 继续：past 的逻辑 kv_len 设成 prefix_len —— buffer 里那
    // prefix_len 行是上一轮写下的历史，内容一个字都不动，所以**不要**再调
    // upload()（那会把 staging 里的旧字节盖上去）。跨轮前缀复用用这个入口：
    // 新的 prefill 只需要算 [prefix_len, prefix_len + S) 那几行。
    //
    // prefix_len 会变成 shader 读 cache 的行数，所以在这里挡住越界 —— 超出
    // 预分配的 max_kv 就是读进了 buffer 外面。
    void resume_from(int prefix_len) {
        if (prefix_len < 0 || prefix_len > max_kv_) {
            throw std::runtime_error("KVCache: resume_from(" +
                                     std::to_string(prefix_len) +
                                     ") is outside [0, max_kv=" +
                                     std::to_string(max_kv_) + "]");
        }
        for (int i = 0; i < nlayers_; ++i) {
            set_past_len_(i, prefix_len);
        }
    }

    // present→past：两个张量一开始就是同一块 buffer，所以这里没有设备拷贝、
    // 也没有 buffer 身份交换，只有「past 的逻辑长度跟上本轮」这一件 host 侧的事。
    // 返回新的 past_len（下一轮的位置/attention_bias 都以它为准）。
    int feedback() {
        int past_len = 0;
        for (int i = 0; i < nlayers_; ++i) {
            auto pres = rt_->GetOutput(present_name(i));
            const int kv_len = num_kv_elems_(pres) / (2 * nkv_ * hd_);
            set_past_len_(i, kv_len);
            past_len = kv_len;
        }
        return past_len;
    }

    int nlayers() const { return nlayers_; }
    int nkv() const { return nkv_; }
    int hd() const { return hd_; }
    core::ElemKind cache_kind() const { return cache_kind_; }

private:
    std::shared_ptr<core::Runtime> rt_;
    std::shared_ptr<VulkanCommandPool> cmdpool_;
    int nlayers_, nkv_, hd_, max_kv_;
    // Layer number per cache slot; 0..nlayers-1 unless the model is hybrid.
    std::vector<int> layers_;
    core::ElemKind cache_kind_;

    // past 的逻辑长度跟上写出的历史。ResizeInput 会把 converted_ 置回 false
    // （buffer 走 prealloc_keep_ 复用，不重分配），所以最后要把 past 标回「数据
    // 在 GPU 上」——那块 buffer 里装的正是本轮的历史，不需要任何上传。
    void set_past_len_(int layer, int kv_len) {
        const std::string name = past_name(layer);
        rt_->ResizeInput(name, {1u, 2u, static_cast<uint32_t>(nkv_),
                                static_cast<uint32_t>(kv_len),
                                static_cast<uint32_t>(hd_)});
        mark_gpu_(rt_->GetInput(name));
    }

    // 按 cache_kind 分派到对应容器的预分配。fp16 走 uint16_t，fp8 走 int8_t
    // （和 bool/int8 同一容器，elem_kind() 记录 fp8 语义）。4-bit 也走 int8_t，
    // 但一个字节装两个值：kv_elems 是逻辑值数，容器槽位（字节）是它的一半
    // （kv_elems 含因子 2，必为偶数）。
    void preallocate_kv_(const std::shared_ptr<core::ITensor>& t,
                         std::shared_ptr<VulkanDevice>& dev,
                         std::size_t kv_elems) {
        if (cache_kind_ == core::ElemKind::kFloat16) {
            core::as_tensor<uint16_t>(t)->preallocate_buffer(dev, kv_elems);
        } else if (cache_kind_ == core::ElemKind::kInt4 ||
                   cache_kind_ == core::ElemKind::kNF4) {
            core::as_tensor<int8_t>(t)->preallocate_buffer(dev, kv_elems / 2);
        } else {
            core::as_tensor<int8_t>(t)->preallocate_buffer(dev, kv_elems);
        }
    }
    void upload_kv_(const std::shared_ptr<core::ITensor>& t) {
        if (cache_kind_ == core::ElemKind::kFloat16) {
            core::as_tensor<uint16_t>(t)->copyToGPU(cmdpool_);
        } else {
            core::as_tensor<int8_t>(t)->copyToGPU(cmdpool_);
        }
    }
    // 一段 KV 的逻辑值数。fp16/fp8 的 num_elements() 就是值数；4-bit 的
    // num_elements() 是字节数（容器槽位），要乘 2 才是值数 —— 否则 feedback()
    // 会把 kv_len 算成一半，下一轮的 attention_bias/位置就全错位。
    int num_kv_elems_(const std::shared_ptr<core::ITensor>& t) {
        if (cache_kind_ == core::ElemKind::kFloat16) {
            return core::as_tensor<uint16_t>(t)->num_elements();
        }
        const int slots = core::as_tensor<int8_t>(t)->num_elements();
        if (cache_kind_ == core::ElemKind::kInt4 ||
            cache_kind_ == core::ElemKind::kNF4) {
            return slots * 2;
        }
        return slots;
    }
    // dst 变成 src 那块 buffer 的第二个句柄，src 原地不动。src 必须是已经
    // preallocate 过的（形状大小的 buffer 装不下按 max_kv 长起来的历史，
    // 见 Tensor::alias_gpu_buffer）。dst 自己可能带着 LoadModel 时开的
    // buffer（present 是图输出）—— 被这里替换掉，由 shared_ptr 释放。
    //
    // 两边的容器类型必须一致：alias 只搬 buffer 身份、不看元素类型，如果
    // cache_kind 把 past 和 present 分到不同容器上，绑定时才会发现宽窄不对。
    // 事实上不会 —— preallocate_kv_ 按同一个 cache_kind_ 分派。
    void share_kv_(const std::shared_ptr<core::ITensor>& dst,
                   const std::shared_ptr<core::ITensor>& src) {
        if (cache_kind_ == core::ElemKind::kFloat16) {
            core::as_tensor<uint16_t>(dst)->alias_gpu_buffer(
                *core::as_tensor<uint16_t>(src));
        } else {
            core::as_tensor<int8_t>(dst)->alias_gpu_buffer(
                *core::as_tensor<int8_t>(src));
        }
    }
    void mark_gpu_(const std::shared_ptr<core::ITensor>& t) {
        if (cache_kind_ == core::ElemKind::kFloat16) {
            core::as_tensor<uint16_t>(t)->toGPU();
        } else {
            core::as_tensor<int8_t>(t)->toGPU();
        }
    }
};

// LinearState — the conv/recurrent state of a hybrid decoder's linear-attention
// layers, held the same way KVCache holds past/present.
//
// Qwen3.5-27B is 48 linear + 16 full-attention layers. Each linear layer owns
// two fixed-shape fp32 tensors that ride the whole conversation:
//
//   conv_state_<L>       [1, conv_dim(=10240), conv_k(=3)]
//   recurrent_state_<L>  [1, v_heads(=48), dk(=128), dv(=128)]
//
// The graph takes them as inputs and returns `conv_state_out_<L>` /
// `recurrent_state_out_<L>`. Unlike the KV cache these are NOT the same bytes
// in and out: the operator reads the whole state and writes a new one, so the
// output cannot alias the input -- one dispatch would read and clobber the same
// buffer. Two buffers per pair, swapped each round, is what gives a disjoint
// read and write.
//
// Two tensor objects per pair (`a`, `b`), each carrying its own VkBuffer. The
// graph's input and output names re-point at whichever pair of buffers the
// round needs:
//
//   round k:  conv_state_<L> -> cur ,  conv_state_out_<L> -> other
//   then swap cur/other
//
// so the output of one round is the input of the next without a device copy.
// Sizes are fixed, so the four buffers are allocated once at load and nothing
// here depends on the context length.
class LinearState {
public:
    // Discover the linear-attention layers from the graph's own input names.
    // A hybrid decoder numbers them by their position in the whole stack
    // (Qwen3.5-27B: 0,1,2, 4,5,6, 8,... -- every 4th layer is full
    // attention), so the set cannot be derived arithmetically; it is read.
    // A dense model has no `conv_state_*` inputs at all, which is how this
    // reports "nothing to do" without the caller having to classify the model.
    LinearState(const std::shared_ptr<core::Runtime>& rt,
                const std::shared_ptr<VulkanCommandPool>& cmdpool)
        : rt_(rt), cmdpool_(cmdpool) {
        const std::string prefix = "conv_state_";
        for (const auto& name : rt_->ListInputNames()) {
            if (name.rfind(prefix, 0) != 0) continue;
            const std::string digits = name.substr(prefix.size());
            if (digits.empty() ||
                digits.find_first_not_of("0123456789") != std::string::npos)
                continue;
            layers_.push_back(std::atoi(digits.c_str()));
        }
        if (layers_.empty()) return;  // a dense decoder has no linear layers
        auto dev = cmdpool->getVulkanDevice();
        for (int layer : layers_) {
            add_pair_("conv_state_", "conv_state_out_", layer, dev);
            add_pair_("recurrent_state_", "recurrent_state_out_", layer, dev);
        }
    }

    // True when this model has linear-attention state at all.
    bool present() const { return !pairs_.empty(); }
    size_t pair_count() const { return pairs_.size(); }

    // Zero every state, in preparation for a fresh conversation.
    void reset() {
        for (auto &p : pairs_) zero_(p.in);
    }

    // reset() already uploaded the zeros; kept so the driver's startup sequence
    // reads the same as the KV cache's.
    void upload() {}

    // Nothing to do before Run: the graph reads `conv_state_<L>` /
    // `recurrent_state_<L>` as ordinary inputs, and those tensors ARE the state.
    // Kept as an explicit no-op so the driver's round shape stays visible.
    void bind_for_round() {}

    // After Run (and after the GPU is done reading): make this round's output
    // next round's input.
    //
    // This copies INTO the graph-input tensor's own buffer. Two cheaper designs
    // were tried and both failed, which is why the obvious ones are not here:
    //
    //  * alias the input onto a private slot  -- the state never reached the
    //    graph. The logits came back BIT-IDENTICAL to a run with a zero state,
    //    and only the private slot showed the value, i.e. the graph kept reading
    //    its own buffer while the copy went elsewhere.
    //  * alias the output onto that slot    -- the producer of
    //    `conv_state_out_<L>` is an Unsqueeze, a pure view, which re-aliases its
    //    output onto Conv1d's buffer at execute time and discards the alias.
    //    Measured: the state stayed exactly zero (nonzero 0, mean|v| 0) round
    //    after round.
    //
    // Copying leaves every buffer identity alone: the input tensor keeps the
    // buffer the runtime allocated for it, so whatever descriptor the op binds
    // is still the right one. Cost is ~153 MB per round across 96 pairs (48 x
    // 120 KB conv + 48 x 3 MB recurrent), about 13 ms of PCIe against a ~1.7 s
    // round.
    void advance() {
        for (auto &p : pairs_) {
            auto from = core::as_tensor<float>(p.out);
            from->copyToCPU(cmdpool_);
            auto to = core::as_tensor<float>(p.in);
            // toCPU() first: copyToGPU returns immediately when the tensor is
            // already resident, and `p.in` has been resident since the first
            // round. Without this the state never reached the graph at all --
            // decode ran every round against whatever in held, and only prefill
            // looked right, because a chunk's whole recurrence happens inside
            // one Run() along seq.
            to->toCPU();
            to->fillToCPU(from->data().data());
            to->copyToGPU(cmdpool_);
            // The readback left the graph output host-resident; next round's
            // producer expects to write a device-resident tensor.
            from->toGPU();
        }
    }

    // Diagnostic: is the state non-zero, i.e. did the last round actually
    // produce something the next one will read?
    void dump_debug(const char *label) {
        if (pairs_.empty()) {
            std::printf("[state] %s: no linear-attention state\n", label);
            std::fflush(stdout);
            return;
        }
        for (const auto &named : {
                 std::pair<const char *, std::shared_ptr<core::ITensor>>{
                     "in", pairs_[0].in},
                 {"out", pairs_[0].out}}) {
            auto f = core::as_tensor<float>(named.second);
            if (!f) continue;
            f->copyToCPU(cmdpool_);
            const auto &d = f->data();
            double sum = 0.0;
            size_t nz = 0;
            for (float v : d) {
                sum += v < 0 ? -static_cast<double>(v) : static_cast<double>(v);
                if (v != 0.0f) ++nz;
            }
            std::printf("[state] %-22s %-3s %zu elem(s), nonzero %zu, "
                        "mean|v| %.6g\n",
                        label, named.first, d.size(), nz,
                        d.empty() ? 0.0 : sum / d.size());
            std::fflush(stdout);
            f->toGPU();
        }
    }

private:
    struct Pair {
        std::shared_ptr<core::ITensor> in;   // graph input:  the state the graph reads
        std::shared_ptr<core::ITensor> out;  // graph output: the state it produced
    };

    void add_pair_(const char *in_prefix, const char *out_prefix, int layer,
                   std::shared_ptr<VulkanDevice>& dev) {
        const std::string in_name = in_prefix + std::to_string(layer);
        const std::string out_name = out_prefix + std::to_string(layer);
        auto tin = rt_->GetInput(in_name);
        auto tout = rt_->GetOutput(out_name);
        if (!tin) {
            throw std::runtime_error("LinearState: no graph input " + in_name);
        }
        if (!tout) {
            throw std::runtime_error("LinearState: no graph output " + out_name);
        }
        if (tin->elem_kind() != core::ElemKind::kFloat32) {
            throw std::runtime_error(
                "LinearState: " + in_name + " must be float32, got " +
                std::string(core::elem_name(tin->elem_kind())));
        }
        size_t elems = 1;
        for (auto d : tin->getShape()) elems *= static_cast<size_t>(d);
        if (elems == 0) {
            throw std::runtime_error("LinearState: " + in_name + " is empty");
        }
        // The graph's own tensors ARE the state; nothing extra is allocated.
        // `elems` is kept only for the emptiness check above.
        (void)elems;
        (void)dev;
        Pair p;
        p.in = tin;
        p.out = tout;
        pairs_.push_back(std::move(p));
    }

    void zero_(const std::shared_ptr<core::ITensor>& t) {
        auto f = core::as_tensor<float>(t);
        // toCPU() before the upload is load-bearing: copyToGPU returns
        // immediately when the tensor is already resident, so without it this
        // is a NO-OP from the second round onward -- the host vector is zeroed
        // and never sent. Measured: the first request answered correctly and
        // every later one inherited the previous request's state, because
        // reset() had stopped doing anything.
        f->toCPU();
        std::vector<float> zeros(f->num_elements(), 0.0f);
        f->fillToCPU(zeros.data());
        f->copyToGPU(cmdpool_);
    }


    std::shared_ptr<core::Runtime> rt_;
    std::shared_ptr<VulkanCommandPool> cmdpool_;
    std::vector<int> layers_;
    std::vector<Pair> pairs_;
};

} // namespace export_
} // namespace vkop
