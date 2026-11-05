// Copyright 2025 @junka
#ifndef OPS_MATMUL_HPP_
#define OPS_MATMUL_HPP_

#include "Operator.hpp"
#include "ops/BufferBase.hpp"
#include "ops/PimplFacade.hpp"

#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>
namespace vkop {
namespace ops {

namespace matmul {

enum class Method {
    BASIC_ARITHMETIC = 0,
    VK_COOPERATE_MATRIX = 1,
    NV_TENSORCORE = 2,
};

struct alignas(16) GpuMatMulParam {
    int M;
    int N;
    int K;
    int C;        // image path: channel count; buffer path: batch count
    int fp32;     // 1 = fp32, 0 = fp16
    int transB;   // 1 = B is laid out as [batch, N, K] (transposed input); 0 =
                  // [batch, K, N]
    int tile = 0; // 1 = shared-memory tiled GEMM; buffer path only
    // 1 = B is a byte-packed int8 weight with a per-output-column fp32 scale
    // bound at binding 4 (buffer path only; the image shaders declare the slot
    // but never read it). Defaulted because MatMulImage never assigns it and an
    // indeterminate byte in the push range is a wrong answer waiting to happen.
    int weight_int8 = 0;
    // 1 = the int8 kernel owns four output columns per thread (N % 4 == 0 and B
    // is [batch, K, N], where that quad is exactly one 32-bit weight word). The
    // grid x extent counts column quads instead of column pairs, so the host
    // and the shader must agree on it; defaulting keeps MatMulImage out of it.
    int w8_quad = 0;
    // 1 = the buffer kernel splits K across the workgroup's lanes instead of
    // across output rows, because at decode time there is only one A row and
    // the column-parallel kernels then leave 15 of every 16 lanes idle. Buffer
    // path, fp16 build only; the image shaders never see it set.
    int ksplit = 0;
    // Column quads per block for the split-K GEMV (buffer path, fp16 build).
    // The block's 256 lanes are laid out flat as quad = lane % ksplit_q,
    // K slice = lane / ksplit_q, so this trades column coverage for K
    // parallelism: at 16 (the original layout) down_proj's N = 896 put only 14
    // blocks on the device and ran at 21 GB/s while lm_head's 2374 blocks
    // reached 25. Host picks it in execute(); 16 keeps the original layout.
    int ksplit_q = 16;
    // 1 = B is a 4-bit weight-only payload (two nibbles per byte, even element
    // in the low nibble) with a per-K-group fp32 scale at binding 4, laid out
    // [K/group, N]. Buffer path only, and the host gates it on transB == 0,
    // N % 8 == 0 and K % group == 0 (see MatMulBuffer::execute).
    int w4 = 0;
    // 1 = the nibble is an unsigned NF4 codebook index, not a signed int4
    // value.
    int nf4 = 0;
    // Values of K sharing one scale row (K / n_groups). Even, and a divisor of
    // K.
    int group = 0;
    // 1 = the weight bytes are fp8 (E4M3, or E5M2 when fp8_e5m2 below) rather
    // than two's-complement int8. Same per-column scale, same kernels -- only
    // the byte-to-float step differs -- so this flag rides on weight_int8 being
    // set as well. Buffer path only.
    int fp8 = 0;
    // 1 = the fp8 layout is E5M2 (5 exponent, 2 mantissa bits) instead of E4M3.
    int fp8_e5m2 = 0;
    // 1 = NVFP4: the w4 nibble is an E2M1 code, binding 4 holds one fp8 E4M3
    // block scale per (16-value K block, output column) instead of an fp32
    // table, and binding 5 holds the tensor's single fp32 factor that every
    // block scale is multiplied by. Buffer path only; the image shader declares
    // neither slot and never reads them.
    int nvfp4 = 0;
    // 1 = the 4-bit weight's nibble is an unsigned value in [0, 15] scaled by
    // the per-group fp32 scale table — the unsigned counterpart of int4's
    // signed [-8, 7] grid (zero point 0, the scale carrying the group absmax/15
    // rather than /7). Same packing, same kernels, same scale table shape as
    // int4; only the nibble-to-float step differs. Buffer path only, and only
    // meaningful with w4 set.
    int uint4 = 0;
    // 1 = fp16 split-K GEMV uses the 16-byte-load variant: each lane owns
    // eight contiguous columns (one uvec4 weight load per k, two accumulator
    // chains) and ksplit_q counts 8-column groups per block. Host requires
    // N % 8 == 0; see MatMulBuffer::execute. Buffer path only.
    int wide = 0;
    // Grouped leading batch axis (repeat_kv folded into this MatMul by the
    // converter): exactly one leading axis of A is `rep` consecutive copies of
    // B's axis (kv heads repeated into q heads), so B's flat batch is rep times
    // smaller than the dispatch's. The kernels map A's flat batch index onto
    // B's slices with bmap() instead of reading a B slice that does not exist.
    // rep <= 1 (the default) leaves every kernel's addressing untouched. Buffer
    // path only; the image shaders never read these.
    int rep = 1;
    // Product of A's leading dims after the grouped axis (1 when the grouped
    // axis is the last leading axis, which is the repeat_kv case).
    int rep_stride = 1;
    // A's dim on the grouped axis (= b_dim * rep). 0 when rep <= 1.
    int a_dim = 0;
    // B's dim on the grouped axis. 0 when rep <= 1.
    int b_dim = 0;
    // Attention QK^T epilogue: the converter folds `MatMul * epi_scale +
    // epi_bias[col]` (the per-layer 1/sqrt(head_dim) scale and the causal
    // attention_bias) into the MatMul so the FusedElemwise dispatch disappears.
    // epi_bias=1 means a bias tensor is bound at binding 6 (broadcast over the
    // M and head axes — shape [B,1,q,kv] or narrower); epi_scale multiplies
    // every output element before the bias add. Both default off so every
    // non-attention MatMul is untouched. Buffer path only.
    float epi_scale = 1.0f;
    int epi_bias = 0;
    // K as the converter delivered it. In the SwiGLU down fold the kernel's
    // two operands disagree on K: uPC.K counts the weight rows (the up width
    // N_up, the down GEMM's contraction), and K_act counts x's columns
    // (K_out). Unlike every other kernel, the generic paths of the DOWN_ACT
    // module need BOTH -- their output size is the down GEMM's (M x N), while
    // the x row they read is K_out wide -- so the host leaves K at K_out (the
    // converter's value) and the fold route alone overwrites it with N_up.
    // Buffer path only.
    //
    // K_act and mind_k live here, in the padding hole that follows epi_bias
    // (epi_scale's 4 bytes align the next int), rather than at the struct tail:
    // a 136-byte struct rounds to 144 under alignas(16) and the push_constant
    // block would collapse to a 4-byte range, zeroing every constant after it.
    int K_act = 0;
    // The SwiGLU down fold's deposit grid-x extent GX = ceil(K / 512): each
    // deposit block owns 512 up columns and parks fp32 partials for all N
    // output columns, and the fold pass sums GX of them. Also the fold's loop
    // bound, so the two passes cannot disagree about the partial count.
    // Buffer path only.
    int mind_k = 0;
    // The bias tensor's flat batch count (== total batch / head-broadcast
    // ratio). 0 when epi_bias is off.
    //
    // The SwiGLU down fold reuses this slot for `ks`: the padded group width
    // of its lane-major Wg/Wu, i.e. ceil(N_up/8)*8. The fold never raises
    // epi_bias (its converter pass deletes the scale/bias chain), so the two
    // meanings can never be live at once, and a separate field would push the
    // struct past the 136-byte boundary the comment above records.
    union {
        int epi_bias_batch;
        int ks;
    };
    // Plan B: fuse the trailing Softmax into this MatMul. When set, the kernel
    // reduces each output row across all N columns (online max→sum→normalize)
    // and stores exp(x-max)/sum, so the fp16 2-dispatch Softmax vanishes. Host
    // only raises this on attention QK^T (small M, even N, single-consumer).
    // Buffer path only; requires the softmax workgroup to cover a full row.
    int attn_softmax = 0;
    // 1 = B is broadcast over the batch axes: a [K, N] weight (or one whose
    // leading dims are all 1) shared by every output slice, so B's slice index
    // folds to 0 regardless of the flat batch. The buffer kernels fold it in
    // bmap()'s early-out and the coop shaders at their B base; without it they
    // would index b * K * N past the weight for every batch >= 1 workgroup.
    // Sits at the struct tail (offset 108): sizeof stays 112 after the
    // alignas(16) rounding, so the pushed range is unchanged for archives
    // whose shaders do not read it. Buffer path only.
    int b_shared = 0;
    // The plain split-K GEMV's fold pass, when the host routes through the
    // standalone fold module. Its lane math (quad = lane % ksplit_q) is
    // identical to a deposit lane's -- every lane publishes a partial at
    // slot (slice * q + quad) -- so the shaders tell the two passes apart only
    // because the host says so. Raising it for a deposit dispatch would make
    // every lane read slot 0..n_slice-1 of its own quad and write an
    // unsplit-K result.
    //
    // The SwiGLU down fold's three passes run on ONE module
    // (buffer_matmul_down_act_fp16), so this flag is also the entry-point
    // selector there: 0 = deposit (K-split up GEMV), 1 = down (Wd walk), 2 =
    // reduce (partial sum + fp16 pack). A shape that reaches the same module
    // with ksplit=1 and a stale ks_part would run a later pass in place of
    // the deposit; the host keeps them in step.
    //
    // Sits at the struct tail (offset 112): sizeof stays 128 after the
    // alignas(16) rounding, so the pushed range is unchanged.
    int ks_part = 0;
};

} // namespace matmul

// Image (image2DArray NCHW->RGBA) implementation. Uses cooperative matrix
// extensions on supported GPUs (KHR coop / NV tensorcore).
class MatMulImage : public Operator {
  public:
    MatMulImage(const MatMulImage &) = delete;
    MatMulImage &operator=(const MatMulImage &) = delete;
    MatMulImage(MatMulImage &&) = delete;
    MatMulImage &operator=(MatMulImage &&) = delete;

    // use_tensorcore still selects the host-side schedule (method_ below), but
    // the image backend's SPIR-V arm no longer varies with it: with no shader
    // symbols linked in, the image MatMul resolves the single "matmul" key
    // from the archive like every other op. The nv/coop image shaders are only
    // reachable if the archive names them, which the buffer backend does.
    explicit MatMulImage(int use_tensorcore = 0)
        : Operator(OpType::MATMUL,
                   {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER},
                   sizeof(matmul::GpuMatMulParam)) {
        method_ =
            use_tensorcore == 2
                ? matmul::Method::NV_TENSORCORE
                : (use_tensorcore == 1 ? matmul::Method::VK_COOPERATE_MATRIX
                                       : matmul::Method::BASIC_ARITHMETIC);
    };

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        // This kernel reads both operands as float or half; a quantized operand
        // has to reach a kernel that dequantizes it first. Silent fallback to
        // the fp32 loader would read payload bytes as exponents.
        for (size_t i = 0; i < inputs.size(); ++i) {
            core::require_float_elem(inputs[i]->elem_kind(), "MatMul",
                                     ("input " + std::to_string(i)).c_str());
        }
        int chan = inputs[0]->get_channel();
        int m = inputs[0]->get_height();
        int n = inputs[1]->get_width();
        int k = inputs[0]->get_width();
        int rank = inputs[0]->num_dims();
        if (rank < 2) {
            // This path takes M and K straight off A's height/width, so a
            // rank-1 (or scalar) A has no M axis: shape[rank - 2] below would
            // write out of bounds. The buffer backend promotes rank-1 operands;
            // the image path does not, so say so loudly rather than corrupt the
            // shape vector.
            throw std::runtime_error(
                "MatMul (image backend) needs rank >= 2 for A, got rank " +
                std::to_string(rank));
        }
        auto shape = inputs[0]->getShape();
        shape[rank - 1] = n;
        shape[rank - 2] = m;
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto t) {
            using T = decltype(t);
            auto outputptr = core::as_tensor<T>(outputs[0]);
            if (outputptr->size() == 0) {
                outputptr->resize(shape);
            }
            auto output_image = outputptr->as_output_image(m_dev_, m_cmd_);
            objs_.emplace_back(output_image);
        });
        for (const auto &input : inputs) {
            dispatch_by_dtype(input->dtype(), [&](auto t) {
                using T = decltype(t);
                auto inputptr = core::as_tensor<T>(input);
                auto input_image = inputptr->as_input_image(m_dev_, m_cmd_);
                objs_.emplace_back(input_image);
                if (typeid(uint16_t) == typeid(T)) {
                    para_.fp32 = 0;
                } else if (typeid(float) == typeid(T)) {
                    para_.fp32 = 1;
                }
            });
        }
        para_.M = m;
        para_.N = n;
        para_.K = k;
        para_.C = chan;
        if (method_ == matmul::Method::VK_COOPERATE_MATRIX) {
            submit(&para_, UP_DIV(n, 32), UP_DIV(m, 16), UP_DIV(chan, 4));
        } else if (method_ == matmul::Method::NV_TENSORCORE) {
            submit(&para_, UP_DIV(n, 32), UP_DIV(m, 16), UP_DIV(chan, 4));
        } else {
            submit(&para_, UP_DIV(n, 16), UP_DIV(m, 16), UP_DIV(chan, 4));
        }
    }

    matmul::GpuMatMulParam para_;
    matmul::Method method_ = matmul::Method::BASIC_ARITHMETIC;
};

// Buffer (SSBO, compact row-major) implementation. fp32: one thread per
// output element. fp16 with even N: single pass — one thread per output
// half2 word (two adjacent columns, A load shared), packed straight into
// the output. fp16 with odd N: the half2 words straddle row boundaries, so
// fall back to 2-pass (reduce to fp32 scratch + pack pass) with the pack
// pass on a SEPARATE pipeline (buffer_matmul_pack_spv) to avoid Intel ANV
// push-constant interference between dispatches of the same pipeline.
class MatMulBuffer : public BufferFactory {
  public:
    explicit MatMulBuffer(int fp16)
        : BufferFactory(OpType::MATMUL,
                        std::vector<VkDescriptorType>{
                            DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                            DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                            DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                            DESCRIPTOR_TYPE_STORAGE},
                        sizeof(matmul::GpuMatMulParam), fp16) {
        update_after_bind_ = true;
    }

    // Kernel-route readout for VKOP_MATMUL_ROUTE. Set at each dispatch site in
    // execute(); the tags are the shader family, not the SPIR-V entry name.
    std::string route_tag() const override { return route_tag_; }
    std::string route_info() const override { return route_info_; }
    bool has_host_side_effects() const override { return true; }
    void set_runtime_device(const std::shared_ptr<VulkanDevice> &dev,
                            const std::shared_ptr<VulkanCommandPool> &cmdpool,
                            VkPipelineCache cache = VK_NULL_HANDLE) override {
        BufferFactory::set_runtime_device(dev, cmdpool, cache);
        // MoltenVK lists KHR_cooperative_matrix but its SPIR-V -> MSL backend
        // only lowers 8x8 coopmat dims, so the 8x16/16x8 A/B tiles here fail
        // pipeline creation ("MSL cooperative matrices only support 8x8
        // dimensions"). VKOP_NO_COOP=1 skips the coop paths entirely and keeps
        // the reduce fallback usable on such portability drivers.
        const bool coop_disabled = coop_env_off();
        if (fp16_ != 0 && !coop_pipeline_ && !coop_disabled &&
            // Skip the eager build on devices whose coopmat support is
            // nominal only: MoltenVK advertises the feature bool but reports
            // 0 supported stages (see VulkanDevice's feature gate), and its
            // MSL backend only lowers 8x8 coopmat dims, so pipeline creation
            // here would throw and take the whole MatMul down with it.
            dev->is_support_cooperate_matrix()) {
            // fp16 cooperative-matrix kernel: a single dispatch reads fp16 A/B
            // from SSBO, accumulates in fp32 via a subgroup MMA
            // (coopMatMulAdd), and writes FLAT fp32 results to the scratch
            // buffer (binding 3). A separate pack pass (pack_pipeline_ below)
            // then repacks adjacent flat element pairs into packed half2 words
            // — kept as a distinct flat pass because when N is odd one output
            // word straddles two rows (element 2w+1 of word w is the next row's
            // col 0), and in-shader (col, col+1) packing would zero-fill the
            // straddling half and clobber the next row's first element. The
            // pipeline's descriptor set declares 7 bindings to match the
            // quantized-weight layouts: only 0-3 are read, 4-5 are
            // bound-but-unused (filled from the same objs_ vector as every
            // other MatMul pipeline).
            //
            // Which tile the kernel declares is probed, not assumed: the KHR
            // property list is the only source of truth for whether an MMA
            // shape works, and a shape the device does not list is NOT a safe
            // fallback -- NVIDIA Ada validates the 8x8x16 pipeline and then
            // makes coopMatMulAdd return zeros (all-zero logits, no error).
            // build_coop_pipeline() walks kFp16Modules largest-tile first,
            // keeps only shapes the device lists, and skips a module whose
            // bytes this archive lacks or whose pipeline the driver rejects.
            // No listed shape -> no coopmat path, and the tiled/ksplit
            // kernels below take over.
            int bm = 0;
            int bn = 0;
            bool tile16 = false;
            if (auto pipe = build_coop_pipeline(kFp16Modules, false, cache, bm,
                                                bn, tile16)) {
                coop_pipeline_ = std::move(pipe);
                coop_bm_ = bm;
                coop_bn_ = bn;
                coop_acc_geo_ = tile16;
                for (auto &ds : coop_ds_) {
                    ds = coop_pipeline_->allocDescriptorSets();
                }
            }
        }
        if (fp16_ != 0 && !coop_i8_pipeline_ &&
            dev->is_support_cooperate_matrix()) {
            // W8A8 cooperative-matrix kernel: int8 weight bytes x an int8
            // activation the kernel quantizes itself (per output row, see the
            // shader header), accumulated in int32 by the SINT8 subgroup MMA.
            // Same probe as the fp16 side, against kI8Modules: only a tile the
            // device lists is built. That distinction is load-bearing -- Ada
            // lists 16x16x32 and not 8x8x32, and the ARL 8x8x32 kernel it
            // used to get compiles there and returns zeros.
            int bm = 0;
            int bn = 0;
            bool tile16 = false;
            if (auto pipe = build_coop_pipeline(kI8Modules, true, cache, bm, bn,
                                                tile16)) {
                coop_i8_pipeline_ = std::move(pipe);
                coop_i8_bm_ = bm;
                coop_i8_bn_ = bn;
                for (auto &ds : coop_i8_ds_) {
                    ds = coop_i8_pipeline_->allocDescriptorSets();
                }
            }
        }
        if (fp16_ != 0 && !pack_pipeline_) {
            // Pack pass: reads flat fp32 from scratch, packs half2 to output.
            // Separate pipeline to avoid Intel ANV push-constant interference
            // between the coopmat dispatch and the pack dispatch.
            bool use_uab = update_after_bind_ &&
                           dev->is_support_descriptor_update_after_bind();
            // Same binding count as the reduce pipeline: the pack set is filled
            // straight from objs_, so the two layouts must have the same slots
            // even though the pack shader only reads scratch and writes out.
            auto pack = resolve_spv("buffer_matmul_pack");
            pack_pipeline_ = std::make_unique<VulkanPipeline>(
                dev->getLogicalDevice(),
                std::vector<VkDescriptorType>{
                    DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                    DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                    DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                    DESCRIPTOR_TYPE_STORAGE},
                sizeof(MatMulPackPC), pack.first, static_cast<int>(pack.second),
                use_uab, 0, cache);
            for (auto &ds : pack_ds_) {
                ds = pack_pipeline_->allocDescriptorSets();
            }
        }
        // SK_LASTBLOCK pipelines: per-cs precompiled variants (SK_CS baked in)
        // avoid glslc's register-allocator bug. Try cs=2,4,8 first, then the
        // default (SK_CS=1, full-K path) as fallback. Each variant has both
        // fp32 and half2 epilogue builds; half2 only used when N is even.
        if (fp16_ != 0 && !coop_disabled && dev->is_support_cooperate_matrix() &&
            sklast_fp32_pipes_.empty()) {
            bool use_uab = update_after_bind_ &&
                           dev->is_support_descriptor_update_after_bind();
            std::vector<VkDescriptorType> sklast_types{
                DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                DESCRIPTOR_TYPE_STORAGE};
            // List of (cs, fp32_key, half2_key) to try. cs=0 means default.
            struct SkLastKey { int cs; const char *fp32; const char *half2; };
            SkLastKey keys[] = {
                { 2, "buffer_matmul_coop16_sklast_fp32_cs2",
                    "buffer_matmul_coop16_sklast_cs2" },
                { 4, "buffer_matmul_coop16_sklast_fp32_cs4",
                    "buffer_matmul_coop16_sklast_cs4" },
                { 8, "buffer_matmul_coop16_sklast_fp32_cs8",
                    "buffer_matmul_coop16_sklast_cs8" },
                { 1, "buffer_matmul_coop16_sklast_fp32",
                    "buffer_matmul_coop16_sklast" },
            };
            for (auto &k : keys) {
                // Try fp32 first (always usable)
                try {
                    auto spv = resolve_spv(k.fp32);
                    sklast_fp32_pipes_[k.cs] = std::make_unique<VulkanPipeline>(
                        dev->getLogicalDevice(), sklast_types,
                        sizeof(matmul::GpuMatMulParam), spv.first,
                        static_cast<int>(spv.second), use_uab, 0, cache);
                    std::fprintf(stderr, "[vkop] sklast_fp32 cs=%d pipeline init OK\n", k.cs);
                } catch (const std::exception &e) {
                    // fp32 variant unavailable — that cs value won't be usable
                    std::fprintf(stderr, "[vkop] sklast_fp32 cs=%d not found (%s)\n",
                                 k.cs, e.what());
                    continue;
                }
                // Try half2 variant (bonus)
                try {
                    auto spv = resolve_spv(k.half2);
                    sklast_half2_pipes_[k.cs] = std::make_unique<VulkanPipeline>(
                        dev->getLogicalDevice(), sklast_types,
                        sizeof(matmul::GpuMatMulParam), spv.first,
                        static_cast<int>(spv.second), use_uab, 0, cache);
                    std::fprintf(stderr, "[vkop] sklast_half2 cs=%d pipeline init OK\n", k.cs);
                } catch (...) {
                    // half2 optional — only fp32 needed
                }
            }
            // Alloc descriptor sets from the first available fp32 pipeline
            // (all share the same 7-binding layout).
            for (auto &ds : sklast_fp32_ds_) ds = nullptr;
            for (auto &ds : sklast_half2_ds_) ds = nullptr;
            if (!sklast_fp32_pipes_.empty()) {
                auto &first_fp32 = sklast_fp32_pipes_.begin()->second;
                for (auto &ds : sklast_fp32_ds_) {
                    ds = first_fp32->allocDescriptorSets();
                }
            }
            if (!sklast_half2_pipes_.empty()) {
                auto &first_h2 = sklast_half2_pipes_.begin()->second;
                for (auto &ds : sklast_half2_ds_) {
                    ds = first_h2->allocDescriptorSets();
                }
            }
        }
    }

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("transB") != attributes.end()) {
            transB_ = std::stol(attributes.at("transB")) != 0;
        }
        if (attributes.find("epi_scale") != attributes.end()) {
            epi_scale_ = std::stof(attributes.at("epi_scale"));
        }
        if (attributes.find("epi_bias") != attributes.end()) {
            epi_bias_ = std::stol(attributes.at("epi_bias")) != 0;
        }
        if (attributes.find("attn_softmax") != attributes.end()) {
            attn_softmax_ = std::stol(attributes.at("attn_softmax")) != 0;
        }
        auto da = attributes.find("down_act");
        if (da != attributes.end() && !da->second.empty()) {
            // The converter writes this attribute as a bare integer rather
            // than the bracketed list parse_attr_list wants: a 1-element ints
            // list stores as the scalar "1", and a bare list parses to empty
            // instead of throwing. Accept both spellings so an exporter node
            // and a hand-built test node agree.
            const std::string &raw = da->second;
            auto list = parse_attr_list<int>(raw);
            down_act_ = !list.empty() ? list[0] != 0 : std::stol(raw) != 0;
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
        // Element formats. A and the output must be float or half; B may also
        // be a quantized weight, in which case the graph carries its dequant
        // scale as the last input — the same appended-scale convention the
        // optimizer and Conv2d use:
        //   fp32/fp16 : [A, B]
        //   int8/fp8  : [A, B_byte, scale(N)]             one scale per column
        //   int4 / nf4 : [A, B_4bit, scale(n_groups*N)]  one per (K group,
        //   column)
        //   nvfp4      : [A, B_4bit, block_scale, global_scale], block_scale
        //   being one fp8 E4M3 byte per (16-value K block, column) and
        //   global_scale one fp32 for the whole tensor — the format's two scale
        //   levels, both carried as tensors so neither has to be read back.
        // Everything else fails here instead of being handed to a float loader,
        // which would read a quantized payload's bytes as exponents and return
        // a plausible wrong answer.

        // Weight-residency hit-rate profiling (VKOP_WEIGHT_PROF). Signalled
        // here, at the top of the launcher, rather than at the bind site: under
        // graph submit (the default) Operator::onExecute runs this via its
        // graph_cmd_ branch, which clears objs_ and never repopulates it with a
        // weight we could read back. inputs[1] is the B operand on every path
        // (fp32/fp16 weights and quantized payloads alike; the dequant scales
        // follow it), so it is the one place a weight is visible to both modes.
        if (bind_prof_enabled_ && bind_prof_hook_ && inputs.size() > 1 &&
            inputs[1]) {
            bind_prof_hook_({nullptr, nullptr, inputs[1]->gpu_buffer()});
        }

        const core::ElemKind bkind = inputs.size() > 1
                                         ? inputs[1]->elem_kind()
                                         : core::ElemKind::kInvalid;
        const bool weight_int8 = bkind == core::ElemKind::kInt8;
        const bool weight_fp8 = bkind == core::ElemKind::kFloat8E4M3FN ||
                                bkind == core::ElemKind::kFloat8E5M2;
        // int8 and fp8 are one byte per value and scale per column, so they
        // share every gate, every kernel and the scale-length check below.
        const bool weight_byte = weight_int8 || weight_fp8;
        const bool weight_nf4 = bkind == core::ElemKind::kNF4;
        const bool weight_int4 = bkind == core::ElemKind::kInt4;
        const bool weight_nvfp4 = bkind == core::ElemKind::kFloat4E2M1;
        // An unsigned 4-bit weight: the same packing, the same per-group fp32
        // scale table and the same kernels as int4, with the nibble read as
        // [0, 15] instead of [-8, 7]. Nothing about the geometry changes, so it
        // joins weight_4bit and rides every gate below unchanged.
        const bool weight_uint4 = bkind == core::ElemKind::kUint4;
        // NVFP4 packs its nibbles exactly like int4 and nf4 and walks K in
        // groups like both, so it shares those gates; what differs is the
        // nibble's meaning and a two-level, partly fp8 scale table — checked
        // separately below rather than folded into the shared ones.
        const bool weight_4bit =
            weight_nf4 || weight_int4 || weight_nvfp4 || weight_uint4;
        core::require_float_elem(inputs[0]->elem_kind(), "MatMul", "input 0");
        if (!weight_byte && !weight_4bit) {
            core::require_float_elem(inputs[1]->elem_kind(), "MatMul",
                                     "input 1");
        }
        core::require_float_elem(outputs[0]->elem_kind(), "MatMul", "output");
        const std::string weight_label = weight_int8    ? "int8"
                                         : weight_fp8   ? "fp8"
                                         : weight_nvfp4 ? "nvfp4"
                                         : weight_uint4 ? "uint4"
                                         : weight_4bit  ? "4-bit"
                                                        : "fp16";
        size_t scale_index = 0;
        size_t global_index = 0;
        // The SwiGLU fold carries THREE weights in one MatMul, so a quantized
        // fold is [x, Wd, Wg, Wu, scale] -- five inputs, with one fp32 table
        // holding all three per-column scales (the [Sd | Sg | Su] layout is
        // documented in the DOWN_ACT module header of shaders/buffer/matmul.comp
        // and built by the converter). That shape is accepted here so the
        // plain quantized gate below (exactly [A, B, scale]) stays strict; the
        // table's LENGTH is checked in the fold block, where n / n_up are known.
        const bool down_act_byte = down_act_ && weight_byte && inputs.size() == 5;
        if (weight_byte || weight_4bit) {
            const size_t want = weight_nvfp4 ? 4 : 3;
            if (down_act_byte) {
                if (inputs[4]->elem_kind() != core::ElemKind::kFloat32) {
                    throw std::runtime_error(
                        std::string("vkop: MatMul down_act ") + weight_label +
                        " scale must be float32, got " +
                        core::elem_name(inputs[4]->elem_kind()));
                }
                scale_index = 4;
            } else if (inputs.size() != want) {
                throw std::runtime_error(
                    std::string("vkop: MatMul with a ") + weight_label +
                    " weight needs exactly " +
                    (weight_nvfp4 ? "[A, B, block_scale, global_scale]"
                                  : "[A, B, scale]") +
                    ", got " + std::to_string(inputs.size()) + " inputs");
            }
            if (down_act_byte) {
                // Validated above; nothing else to check until the shapes are.
            } else if (weight_nvfp4) {
                // A block scale that is not an e4m3 byte is not NVFP4: the
                // format's whole point is that 16-value groups cost one byte
                // instead of four.
                if (inputs[2]->elem_kind() != core::ElemKind::kFloat8E4M3FN) {
                    throw std::runtime_error(
                        std::string("vkop: MatMul nvfp4 block scale must be "
                                    "float8e4m3fn, got ") +
                        core::elem_name(inputs[2]->elem_kind()));
                }
                if (inputs[3]->elem_kind() != core::ElemKind::kFloat32) {
                    throw std::runtime_error(
                        std::string("vkop: MatMul nvfp4 global scale must be "
                                    "float32, got ") +
                        core::elem_name(inputs[3]->elem_kind()));
                }
                if (inputs[3]->size() !=
                    core::elem_bytes(core::ElemKind::kFloat32, 1)) {
                    throw std::runtime_error(
                        "vkop: MatMul nvfp4 global scale is " +
                        std::to_string(inputs[3]->size()) +
                        " bytes, not the single fp32 the format defines");
                }
                global_index = 3;
            } else if (inputs[2]->elem_kind() != core::ElemKind::kFloat32) {
                throw std::runtime_error(
                    std::string("vkop: MatMul ") + weight_label +
                    " dequant scale must be float32, got " +
                    core::elem_name(inputs[2]->elem_kind()));
            }
            if (!down_act_byte) {
                scale_index = 2;
            }
        }
        auto shape_a = inputs[0]->getShape();
        auto shape_b = inputs[1]->getShape();
        const int rank_a = static_cast<int>(shape_a.size());
        const int rank_b = static_cast<int>(shape_b.size());

        // ONNX MatMul allows 1-D operands: a rank-1 A is a single row [K]
        // (promoted to [1, K]) and a rank-1 B is a single column [K] (promoted
        // to [K, 1]); the promoted dim is then dropped from the output shape
        // again. This used to `return` for rank < 2, silently leaving the
        // output buffer unwritten — a bias-vector or single-row matmul came out
        // as zeros. The kernels only ever see a flat [batch, M, K] x [batch, K,
        // N], so the promotion is pure metadata and M=1 / N=1 are already
        // first-class (the ksplit GEMV and the reduce path both handle them).
        std::vector<int> a_eff = shape_a;
        std::vector<int> b_eff = shape_b;
        if (rank_a < 2)
            a_eff = rank_a == 1 ? std::vector<int>{1, shape_a[0]}
                                : std::vector<int>{1, 1};
        if (rank_b < 2)
            b_eff = rank_b == 1 ? std::vector<int>{shape_b[0], 1}
                                : std::vector<int>{1, 1};
        const int ra = static_cast<int>(a_eff.size());
        const int rb = static_cast<int>(b_eff.size());

        // ONNX MatMul: A is [..., M, K], B is [..., K, N]. The leading
        // ("batch") dims are broadcast (right-aligned). The reduce shader works
        // on a flat [batch, M, K] / [batch, K, N] layout where `batch` is the
        // product of the leading broadcast dims, so we still compute that
        // scalar product for the dispatch/total. BUT the output *shape* must
        // preserve the un-collapsed leading dims: collapsing to {batch, m, n}
        // loses the rank and breaks downstream shape-meta ops that carry a
        // multi-D view (e.g. a Transpose with a 4-D perm reading a 3-D MatMul
        // output does an OOB shape read -> 0 -> corrupts the whole
        // rotary/attention chain). The output SSBO is still a flat batch*m*n
        // buffer — only the reported logical shape (rank) carries the leading
        // dims. The runtime guarantees both operands share the same
        // leading-broadcast shape by the time MatMul runs (graph shape
        // inference / the preceding Expand materializes it), so the per-dim max
        // is exact and product(batch) is exact.
        int m = a_eff[ra - 2];
        int k = a_eff[ra - 1];
        // transB: B is laid out as [batch, N, K] (a Transpose(perm=[..,last2
        // swapped]) was folded into this MatMul). N comes from B's 2nd-to-last
        // dim instead of the last; K from B's last dim must equal A's K. A
        // promoted rank-1 B is [K, 1] and has no transposed interpretation —
        // the Transpose that sets transB necessarily had rank >= 2.
        if (transB_ && rank_b < 2) {
            throw std::runtime_error(
                "vkop: MatMul with a rank-" + std::to_string(rank_b) +
                " B cannot carry transB (the folded Transpose had rank >= 2)");
        }
        int n = transB_ ? b_eff[rb - 2] : b_eff[rb - 1];

        int batch = 1;
        int lead_a = ra - 2;
        int lead_b = rb - 2;
        int lead = std::max(lead_a, lead_b);
        // Leading broadcast dims in natural (left-to-right) order, taking the
        // per-axis max (ONNX broadcast: a dim of 1 matches the other's value).
        // Built left-to-right by indexing from the front of each operand's
        // leading dims.
        std::vector<int> lead_shape;
        lead_shape.reserve(lead);
        // Grouped leading-axis detection (repeat_kv elimination): after the
        // converter folds the Expand(Unsqueeze(KV)) broadcast into this MatMul,
        // exactly one leading axis of A is `rep` consecutive copies of B's
        // axis. Detected per execute, so a growing kv dim re-fires every decode
        // round. Strict on purpose: one axis with db > 1 && da == db * rep and
        // every other axis exactly equal; anything else keeps the plain
        // broadcast behavior (the db > 1 guard also keeps a 2-D weight's
        // virtual leading axes — db reads as 1 past B's rank — from posing as
        // rep copies, and a quantized weight never qualifies at all). Every
        // buffer kernel that can take rep > 1 maps the flat batch through
        // bmap() -- matmul.comp and the three coop shaders; matmul_coop_i8
        // additionally requires transB == 0.
        int rep = 1;
        int rep_stride = 1;
        int a_dim = 0;
        int b_dim = 0;
        int b_shared = 0;
        int grouped_axis = -1;
        bool grouped_ok = !weight_byte && !weight_4bit;
        for (int i = 0; i < lead; ++i) {
            int da = (i < lead_a) ? a_eff[i] : 1;
            int db = (i < lead_b) ? b_eff[i] : 1;
            if (grouped_ok && da != db) {
                if (grouped_axis < 0 && db > 1 && da > db && da % db == 0) {
                    grouped_axis = i;
                    a_dim = da;
                    b_dim = db;
                } else {
                    grouped_ok = false;
                }
            }
            int dmax = std::max(da, db);
            batch *= dmax;
            lead_shape.push_back(dmax);
        }
        if (grouped_axis >= 0) {
            rep = a_dim / b_dim;
            for (int i = grouped_axis + 1; i < lead; ++i) {
                rep_stride *= lead_shape[i];
            }
        } else if (batch > 1) {
            // Broadcast B (every leading dim of B is 1, or B has no leading
            // dims at all -- the plain 2-D weight): all `batch` output slices
            // read the SAME weight, but the kernels map B's slice as
            // bmap(flat_batch) * K * N (buffer) / b * K * N (coop), which for
            // batch index >= 1 walks off the [K, N] weight into whatever
            // follows it (a fresh buffer reads zeros; in the unified weight
            // region it reads the next tensor's bytes) and every batch >= 1
            // output row comes out wrong. b_shared folds B's slice index to 0
            // in every kernel -- bmap()'s early-out on the buffer path, the B
            // base check in the coop shaders. Unlike the grouped axis above
            // it moves no addressing but B's slice base, so the rep fields
            // stay at their defaults and every kernel's bmap() early-out
            // handles it like a per-batch 3-D B would.
            long long b_flat_batch = 1;
            for (int i = 0; i < lead_b; ++i) {
                b_flat_batch *= b_eff[i];
            }
            if (b_flat_batch == 1) {
                b_shared = 1;
            }
        }

        // ONNX drops the output dim contributed by a rank-1 operand; when both
        // are rank-1 the result is a scalar (kept as a 1-element vector, the
        // form the converter materializes scalars in).
        std::vector<int> out_shape = lead_shape;
        if (rank_a >= 2)
            out_shape.push_back(m);
        if (rank_b >= 2)
            out_shape.push_back(n);
        if (out_shape.empty())
            out_shape = {1};

        int total = batch * m * n;

        if (weight_byte && !down_act_byte) {
            // The scale indexes output columns, so its length IS N. A mismatch
            // means the weight was quantized along the wrong axis (a [N, K]
            // weight treated as [K, N], which is what a folded Transpose
            // produces) and no kernel can recover the intended values. size()
            // is bytes and the scale was just proven float32, so 4 bytes per
            // entry.
            const size_t entries =
                inputs[scale_index]->size() /
                core::elem_bytes(core::ElemKind::kFloat32, 1);
            if (entries != static_cast<size_t>(n)) {
                throw std::runtime_error(
                    "vkop: MatMul " + weight_label + " scale has " +
                    std::to_string(entries) + " entries but the output has " +
                    std::to_string(n) +
                    " columns — the weight was quantized along the wrong axis");
            }
            // No batch check: the converter only quantizes a single 2-D weight,
            // so a batched B here is that matrix broadcast (identical slices),
            // and one scale per column stays correct for every batch index.
        }

        // With a 4-bit weight the scale table is [n_groups, N] fp32 — one
        // absmax per output column per slice of K — so its length has to be a
        // whole number of N-rows, and K a whole number of those slices. Both
        // derive the group length the kernels walk; anything else means the
        // weight was grouped along the wrong axis or with a group size that
        // does not divide K, which no kernel can undo.
        int group_size = k;
        if (weight_4bit) {
            // int4 and nf4 carry an fp32 scale per (group, column); NVFP4
            // carries one e4m3 BYTE per (block, column), so its table's byte
            // count is its entry count. Both are [rows, N] row-major, which is
            // what keeps a thread's four columns contiguous either way.
            const size_t entry_bytes =
                weight_nvfp4 ? 1u
                             : core::elem_bytes(core::ElemKind::kFloat32, 1);
            const size_t entries = inputs[scale_index]->size() / entry_bytes;
            const std::string what = weight_nvfp4   ? "nvfp4"
                                     : weight_nf4   ? "nf4"
                                     : weight_uint4 ? "uint4"
                                                    : "int4";
            if (n == 0 || entries % static_cast<size_t>(n) != 0) {
                throw std::runtime_error("vkop: MatMul " + what +
                                         " scale has " +
                                         std::to_string(entries) +
                                         " entries, not a whole number of rows "
                                         "of " +
                                         std::to_string(n) + " columns");
            }
            const int n_groups = static_cast<int>(entries / n);
            if (n_groups == 0 || k % n_groups != 0) {
                throw std::runtime_error(
                    "vkop: MatMul " + what + " scale has " +
                    std::to_string(n_groups) + " K groups but K is " +
                    std::to_string(k));
            }
            group_size = k / n_groups;
            // A 16-value block is part of what NVFP4 *is*: its scale byte is
            // the block's absmax over 6. Any other grouping is a different
            // format, and decoding it as this one would apply each scale to the
            // wrong span of values.
            if (weight_nvfp4 && group_size != 16) {
                throw std::runtime_error(
                    "vkop: MatMul nvfp4 blocks 16 values of K, but " +
                    std::to_string(n_groups) + " scale rows over K = " +
                    std::to_string(k) + " say " + std::to_string(group_size));
            }
            // transB: a [N, K] weight would put a column's nibbles side by side
            // along K instead of across columns, which is a different
            // addressing (and a different word-alignment story) than the
            // kernels below.
            if (transB_) {
                throw std::runtime_error(
                    "vkop: MatMul " + what +
                    " weight with transB is not supported "
                    "(only the [K, N] layout is unpacked)");
            }
            // N % 8: the packed row must start on a 32-bit word boundary, since
            // one word is the unit of addressable weight data. It also makes
            // the group scale rows contiguous for a thread's four columns.
            if ((n % 8) != 0) {
                throw std::runtime_error(
                    "vkop: MatMul " + what +
                    " needs N % 8 == 0, got N = " + std::to_string(n));
            }
            // Even group: the fp16 kernels read A two halves per word, so a
            // group's taps have to come in pairs.
            if ((group_size % 2) != 0) {
                throw std::runtime_error("vkop: MatMul " + what +
                                         " needs an even group size, got " +
                                         std::to_string(group_size));
            }
        }

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->size() == 0) {
                // Fresh/empty output (recorded shape had 0 sentinels): allocate
                // at the concrete flat size.
                output->resize(out_shape);
            } else if (output->num_elements() == total) {
                // Recycled buffer with the right element count but possibly the
                // wrong rank (e.g. collapsed to {batch,m,n} on a prior round):
                // metadata-only reshape — no GPU realloc. The reshape_view
                // element-count guard is a backstop that silently skips on any
                // mismatch.
                output->reshape_view(out_shape);
            } else {
                // Element count changed (growing kv_len across decode rounds,
                // etc.): reallocate.
                output->resize(out_shape);
            }
            bind_ssbo<T>(outputs[0], /*is_output=*/true);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            bind_ssbo<T>(inputs[0], /*is_output=*/false);
        });
        dispatch_by_dtype(inputs[1]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            bind_ssbo<T>(inputs[1], /*is_output=*/false);
        });

        // fp16 needs a scratch fp32 buffer (binding 3) only for the odd-N
        // reduce->pack fallback; even N packs in the reduce shader itself.
        // int8 weights use that same even-N single pass: the fused path's word
        // alignment comes from the OUTPUT (N even), while byte packing only
        // changes how B is loaded. Sending int8 through reduce->pack anyway
        // cost a second dispatch per GEMM (~170 per decode token) and an fp32
        // scratch round trip for no benefit.
        const bool fused_fp16 = (fp16_ != 0) && ((n & 1) == 0);
        // ------------------------------------------------------------------
        // Kernel routing. core/KernelSelect owns the policy: one ordered rule
        // table keyed on (device class, shape, weight kind, epilogue flags),
        // first match wins. The predicates that used to live here are rows in
        // that table; everything below is dispatch.
        //
        // The fold's shape mapping is not the plain MatMul's: this node's k is
        // x's true width (K_out), while the down GEMM's contraction is Wd's row
        // count (N_up), read from B. The selector's down_act row wants both,
        // and the dispatch below reuses them.
        int down_k_out = 0;
        int down_n_up = -1;
        // Two weight flavours reach the fold: fp16 [x, Wd, Wg, Wu] (four inputs)
        // and a byte-quantized [x, Wd, Wg, Wu, scale] (five inputs, the single
        // concatenated per-column table at binding 4). Anything else is not a
        // folded node at all.
        const bool down_inputs_ok = down_act_ &&
                                    (inputs.size() == 4 || down_act_byte) &&
                                    rank_a >= 2 && rank_b >= 2;
        if (down_inputs_ok) {
            // Wg/Wu are [K_out, N_up] (plain row-major) or lane-major
            // [K_out, ceil(N_up/8), 8], which optimizer.py's
            // fold_swiglu_down_proj emits so a lane's eight columns are one
            // 16-byte word. K_out is the LEADING dim in both layouts: reading
            // shape[rank - 2] returns the band count (N_up/8, i.e. 768) for the
            // 3-D form and silently shrank the fold's contraction to the first
            // 768 K. Wd is [N_up, N] either way, so its row count is N_up.
            const auto &gated_shape = inputs[2]->getShape();
            const auto &wd_shape = inputs[1]->getShape();
            down_k_out = gated_shape.size() >= 2
                             ? static_cast<int>(gated_shape[0])
                             : 0;
            down_n_up = wd_shape.size() >= 2
                            ? static_cast<int>(wd_shape[0])
                            : -1;
            // x's last dim IS the fold's K_out (see the shape mapping above).
            // A gate/up weight that disagrees means the layout read here and
            // the operator's shape mapping have diverged, so refuse the fold
            // rather than emit a silently truncated down_proj.
            if (gated_shape.size() >= 2 && down_k_out != k) {
                throw std::runtime_error(
                    "vkop: MatMul down_act gate/up weight of rank " +
                    std::to_string(gated_shape.size()) + " reports K_out " +
                    std::to_string(down_k_out) + " but x contracts " +
                    std::to_string(k) + " (unexpected weight layout)");
            }
            // The fold kernels address Wg/Wu with the up width as their row
            // stride: uPC.K = n_up and matmul_down_act.comp's da_b2/da_ba read
            // (row * n_up + column). The converter's lane-major layout rounds
            // each 8-column group up to a whole 16-byte word, so with
            // n_up % 8 != 0 the declared row width is pad = ceil(n_up/8)*8, not
            // n_up -- host and shader would disagree about the stride and read
            // the wrong columns. No layout reconciles that today, so refuse the
            // fold instead of emitting a silently wrong down_proj.
            if (down_n_up >= 1 && (down_n_up % 8) != 0) {
                throw std::runtime_error(
                    "vkop: MatMul down_act fold needs N_up % 8 == 0, got N_up = " +
                    std::to_string(down_n_up) +
                    " (lane-major rows would be padded past n_up)");
            }
        }

        core::MatmulShape sel_shape;
        sel_shape.batch = batch;
        sel_shape.m = m;
        sel_shape.k = k;
        sel_shape.n = n;
        sel_shape.n_up = down_n_up;
        core::MatmulFeatures sel_feat;
        sel_feat.fp16 = fp16_ != 0;
        sel_feat.transB = transB_;
        sel_feat.weight = weight_int8    ? core::WeightKind::kInt8
                          : weight_fp8   ? core::WeightKind::kFp8
                          : weight_nvfp4 ? core::WeightKind::kNvfp4
                          : weight_4bit  ? core::WeightKind::kInt4
                                         : core::WeightKind::kFp16;
        sel_feat.group_size = weight_4bit ? group_size : 0;
        sel_feat.epi_bias = epi_bias_;
        sel_feat.attn_softmax = attn_softmax_;
        // The fold's structural half (the 4-input rewrite and the da_ok
        // geometry) is not expressible as a shape, so fold it into the flag.
        sel_feat.down_act = down_inputs_ok && down_k_out >= 1 && (n % 4 == 0) &&
                            n <= 2048 && (down_n_up % 4 == 0);

        // Modules set_runtime_device() already built are facts. The ones built
        // on first use (WIDE GEMV, 512-lane split-K, tile-split-K, the down
        // fold) start optimistic and are probed only when the selected route
        // needs them; a refusal clears that entry and re-runs the table, which
        // is the fall-through the old cascade wrote by hand.
        core::MatmulAvailability sel_av;
        sel_av.coopmat = coop_pipeline_ != nullptr;
        sel_av.coop_bm = coop_bm_;
        sel_av.coop_bn = coop_bn_;
        sel_av.coopmat_i8 = coop_i8_pipeline_ != nullptr;
        sel_av.ksplit512 = true;
        sel_av.w8_wide = true;
        sel_av.w8_wide512 = true;
        sel_av.tile_ks_split = true;
        sel_av.tile_ks_fold = true;
        sel_av.down_act = true;

        // Process-wide: the env knobs are resolved once, by core/KernelSelect.
        const core::MatmulKnobs &sel_knobs = core::matmul_knobs();
        VulkanPipeline *sel_wide = nullptr;   // W8 WIDE (either build)
        VulkanPipeline *sel_ks512 = nullptr;  // 512-lane split-K
        VulkanPipeline *sel_ks_split = nullptr;
        VulkanPipeline *sel_ks_fold = nullptr;
        VulkanPipeline *sel_down_act = nullptr;

        core::MatmulPlan plan = core::select_matmul(
            m_dev_->device_traits(), sel_shape, sel_feat, sel_av, sel_knobs);
        for (int attempt = 0; attempt < 8; ++attempt) {
            bool retry = false;
            switch (plan.route) {
            case core::MatmulRoute::kKsplitW8Wide512:
                sel_wide = w8_wide_pipeline();
                if (sel_wide == nullptr) {
                    sel_av.w8_wide = false;
                    retry = true;
                    break;
                }
                if (VulkanPipeline *p = secondary_pipeline(
                        1, "buffer_matmul_ksplit512_w8_wide_fp16",
                        w8_wide512_pipeline_, w8_wide512_ds_)) {
                    sel_wide = p;
                } else {
                    sel_av.w8_wide512 = false;
                    retry = true;
                }
                break;
            case core::MatmulRoute::kKsplitW8Wide:
                sel_wide = w8_wide_pipeline();
                if (sel_wide == nullptr) {
                    sel_av.w8_wide = false;
                    retry = true;
                }
                break;
            case core::MatmulRoute::kKsplitWide512:
            case core::MatmulRoute::kKsplit512:
                sel_ks512 = secondary_pipeline(0,
                                               "buffer_matmul_ksplit512_fp16",
                                               ksplit512_pipeline_,
                                               ksplit512_ds_);
                if (sel_ks512 == nullptr) {
                    sel_av.ksplit512 = false;
                    retry = true;
                }
                break;
            case core::MatmulRoute::kTiledKs:
                sel_ks_split = ks_split_pipeline();
                if (sel_ks_split == nullptr) {
                    sel_av.tile_ks_split = false;
                    retry = true;
                    break;
                }
                sel_ks_fold = ks_fold_pipeline();
                if (sel_ks_fold == nullptr) {
                    sel_av.tile_ks_fold = false;
                    retry = true;
                }
                break;
            case core::MatmulRoute::kDownActFold:
                sel_down_act = down_act_pipeline();
                if (sel_down_act == nullptr) {
                    sel_av.down_act = false;
                    retry = true;
                } else {
                    // Build the M>1 module now, while we are still outside a
                    // command buffer. A miss only costs the fallback walk.
                    down_act_pipeline(true);
                }
                break;
            case core::MatmulRoute::kCoopmatSkLast:
                if (sklast_fp32_pipes_.empty()) {
                    sel_av.coopmat = false;
                    retry = true;
                }
                break;
            case core::MatmulRoute::kCoopmatInwg:
                // INWG not yet dispatched on the host; treat like unavailable.
                sel_av.coopmat = false;
                retry = true;
                break;
            default:
                break;
            }
            if (!retry) {
                break;
            }
            plan = core::select_matmul(m_dev_->device_traits(), sel_shape,
                                       sel_feat, sel_av, sel_knobs);
        }

        // The dispatch below reads the plan through these aliases so the
        // kernels' own flags keep their names.
        const bool tiled = plan.route == core::MatmulRoute::kTiled ||
                           plan.route == core::MatmulRoute::kTiledKs;
        const bool ksplit =
            plan.route == core::MatmulRoute::kKsplit ||
            plan.route == core::MatmulRoute::kKsplit512 ||
            plan.route == core::MatmulRoute::kKsplitWide ||
            plan.route == core::MatmulRoute::kKsplitWide512 ||
            plan.route == core::MatmulRoute::kKsplitW8Wide ||
            plan.route == core::MatmulRoute::kKsplitW8Wide512;
        const bool coopmat = plan.route == core::MatmulRoute::kCoopmat;
        const bool coopmat_i8 = plan.route == core::MatmulRoute::kCoopmatI8;
        const bool coopmat_sklast =
            plan.route == core::MatmulRoute::kCoopmatSkLast;
        const bool w8_quad =
            plan.route == core::MatmulRoute::kQuad4 && weight_byte;
        const bool w4_quad =
            plan.route == core::MatmulRoute::kQuad4 && weight_4bit;
        // The tile-split-K cut is made here, before the scratch sizing below:
        // the deposit blocks, the fold pass and this host staging must agree,
        // and the cut is a function of the shape alone. The selector returns it
        // only when both modules exist, so an archive without them keeps the
        // unsplit tile and allocates no partials.
        const int ks_n = plan.ks_n;
        // A grouped weight (rep > 1) reads bmap(b) from the shader's unscaled
        // b, and the tile-split-K kernel derives b = gid.z / ks_n -- so it
        // needs the unscaled batch. rep_stride is the product of the leading
        // dims AFTER the grouped axis, so folding it in rescales z's split
        // ring by exactly that stride. Without it, any grouped shape whose
        // grouped axis is not the last leading axis (kv-prefill attention)
        // maps every rep group past the first onto B's wrong slices.
        if (rep > 1)
            para_.rep_stride *= rep;
        // The tile-split-K cut: an even, ordered tiling of [0, K) into ks_n
        // slices. It is derived from the shape alone and lives in no
        // per-dispatch state, because THREE consumers must agree on it -- the
        // deposit blocks (which look their own slice up from gid.z), the fold
        // pass that reduces them, and this host staging that publishes the
        // table. A cut keyed on the replay lane would silently change between
        // an execute round and a cached replay round (m_id_ is not stable),
        // and the fold would then reduce a different partition than the
        // deposit wrote. Allocate the scratch size from the same numbers, so
        // the cut must be computed before the buffer sizing below.
        ks_n_ = ks_n;
        ks_k0_ = 0;
        ks_kstop_ = 0;
        ks_slices_.clear();
        if (ks_n > 1) {
            const int base = k / ks_n;
            const int rem = k - base * ks_n;
            int at = 0;
            for (int z = 0; z < ks_n; ++z) {
                const int lo = at;
                at += base + (z < rem ? 1 : 0);
                ks_slices_.push_back(lo);
                ks_slices_.push_back(at);
            }
            ks_k0_ = ks_slices_[0];
            ks_kstop_ = ks_slices_[1];
        }
        // The odd-N reduce->pack fallback needs scratch; the cooperative-matrix
        // path also needs it (it writes flat fp32 results that the pack pass
        // repacks to half2, regardless of N parity). Even-N fused/reduce paths
        // pack in-shader and bind the dummy for the 4th slot.
        // total may be 0 for a dynamic-shape output that resolved empty (a 0
        // dim). vkCreateBuffer rejects size 0 with
        // VK_ERROR_INITIALIZATION_FAILED on Intel, so clamp to a minimal
        // 16-byte dummy — the dispatch is 0 threads anyway.
        if (fp16_ != 0 &&
            (!fused_fp16 || coopmat || coopmat_i8 || coopmat_sklast) &&
            ks_n <= 1) {
            size_t scratch_bytes = static_cast<size_t>(total) * sizeof(float);
            if (scratch_bytes == 0) {
                scratch_bytes = 16;
            }
            // Grow-only pool: recreating a GPU buffer on every execute churns
            // allocations when shapes are stable across rounds.
            if (!scratch_ || scratch_bytes_ < scratch_bytes) {
                scratch_ = std::make_shared<VulkanBuffer>(
                    m_dev_, scratch_bytes,
                    STORAGE | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                scratch_bytes_ = scratch_bytes;
            }
            objs_.emplace_back(scratch_);
        } else if (ks_n > 1) {
            // Tile-split-K: the deposit pass writes one fp32 MxN plane per K
            // range, for every batch, and the fold pass walks them as ks_n
            // contiguous M*N planes per batch. Only valid rows/columns are
            // stored -- a padded 64x64 slot per output tile made the scratch
            // traffic equal the weight at M=4 and erased the parallelism win.
            // Words 0..3 are the per-op control header: the debug flag
            // register, then ks_n and the first slice's bounds (the fold pass's
            // own read). The slice table follows at KS_TABLE_OFF, then the
            // partials at KS_PART_OFF -- both word addresses, so the header,
            // the table and the tile data can never alias.
            const size_t ks_table_words = 2 * static_cast<size_t>(ks_n);
            const size_t ks_part_off = ks_table_off_ + ks_table_words;
            size_t ks_bytes =
                (ks_part_off +
                 static_cast<size_t>(batch) * static_cast<size_t>(ks_n) *
                     static_cast<size_t>(m) * static_cast<size_t>(n)) *
                sizeof(float);
            if (!scratch_ || scratch_bytes_ < ks_bytes) {
                scratch_ = std::make_shared<VulkanBuffer>(
                    m_dev_, ks_bytes,
                    STORAGE | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                scratch_bytes_ = ks_bytes;
                ks_words_.clear();
            }
            // Stage the control header. HOST_COHERENT host writes are visible
            // to the device at the next submit without an explicit flush, and
            // the dispatch reads them after any queued writes from earlier ops.
            //
            // The header and the slice table are staged HERE, at record time,
            // not at the dispatch: a cached replay round re-runs the recorded
            // commands but skips this function entirely, so a table written at
            // the dispatch site would still be the recording round's cut. The
            // cut is a function of the shape, so the two rounds agree -- but
            // the buffer must already hold it when the replayed dispatch runs.
            // Writing only on a real change keeps the cached path free.
            if (auto *words =
                    static_cast<uint32_t *>(scratch_->getMappedMemory())) {
                std::vector<uint32_t> hdr;
                hdr.reserve(ks_part_off);
                hdr.push_back(0);
                hdr.push_back(static_cast<uint32_t>(ks_n));
                hdr.push_back(static_cast<uint32_t>(ks_k0_));
                hdr.push_back(static_cast<uint32_t>(ks_kstop_));
                // The whole cut, so the deposit can find its slice from gid.z
                // and the fold can reject a set of K ranges that do not tile
                // [0, K) exactly. KS_HDR words precede it; the partials follow.
                for (int v : ks_slices_) {
                    hdr.push_back(static_cast<uint32_t>(v));
                }
                hdr.resize(ks_part_off, 0);
                if (hdr != ks_words_) {
                    ks_words_ = hdr;
                    for (size_t t = 0; t < hdr.size(); ++t) {
                        words[t] = hdr[t];
                    }
                    // HOST_COHERENT makes the bytes device-visible without a
                    // flush, but RenderDoc only reconstructs buffer contents
                    // from map/unmap/flush events — without this the captured
                    // header (and the indirect counts derived from it) replay
                    // as stale/zero bytes.
                    scratch_->flushMappedRange(0, hdr.size() * sizeof(uint32_t));
                }
            }
            objs_.emplace_back(scratch_);
        } else {
            objs_.emplace_back(dummy_buffer_);
        }

        // binding 4: the quantized weight's dequant scale table (per column for
        // int8 and fp8, per [K group, column] for the 4-bit formats), or a
        // dummy to keep the descriptor set fully bound (same convention as
        // Conv2d).
        if (weight_byte || weight_4bit) {
            dispatch_by_dtype(inputs[scale_index]->dtype(), [&](auto dummy) {
                using T = decltype(dummy);
                bind_ssbo<T>(inputs[scale_index], /*is_output=*/false);
            });
        } else {
            objs_.emplace_back(dummy_buffer_);
        }

        // binding 5: the fp32 factor that multiplies NVFP4's block scales, or a
        // dummy so the descriptor set stays fully bound for every other format.
        if (weight_nvfp4) {
            dispatch_by_dtype(inputs[global_index]->dtype(), [&](auto dummy) {
                using T = decltype(dummy);
                bind_ssbo<T>(inputs[global_index], /*is_output=*/false);
            });
        } else {
            objs_.emplace_back(dummy_buffer_);
        }

        // binding 6: the attention-bias tensor appended to a plain 2-input
        // MatMul by fuse_scale_bias_into_matmul (inputs[2]). The converter only
        // folds onto a plain [A,B] MatMul, so a quantized weight (which already
        // uses inputs[2..] for its scale table) never lands here. A dummy keeps
        // the descriptor set fully bound for every other MatMul.
        if (epi_bias_ && inputs.size() > 2 && inputs[2] != nullptr) {
            dispatch_by_dtype(inputs[2]->dtype(), [&](auto dummy) {
                using T = decltype(dummy);
                bind_ssbo<T>(inputs[2], /*is_output=*/false);
            });
        } else {
            objs_.emplace_back(dummy_buffer_);
        }

        para_.M = m;
        para_.N = n;
        para_.K = k;
        para_.C = batch;
        para_.fp32 = (fp16_ != 0) ? 0 : 1;
        para_.transB = transB_ ? 1 : 0;
        para_.tile = tiled ? 1 : 0;
        para_.weight_int8 = weight_byte ? 1 : 0;
        para_.fp8 = weight_fp8 ? 1 : 0;
        para_.fp8_e5m2 = bkind == core::ElemKind::kFloat8E5M2 ? 1 : 0;
        para_.w8_quad = (w8_quad && !ksplit && !tiled) ? 1 : 0;
        // main() tests uPC.ksplit before uPC.tile, so raising both flags makes
        // the host's tile dispatch (y = UP_DIV(m,64)) run the split-K GEMV,
        // which takes its output row from gl_WorkGroupID.y and so writes only
        // row 0 -- rows 1..m-1 keep whatever the buffer held. The two gates
        // overlap for 3 <= batch*m <= 8; there the tile is also the faster
        // kernel (m=8, K=2048 N=6144: 1.54 ms tiled vs ~3.0 ms of GEMV
        // re-reading the weight per row). The selector's tile row pre-empts the
        // split-K rows, so `ksplit` is already false whenever `tiled` is set;
        // the two flags cannot disagree.
        // SwiGLU down fold: x's true column count (see K_act above). Set for
        // every route of a down_act op; the non-fold routes read it with the
        // plain kernels' x stride, the fold route overwrites K with the up
        // width and reads it directly.
        para_.K_act = k;
        para_.ksplit = (ksplit && !tiled) ? 1 : 0;
        // para_ persists across executes and the down fold raises ks_part for
        // its fold pass without lowering it again, so a later plain split-K
        // dispatch on this module would run the fold entry in place of its
        // deposit. Clear it with every other per-dispatch constant; the fold
        // block sets its own 0/1 pair just below.
        para_.ks_part = 0;
        para_.wide = plan.wide ? 1 : 0;
        para_.ksplit_q = plan.ksplit_q;
        para_.w4 = weight_4bit ? 1 : 0;
        para_.nf4 = weight_nf4 ? 1 : 0;
        para_.nvfp4 = weight_nvfp4 ? 1 : 0;
        para_.uint4 = weight_uint4 ? 1 : 0;
        para_.group = weight_4bit ? group_size : 0;
        para_.rep = rep;
        para_.rep_stride = rep_stride;
        para_.a_dim = a_dim;
        para_.b_dim = b_dim;
        para_.b_shared = b_shared;
        // Plan C: tile-split-K hands the block its own K range through the
        // SCRATCH buffer (word 1 = ks_n, words 2/3 = the first slice, then the
        // slice table). Push constants cannot carry it: four ints at the struct
        // tail push sizeof(GpuMatMulParam) past the 128-byte limit, which
        // collapses the push_constant block to a 4-byte range and zeroes EVERY
        // constant after it -- including K, on which the tiled K window
        // depends.
        para_.epi_scale = epi_scale_;
        para_.epi_bias = epi_bias_ ? 1 : 0;
        // The bias is [B,1,q,kv] broadcast over the head axis, so its element
        // count is B*M*N while the output's is batch*M*N (batch = B*heads).
        // epi_bias_batch = B = bias_elems / (M*N); the shader divides the flat
        // batch by this to land each output row on its bias row.
        if (epi_bias_ && inputs.size() > 2 && inputs[2] != nullptr && m > 0 &&
            n > 0) {
            const size_t bias_elems =
                inputs[2]->size() / core::elem_bytes(inputs[2]->elem_kind(), 1);
            para_.epi_bias_batch =
                static_cast<int>(bias_elems / (static_cast<size_t>(m) * n));
        } else {
            para_.epi_bias_batch = 0;
        }
        // Plan B: when the host decides this MatMul is an attention QK^T whose
        // trailing Softmax can be fused, the kernel reduces each row over all N
        // columns in one workgroup and writes normalized values. The dispatch
        // is (batch*M, 1, 1) -- one workgroup per output row. The selector's
        // attn_softmax row owns the gate (fp16, even N, fp16 weight, N and K
        // bounds); this only publishes the flag the shader tests.
        para_.attn_softmax =
            plan.route == core::MatmulRoute::kAttnSoftmax ? 1 : 0;

        // Plan B: fused QK^T+scale+bias+Softmax. One workgroup per output row
        // (batch*M rows); the kernel does the K-dot, applies the epilogue, and
        // reduces each row to exp(x-max)/sum in-place, so the standalone
        // FusedElemwise + 2-dispatch Softmax all vanish. Must pre-empt coopmat
        // (which writes flat fp32 to scratch + a separate pack pass and cannot
        // do the row reduction). Gating on attn_softmax already excludes quant
        // and odd-N; this only fires on attention QK^T decode (small M).
        if (para_.attn_softmax) {
            // matmul.comp tests uPC.ksplit *before* uPC.attn_softmax, and the
            // split-K gate above is true for a wide-K attention head (K >= 256
            // with batch*M <= 8). Left raised, this grid (one workgroup per
            // output row) would run the split-K GEMV instead, which takes its
            // output row from WorkGroupID.y -- so only row 0 was written and
            // without the epilogue or the softmax. The fused path owns the
            // dispatch, so clear the flags it pre-empts.
            para_.ksplit = 0;
            set_route(plan.tag, batch, m, n, k, rep);
            submit(&para_, batch * m, 1, 1);
            return;
        }

        // Cooperative-matrix fp16 path (Intel ARL subgroup MMA): ONE dispatch
        // reads fp16 A/B, accumulates in fp32 via coopMatMulAdd, and writes
        // flat fp32 to the scratch buffer. Workgroup footprint is BM=16 (M) x
        // BN=32 (N); batch is the z dispatch dim (shader reads gl_WorkGroupID.z
        // as the batch index). The accumulator replaces the naive per-element
        // MAC loop; the scratch+pack structure is preserved for odd-N safety.
        // This is the highest-priority fp16 path — it pre-empts
        // tiled/ksplit/fused/reduce, which remain the fallback for quantized or
        // non-coopmat hardware.
        if (coopmat_i8) {
            // int8 weight x runtime-quantized int8 activation (SINT8 MMA).
            // Same scratch + pack tail as the fp16 coopmat path. The grid uses
            // the footprint of whichever SINT8 tile was probed (32x64 on Ada's
            // 16x16x32, 16x32 on ARL's 8x8x32).
            set_route(plan.tag, batch, m, n, k, rep);
            fillDescriptorWrites(coop_i8_ds_[m_id_]);
            coop_i8_pipeline_->updateDescriptorSets(ds_writes_);
            m_cmd_->bind(*coop_i8_pipeline_, coop_i8_ds_[m_id_]);
            m_cmd_->push_constants(*coop_i8_pipeline_,
                                   sizeof(matmul::GpuMatMulParam), &para_);
            dispatch_and_record(UP_DIV(n, coop_i8_bn_), UP_DIV(m, coop_i8_bm_),
                                batch, &para_);

            scratch_->shaderWriteBarrier(m_cmd_->get());

            int nwords = (total + 1) / 2;
            MatMulPackPC pack_pc{};
            pack_pc.total = total;
            fillDescriptorWrites(pack_ds_[m_id_]);
            pack_pipeline_->updateDescriptorSets(ds_writes_);
            m_cmd_->bind(*pack_pipeline_, pack_ds_[m_id_]);
            m_cmd_->push_constants(*pack_pipeline_, sizeof(MatMulPackPC),
                                   &pack_pc);
            dispatch_and_record(UP_DIV(nwords, 256), 1, 1, &pack_pc);
            return;
        }

        if (coopmat) {
            // The tile this shape measures best on, or one of the larger builds
            // of the same 16x16x16 module when the shape calls for it. A bigger
            // block halves the traffic per FLOP but divides the grid by the
            // same factor, so pick_coop_geo gates each on the shape. Only the
            // 16x16x16 tile has ACC builds, hence the tile16 guard.
            CoopPick pick = pick_coop_tile(m, n);
            VulkanPipeline *pipe = pick.pipe;
            VkDescriptorSet *ds = pick.ds;
            int bm = pick.bm;
            int bn = pick.bn;
            if (pick.tile16) {
                const int acc_mode = coop_acc_mode();
                CoopGeo *geo = nullptr;
                if (acc_mode == 0) {
                    geo = pick_coop_geo(m, n, batch);
                } else if (acc_mode != 11) {
                    for (auto &g : coop_geos_) {
                        if (g.acc == acc_mode) {
                            geo = &g;
                        }
                    }
                }
                if (geo != nullptr && coop_geo_pipeline(*geo) != nullptr) {
                    pipe = geo->pipe;
                    ds = geo->ds;
                    bm = geo->bm;
                    bn = geo->bn;
                }
            }
            set_route(plan.tag, batch, m, n, k, rep);
            fillDescriptorWrites(ds[m_id_]);
            pipe->updateDescriptorSets(ds_writes_);
            m_cmd_->bind(*pipe, ds[m_id_]);
            m_cmd_->push_constants(*pipe, sizeof(matmul::GpuMatMulParam),
                                   &para_);
            dispatch_and_record(UP_DIV(n, bn), UP_DIV(m, bm), batch, &para_);

            // Barrier: flush coopmat's scratch writes for pack's reads.
            scratch_->shaderWriteBarrier(m_cmd_->get());

            // Pack pass: separate pipeline (no PC interference). One thread per
            // output word; reads flat fp32 from scratch, packs half2 to output.
            int nwords = (total + 1) / 2;
            MatMulPackPC pack_pc{};
            pack_pc.total = total;
            fillDescriptorWrites(pack_ds_[m_id_]);
            pack_pipeline_->updateDescriptorSets(ds_writes_);
            m_cmd_->bind(*pack_pipeline_, pack_ds_[m_id_]);
            m_cmd_->push_constants(*pack_pipeline_, sizeof(MatMulPackPC),
                                   &pack_pc);
            dispatch_and_record(UP_DIV(nwords, 256), 1, 1, &pack_pc);
            return;
        }

        // -------------------------------------------------------------------
        // SK_LASTBLOCK dispatch: coopmat + grid.z parallel split-K.
        // Select pipeline by cs value — per-cs variants are precompiled with
        // SK_CS baked in to avoid glslc's register-allocator bug.
        if (coopmat_sklast && !sklast_fp32_pipes_.empty()) {
            const int cs = plan.coop_split;
            const int bm = coop_bm_;
            const int bn = coop_bn_;
            // Select the pipeline for this cs — fall back to default (key 1).
            int cs_key = cs;
            if (sklast_fp32_pipes_.find(cs_key) == sklast_fp32_pipes_.end()) {
                cs_key = 1; // default / fallback
            }
            VkDescriptorSet *ds = sklast_fp32_ds_;
            VulkanPipeline *pipe = nullptr;
            // Try half2 first (faster, direct uOutput write, if N even and available)
            if ((n & 1) == 0) {
                auto it = sklast_half2_pipes_.find(cs_key);
                if (it != sklast_half2_pipes_.end() && it->second) {
                    pipe = it->second.get();
                    ds = sklast_half2_ds_;
                }
            }
            // Fall back to fp32
            if (!pipe) {
                auto it = sklast_fp32_pipes_.find(cs_key);
                if (it != sklast_fp32_pipes_.end()) pipe = it->second.get();
            }
            if (!pipe) {
                // No usable pipeline — skip sklast dispatch
                std::fprintf(stderr, "[sklast] no pipeline for cs=%d (cs_key=%d)\n", cs, cs_key);
                pipe = nullptr;
            } else {
            set_route(plan.tag, batch, m, n, k, rep);
            para_.ksplit = cs;
            // uPartial (objs_[4]) and uCounter (objs_[5]) are now UNUSED —
            // serial in-workgroup split-K (grid.z=batch) accumulates fully
            // inside each workgroup and writes directly to scratch.
            if (objs_.size() < 7) objs_.resize(7, nullptr);
            fillDescriptorWrites(ds[m_id_]);
            pipe->updateDescriptorSets(ds_writes_);
            m_cmd_->bind(*pipe, ds[m_id_]);
            m_cmd_->push_constants(*pipe, sizeof(matmul::GpuMatMulParam), &para_);
            dispatch_and_record(UP_DIV(n, bn), UP_DIV(m, bm), batch, &para_);
            scratch_->shaderWriteBarrier(m_cmd_->get());
            int nwords = (total + 1) / 2;
            MatMulPackPC pack_pc{};
            pack_pc.total = total;
            fillDescriptorWrites(pack_ds_[m_id_]);
            pack_pipeline_->updateDescriptorSets(ds_writes_);
            m_cmd_->bind(*pack_pipeline_, pack_ds_[m_id_]);
            m_cmd_->push_constants(*pack_pipeline_, sizeof(MatMulPackPC), &pack_pc);
            dispatch_and_record(UP_DIV(nwords, 256), 1, 1, &pack_pc);
            return;
            } // end else (pipe valid)
        } // end outer coopmat_sklast if

        // -------------------------------------------------------------------
        // SwiGLU down-projection fold (down_act). The converter matched
        //     down = MatMul(Mul(Mul(gate, Sigmoid(gate)), up), Wd)
        //     gate = MatMul(x, Wg),  up = MatMul(x, Wu)
        // and rewrote down's inputs to [x, Wd, Wg, Wu] (bound 1, 2, 7, 8
        // below), deleting the gate and up nodes -- so there is no "fall back
        // to the three plain GEMMs" here: by the time this op runs they no
        // longer exist. If the module cannot run, [x, Wd] is dispatched as the
        // plain down GEMM and the output is only right for the part of the
        // graph that the folded operator was not doing; the converter's only
        // job is to not emit this tag on an archive without the key.
        //
        // The four shapes:
        //   A  = inputs[0] = x        [batch, M, K_out]  fp16
        //   B  = inputs[1] = Wd       [N_up, N] = [K, N], transB == 0
        //   B2 = inputs[2] = Wg       [K_out, N_up]   = x's K, this K
        //   B3 = inputs[3] = Wu       [K_out, N_up]   likewise
        //   S  = inputs[4] = scale    [N + 2*N_up]    fp32, byte weights only:
        //        the three per-column tables concatenated (Sd | Sg | Su), see
        //        the DOWN_ACT module header in shaders/buffer/matmul_down_act.comp.
        // NOTE the shape mapping is NOT the plain MatMul's: this node's `k`
        // (computed above from x's contraction) is K_out, the UP GEMM's
        // contraction -- x is the up projection's A operand, not the down
        // GEMM's. The down GEMM's own contraction (the kernel's K) is Wd's
        // ROW count N_up, read from Wd's shape below. So of this MatMul's own
        // (m, n, k) only (m, n) carry over to the fold's (M, N); N_up comes
        // from B, and K_out rides in K_act while Wg/Wu are read with their
        // true [K_out, N_up] layout.
        // The selector's down_act row already proved the 4/5-input rewrite, the
        // da_ok geometry and that the module is available: a refused pipeline
        // re-ran the table, so this route implies a live module.
        if (plan.route == core::MatmulRoute::kDownActFold) {
            const bool da_byte = down_act_byte;
            const int k_out = down_k_out;
            // The kernel's K is Wd's row count (N_up), which is also its
            // contraction length; this node's k is x's K_out (see above), so
            // N_up cannot come from it.
            const int n_up = down_n_up;
            // The fold GEMV's geometry (-DKSPLIT512: 512-lane blocks). Each
            // deposit block owns a contiguous run of up columns (gx = grid.x)
            // for a 16-row tile (grid.y = batch * ceil(M / 16)); the deposit
            // parks its slice partials per (row, gx, slice) and the down pass
            // sums them. GX rides to the shader in mind_k. Decode (M == 1) does
            // not halve the grid: the wide deposit maps the ks_n K slices onto
            // the block's lanes (lane/64) and gives each lane an octet of up
            // columns, so the block still covers 512 columns.
            const int da_gx = UP_DIV(n_up, 512);
            // N % 4 == 0 lets the fold lane own four columns and pack fp16
            // pairs; N <= 2048 covers the output with the block's 512 lanes
            // (2048 = 512 * 4); N_up % 4 == 0 keeps Wg/Wu's RowAlign row
            // stride equal to N_up so the [K_out, N_up] indexing is dense.
            // The da_ok gate (N % 4, N <= 2048, N_up % 4, K_out >= 1) lives in
            // the selector's down_act row now; the byte flavour owes it one more
            // check the selector cannot make -- the concatenated [Sd | Sg | Su]
            // table's length, since the kernel indexes Sg/Su at [N, N + N_up).
            if (da_byte &&
                static_cast<size_t>(inputs[4]->size()) !=
                    static_cast<size_t>(n + 2 * n_up) * sizeof(float)) {
                // A conversion-time bug, not a user-data problem: the kernel
                // indexes Sd at [0, N), Sg at [N, N + N_up) and Su at
                // [N + N_up, N + 2*N_up), so a short table would read past it.
                throw std::runtime_error(
                    "vkop: MatMul down_act int8 scale is " +
                    std::to_string(inputs[4]->size()) + " bytes, expected " +
                    std::to_string(static_cast<size_t>(n + 2 * n_up) *
                                   sizeof(float)) +
                    " (N + 2*N_up fp32)");
            }
            {
                // bind_ssbo APPENDS to objs_, so binding Wg/Wu here would land
                // them at slots 3 and 4 and leave the pipeline's 9-slot set
                // half unbound. The block below assigns every slot by index.
                para_.M = m;
                para_.N = n;
                para_.K = n_up;  // the down GEMM's contraction = the up width
                // x's TRUE column count. The kernels read x with this row
                // stride and stop the gate/up loop at it; Wg/Wu's RowAlign
                // pad rows past it are zero and simply never loaded.
                para_.K_act = k_out;
                para_.ks = 0;      // the K-split module reads no group width
                para_.ksplit = 1;  // the DOWN_ACT module's route selector
                para_.ksplit_q = 0;
                para_.mind_k = da_gx; // the deposit grid's x extent = the
                                      // fold pass's partial count
                para_.tile = 0;
                para_.ks_part = 0; // the module's deposit entry
                // Byte vs half weights. Already derived from inputs[1] above,
                // restated here because it is what selects the module's
                // deposit entry (main() tests weight_int8 after ks_part).
                para_.weight_int8 = (da_byte && weight_int8) ? 1 : 0;
                para_.fp8 = (da_byte && weight_fp8) ? 1 : 0;
                para_.fp8_e5m2 =
                    (da_byte && bkind == core::ElemKind::kFloat8E5M2) ? 1 : 0;
                set_route(plan.tag, batch, m, n, n_up, rep);
                // The K cut. K-split is what makes Wg/Wu read once per layer:
                // a deposit block walks only its slice, so the grid covers K_out
                // exactly instead of having every block re-walk it. The cut is
                // a pure function of k_out -- a cut keyed on anything that moves
                // between a record round and a replay round would have the fold
                // reduce a different partition than the deposit wrote.
                const int da_ks_n = down_act_ks_n(k_out);
                da_ks_n_ = da_ks_n;
                // The down pass's j sub-split (grid z, uPC.tile) was first
                // measured monotonically WORSE at M=1 (ds=8: 224-250 ms/step)
                // -- but that was the old shader's 16-row walk burning 15/16
                // of its FMAs on dead rows. With da_down's mt==1 fast path a
                // same-run layer%2 A/B (both arms full nodes) still favors
                // ds=1: full-node medians 1.536 (ds=1) vs 1.654 ms (ds=8).
                // 12 blocks already keep enough Wd loads in flight; the
                // sub-split only adds h-assembly barriers and 8x dp slots.
                // A WIDE-2 retry (the j split on the lane axis, 8 accs, no
                // duplicate loads) measured dead even with the tile=1 WIDE
                // walk (1.159 vs 1.164 ms median) -- the walk is at the
                // memory system's limit for this shape. ds stays 1.
                const int da_ds = 1;
                const int da_hdr = 4 + 2 * da_ks_n; // KS_HDR + the slice table
                // Each partial slot group is 512 words: the deposit parks TWO
                // fp32 per (row, gx block, slice, block-local up column) --
                // the raw gate and up slice sums -- and the block-local axis
                // is 512 wide by construction (the grid-x block size) -- not
                // N, and not N_up. The silu is nonlinear in the gate, so the
                // slices must sum BEFORE it applies: the down pass reassembles
                // h = silu(sum g_s) * sum u_s and then appends its own
                // [row][gx][column] fp32 partials, keeping the fold's Wd walk
                // at a single float per (j, column) instead of re-summing ns
                // words for every output column.
                const size_t da_up_words =
                    2 * static_cast<size_t>(batch) * m * da_gx * da_ks_n * 512;
                const size_t da_dp_words =
                    static_cast<size_t>(batch) * m * da_gx * da_ds * n;
                const size_t da_words =
                    static_cast<size_t>(da_hdr) + da_up_words + da_dp_words;
                if (!scratch_da_ || scratch_da_bytes_ != da_words) {
                    // HOST_COHERENT, not DEVICE_LOCAL: the slice table is
                    // written through the mapping below at RECORD time (a cached
                    // replay round re-runs the recorded dispatches without
                    // re-entering this function, so a table written at dispatch
                    // time would still be the recording round's). The partials
                    // live past the header in the same buffer.
                    scratch_da_ = std::make_shared<VulkanBuffer>(
                        m_dev_, da_words * sizeof(float),
                        STORAGE | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                    scratch_da_bytes_ = da_words;
                    da_words_hdr_.clear();
                }
                // Stage the header + slice table. Byte-identical every round
                // for a given shape, so the compare against da_words_hdr_
                // keeps the steady state to one vector comparison.
                {
                    std::vector<uint32_t> hdr(da_hdr, 0);
                    const int base = k_out / da_ks_n;
                    const int rem = k_out - base * da_ks_n;
                    int at = 0;
                    for (int z = 0; z < da_ks_n; ++z) {
                        const int lo = at;
                        const int hi = lo + base + (z < rem ? 1 : 0);
                        hdr[4 + 2 * z] = static_cast<uint32_t>(lo);
                        hdr[5 + 2 * z] = static_cast<uint32_t>(hi);
                        at = hi;
                    }
                    hdr[1] = static_cast<uint32_t>(da_ks_n);
                    hdr[2] = 0;
                    hdr[3] = static_cast<uint32_t>(k_out);
                    if (hdr != da_words_hdr_) {
                        if (auto *words = static_cast<uint32_t *>(
                                scratch_da_->getMappedMemory())) {
                            for (size_t t = 0; t < hdr.size(); ++t)
                                words[t] = hdr[t];
                            da_words_hdr_ = hdr;
                        }
                    }
                }
                objs_.resize(9, nullptr);
                // Binding 0 is the reduce pass's output; binding 3 is the
                // scratch the three passes share. The passes share the module,
                // so both bindings are always live and neither may double as
                // the other.
                objs_[0] = std::static_pointer_cast<VulkanResource>(
                    outputs[0]->gpu_buffer());
                objs_[1] = std::static_pointer_cast<VulkanResource>(
                    inputs[0]->gpu_buffer());
                objs_[2] = std::static_pointer_cast<VulkanResource>(
                    inputs[1]->gpu_buffer());
                objs_[3] =
                    std::static_pointer_cast<VulkanResource>(scratch_da_);
                objs_[7] = std::static_pointer_cast<VulkanResource>(
                    inputs[2]->gpu_buffer());
                objs_[8] = std::static_pointer_cast<VulkanResource>(
                    inputs[3]->gpu_buffer());
                // Binding 4 already holds the byte weight's concatenated
                // [Sd | Sg | Su] table -- or the dummy the general binding
                // pass put there for fp16 -- so the fold leaves it untouched.
                // Bindings 5/6 keep their dummies too (this module reads
                // neither the NVFP4 global factor nor the attention bias).
                // (objs_ was sized by the general binding pass; resize(9)
                // only adds the Wg/Wu slots when they are not there yet.)
                // Deposit: grid-x over the up-axis blocks, grid-y over the
                // 16-row tiles of every batch slice, grid-z over the K
                // slices -- except when the shader will take the wide
                // deposit at M==1 (it maps the slices into the block's
                // lanes, lane/64, and needs grid-z = 1). The host mirrors
                // the shader's wide gate exactly (uPC.M == 1, K % 8 == 0,
                // ks_n <= 8, 512 % ks_n == 0, byte and half forms alike): a
                // shape that fails it must dispatch all ks_n slices, or the
                // scalar form writes only slice 0 and the down pass sums
                // stale slots into h.
                const bool da_wide_ok = (m == 1) && ((n_up % 8) == 0) &&
                                        da_ks_n <= 8 && (512 % da_ks_n) == 0;
                // M==1 runs the trimmed module; M>1 the one that keeps the
                // multi-row deposit. An archive without the _multi key leaves
                // da_pipe on the M==1 module, whose per-row walk covers M>1.
                VulkanPipeline *da_pipe = sel_down_act;
                VkDescriptorSet *da_dset = down_act_ds_;
                if (m > 1) {
                    if (VulkanPipeline *p = down_act_pipeline(true)) {
                        da_pipe = p;
                        da_dset = down_act_multi_ds_;
                    }
                }
                ks_submit(da_pipe, da_dset[m_id_], da_gx,
                          batch * ((m + 15) / 16), da_wide_ok ? 1 : da_ks_n);
                // Down: sums the deposit's slice partials per (row, j) into
                // shared, walks the sub-block's Wd j-slice for every output
                // column, parks fp32 down partials keyed (row, gx*ds+z, c).
                // Grid z splits the j contraction into da_ds sub-blocks.
                // Reading another dispatch's SSBO stores needs an execution
                // dependency between the two.
                scratch_da_->shaderWriteBarrier(m_cmd_->get());
                para_.ks_part = 1; // the module's down entry
                para_.tile = da_ds;
                ks_submit(da_pipe, da_dset[m_id_], da_gx,
                          batch * ((m + 15) / 16), da_ds);
                // Reduce: sums the GX * da_ds down partials and packs fp16.
                // tile still carries da_ds for the dp stride. Reading what
                // the down pass wrote needs another dependency.
                scratch_da_->shaderWriteBarrier(m_cmd_->get());
                para_.ks_part = 2; // the module's reduce entry
                dispatch_on(da_pipe, da_dset[m_id_],
                            UP_DIV(n, 2048), batch * m);
                return;
            }
        }

        if (plan.route == core::MatmulRoute::kTiledKs) {
            // x = 64-column tiles, y = 64-row tiles, z = batch*ks_n: a block
            // never straddles a batch boundary, and each block accumulates only
            // its K slice into the scratch partials. The fold pass over
            // (n/64, m/64, batch) sums them through the shared epilogue. The
            // two halves are separate modules (-DTILED_KS_SPLIT /
            // -DTILED_KS_FOLD) because a pipeline binds one entry point, and a
            // barrier() that ordered storage across workgroups measured 1-in-8
            // wrong at ks_n=2. Both read the same slice table from the scratch
            // header, so they cannot disagree about the split.
            set_route(plan.tag, batch, m, n, k, rep);
            ks_submit(sel_ks_split, ks_split_ds_[m_id_], UP_DIV(n, 64),
                      UP_DIV(m, 64), batch * ks_n);
            // The fold reads the deposit's SSBO stores, so the pair needs an
            // explicit memory dependency between the two dispatches.
            scratch_->shaderWriteBarrier(m_cmd_->get());
            ks_submit(sel_ks_fold, ks_fold_ds_[m_id_], UP_DIV(n, 64),
                      UP_DIV(m, 64), batch);
            return;
        }
        if (plan.route == core::MatmulRoute::kTiled) {
            // x = 64-column tiles, y = 64-row tiles, z = batch: a block never
            // straddles a batch boundary, so no per-row batch fixups in the
            // shader.
            set_route(plan.tag, batch, m, n, k, rep);
            submit(&para_, UP_DIV(n, 64), UP_DIV(m, 64), batch);
            return;
        }
        if (ksplit) {
            // x = column-quad groups, y = one output row per block: the block's
            // own lanes are the K slices, so y carries the row index the shader
            // reads as gl_WorkGroupID.y. ksplit_q is the occupancy knob -- the
            // selector picks the largest column coverage whose block count
            // still clears the device profile's floor -- and the WIDE variants
            // are chosen per N by the same table.
            const int q = plan.ksplit_q;
            switch (plan.route) {
            case core::MatmulRoute::kKsplitW8Wide512:
                // Byte WIDE (16 columns/lane) in the 512-lane build.
                set_route(plan.tag, batch, m, n, k, rep);
                dispatch_on(sel_wide, w8_wide512_ds_[m_id_],
                            UP_DIV(n / 16, q), batch * m);
                return;
            case core::MatmulRoute::kKsplitW8Wide:
                set_route(plan.tag, batch, m, n, k, rep);
                dispatch_on(sel_wide, w8_wide_ds_[m_id_], UP_DIV(n / 16, q),
                            batch * m);
                return;
            case core::MatmulRoute::kKsplitWide512:
                // fp16 WIDE (8 columns/lane) in the 512-lane build.
                set_route(plan.tag, batch, m, n, k, rep);
                dispatch_on(sel_ks512, ksplit512_ds_[m_id_],
                            UP_DIV(n / 8, q), batch * m);
                return;
            case core::MatmulRoute::kKsplit512:
                set_route(plan.tag, batch, m, n, k, rep);
                dispatch_on(sel_ks512, ksplit512_ds_[m_id_],
                            UP_DIV(n / 4, q), batch * m);
                return;
            case core::MatmulRoute::kKsplitWide:
                set_route(plan.tag, batch, m, n, k, rep);
                submit(&para_, UP_DIV(n / 8, q), batch * m, 1);
                return;
            default:
                set_route(plan.tag, batch, m, n, k, rep);
                submit(&para_, UP_DIV(n / 4, q), batch * m, 1);
                return;
            }
        }
        if (fused_fp16) {
            // Single pass: x covers output WORDS (column pairs), or DWORDs for
            // the four-column quantized kernels.
            set_route(plan.tag, batch, m, n, k, rep);
            submit(&para_, UP_DIV(n / ((w8_quad || w4_quad) ? 4 : 2), 16),
                   UP_DIV(batch * m, 16), 1);
            return;
        }
        // Reduce pass: one thread per output element. Uses the main pipeline.
        // For fp16 inputs without coopmat this is the fp16-in reduce shader;
        // for fp32 inputs this is the naive fp32 reduce (batch*m collapsed into
        // y).
        set_route(plan.tag, batch, m, n, k, rep);
        submit(&para_, UP_DIV(n, 16), UP_DIV(batch * m, 16), 1);

        if (fp16_ != 0) {
            // Reaching here with fp16_ means odd N (even N returned above) and
            // not coopmat (it dispatches its own pack and returns), so the
            // reduce wrote flat fp32 scratch that nothing has repacked.
            scratch_->shaderWriteBarrier(m_cmd_->get());

            int nwords = (total + 1) / 2;
            MatMulPackPC pack_pc{};
            pack_pc.total = total;
            fillDescriptorWrites(pack_ds_[m_id_]);
            pack_pipeline_->updateDescriptorSets(ds_writes_);
            m_cmd_->bind(*pack_pipeline_, pack_ds_[m_id_]);
            m_cmd_->push_constants(*pack_pipeline_, sizeof(MatMulPackPC),
                                   &pack_pc);
            dispatch_and_record(UP_DIV(nwords, 256), 1, 1, &pack_pc);
        }
    }

    // One build of a coopmat kernel source: the tile it declares with
    // coopmat<...>, and the workgroup M/N footprint that tile produces with
    // the 2x4 subgroup grid every module here uses (bm = 2*cm, bn = 4*cn).
    // The 16x16x16 module also has the wider ACC builds in coop_geos_; those
    // are geometry variants of one listed tile, not extra tiles.
    struct CoopModule {
        uint32_t cm, cn, ck;
        const char *key;
        int bm, bn;
    };
    // fp16 -> fp32 modules, largest output tile first. Which ones a device can
    // run is a property-list query, not something to infer from the vendor:
    // Intel ARL lists 8x8x16 only, Ada lists 16x16x16 / 16x8x16 / 16x8x8 and
    // NOT 8x8x16. The 8x8x16 tile keeps matmul_coop.comp, the single-buffered
    // dedicated source: the pipelined matmul_coop16.comp build of the same
    // tile is 10-36% slower on ARL (m=16..63, k=2048), because one 8x8 MMA
    // per K tile leaves nothing for the prefetch to hide behind.
    static constexpr CoopModule kFp16Modules[] = {
        {16, 16, 16, "buffer_matmul_coop16", 32, 64},
        {16, 8, 16, "buffer_matmul_coop16_16x8x16", 32, 32},
        {16, 8, 8, "buffer_matmul_coop16_16x8x8", 32, 32},
        {8, 8, 16, "buffer_matmul_coop", 16, 32},
    };
    // SINT8 -> SINT32 builds of shaders/buffer/matmul_coop_i8.comp. ARL
    // lists 8x8x32, Ada 16x16x32, and each device rejects the other's tile at
    // the hardware level (validates, computes zeros), so both are needed.
    static constexpr CoopModule kI8Modules[] = {
        {16, 16, 32, "buffer_matmul_coop_i8_16x16x32", 32, 64},
        {8, 8, 32, "buffer_matmul_coop_i8", 16, 32},
    };

    // First module whose tile the device lists and whose SPIR-V this op's
    // archive resolves and the driver accepts. A missing key (older archive)
    // or a rejected pipeline moves on to the next LISTED tile, never to an
    // unlisted one. nullptr means no coopmat kernel is available at all.
    template <size_t N>
    std::unique_ptr<VulkanPipeline>
    build_coop_pipeline(const CoopModule (&mods)[N], bool sint8,
                        VkPipelineCache cache, int &bm, int &bn, bool &tile16) {
        for (const CoopModule &m : mods) {
            const bool listed =
                sint8 ? m_dev_->supports_coopmat_sint8_tile(m.cm, m.cn, m.ck)
                      : m_dev_->supports_coopmat_fp16_tile(m.cm, m.cn, m.ck);
            if (!listed) {
                continue;
            }
            std::pair<const uint32_t *, uint32_t> spv;
            try {
                spv = resolve_spv(m.key);
            } catch (const std::exception &e) {
                // Names the key: this model's archive predates it.
                std::fprintf(stderr, "[vkop] %s\n", e.what());
                continue;
            }
            try {
                bool use_uab =
                    update_after_bind_ &&
                    m_dev_->is_support_descriptor_update_after_bind();
                auto pipe = std::make_unique<VulkanPipeline>(
                    m_dev_->getLogicalDevice(),
                    std::vector<VkDescriptorType>{
                        DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                        DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                        DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                        DESCRIPTOR_TYPE_STORAGE},
                    sizeof(matmul::GpuMatMulParam), spv.first,
                    static_cast<int>(spv.second), use_uab, 0, cache);
                bm = m.bm;
                bn = m.bn;
                tile16 = m.cm == 16 && m.cn == 16 && m.ck == 16;
                return pipe;
            } catch (const std::exception &e) {
                std::fprintf(stderr,
                             "[vkop] coopmat %ux%ux%u MatMul module %s "
                             "rejected (%s); trying the next listed shape\n",
                             m.cm, m.cn, m.ck, m.key, e.what());
            }
        }
        return nullptr;
    }

    // VKOP_NO_COOP=1 drops every coopmat path (MoltenVK cannot lower the A/B
    // tile dims; see set_runtime_device). The flag is one of the selector's
    // knobs, so read it from there: the module builder and the router must not
    // reach different verdicts on the same environment.
    static bool coop_env_off() {
        return core::matmul_knobs().coop_disabled;
    }

    // VKOP_COOP_ACC=1x1|2x1|2x2 pins the coopmat geometry for an A/B;
    // unset (0) picks it from the shape. Diagnostic only, like VKOP_NO_COOP:
    // every geometry computes the same output.
    static int coop_acc_mode() {
        static const int mode = [] {
            const char *v = std::getenv("VKOP_COOP_ACC");
            if (v == nullptr) {
                return 0;
            }
            const std::string s(v);
            if (s == "1x1") {
                return 11;
            }
            if (s == "2x1") {
                return 21;
            }
            if (s == "2x2") {
                return 22;
            }
            return 0;
        }();
        return mode;
    }

    // Large-footprint builds of the 16x16x16 module (-DACC_M/-DACC_N over
    // shaders/buffer/matmul_coop16.comp): one workgroup covers more of the
    // output with the same 256 threads, which cuts the traffic per FLOP
    // (1/BM + 1/BN) and gives each subgroup more MMAs per K tile to hide the
    // next tile's loads behind -- at the cost of a proportionally coarser
    // dispatch grid. The default 1x1 geometry is the coop_pipeline_ built in
    // set_runtime_device; these are probed on first use because only archives
    // built after a key was added carry its bytes, and an older model must keep
    // running (on the 32x64 tile) rather than fail to load.
    struct CoopGeo {
        const char *key;
        int bm;
        int bn;
        int acc; // VKOP_COOP_ACC value that pins this geometry
        VulkanPipeline *pipe = nullptr;
        VkDescriptorSet ds[vkop::kInflight] = {nullptr};
        bool no_bytes = false; // this op's archive lacks the key
        bool refused = false;  // driver rejected the module
    };
    CoopGeo coop_geos_[2] = {
        {"buffer_matmul_coop16_2x2", 64, 128, 22},
        {"buffer_matmul_coop16_2x1", 64, 64, 21},
    };
    std::vector<std::unique_ptr<VulkanPipeline>> coop_geo_pipes_;

    VulkanPipeline *coop_geo_pipeline(CoopGeo &g) {
        if (g.pipe != nullptr) {
            return g.pipe;
        }
        if (m_dev_ == nullptr || g.no_bytes || g.refused ||
            !m_dev_->supports_coopmat_fp16_tile(16, 16, 16)) {
            return nullptr;
        }
        std::pair<const uint32_t *, uint32_t> spv;
        try {
            spv = resolve_spv(g.key);
        } catch (const std::exception &e) {
            g.no_bytes = true;
            std::fprintf(stderr,
                         "[vkop] %s\n[vkop] coopmat MatMul stays on the "
                         "geometries this archive carries\n",
                         e.what());
            return nullptr;
        }
        try {
            bool use_uab = update_after_bind_ &&
                           m_dev_->is_support_descriptor_update_after_bind();
            auto pipe = std::make_unique<VulkanPipeline>(
                m_dev_->getLogicalDevice(),
                std::vector<VkDescriptorType>{
                    DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                    DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                    DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                    DESCRIPTOR_TYPE_STORAGE},
                sizeof(matmul::GpuMatMulParam), spv.first,
                static_cast<int>(spv.second), use_uab, 0);
            for (auto &ds : g.ds) {
                ds = pipe->allocDescriptorSets();
            }
            g.pipe = pipe.get();
            coop_geo_pipes_.push_back(std::move(pipe));
        } catch (const std::exception &e) {
            g.refused = true;
            std::fprintf(stderr,
                         "[vkop] coopmat %dx%d MatMul pipeline rejected "
                         "(%s); smaller geometries stay\n",
                         g.bm, g.bn, e.what());
            return nullptr;
        }
        return g.pipe;
    }

    // The half-width fp16 tile (16x8x16 -> 32x32 workgroup footprint): the same
    // MMA depth as the 32x64 default but half the N, so a short prompt keeps
    // twice as many workgroups busy. Measured on RTX 4000 Ada, k=2048,
    // interleaved min-of-N against 32x64: 1.27-1.31x faster for m <= 128 and
    // n <= 2048 (.083 vs .109 ms at m=8, .096 vs .122 at m=32, .145 vs .158 at
    // m=128), and slower outside that band (.208 vs .173 at m=64/n=6144, .346
    // vs .260 at m=128/n=6144, .462 vs .385 at n=8192); with k=6144 it still
    // wins 1.10-1.34x at m <= 128/n=2048. The win is occupancy, not the MMA
    // shape -- the same 256 threads cover half the output, and 16x8x16 is the
    // only half-N tile the device lists. The 16x8x8 build was measured too and
    // never won, so it is not routed.
    struct CoopAlt {
        VulkanPipeline *pipe = nullptr;
        VkDescriptorSet ds[vkop::kInflight] = {nullptr};
        int bm = 32;
        int bn = 32;
        bool no_bytes = false; // this op's archive lacks the key
        bool refused = false;  // driver rejected the module
    };
    CoopAlt coop_alt_;
    std::unique_ptr<VulkanPipeline> coop_alt_owner_;

    // Built on first use: a graph whose shapes all stay outside the band must
    // never pay to compile it. nullptr means the shape keeps the 32x64 tile --
    // the device does not list 16x8x16, this archive lacks its bytes, or the
    // driver refused it.
    CoopAlt *coop_alt_pipeline() {
        if (coop_alt_.pipe != nullptr) {
            return &coop_alt_;
        }
        if (coop_alt_.no_bytes || coop_alt_.refused || m_dev_ == nullptr ||
            // Only an alternative to the 16x16x16 module: a device whose probed
            // tile already is 16x8x16 has nothing narrower to switch to.
            !coop_acc_geo_ || !m_dev_->supports_coopmat_fp16_tile(16, 8, 16)) {
            return nullptr;
        }
        std::pair<const uint32_t *, uint32_t> spv;
        try {
            spv = resolve_spv("buffer_matmul_coop16_16x8x16");
        } catch (const std::exception &e) {
            coop_alt_.no_bytes = true;
            // Once per process: a model converted before the key shipped has a
            // dozen-plus MatMul nodes, all with the same answer.
            static bool warned = false;
            if (!warned) {
                warned = true;
                std::fprintf(
                    stderr,
                    "[vkop] %s\n[vkop] MatMul keeps the 32x64 coopmat tile "
                    "for short prompts\n",
                    e.what());
            }
            return nullptr;
        }
        try {
            bool use_uab = update_after_bind_ &&
                           m_dev_->is_support_descriptor_update_after_bind();
            auto pipe = std::make_unique<VulkanPipeline>(
                m_dev_->getLogicalDevice(),
                std::vector<VkDescriptorType>{
                    DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                    DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                    DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                    DESCRIPTOR_TYPE_STORAGE},
                sizeof(matmul::GpuMatMulParam), spv.first,
                static_cast<int>(spv.second), use_uab, 0);
            for (auto &ds : coop_alt_.ds) {
                ds = pipe->allocDescriptorSets();
            }
            coop_alt_.pipe = pipe.get();
            coop_alt_owner_ = std::move(pipe);
        } catch (const std::exception &e) {
            coop_alt_.refused = true;
            std::fprintf(stderr,
                         "[vkop] coopmat 16x8x16 MatMul module rejected (%s); "
                         "the 32x64 tile keeps every shape\n",
                         e.what());
            return nullptr;
        }
        return &coop_alt_;
    }

    // The one place the fp16 tile is chosen. Everything else in the coopmat
    // path (gate, scratch, pack pass) is tile-agnostic.
    struct CoopPick {
        VulkanPipeline *pipe;
        VkDescriptorSet *ds;
        int bm;
        int bn;
        bool tile16;
    };
    CoopPick pick_coop_tile(int m, int n) {
        CoopPick pick{coop_pipeline_.get(), coop_ds_, coop_bm_, coop_bn_,
                      coop_acc_geo_};
        if (m <= kCoopAltMaxM && n <= kCoopAltMaxN) {
            if (CoopAlt *alt = coop_alt_pipeline()) {
                pick = {alt->pipe, alt->ds, alt->bm, alt->bn, false};
            }
        }
        return pick;
    }

    // Geometry for a shape. Both larger tiles are 64 rows tall, so a shape
    // that does not fill one row block keeps the 32x64 geometry rather than
    // masking most of the workgroup. 64x64 is the default above that because
    // it halves the traffic per FLOP of 32x64. 64x128 only widens N: it needs
    // 48 N-tiles of work to amortize the A reads and 128 workgroups to keep
    // both dimensions busy, and outside that it loses to 64x64 -- RTX 4000 Ada,
    // k=2048, m=128: .288 vs .259 ms at n=6144 (96 waves), .378 vs .400 at
    // n=8192, 6.22 vs 6.68 at n=151936; m=701: .946 vs .896 at n=4096 (352
    // waves, N too narrow), 1.363 vs 1.407 at n=6144, 33.1 vs 35.5 at n=151936.
    // 64x64 is in turn gated on two workgroups per SM: at 32 waves (m=64,
    // n=2048) it loses .148 to .141 and wins from 128 up (m=64, n=8192:
    // .210 vs .239).
    CoopGeo *pick_coop_geo(int m, int n, int batch) {
        if (m < 64) {
            return nullptr;
        }
        auto waves = [&](const CoopGeo &g) {
            return UP_DIV(m, g.bm) * UP_DIV(n, g.bn) * batch;
        };
        CoopGeo &wide = coop_geos_[0];
        CoopGeo &square = coop_geos_[1];
        if (m >= 128 && n >= kWideN && waves(wide) >= kWideMinWaves &&
            coop_geo_pipeline(wide) != nullptr) {
            return &wide;
        }
        if (waves(square) >= kCoopMinWaves &&
            coop_geo_pipeline(square) != nullptr) {
            return &square;
        }
        return nullptr;
    }

    // The int8/fp8 weight-only WIDE GEMV, which lives in a module of its own
    // (matmul16_w8_ksplit_wide is not in the general fp16 module -- MoltenVK
    // cannot compile it there, and as part of that module its failed pipeline
    // creation took every buffer MatMul down; see the W8_WIDE entry in
    // shaders/buffer/matmul.comp). nullptr means this weight has no wide path,
    // so byte weights stay on the 4-column split-K GEMV: correct, but without
    // the 16-columns-per-lane load that measured -15.9% on the big-N decode
    // layers above the threshold.
    //
    // Built on first use, not in set_runtime_device: a graph with no quantized
    // weight must never pay to compile a kernel it will not dispatch.
    //
    // Two verdicts reach nullptr and they are not the same width of state.
    // A driver that cannot compile the kernel is the DEVICE's answer, recorded
    // once for the process. A model whose archive has no bytes for the key is
    // that MODEL's answer -- the SPIR-V comes from the .vkopbin's own archive,
    // and Operator::resolve_spv throws by name when neither it nor the dev
    // archive carries the key -- so a graph converted before shader_map listed
    // buffer_matmul_w8_wide_fp16 loses the path while a reconverted one keeps
    // it, in the same process.
    VulkanPipeline *w8_wide_pipeline() {
        static bool refused_by_driver = false;
        static bool warned_missing_bytes = false;
        if (w8_wide_pipeline_) {
            return w8_wide_pipeline_.get();
        }
        if (refused_by_driver || w8_wide_no_bytes_) {
            return nullptr;
        }
        bool use_uab = update_after_bind_ &&
                       m_dev_->is_support_descriptor_update_after_bind();
        std::pair<const uint32_t *, uint32_t> wide;
        try {
            wide = resolve_spv("buffer_matmul_w8_wide_fp16");
        } catch (const std::exception &e) {
            // Thrown, not returned: resolve_spv treats a secondary it cannot
            // name as a broken model. Stay per-op and say it once -- the
            // fallback is still the right answer, and a silent 15% is the
            // failure mode this whole probe exists to avoid.
            w8_wide_no_bytes_ = true;
            if (!warned_missing_bytes) {
                warned_missing_bytes = true;
                std::fprintf(stderr,
                             "[vkop] %s\n[vkop] byte-weight decode GEMVs use "
                             "the 4-column kernel until that key is shipped\n",
                             e.what());
            }
            return nullptr;
        }
        try {
            w8_wide_pipeline_ = std::make_unique<VulkanPipeline>(
                m_dev_->getLogicalDevice(),
                std::vector<VkDescriptorType>{
                    DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                    DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                    DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                    DESCRIPTOR_TYPE_STORAGE},
                sizeof(matmul::GpuMatMulParam), wide.first,
                static_cast<int>(wide.second), use_uab, 0);
            for (auto &ds : w8_wide_ds_) {
                ds = w8_wide_pipeline_->allocDescriptorSets();
            }
        } catch (const std::exception &e) {
            // Loud: the fallback computes the right answer more slowly, so
            // nothing else would ever report that this device lost the path.
            refused_by_driver = true;
            w8_wide_pipeline_.reset();
            std::fprintf(stderr,
                         "[vkop] int8/fp8 WIDE GEMV pipeline rejected (%s); "
                         "byte-weight decode GEMVs use the 4-column kernel\n",
                         e.what());
            return nullptr;
        }
        return w8_wide_pipeline_.get();
    }

    // The SwiGLU down-projection fold (buffer_matmul_down_act_fp16). Nine
    // descriptors, because this module is the only MatMul build that reads the
    // gate and up weights: 7/8 are beyond every other module's count, so a
    // pipeline that declares them cannot be shared. Probed per process like the
    // 512-lane pair, and for a stronger reason -- the whole point of the fold
    // is an allocation the host has already skipped, so a refusal must be
    // visible and must fall back to routing the plain GEMMs.
    // Builds one of the two fold modules. The M==1 key is compiled with
    // DA_M1_ONLY: the multi-row tile walk is gone from it, which is what made
    // the decode fold regress 4.2 ms/step when both walks shared one module
    // (2026-11-05: same executed code, bigger SPIR-V, slower). M>1 dispatches
    // use _multi, which keeps the unrolled walk. An archive converted before
    // the split has only the M==1 key; M>1 then falls back to it, whose
    // per-row walk is still correct -- just the pre-split speed -- so an old
    // model keeps folding instead of silently writing one row per tile.
    VulkanPipeline *down_act_pipeline(bool multi = false) {
        const int id = multi ? 1 : 0;
        static bool refused_by_driver[2] = {false, false};
        static bool warned_missing_bytes[2] = {false, false};
        std::unique_ptr<VulkanPipeline> &cached =
            multi ? down_act_multi_pipeline_ : down_act_pipeline_;
        bool &no_bytes = multi ? down_act_multi_no_bytes_ : down_act_no_bytes_;
        VkDescriptorSet *ds = multi ? down_act_multi_ds_ : down_act_ds_;
        if (cached) {
            return cached.get();
        }
        if (refused_by_driver[id] || no_bytes) {
            return nullptr;
        }
        bool use_uab = update_after_bind_ &&
                       m_dev_->is_support_descriptor_update_after_bind();
        std::pair<const uint32_t *, uint32_t> spv;
        try {
            spv = resolve_spv(multi ? "buffer_matmul_down_act_multi_fp16"
                                    : "buffer_matmul_down_act_fp16");
        } catch (const std::exception &e) {
            no_bytes = true;
            if (!warned_missing_bytes[id]) {
                warned_missing_bytes[id] = true;
                std::fprintf(
                    stderr,
                    "[vkop] %s\n[vkop] %s\n", e.what(),
                    multi ? "M>1 keeps the M==1 module's per-row deposit walk"
                          : "SwiGLU down_proj folds stay off until that key "
                            "is shipped");
            }
            return nullptr;
        }
        try {
            cached = std::make_unique<VulkanPipeline>(
                m_dev_->getLogicalDevice(),
                std::vector<VkDescriptorType>{
                    DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                    DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                    DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                    DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                    DESCRIPTOR_TYPE_STORAGE},
                sizeof(matmul::GpuMatMulParam), spv.first,
                static_cast<int>(spv.second), use_uab, 0);
            for (int i = 0; i < vkop::kInflight; ++i) {
                ds[i] = cached->allocDescriptorSets();
            }
        } catch (const std::exception &e) {
            refused_by_driver[id] = true;
            cached.reset();
            std::fprintf(stderr,
                         "[vkop] SwiGLU down fold %s pipeline rejected (%s); "
                         "%s\n",
                         multi ? "M>1" : "M==1", e.what(),
                         multi ? "M>1 uses the M==1 module's per-row walk"
                               : "down_proj runs as three standalone GEMMs");
            return nullptr;
        }
        return cached.get();
    }

    // Shared probe/build for the two 512-lane keys. id pins the process-wide
    // verdict (0 = ksplit-only module, 1 = int8 WIDE module) so a device
    // refusal or a missing-archive-bytes answer is reached once per process,
    // not once per MatMul instance.
    VulkanPipeline *secondary_pipeline(int id, const char *key,
                                       std::unique_ptr<VulkanPipeline> &slot,
                                       VkDescriptorSet *ds_slot) {
        static bool no_bytes[4] = {false, false, false, false};
        static bool refused[4] = {false, false, false, false};
        static bool warned_missing[4] = {false, false, false, false};
        if (slot) {
            return slot.get();
        }
        if (no_bytes[id] || refused[id]) {
            return nullptr;
        }
        bool use_uab = update_after_bind_ &&
                       m_dev_->is_support_descriptor_update_after_bind();
        std::pair<const uint32_t *, uint32_t> spv;
        try {
            spv = resolve_spv(key);
        } catch (const std::exception &e) {
            no_bytes[id] = true;
            if (!warned_missing[id]) {
                warned_missing[id] = true;
                std::fprintf(stderr,
                             "[vkop] %s\n[vkop] %s is unavailable until that "
                             "key is shipped; using its fallback\n",
                             e.what(), key);
            }
            return nullptr;
        }
        try {
            slot = std::make_unique<VulkanPipeline>(
                m_dev_->getLogicalDevice(),
                std::vector<VkDescriptorType>{
                    DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                    DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                    DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                    DESCRIPTOR_TYPE_STORAGE},
                sizeof(matmul::GpuMatMulParam), spv.first,
                static_cast<int>(spv.second), use_uab, 0);
            for (int i = 0; i < vkop::kInflight; ++i) {
                ds_slot[i] = slot->allocDescriptorSets();
            }
        } catch (const std::exception &e) {
            refused[id] = true;
            slot.reset();
            std::fprintf(stderr,
                         "[vkop] %s pipeline rejected (%s); using its "
                         "fallback\n",
                         key, e.what());
            return nullptr;
        }
        return slot.get();
    }

    // Record which kernel route the current execute() took. The tag is the
    // plan's, never a literal: core/KernelSelect decided the route and the
    // dispatch below only executes it, so the readout cannot drift from the
    // table. Cheap either way -- one snprintf per recording, not per round: on
    // a CACHED replay round execute() is skipped and the member still holds the
    // recording round's value for Runtime to read back.
    void set_route(const char *tag, int batch, int m, int n, int k, int rep) {
        route_tag_ = tag;
        char buf[192];
        snprintf(buf, sizeof(buf),
                 "%s batch=%d m=%d n=%d k=%d transB=%d rep=%d mid=%d", tag,
                 batch, m, n, k, transB_ ? 1 : 0, rep, m_id_);
        route_info_ = buf;
    }

    // The two tile-split-K modules (buffer_matmul_tiled_ks_split_fp16,
    // buffer_matmul_tiled_ks_fold_fp16: the same source as the general fp16
    // module, built with a different define so each carries exactly one entry
    // point). A pipeline holds one entry point, and the split form is
    // barrier-free by construction -- each block returns after depositing its
    // partial -- so the deposit and the fold cannot share a dispatch. Probed
    // through secondary_pipeline() like the 512-lane pair: a driver refusal or
    // a model archive without the keys leaves ks_n_ at 1, i.e. the unsplit
    // tiled kernel, which is the same kernel without the K cut.
    VulkanPipeline *ks_split_pipeline() {
        return secondary_pipeline(2, "buffer_matmul_tiled_ks_split_fp16",
                                  ks_split_pipeline_, ks_split_ds_);
    }
    VulkanPipeline *ks_fold_pipeline() {
        return secondary_pipeline(3, "buffer_matmul_tiled_ks_fold_fp16",
                                  ks_fold_pipeline_, ks_fold_ds_);
    }

    // The SwiGLU down fold's K-split factor. A deposit block walks only its
    // slice, so the M=1 grid is GX*ks_n workgroups instead of the GX=12 the
    // column-partitioned form had -- the fold's decode problem was never
    // weight traffic (the column slices are disjoint, every byte was already
    // read once) but occupancy: 12 blocks cannot fill the device. The cost is
    // one fp32 partial per (row, gx block, slice, block-local column):
    // batch*m*GX*ks_n*512 words, ~196 KiB at M=1/ks_n=8.
    //
    // Shape-derived on purpose. Both the deposit (which looks its slice up from
    // gid.z) and the fold (which reduces all ks_n of them) read the table the
    // host stages, so a cut that moved between a record round and a replay
    // round would have the fold reduce a partition the deposit never wrote.
    static int down_act_ks_n(int k_out) {
        // Aim for ~256 K per slice: deep enough that the partial traffic stays
        // rounding error, shallow enough to keep the grid wide at M=1.
        const int want = 256;
        const int n = (k_out + want - 1) / want;
        return n < 1 ? 1 : n;
    }

    // Bind, push and dispatch one half of the tile-split-K pair. Same body as
    // dispatch_on above; separate only because the two dispatches in that pair
    // each need their own descriptor set (a set is written once per dispatch
    // through fillDescriptorWrites, so the deposit and the fold cannot share
    // one). The writes live in this call's own vector: the base dispatch's
    // writes_ is still waiting behind a not-yet-existing replay entry, and
    // refilling the shared vector would leave that entry describing the fold's
    // set.
    void ks_submit(VulkanPipeline *pipe, VkDescriptorSet ds, int gx, int gy,
                   int gz) {
        fillDescriptorWrites(ds);
        pipe->updateDescriptorSets(ds_writes_);
        m_cmd_->bind(*pipe, ds);
        m_cmd_->push_constants(*pipe, sizeof(matmul::GpuMatMulParam), &para_);
        dispatch_and_record(gx, gy, gz, &para_);
    }

    // layout and push constants as the wide path above).
    void dispatch_on(VulkanPipeline *pipe, VkDescriptorSet ds, int gx, int gy) {
        fillDescriptorWrites(ds);
        pipe->updateDescriptorSets(ds_writes_);
        m_cmd_->bind(*pipe, ds);
        m_cmd_->push_constants(*pipe, sizeof(matmul::GpuMatMulParam), &para_);
        // Recorded like every other dispatch. Dropping the recording here was
        // tried and is the wrong direction: the fingerprint is the only
        // host-side evidence of which kernel ran, so an unrecorded dispatch
        // makes a 512-lane recording compare equal to a 256-lane one and lets
        // the replay cache hand back the wrong command buffer.
        dispatch_and_record(gx, gy, 1, &para_);
    }

    // Fill descriptor-set writes from the current objs_ vector (one SSBO per
    // binding). Used by the fp16 cooperative-matrix dispatch path, which binds
    // its own pipeline (coop_pipeline_) instead of the base Operator::submit().
    void fillDescriptorWrites(VkDescriptorSet ds) {
        ds_writes_.resize(objs_.size());
        for (size_t i = 0; i < objs_.size(); ++i) {
            ds_writes_[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            ds_writes_[i].dstSet = ds;
            ds_writes_[i].dstBinding = static_cast<uint32_t>(i);
            ds_writes_[i].dstArrayElement = 0;
            ds_writes_[i].descriptorCount = 1;
            ds_writes_[i].descriptorType = DESCRIPTOR_TYPE_STORAGE;
            switch (objs_[i]->getResourceType()) {
            case ResourceType::VK_BUFFER:
                ds_writes_[i].pBufferInfo = std::get<VkDescriptorBufferInfo *>(
                    objs_[i]->getDescriptorInfo());
                break;
            default:
                break;
            }
        }
    }

    matmul::GpuMatMulParam para_;
    std::shared_ptr<VulkanBuffer> scratch_;
    // SwiGLU-fold scratch: the split-K GEMV's fp32 partials. One 64x64-word
    // slot per (batch, output tile, K slice).
    std::shared_ptr<VulkanBuffer> scratch_da_;
    size_t scratch_da_bytes_ = 0;
    // The down fold's K cut: 2*da_ks_n_ (lo, hi) pairs tiling [0, K_out). The
    // deposit blocks find their own slice from gid.z and the fold pass reduces
    // exactly those slices, so both take the cut from one host-built table.
    // It is a pure function of K_out, which is what lets the record path
    // compare against da_words_hdr_ and skip the rewrite on every later round
    // (a replay round re-runs the recorded dispatches without re-entering this
    // path, so the header has to survive in the buffer).
    int da_ks_n_ = 0;
    std::vector<uint32_t> da_words_hdr_;
    size_t scratch_bytes_ = 0;
    // Tile-split-K's own K range for THIS dispatch, staged into the scratch
    // header (words 1..3) instead of a push constant: four more ints at the PC
    // tail push sizeof past 128 bytes and collapse the push range (see the
    // assignment site). ks_n <= 1 is the unsplit form.
    int ks_n_ = 0;
    int ks_k0_ = 0;
    int ks_kstop_ = 0;
    // The cut: 2*ks_n_ even-odd pairs (lo, hi) tiling [0, K), written to the
    // scratch header at ks_table_off_ so the deposit pass can find its slice
    // from gid.z. Shape-derived, so a replay round stages the same table an
    // execute round did.
    std::vector<int> ks_slices_;
    size_t ks_table_off_ = 4;
    // The scratch words 0..ks_table_off_ as last staged. A cached replay round
    // re-runs the recorded dispatches without re-entering the record path, so
    // the header must survive in the buffer; comparing against this shadow
    // keeps the rewrite off the steady-state path.
    std::vector<uint32_t> ks_words_;
    // coop_pipeline_ runs the fp16 cooperative-matrix kernel (fp16-in, fp32
    // accumulate -> flat scratch). pack_pipeline_ runs the half2 pack pass.
    // ds_writes_/fillDescriptorWrites fill either pipeline's descriptor set
    // from the same objs_ vector.
    std::unique_ptr<VulkanPipeline> coop_pipeline_;
    VkDescriptorSet coop_ds_[vkop::kInflight] = {nullptr};
    // Workgroup output footprint of the probed coopmat tile (bm = 2*cm,
    // bn = 4*cn): 16x32 for ARL's 8x8x16, 32x64 for Ada's 16x16x16, 32x32 for
    // the 16x8 tiles. Set when coop_pipeline_ is built; the dispatch grid
    // divides M/N by these.
    int coop_bm_ = 16;
    int coop_bn_ = 32;
    // The probed tile is 16x16x16, the only one with the wider ACC builds in
    // coop_geos_ (they are extra accumulators over one tile, not extra tiles).
    bool coop_acc_geo_ = false;
    // Large 16x16x16 geometries, see coop_geo_pipeline. kCoopMinWaves and
    // kWideMinWaves are the workgroup counts below which a bigger footprint
    // loses more to the idle tail than it gains in traffic; kWideN is the
    // width below which 64x128 has too few N-tiles to amortize its A reads.
    static constexpr int kCoopMinWaves = 96;
    static constexpr int kWideMinWaves = 128;
    static constexpr int kWideN = 6144;
    // Shape band where the half-width 16x8x16 tile beats the 32x64 one, see
    // coop_alt_pipeline. Above either bound the 32x64 tile's lower traffic per
    // FLOP takes over.
    static constexpr int kCoopAltMaxM = 128;
    static constexpr int kCoopAltMaxN = 2048;
    // coop_i8_pipeline_ runs the W8A8 SINT8 cooperative-matrix kernel (int8
    // weight x runtime-quantized int8 activation -> flat fp32 scratch). Built
    // only for a tile the device lists; the grid divides M/N by its footprint.
    std::unique_ptr<VulkanPipeline> coop_i8_pipeline_;
    VkDescriptorSet coop_i8_ds_[vkop::kInflight] = {nullptr};
    int coop_i8_bm_ = 16;
    int coop_i8_bn_ = 32;
    std::unique_ptr<VulkanPipeline> pack_pipeline_;
    VkDescriptorSet pack_ds_[vkop::kInflight] = {nullptr};
    // The int8/fp8 weight-only WIDE GEMV module (buffer_matmul_w8_wide_fp16,
    // a second build of buffer/matmul.comp whose entry reaches only that
    // kernel) and its descriptor sets. Built on first use, see
    // w8_wide_pipeline.
    std::unique_ptr<VulkanPipeline> w8_wide_pipeline_;
    VkDescriptorSet w8_wide_ds_[vkop::kInflight] = {nullptr};
    // The 512-lane split-K modules (VKOP_KSPLIT512=1), probed through
    // secondary_pipeline(): the ksplit-only general module and the int8 WIDE
    // module, both -DKSPLIT512 builds of shaders/buffer/matmul.comp.
    std::unique_ptr<VulkanPipeline> ksplit512_pipeline_;
    VkDescriptorSet ksplit512_ds_[vkop::kInflight] = {nullptr};
    std::unique_ptr<VulkanPipeline> w8_wide512_pipeline_;
    VkDescriptorSet w8_wide512_ds_[vkop::kInflight] = {nullptr};
    // The tile-split-K pair (ks_split_pipeline / ks_fold_pipeline above).
    std::unique_ptr<VulkanPipeline> ks_split_pipeline_;
    VkDescriptorSet ks_split_ds_[vkop::kInflight] = {nullptr};
    std::unique_ptr<VulkanPipeline> ks_fold_pipeline_;
    VkDescriptorSet ks_fold_ds_[vkop::kInflight] = {nullptr};
    // SK_LASTBLOCK: coopmat + parallel grid.z split-K. Per-cs precompiled
    // variants (SK_CS baked in) avoid glslc's register-allocator bug with
    // runtime ints near coopMat accumulators. Map key = coop_split value.
    std::unordered_map<int, std::unique_ptr<VulkanPipeline>> sklast_fp32_pipes_;
    std::unordered_map<int, std::unique_ptr<VulkanPipeline>> sklast_half2_pipes_;
    VkDescriptorSet sklast_fp32_ds_[vkop::kInflight] = {nullptr};
    VkDescriptorSet sklast_half2_ds_[vkop::kInflight] = {nullptr};
    // This op's archive supplies no bytes for the WIDE key, so it stays on the
    // 4-column kernel; another MatMul from a reconverted model still tries.
    bool w8_wide_no_bytes_ = false;
    // The SwiGLU down-projection fold module (see down_act_pipeline). The
    // converter's "down_act" attribute (an ints list, [1]) marks a MatMul whose
    // inputs are [x, Wd, Wg, Wu] and whose result is
    // silu(x . Wg) * (x . Wu) . Wd; down_act_ holds that flag.
    std::unique_ptr<VulkanPipeline> down_act_pipeline_;
    VkDescriptorSet down_act_ds_[vkop::kInflight] = {nullptr};
    bool down_act_no_bytes_ = false;
    // M>1 fold module (keeps the multi-row tile walk; see down_act_pipeline).
    std::unique_ptr<VulkanPipeline> down_act_multi_pipeline_;
    VkDescriptorSet down_act_multi_ds_[vkop::kInflight] = {nullptr};
    bool down_act_multi_no_bytes_ = false;
    bool down_act_ = false;
    std::vector<VkWriteDescriptorSet> ds_writes_;
    bool transB_ = false; // B laid out as [batch, N, K] (folded Transpose)
    // Attention QK^T epilogue (folded from a FusedElemwise by the converter):
    // epi_scale_ multiplies every output element, epi_bias_ flags that a bias
    // tensor is appended as inputs[2] and bound at binding 6.
    float epi_scale_ = 1.0f;
    bool epi_bias_ = false;
    // Plan B: fuse the trailing Softmax into this MatMul (online softmax in the
    // kernel). Set by the converter via the "attn_softmax" attribute.
    bool attn_softmax_ = false;
    // Which kernel route the last execute() actually dispatched, and the shape
    // contract it was picked from. Filled at every dispatch site below (the
    // ksplit sub-routes are decided mid-branch, so a top-level flag cannot name
    // them). Cached as a member, not printed, because execute() is skipped on a
    // CACHED replay round: Runtime reads these back per round behind
    // VKOP_MATMUL_ROUTE. See Operator::route_tag.
    std::string route_tag_;
    std::string route_info_;
};

// PIMPL façade: buffer SSBO impl when backend_buffer is set, else image.
class MatMul : public PimplFacade {
  public:
    MatMul(int use_tensorcore, int fp16, bool backend_buffer)
        : PimplFacade(OpType::MATMUL) {
        if (backend_buffer) {
            impl_ = std::make_unique<MatMulBuffer>(fp16);
        } else {
            impl_ = std::make_unique<MatMulImage>(use_tensorcore);
        }
    }
    std::string route_tag() const override { return impl_->route_tag(); }
    std::string route_info() const override { return impl_->route_info(); }
};

} // namespace ops
} // namespace vkop
#endif // OPS_MATMUL_HPP_
