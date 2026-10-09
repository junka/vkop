// Copyright 2025 @junka
#ifndef OPS_MATMUL_HPP_
#define OPS_MATMUL_HPP_

#include "Operator.hpp"
#include "ops/BufferBase.hpp"
#include "ops/PimplFacade.hpp"

#include <string>
#include <vector>
extern "C" {
extern unsigned char image_matmul_spv[];
extern unsigned int image_matmul_spv_len;
extern unsigned char image_matmul_nv_spv[];
extern unsigned int image_matmul_nv_spv_len;
extern unsigned char image_matmul_coop_spv[];
extern unsigned int image_matmul_coop_spv_len;
extern unsigned char buffer_matmul_spv[];
extern unsigned int buffer_matmul_spv_len;
extern unsigned char buffer_matmul_fp16_spv[];
extern unsigned int buffer_matmul_fp16_spv_len;
extern unsigned char buffer_matmul_pack_spv[];
extern unsigned int buffer_matmul_pack_spv_len;
extern unsigned char buffer_matmul_coop_spv[];
extern unsigned int buffer_matmul_coop_spv_len;
extern unsigned char buffer_matmul_coop_i8_spv[];
extern unsigned int buffer_matmul_coop_i8_spv_len;
}
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

    explicit MatMulImage(int use_tensorcore = 0)
        : Operator(OpType::MATMUL,
                   use_tensorcore == 2
                       ? image_matmul_nv_spv
                       : (use_tensorcore == 1 ? image_matmul_coop_spv
                                              : image_matmul_spv),
                   use_tensorcore == 2
                       ? image_matmul_nv_spv_len
                       : (use_tensorcore == 1 ? image_matmul_coop_spv_len
                                              : image_matmul_spv_len),
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
        : BufferFactory(
              OpType::MATMUL, fp16 ? buffer_matmul_fp16_spv : buffer_matmul_spv,
              fp16 ? buffer_matmul_fp16_spv_len : buffer_matmul_spv_len,
              std::vector<VkDescriptorType>{
                  DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                  DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                  DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
              sizeof(matmul::GpuMatMulParam), fp16) {
        update_after_bind_ = true;
    }

    void set_runtime_device(
        const std::shared_ptr<VulkanDevice> &dev,
        const std::shared_ptr<VulkanCommandPool> &cmdpool) override {
        BufferFactory::set_runtime_device(dev, cmdpool);
        if (fp16_ != 0 && !coop_pipeline_) {
            // fp16 cooperative-matrix kernel: a single dispatch reads fp16 A/B
            // from SSBO, accumulates in fp32 via the 8x8x16 subgroup MMA
            // (coopMatMulAdd), and writes FLAT fp32 results to the scratch
            // buffer (binding 3). A separate pack pass (pack_pipeline_ below)
            // then repacks adjacent flat element pairs into packed half2 words
            // — kept as a distinct flat pass because when N is odd one output
            // word straddles two rows (element 2w+1 of word w is the next row's
            // col 0), and in-shader (col, col+1) packing would zero-fill the
            // straddling half and clobber the next row's first element. Intel
            // ARL exposes the required fp16 coopmat combo (8x8x16, fp16 A/B ->
            // fp32 acc, subgroup scope, sg size 32). The pipeline's descriptor
            // set declares 6 bindings to match the quantized-weight layouts:
            // only 0-3 are read, 4-5 are bound-but-unused (filled from the same
            // objs_ vector as every other MatMul pipeline).
            bool use_uab = update_after_bind_ &&
                           dev->is_support_descriptor_update_after_bind();
            coop_pipeline_ = std::make_unique<VulkanPipeline>(
                dev->getLogicalDevice(),
                std::vector<VkDescriptorType>{
                    DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                    DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                    DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                sizeof(matmul::GpuMatMulParam),
                reinterpret_cast<const uint32_t *>(buffer_matmul_coop_spv),
                static_cast<int>(buffer_matmul_coop_spv_len), use_uab, 0);
            for (auto &ds : coop_ds_) {
                ds = coop_pipeline_->allocDescriptorSets();
            }
        }
        if (fp16_ != 0 && !coop_i8_pipeline_ &&
            dev->is_support_cooperate_matrix() &&
            dev->supports_coopmat_sint8()) {
            // W8A8 cooperative-matrix kernel: int8 weight bytes x an int8
            // activation the kernel quantizes itself (per output row, see the
            // shader header), accumulated in int32 by the 8x8x32 subgroup MMA.
            // Built only when the device actually lists the SINT8 combo — the
            // KHR property list is the only feature query for integer MMA.
            bool use_uab = update_after_bind_ &&
                           dev->is_support_descriptor_update_after_bind();
            coop_i8_pipeline_ = std::make_unique<VulkanPipeline>(
                dev->getLogicalDevice(),
                std::vector<VkDescriptorType>{
                    DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                    DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                    DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                sizeof(matmul::GpuMatMulParam),
                reinterpret_cast<const uint32_t *>(buffer_matmul_coop_i8_spv),
                static_cast<int>(buffer_matmul_coop_i8_spv_len), use_uab, 0);
            for (auto &ds : coop_i8_ds_) {
                ds = coop_i8_pipeline_->allocDescriptorSets();
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
        if (attributes.find("transB") != attributes.end()) {
            transB_ = std::stol(attributes.at("transB")) != 0;
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
                                                        : "4-bit";
        size_t scale_index = 0;
        size_t global_index = 0;
        if (weight_byte || weight_4bit) {
            const size_t want = weight_nvfp4 ? 4 : 3;
            if (inputs.size() != want) {
                throw std::runtime_error(
                    std::string("vkop: MatMul with a ") + weight_label +
                    " weight needs exactly " +
                    (weight_nvfp4 ? "[A, B, block_scale, global_scale]"
                                  : "[A, B, scale]") +
                    ", got " + std::to_string(inputs.size()) + " inputs");
            }
            if (weight_nvfp4) {
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
            scale_index = 2;
        }
        auto shape_a = inputs[0]->getShape();
        auto shape_b = inputs[1]->getShape();
        int rank_a = static_cast<int>(shape_a.size());
        int rank_b = static_cast<int>(shape_b.size());

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
        if (rank_a < 2 || rank_b < 2) {
            // Degenerate; nothing to do (should not happen for a valid model).
            return;
        }
        int m = shape_a[rank_a - 2];
        int k = shape_a[rank_a - 1];
        // transB: B is laid out as [batch, N, K] (a Transpose(perm=[..,last2
        // swapped]) was folded into this MatMul). N comes from B's 2nd-to-last
        // dim instead of the last; K from B's last dim must equal A's K.
        int n = transB_ ? shape_b[rank_b - 2] : shape_b[rank_b - 1];

        int batch = 1;
        int lead_a = rank_a - 2;
        int lead_b = rank_b - 2;
        int lead = std::max(lead_a, lead_b);
        // Leading broadcast dims in natural (left-to-right) order, taking the
        // per-axis max (ONNX broadcast: a dim of 1 matches the other's value).
        // Built left-to-right by indexing from the front of each operand's
        // leading dims.
        std::vector<int> lead_shape;
        lead_shape.reserve(lead);
        for (int i = 0; i < lead; ++i) {
            int da = (i < lead_a) ? shape_a[i] : 1;
            int db = (i < lead_b) ? shape_b[i] : 1;
            int dmax = std::max(da, db);
            batch *= dmax;
            lead_shape.push_back(dmax);
        }

        std::vector<int> out_shape = lead_shape;
        out_shape.push_back(m);
        out_shape.push_back(n);
        if (out_shape.empty()) {
            out_shape = {m, n}; // degenerate guard
        }

        int total = batch * m * n;

        if (weight_byte) {
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
        // Cooperative-matrix GEMM (Intel ARL fp16 8x8x16 subgroup MMA). A
        // device-specific fast path: when the GPU exposes
        // KHR_cooperative_matrix with the fp16->fp32 combo, a single
        // coopMatMulAdd dispatch replaces the scalar reduce. It writes flat
        // fp32 to the scratch buffer (odd-N safe: a word can straddle two
        // rows), then the existing pack pass repacks half2 — so it also needs
        // the scratch below. Only for plain fp16 A/B (a quantized weight has
        // its own dequant kernels) and only when the device supports it;
        // everything else falls through to the tiled / ksplit / fused / reduce
        // path below.
        const bool coopmat = fp16_ != 0 && !weight_byte && !weight_4bit &&
                             m_dev_->is_support_cooperate_matrix() &&
                             coop_pipeline_ != nullptr;
        // W8A8 sibling of the path above: an int8 weight with an int8
        // activation the kernel quantizes itself (per output row), accumulated
        // by the 8x8x32 SINT8 subgroup MMA. Unlike coopmat it does not exclude
        // a byte weight -- that is the whole point -- but it does keep the
        // existing gate's transB == 0 and N % 8 == 0 constraints, since its B
        // loader walks K rows of contiguous N bytes. Requires the device to
        // list the SINT8 combo; the pipeline is built only then.
        const bool coopmat_i8 = fp16_ != 0 && weight_int8 && !transB_ &&
                                (n % 8 == 0) &&
                                m_dev_->is_support_cooperate_matrix() &&
                                coop_i8_pipeline_ != nullptr;
        // Split-K GEMV: the column-parallel kernels above put output rows on
        // grid.y and use 16 of their workgroup's 256 lanes per row, so at
        // decode time (one A row) 15/16 of every workgroup idles and the
        // surviving threads walk K serially. Measured on GLM-Edge that caps
        // int8 at 13.9 GB/s and fp16 at 26.8 GB/s, far under the machine -- the
        // fix is to spend those idle lanes on K, not on more bytes. Needs the
        // quad ownership (N % 4 == 0, B as [batch, K, N], K even so A's k-pairs
        // stay word-aligned) and only pays where columns alone are too few: the
        // small batch*m, long-K regime.
        const bool ksplit = fused_fp16 && !transB_ && (n % 4 == 0) &&
                            (k % 2 == 0) && (batch * m <= 8) && (k >= 256);
        // With B as [batch, K, N] (transB == 0, what an ONNX MatMul weight is)
        // the four columns of one K row are four adjacent weight BYTES, so one
        // 32-bit load feeds four output columns. That needs N % 4 == 0 to keep
        // every row start word-aligned, and it changes how many columns a
        // thread owns, so the grid x extent below has to count quads instead of
        // pairs.
        const bool w8_quad =
            fused_fp16 && weight_byte && !transB_ && (n % 4 == 0);
        // 4-bit weights always take the four-column quad kernel: the validation
        // above already proved transB == 0 and N % 8 == 0, which is what makes
        // a thread's four columns exactly half of one packed weight word.
        const bool w4_quad = fused_fp16 && weight_4bit;
        // Shared-memory tiling pays only in the compute-bound regime, and only
        // where its indexing assumptions hold: k % 16 keeps every tile word
        // aligned on both B layouts. m >= 12 is measured, not guessed: at
        // M <= 8 the naive GEMV is weight-bandwidth-bound and the 64-row tile
        // is ~break-even (0.99-1.01x across N=1024..9728), while from M = 12
        // the tile reuse wins 1.2-2.0x and never regresses (narrow N=64 and
        // batched attention shapes included). A byte-quantized weight joins it
        // through tile_load_b_w8, which stages the decoded value: in the
        // column-parallel quad kernel a weight is loaded and decoded once per
        // output row, which on the DiT (m = 1024) cost int8 4.0x and fp8 5.5x
        // what fp16 pays per step. That loader reads four columns of one K row
        // out of one weight word, so it needs transB == 0 (the only layout the
        // quantizer emits) and n % 4 == 0. 4-bit weights stay out: their scale
        // varies along K, which a per-column epilogue cannot fold.
        const bool tiled = fused_fp16 && !weight_4bit && (k % 16 == 0) &&
                           (m >= 12) &&
                           (!weight_byte || (!transB_ && (n % 4 == 0)));
        // The odd-N reduce->pack fallback needs scratch; the cooperative-matrix
        // path also needs it (it writes flat fp32 results that the pack pass
        // repacks to half2, regardless of N parity). Even-N fused/reduce paths
        // pack in-shader and bind the dummy for the 4th slot.
        if (fp16_ != 0 && (!fused_fp16 || coopmat || coopmat_i8)) {
            // total may be 0 for a dynamic-shape output that resolved empty
            // (a 0 dim). vkCreateBuffer rejects size 0 with
            // VK_ERROR_INITIALIZATION_FAILED on Intel, so clamp to a minimal
            // 16-byte dummy — the dispatch is 0 threads anyway.
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
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                scratch_bytes_ = scratch_bytes;
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
        para_.ksplit = ksplit ? 1 : 0;
        para_.w4 = weight_4bit ? 1 : 0;
        para_.nf4 = weight_nf4 ? 1 : 0;
        para_.nvfp4 = weight_nvfp4 ? 1 : 0;
        para_.uint4 = weight_uint4 ? 1 : 0;
        para_.group = weight_4bit ? group_size : 0;

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
            // Same scratch + pack tail as the fp16 coopmat path.
            fillDescriptorWrites(coop_i8_ds_[m_id_]);
            coop_i8_pipeline_->updateDescriptorSets(ds_writes_);
            m_cmd_->bind(*coop_i8_pipeline_, coop_i8_ds_[m_id_]);
            m_cmd_->push_constants(*coop_i8_pipeline_,
                                   sizeof(matmul::GpuMatMulParam), &para_);
            m_cmd_->dispatch(UP_DIV(n, 32), UP_DIV(m, 16), batch);

            scratch_->shaderWriteBarrier(m_cmd_->get());

            int nwords = (total + 1) / 2;
            MatMulPackPC pack_pc{};
            pack_pc.total = total;
            fillDescriptorWrites(pack_ds_[m_id_]);
            pack_pipeline_->updateDescriptorSets(ds_writes_);
            m_cmd_->bind(*pack_pipeline_, pack_ds_[m_id_]);
            m_cmd_->push_constants(*pack_pipeline_, sizeof(MatMulPackPC),
                                   &pack_pc);
            m_cmd_->dispatch(UP_DIV(nwords, 256), 1, 1);
            return;
        }

        if (coopmat) {
            fillDescriptorWrites(coop_ds_[m_id_]);
            coop_pipeline_->updateDescriptorSets(ds_writes_);
            m_cmd_->bind(*coop_pipeline_, coop_ds_[m_id_]);
            m_cmd_->push_constants(*coop_pipeline_,
                                   sizeof(matmul::GpuMatMulParam), &para_);
            m_cmd_->dispatch(UP_DIV(n, 32), UP_DIV(m, 16), batch);

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
            m_cmd_->dispatch(UP_DIV(nwords, 256), 1, 1);
            return;
        }

        if (tiled) {
            // x = 64-column tiles, y = 64-row tiles, z = batch: a block never
            // straddles a batch boundary, so no per-row batch fixups in the
            // shader.
            submit(&para_, UP_DIV(n, 64), UP_DIV(m, 64), batch);
            return;
        }
        if (ksplit) {
            // x = column quads (16 per block), y = one output row per block:
            // the block's own 16 lanes are the K slices, so y carries the row
            // index the shader reads as gl_WorkGroupID.y.
            submit(&para_, UP_DIV(n / 4, 16), batch * m, 1);
            return;
        }
        if (fused_fp16) {
            // Single pass: x covers output WORDS (column pairs), or DWORDs for
            // the four-column quantized kernels.
            submit(&para_, UP_DIV(n / ((w8_quad || w4_quad) ? 4 : 2), 16),
                   UP_DIV(batch * m, 16), 1);
            return;
        }
        // Reduce pass: one thread per output element. Uses the main pipeline.
        // For fp16 inputs without coopmat this is the fp16-in reduce shader;
        // for fp32 inputs this is the naive fp32 reduce (batch*m collapsed into
        // y).
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
            m_cmd_->dispatch(UP_DIV(nwords, 256), 1, 1);
        }
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
    size_t scratch_bytes_ = 0;
    // coop_pipeline_ runs the fp16 cooperative-matrix kernel (fp16-in, fp32
    // accumulate -> flat scratch). pack_pipeline_ runs the half2 pack pass.
    // ds_writes_/fillDescriptorWrites fill either pipeline's descriptor set
    // from the same objs_ vector.
    std::unique_ptr<VulkanPipeline> coop_pipeline_;
    VkDescriptorSet coop_ds_[vkop::kInflight] = {nullptr};
    // coop_i8_pipeline_ runs the W8A8 SINT8 cooperative-matrix kernel (int8
    // weight x runtime-quantized int8 activation -> flat fp32 scratch). Built
    // only when the device lists the SINT8 coopmat combo.
    std::unique_ptr<VulkanPipeline> coop_i8_pipeline_;
    VkDescriptorSet coop_i8_ds_[vkop::kInflight] = {nullptr};
    std::unique_ptr<VulkanPipeline> pack_pipeline_;
    VkDescriptorSet pack_ds_[vkop::kInflight] = {nullptr};
    std::vector<VkWriteDescriptorSet> ds_writes_;
    bool transB_ = false; // B laid out as [batch, N, K] (folded Transpose)
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
};

} // namespace ops
} // namespace vkop
#endif // OPS_MATMUL_HPP_
