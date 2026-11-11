"""Shader-key manifest for onnx2vkop's per-model archive selection.

Mirrors core/ShaderKeys.hpp on the C++ side. The Python copy is what a
per-model vkop_shaders.bin is filtered by; the C++ copy is what a running
Runtime looks each op up in. They MUST agree — a mismatch means the archive
either lacks a shader the graph will ask for (Runtime falls back to an
embedded pointer, defeating strip mode) or ships shaders the graph never
touches (wasted bytes). model/pypi/tests/test_shader_map_agrees.py parses
both sides and fails on drift.

Since the kernel plan landed, agreement is no longer something the two sides
have to *re-derive* at run time: `plan_for_op` is written into the .vkopbin
as `Model.kernel_plans` and Node.plan points at it, so the runtime reads the
answer the converter committed to (C++'s table is the fallback for files
written before the field existed, and for the host-side post-process ops).
The same plans drive the archive filter above, so "what the file says" and
"what the file carries" cannot disagree.

Keys are the ONNX-style op_type strings a vkopbin actually stores (Conv,
MaxPool, BatchNormalization, ...) — the same inputs C++'s
convert_opstring_to_enum() accepts. Where the vkop-internal name differs
("Conv2d", "Maxpool2d", ...) we accept both spellings so a converter pass
that writes the internal name still maps cleanly.

Selection at conversion time is intentionally wider than what the C++
Runtime actually dispatches on a given GPU. Two kinds of over-inclusion:

  · fp16 variants even for fp32 graphs — a driver that runs the model on a
    GPU with only fp32 shader support picks the fp32 key at load time; a
    GPU that supports fp16 might pick the fp16 key. Emitting both keeps the
    archive portable across targets without a reconvert.
  · Secondary pipelines (Matmul's coop / coop16 / coop_i8 / pack / w8_wide,
    Conv2d's
    pack, the int64 twin of Expand/Concat/Gather/Equal/Where). These are
    built lazily inside each op's set_runtime_device() based on device
    features and input dtype. Conversion time cannot know them, so we ship
    all of them for any op whose class has secondaries.

Backend selection IS knowable at conversion time — the `-r/--rgba` flag
picks image vs buffer, and that flag decides what the whole graph runs on.
Callers pass it through.
"""
from typing import Dict, Iterable, List, Set

# Each OpType string (as it appears in the vkopbin's Node.op_type) maps to
# the shader basename per backend. None means "no such variant"; the runtime
# falls back to whichever side exists. `_fp16` and `_int64` suffixes are
# applied by the helper functions below, not stored here.
_PRIMARY: Dict[str, Dict[str, object]] = {
    # op: (image_basename, buffer_basename, image_has_fp16, buffer_has_fp16)
    "Add": ("add", "add", True, True),
    "Atan": ("atan", "atan", False, True),
    "AveragePool": ("averagepool", None, False, False),
    "BatchNorm": ("batchnorm", None, False, False),
    "BatchNormalization": ("batchnorm", None, False, False),
    "Col2Im": ("col2im", None, False, False),
    "Concat": ("concat", "concat", False, True),
    "Conv": ("conv2d", "conv2d", True, True),
    "Conv2d": ("conv2d", "conv2d", True, True),
    "Cos": ("cos", "cos", False, True),
    "Div": ("div", "div", False, True),
    "EmbeddingForward": (None, "embedding_forward", False, False),
    "Erf": ("erf", "erf", False, True),
    "Equal": (None, "equal", False, False),
    "Expand": (None, "expand", False, True),
    "Floor": ("floor", "floor", False, True),
    "FusedElemwise": (None, "fused_elemwise", False, True),
    "Gather": (None, "gather", False, True),
    "Gemm": (None, "gemm", False, False),
    "GlobalAveragePool": ("globalaveragepool", "globalaveragepool", False, True),
    "GridSample": ("gridsample", None, False, False),
    "LayerNorm": ("layernorm", "layernorm", False, True),
    "LayerNormalization": ("layernorm", "layernorm", False, True),
    "MatMul": ("matmul", "matmul", False, True),
    "Max": (None, "max", False, True),
    "MaxPool": ("maxpool2d", "maxpool2d", True, True),
    "MaxPool2d": ("maxpool2d", "maxpool2d", True, True),
    "Maxpool2d": ("maxpool2d", "maxpool2d", True, True),
    "Min": (None, "min", False, True),
    "Mod": (None, "mod", False, True),
    "Mul": ("mul", "mul", False, True),
    "Neg": ("neg", "neg", False, True),
    "Nms": ("nms", None, False, False),
    "NonZero": (None, "nonzero", False, False),
    "Pow": ("pow", "pow", False, True),
    "PRelu": ("prelu", "prelu", False, True),
    "QuantizeLinear": (None, "quantize_linear", False, False),
    "DequantizeLinear": (None, "quantize_linear", False, False),
    "Range": (None, "range", False, False),
    "Reduce": ("reduce", "reduce", False, True),
    "ReduceMean": ("reduce", "reduce", False, True),
    "Relu": ("relu", "relu", False, True),
    "Reshape": ("reshape", "reshape", False, True),
    "Resize": ("resize", "resize", False, True),
    "RMSNorm": (None, "rmsnorm", False, True),
    "RotaryEmbedding": (None, "rotary", False, True),
    "ScatterElements": (None, "scatter_elements", False, True),
    "ScatterND": (None, "scatternd", False, False),
    "Sigmoid": ("sigmoid", "sigmoid", False, True),
    "Sin": ("sin", "sin", False, True),
    "Slice": ("slice", "slice", False, True),
    "Softmax": ("softmax", "softmax", True, True),
    "Softplus": ("softplus", "softplus", False, True),
    "Split": ("split", "split", False, True),
    "Sqrt": ("sqrt", "sqrt", False, True),
    "Sub": ("sub", "sub", False, True),
    "Tanh": ("tanh", "tanh", False, True),
    "TopK": (None, "topk", False, False),
    "Topk": (None, "topk", False, False),
    "Transpose": ("transpose", "transpose", False, True),
    "Attention": (None, "attention", False, True),
    "Cast": (None, "cast", False, False),
    "Where": (None, "where", False, False),
    # Host-only ops (Shape / Squeeze / Unsqueeze / Identity) have no shader
    # and are absent from this table.
}

# Ops whose set_runtime_device() builds a SECONDARY VulkanPipeline that
# override_spv() does not reach — those extras go through Operator::
# resolve_spv() with a hard-coded key. Every key listed here is emitted
# unconditionally when the op appears in the graph, because whether the
# runtime picks one (e.g. coop16 vs coop) is a device-feature query the
# converter cannot make.
_SECONDARY: Dict[tuple, List[str]] = {
    # Keyed by (op, backend). Buffer and image paths never share a shader,
    # so listing them under the wrong backend just ships dead bytes.
    ("MatMul", "buffer"): [
        "buffer_matmul_coop",
        "buffer_matmul_coop16",
        # Larger 16x16x16 workgroups of the same coop16 source (-DACC_M/-DACC_N):
        # 64x64 and 64x128 output footprints. Which one a MatMul dispatches is a
        # host-side shape decision, so like the coop split itself every MatMul
        # must carry both.
        "buffer_matmul_coop16_2x1",
        "buffer_matmul_coop16_2x2",
        # BDA table builds of the two ACC geometries.
        "buffer_matmul_coop16_2x1_bda",
        "buffer_matmul_coop16_2x2_bda",
        # The other fp16 tiles the same source can declare (-DCM/-DCN/-DCK).
        # The runtime probes the device's cooperative-matrix table and only a
        # listed shape is ever dispatched, but which shapes are listed is
        # decided at load time, so every MatMul carries the ones it may need:
        # 16x8x16 is the Ada half-width tile (Ada does not list 8x8x16, and
        # the ARL 8x8x16 tile stays on buffer_matmul_coop which is faster
        # there). 16x8x8 is NOT listed — it has the same BM=32/BN=32 footprint
        # and CK=16 (16x8x16) always wins the alt-geo lookup on both hardware
        # and perf; we removed it from the archive entirely.
        "buffer_matmul_coop16_16x8x16",
        "buffer_matmul_coop_i8",
        # 16x16x32 SINT8 tile (Ada). 4x the per-MMA output area of the 8x8x32
        # default build, and the only int8 shape Ada lists.
        "buffer_matmul_coop_i8_16x16x32",
        "buffer_matmul_pack",
        # The int8/fp8 weight-only WIDE GEMV: a second fp16 build of
        # buffer/matmul.comp (-DW8_WIDE), not its own .comp, so nothing derives
        # this key from a file name. MatMulBuffer::w8_wide_pipeline builds it
        # lazily only for a byte weight with N >= 896 and one A row, which the
        # converter cannot know, so it ships with every MatMul.
        "buffer_matmul_w8_wide_fp16",
        # The 512-lane split-K GEMV builds of the same file (-DKSPLIT512,
        # VKOP_KSPLIT512=1 host switch): ksplit-only general module plus the
        # int8 WIDE module. Probed lazily like the WIDE one; each falls back
        # to the 256-lane kernel where the driver or this archive refuses.
        "buffer_matmul_ksplit512_fp16",
        "buffer_matmul_ksplit512_w8_wide_fp16",
        # The -DSUBGROUP build of the base module: only matmul16_softmax's row
        # reduce differs (subgroupMax/subgroupAdd instead of the two 8-round
        # shared trees). The host swaps the primary key for this one when the
        # device advertises subgroup arithmetic, so both must ship.
        "buffer_matmul_sg",
        "buffer_matmul_fp16_sg",
        "buffer_matmul_sg_bda",
        "buffer_matmul_fp16_sg_bda",
        # The SwiGLU down-projection fold (-DDOWN_ACT, a third build of the
        # same file): one module holding both the tiled and the split-K GEMV
        # entry points, and the only MatMul build that reads Wg/Wu (bindings
        # 7/8). MatMulBuffer::down_act_pipeline probes it lazily, so like the
        # WIDE and 512 keys it has to ship with every MatMul -- the operator
        # that needs it is a plain MatMul, not a distinct op_type.
        "buffer_matmul_down_act_fp16",
        # The M>1 build of the same source (no DA_M1_ONLY): keeps the
        # multi-row deposit walk that the M==1 module drops. MatMul routes
        # M==1 to _fp16 and M>1 to this key, so both must ship.
        "buffer_matmul_down_act_multi_fp16",
        # BDA table builds of the same two fold modules.
        "buffer_matmul_down_act_bda_fp16",
        "buffer_matmul_down_act_multi_bda_fp16",
        # Device-address table builds (one table descriptor + data pointers in
        # the table). The host only uses these for the base submit() routes.
        "buffer_matmul_bda",
        "buffer_matmul_bda_fp16",
        # Table builds of the secondary same-source variants: WIDE GEMV,
        # 512-lane split-K, and the tiled split/fold pair.
        "buffer_matmul_w8_wide_bda_fp16",
        "buffer_matmul_ksplit512_bda_fp16",
        "buffer_matmul_ksplit512_w8_wide_bda_fp16",
        "buffer_matmul_tiled_ks_split_bda_fp16",
        "buffer_matmul_tiled_ks_fold_bda_fp16",
        # Pack pass (direct addresses in the push constant).
        "buffer_matmul_pack_bda",
        # Coop pipelines: primary + alternative tile geometries.
        "buffer_matmul_coop_bda",
        "buffer_matmul_coop16_bda",
        # Alternative coop tile geometries.
        "buffer_matmul_coop16_16x8x16_bda",
        "buffer_matmul_coop_fp16_8x8x16_bm8bn64_bda",
        "buffer_matmul_coop_fp16_8x8x16_bm32bn16_bda",
        "buffer_matmul_coop_fp16_8x8x16_bm64bn8_bda",
        # SINT8/UINT8 coop modules.
        "buffer_matmul_coop_i8_bda",
        "buffer_matmul_coop_i8_16x16x32_bda",
        "buffer_matmul_coop_i8_16x8x32_bda",
        "buffer_matmul_coop_u8_bda",
        "buffer_matmul_coop_u8_16x16x32_bda",
        "buffer_matmul_coop_u8_16x8x32_bda",
        # The tile-split-K pair for the tiled fp16 kernel (-DTILED_KS_SPLIT
        # deposit pass, -DTILED_KS_FOLD fold pass; one source, two modules,
        # because a pipeline binds a single entry point and the split half is
        # barrier-free). MatMulBuffer::ks_split_pipeline/ks_fold_pipeline probe
        # them lazily and only the host's per-device routing knows whether the
        # K cut is used, so both ship with every MatMul. With either key
        # absent the archive cannot answer and the op falls back to the plain
        # tiled kernel.
        "buffer_matmul_tiled_ks_split_fp16",
        "buffer_matmul_tiled_ks_fold_fp16",
        # coopmat + split-K hybrid: SPLITK_DEPOSIT main pass (fp32 partial to
        # scratch) and SPLITK_FOLD reduce pass (fp32-sum partials, pack half2
        # to binding 0). ARL's 8x8x16 source (matmul_coop.comp) and Ada's
        # 16x16x16 source (matmul_coop16.comp) each ship both builds; runtime
        # picks coop vs coop16 the same way it always did, then resolves the
        # matching *splitk_{deposit,fold} key.
        "buffer_matmul_coop_splitk_deposit",
        "buffer_matmul_coop_splitk_fold",
        # coopmat + split-K in-workgroup: SPLITK_INWG runs cs serial K-slice
        # passes inside one workgroup, cross-pass fp32 accumulate in shared
        # memory, single dispatch, no global scratch. coop_split (uPC.ksplit)
        # clamped to 1..8 on the host. Guard: ACC_M==ACC_N==1 + N even + K/cs
        # >= 2*K_BLOCK. Only the coop build exists -- the coop16 INWG variant
        # was never shipped (CMakeLists builds coop only) and KernelSelect's
        # kCoopmatInwg route treats it as unavailable, so no coop16 key ships.
        "buffer_matmul_coop_inwg",
        # coopmat + split-K grid-parallel "last block reduce": grid.z = batch
        # * SK, single dispatch, fp32 partials in binding 4, per-tile arrival
        # counter in binding 5 (host vkCmdFillBuffer 0 before dispatch). Host
        # guard: SK ∈ [2, 8], K/SK >= 2*K_BLOCK, N even, ACC_M==ACC_N==1.
        # Only fp32 builds exist (the half2 builds were byte-identical and
        # dropped from CMake; MatMul.hpp still probes the half2 keys but they
        # are optional). Per-cs modules bake SK_CS in at compile time; the
        # runtime picks by cs and falls back to the default (cs=1) module, so
        # all four must ship.
        "buffer_matmul_coop16_sklast_fp32",
        "buffer_matmul_coop16_sklast_fp32_cs2",
        "buffer_matmul_coop16_sklast_fp32_cs4",
        "buffer_matmul_coop16_sklast_fp32_cs8",
        # BDA table builds of the four cs variants (Ada-only route).
        "buffer_matmul_coop16_sklast_fp32_bda",
        "buffer_matmul_coop16_sklast_fp32_cs2_bda",
        "buffer_matmul_coop16_sklast_fp32_cs4_bda",
        "buffer_matmul_coop16_sklast_fp32_cs8_bda",
    ],
    ("MatMul", "image"): [
        "image_matmul_nv",
        "image_matmul_coop",
    ],
    # Conv2d's fp16 pack pass reuses buffer_matmul_pack (same shader), but
    # ship it explicitly so removing that shader from MatMul would trip a
    # loud test failure here rather than a silent miss at load time.
    ("Conv2d", "buffer"): [
        "buffer_matmul_pack",
        "buffer_matmul_pack_bda",
        "buffer_conv2d_bda",
        "buffer_conv2d_bda_fp16",
    ],
    ("Conv", "buffer"): [
        "buffer_matmul_pack",
        "buffer_matmul_pack_bda",
        "buffer_conv2d_bda",
        "buffer_conv2d_bda_fp16",
    ],
    # int64 shape-meta chains built by Expand/Concat/Gather/Equal/Where on
    # an int64 input; their set_runtime_device() creates a second pipeline
    # keyed by *_int64. Only present on the buffer backend -- the image
    # backend has no int64 path.
    # NOTE: the int64 key and the BDA builds share this row -- a second
    # ("Expand", "buffer") dict-literal key would silently overwrite it.
    ("Expand", "buffer"): [
        "buffer_expand_int64",
        "buffer_expand_int64_bda",
        "buffer_expand_bda",
        "buffer_expand_bda_fp16",
    ],
    ("Concat", "buffer"): [
        "buffer_concat_int64",
        "buffer_concat_int64_bda",
        "buffer_concat_bda",
        "buffer_concat_bda_fp16",
    ],
    ("Gather", "buffer"): [
        "buffer_gather_int64",
        "buffer_gather_int64_bda",
        "buffer_gather_bda",
        "buffer_gather_bda_fp16",
    ],
    ("Equal", "buffer"): [
        "buffer_equal_int64",
        "buffer_equal_int64_bda",
        "buffer_dispatch_from_shape",
        "buffer_dispatch_from_shape_bda",
        "buffer_equal_bda",
    ],
    ("Where", "buffer"): [
        "buffer_where_int64",
        "buffer_where_int64_bda",
        "buffer_where_bda",
    ],
    # The GPU-shape indirect dispatch pre-pass (VKOP_GPU_SHAPE):
    # dispatch_from_shape.comp turns the out shape SSBO into a
    # VkDispatchIndirectCommand so BufferBinaryFactory's broadcast==2 path can
    # dispatch via vkCmdDispatchIndirect. Built lazily inside
    # execute_gpu_shape's ensure_indirect_prepass(); whether the env flag is
    # on is unknowable at conversion time, so it ships with every op whose
    # class routes through execute_gpu_shape.
    # NOTE: Equal's entry is merged with its int64 entry above — a second
    # ("Equal", "buffer") dict-literal key would silently overwrite it.
    ("Add", "buffer"): [
        "buffer_dispatch_from_shape",
        "buffer_dispatch_from_shape_bda",
        "buffer_add_bda",
        "buffer_add_bda_fp16",
    ],
    ("Sub", "buffer"): [
        "buffer_dispatch_from_shape",
        "buffer_dispatch_from_shape_bda",
        "buffer_sub_bda",
        "buffer_sub_bda_fp16",
    ],
    ("Mul", "buffer"): [
        "buffer_dispatch_from_shape",
        "buffer_dispatch_from_shape_bda",
        "buffer_mul_bda",
        "buffer_mul_bda_fp16",
    ],
    ("Div", "buffer"): [
        "buffer_dispatch_from_shape",
        "buffer_dispatch_from_shape_bda",
        "buffer_div_bda",
        "buffer_div_bda_fp16",
    ],
    ("Pow", "buffer"): [
        "buffer_dispatch_from_shape",
        "buffer_dispatch_from_shape_bda",
        "buffer_pow_bda",
        "buffer_pow_bda_fp16",
    ],
    ("Min", "buffer"): [
        "buffer_dispatch_from_shape",
        "buffer_dispatch_from_shape_bda",
        "buffer_min_bda",
        "buffer_min_bda_fp16",
    ],
    ("Max", "buffer"): [
        "buffer_dispatch_from_shape",
        "buffer_dispatch_from_shape_bda",
        "buffer_max_bda",
        "buffer_max_bda_fp16",
    ],
    ("Mod", "buffer"): [
        "buffer_dispatch_from_shape",
        "buffer_dispatch_from_shape_bda",
        "buffer_mod_bda",
        "buffer_mod_bda_fp16",
    ],
    # The subgroup-arithmetic builds of the hot workgroup reductions
    # (-DSUBGROUP: attention's per-tile max/sum, the norm passes, softmax).
    # Which module runs is a device-feature query at load time
    # (VK_SUBGROUP_FEATURE_ARITHMETIC_BIT), which the converter cannot make; the
    # runtime swaps the primary key for <key>_sg and falls back to the base
    # module when this archive lacks it. Both must therefore ship whenever the
    # op does.
    ("Attention", "buffer"): [
        "buffer_attention_sg",
        "buffer_attention_fp16_sg",
        "buffer_attention_bda",
        "buffer_attention_bda_fp16",
        "buffer_attention_sg_bda",
        "buffer_attention_fp16_sg_bda",
    ],
    ("FusedElemwise", "buffer"): [
        "buffer_fused_elemwise_bda",
        "buffer_fused_elemwise_bda_fp16",
    ],
    ("Gemm", "buffer"): ["buffer_gemm_bda"],
    ("MaxPool", "buffer"): [
        "buffer_maxpool2d_bda",
        "buffer_maxpool2d_bda_fp16",
    ],
    ("MaxPool2d", "buffer"): [
        "buffer_maxpool2d_bda",
        "buffer_maxpool2d_bda_fp16",
    ],
    ("Maxpool2d", "buffer"): [
        "buffer_maxpool2d_bda",
        "buffer_maxpool2d_bda_fp16",
    ],
    ("Reshape", "buffer"): ["buffer_reshape_bda", "buffer_reshape_bda_fp16"],
    ("ScatterElements", "buffer"): [
        "buffer_scatter_elements_bda",
        "buffer_scatter_elements_bda_fp16",
    ],
    ("ScatterND", "buffer"): ["buffer_scatternd_bda"],
    ("Slice", "buffer"): ["buffer_slice_bda", "buffer_slice_bda_fp16"],
    ("Split", "buffer"): ["buffer_split_bda", "buffer_split_bda_fp16"],
    ("TopK", "buffer"): ["buffer_topk_bda"],
    ("Topk", "buffer"): ["buffer_topk_bda"],
    ("LayerNorm", "buffer"): [
        "buffer_layernorm_sg",
        "buffer_layernorm_fp16_sg",
        "buffer_layernorm_bda",
        "buffer_layernorm_bda_fp16",
        "buffer_layernorm_sg_bda",
        "buffer_layernorm_fp16_sg_bda",
    ],
    ("LayerNormalization", "buffer"): [
        "buffer_layernorm_sg",
        "buffer_layernorm_fp16_sg",
        "buffer_layernorm_bda",
        "buffer_layernorm_bda_fp16",
        "buffer_layernorm_sg_bda",
        "buffer_layernorm_fp16_sg_bda",
    ],
    ("RMSNorm", "buffer"): [
        "buffer_rmsnorm_sg",
        "buffer_rmsnorm_fp16_sg",
        "buffer_rmsnorm_bda",
        "buffer_rmsnorm_bda_fp16",
        "buffer_rmsnorm_sg_bda",
        "buffer_rmsnorm_fp16_sg_bda",
    ],
    ("Reduce", "buffer"): ["buffer_reduce_bda", "buffer_reduce_bda_fp16"],
    ("ReduceMean", "buffer"): [
        "buffer_reduce_bda",
        "buffer_reduce_bda_fp16",
    ],
    ("Transpose", "buffer"): [
        "buffer_transpose_int64",
        "buffer_transpose_int64_bda",
        "buffer_transpose_bda",
        "buffer_transpose_bda_fp16",
    ],
    ("GlobalAveragePool", "buffer"): [
        "buffer_globalaveragepool_bda",
        "buffer_globalaveragepool_bda_fp16",
    ],
    ("RotaryEmbedding", "buffer"): [
        "buffer_rotary_bda",
        "buffer_rotary_bda_fp16",
    ],
    ("EmbeddingForward", "buffer"): ["buffer_embedding_forward_bda"],
    ("PRelu", "buffer"): ["buffer_prelu_bda", "buffer_prelu_bda_fp16"],
    ("Resize", "buffer"): ["buffer_resize_bda", "buffer_resize_bda_fp16"],
    ("Softmax", "buffer"): [
        "buffer_softmax_sg",
        "buffer_softmax_fp16_sg",
        "buffer_softmax_bda",
        "buffer_softmax_bda_fp16",
        "buffer_softmax_sg_bda",
        "buffer_softmax_fp16_sg_bda",
    ],
    # The image backend's LayerNorm has no fp16 build, so there is a single
    # subgroup module to carry (its three shapes -- W / HW / CHW -- all fold
    # through the vec4 reduce).
    ("LayerNorm", "image"): ["image_layernorm_sg"],
    ("LayerNormalization", "image"): ["image_layernorm_sg"],
    # Buffer-device-address variants. The runtime builds this pipeline only
    # when the device advertises BDA and the key is present, and always keeps
    # the descriptor pipeline as the per-dispatch fallback; shipping both
    # therefore cannot break a non-BDA target. The unary ops below all share
    # shaders/common/unary_elementwise_body.comp; add each new one here so a converted
    # model actually embeds it instead of silently falling back.
    ("Neg", "buffer"): ["buffer_neg_bda", "buffer_neg_bda_fp16"],
    ("Sigmoid", "buffer"): ["buffer_sigmoid_bda", "buffer_sigmoid_bda_fp16"],
    ("Tanh", "buffer"): ["buffer_tanh_bda", "buffer_tanh_bda_fp16"],
    ("Erf", "buffer"): ["buffer_erf_bda", "buffer_erf_bda_fp16"],
    ("Sqrt", "buffer"): ["buffer_sqrt_bda", "buffer_sqrt_bda_fp16"],
    ("Atan", "buffer"): ["buffer_atan_bda", "buffer_atan_bda_fp16"],
    ("Cos", "buffer"): ["buffer_cos_bda", "buffer_cos_bda_fp16"],
    ("Sin", "buffer"): ["buffer_sin_bda", "buffer_sin_bda_fp16"],
    ("Softplus", "buffer"): [
        "buffer_softplus_bda",
        "buffer_softplus_bda_fp16",
    ],
    ("Relu", "buffer"): ["buffer_relu_bda", "buffer_relu_bda_fp16"],
    ("Floor", "buffer"): ["buffer_floor_bda", "buffer_floor_bda_fp16"],
    # Cast's single module handles both directions, so only the fp32 key.
    ("Cast", "buffer"): [
        "buffer_cast_int64",
        "buffer_cast_int64_bda",
        "buffer_cast_bda",
    ],
    # QuantizeLinear/DequantizeLinear share one format-polymorphic module.
    ("QuantizeLinear", "buffer"): ["buffer_quantize_linear_bda"],
    ("DequantizeLinear", "buffer"): ["buffer_quantize_linear_bda"],
}


def _primary_keys(op: str, backend: str, emit_fp16: bool) -> Iterable[str]:
    spec = _PRIMARY.get(op)
    if spec is None:
        return ()
    image, buffer_, img_fp16, buf_fp16 = spec
    if backend == "buffer":
        base, has_fp16 = buffer_, buf_fp16
        prefix = "buffer_"
    else:
        base, has_fp16 = image, img_fp16
        prefix = "image_"
    if base is None:
        # The op has no variant for this backend; if it has a variant on the
        # other one, ship that too. Runtime falls back to the same symbol.
        base = image or buffer_
        prefix = "image_" if image else "buffer_"
        has_fp16 = (image and img_fp16) or (buffer_ and buf_fp16)
        if base is None:
            return ()
    keys = [f"{prefix}{base}"]
    if emit_fp16 and has_fp16:
        keys.append(f"{prefix}{base}_fp16")
    return keys


def shader_keys_for_model(
    op_types: Iterable[str],
    backend: str = "buffer",
    emit_fp16: bool = True,
) -> Set[str]:
    """Return the archive keys a model built from `op_types` may load.

    `op_types` is any iterable of ONNX-style op-type strings — the same
    strings written to `Node.op_type` in the vkopbin. Unknown names are
    silently skipped (a converter pass that emits a not-yet-registered op is
    a shader_map maintenance failure, not a conversion crash; the C++ side
    has the same policy in `convert_opstring_to_enum`).

    `backend` picks image or buffer primary (plus "auto" for both); the
    secondary table is emitted regardless because those pipelines are selected
    on device features the converter cannot know. `emit_fp16` broadens primary
    keys to include the fp16 dual build; False keeps the archive minimal when
    the target is known to be fp32.

    Derived from `kernel_plans_for_model`, which is the same table written
    into the .vkopbin as `Model.kernel_plans`. That is the point: the file
    records the keys, and the archive is filtered to exactly those keys, so
    the two can no longer drift apart (with no built-in SPIR-V in libvkop a
    drift is a hard failure at pipeline creation, not a fallback).
    """
    if backend not in ("buffer", "image", "auto"):
        raise ValueError(f"unknown backend {backend!r}, want buffer|image|auto")
    keys: Set[str] = set()
    for op in set(op_types):
        plan = plan_for_op(op)
        if plan is None:
            continue
        columns = ("image", "buffer") if backend == "auto" else (backend,)
        for col in columns:
            cells = (col, col + "_fp16") if emit_fp16 else (col,)
            for cell in cells:
                if plan[cell]:
                    keys.add(plan[cell])
        for v in plan["variants"]:
            # Variants carry their backend's prefix (enforced by
            # test_shader_map_agrees), so a pinned backend takes only its own
            # half -- the same selection _SECONDARY used to make.
            if backend == "auto" or v.startswith(backend + "_"):
                keys.add(v)
    return keys


def _column(op: str, backend: str) -> tuple:
    """(fp32 key, fp16 key) for one op on one backend.

    The fp16 key IS the fp32 key when the op's variant is not in
    DUAL_FP16_SHADERS: the runtime asks for "..._fp16" only when the op
    actually has one, so recording the fallback here keeps the cell
    meaningful ("what key does fp16 resolve to") instead of empty.
    """
    keys = tuple(_primary_keys(op, backend, emit_fp16=True))
    if not keys:
        return "", ""
    return keys[0], (keys[1] if len(keys) > 1 else keys[0])


def plan_for_op(op: str):
    """The kernel plan for one op_type, or None when it has no GPU shader.

    None covers the host-only ops (Shape / Squeeze / Unsqueeze / Identity) and
    any name the tables do not know; both mean "no key for this op", which is
    also what the C++ fallback table returns for them, so a planless node
    behaves exactly as it did before plans existed.

    The returned dict mirrors the KernelPlan flatbuffer table field for field.
    """
    if op not in _PRIMARY:
        return None
    img, img16 = _column(op, "image")
    buf, buf16 = _column(op, "buffer")
    if not (img or buf):
        return None
    variants = sorted(set(_SECONDARY.get((op, "image"), ())) |
                      set(_SECONDARY.get((op, "buffer"), ())))
    return {
        "op_type": op,
        "image": img,
        "image_fp16": img16,
        "buffer": buf,
        "buffer_fp16": buf16,
        "variants": variants,
    }


def kernel_plans_for_model(op_types: Iterable[str]):
    """Return (rows, index) for a graph's op census.

    `rows` is the deduplicated KernelPlan list to write into Model.kernel_plans
    (first-appearance order, so two conversions of the same graph produce the
    same table) and `index` maps op_type -> row index for Node.plan. An
    op_type with no plan is absent from `index`, i.e. its nodes carry -1.
    """
    rows: List[dict] = []
    index: Dict[str, int] = {}
    for op in op_types:
        if op in index:
            continue
        plan = plan_for_op(op)
        if plan is None:
            continue
        index[op] = len(rows)
        rows.append(plan)
    return rows, index


def known_ops() -> Set[str]:
    """Every op_type string the shader map carries a shader for. Tests use
    this to catch drift when a new operator lands on one side but not the
    other. (`_SECONDARY` is keyed by (op, backend) tuples, not names, so it
    contributes nothing here.)"""
    return set(_PRIMARY.keys())


# The ops whose buffer shader build follows the DTYPE OF THE TENSORS they are
# handed, rather than the --precision flag: exactly the case list of
# core/runtime.cpp's `op_shader_fp16`, spelled with the ONNX op_type strings
# this table is keyed by (all aliases of an enum, e.g. Conv and Conv2d, are
# listed). Every name here returns
# `data_input->dtype() == typeid(uint16_t) ? 1 : 0` for its fp16 choice; an op
# NOT here falls through to `default: return precision;`, so its build is
# picked from the flag while its buffers are allocated 2-bytes-per-element from
# the declaration -- an fp32 kernel over half-typed storage is garbage, not an
# error. That asymmetry is why an fp16 artifact can only be built out of ops
# on this list (see fp16_graph.py).
#
# The shipped converter cannot read C++, so this is a copy, guarded in both
# directions by test_shader_map_agrees.py, which parses the switch out of
# core/runtime.cpp and the enum→string map out of ops/Ops.hpp.
# The list is the switch's case labels translated verbatim, including TopK. It
# is the necessary half only: `fp16_dtype_driven` below adds the build half,
# which is where TopK -- in the switch, but with a single buffer shader -- is
# put back outside the domain.
_FP16_DTYPE_DRIVEN: Set[str] = {
    "Add", "Atan", "Attention", "Concat", "Conv", "Conv2d", "Cos", "Div",
    "Erf", "Expand", "Floor", "FusedElemwise", "Gather", "GlobalAveragePool",
    "LayerNorm", "LayerNormalization", "MatMul", "Max", "MaxPool", "MaxPool2d",
    "Maxpool2d", "Min", "Mod", "Mul", "Neg", "PRelu", "Pow", "Reduce",
    "ReduceMean", "Relu", "Reshape", "Resize", "RMSNorm", "RotaryEmbedding",
    "ScatterElements", "Sigmoid", "Sin", "Slice", "Softmax", "Softplus",
    "Split", "Sqrt", "Sub", "Tanh", "TopK", "Topk", "Transpose",
}


def fp16_dtype_driven(op_type: str) -> bool:
    """Can an fp16 artifact put this op inside its FP16 domain?

    True needs both halves: the runtime must choose this op's build from the
    tensor dtype (the list above), and the op must actually HAVE a distinct
    fp16 buffer build. Either one missing means the graph stays in the FP32
    domain at this node.

    TopK is the case the second half exists for: it IS in the dispatch switch,
    so declaring its tensors fp16 sends the runtime looking for a build the
    archive never carried -- `plan_for_op` returns the fp32 key in its
    `buffer_fp16` cell when the op has no dual build, and an op that reads
    half-typed storage through the fp32 kernel is the failure this whole
    question is trying to avoid.
    """
    spec = _PRIMARY.get(op_type)
    if spec is None:
        return False
    return op_type in _FP16_DTYPE_DRIVEN and bool(spec[3])


# Spellings `_PRIMARY` accepts for the archive filter but that the runtime's
# convert_opstring_to_enum does NOT resolve (it wants TopK / MaxPool2d). A
# graph node written with one of these maps to OpType::UNKNOWN at load time, so
# they must not be offered as accepted names to front-end validation.
_ARCHIVE_ONLY_SPELLINGS = {"Topk", "Maxpool2d"}

# Host-only ops: real OpType values the runtime handles on the host with no
# shader, hence absent from `_PRIMARY` (see the closing comment there).
HOST_ONLY_OPS = {"Shape", "Squeeze", "Unsqueeze"}


def runtime_op_names() -> Set[str]:
    """Every op_type string core/runtime.cpp resolves to a real OpType.

    This is the single source of truth for "what a converted graph may
    contain": converter-side validation, the coverage tools, and Ops.hpp's
    convert_opstring_to_enum must all agree on it. It is deliberately NOT
    `known_ops()` — that set carries archive-filter aliases the runtime
    rejects (see `_ARCHIVE_ONLY_SPELLINGS`), and omits the host-only ops.

    test_runtime_op_names.py parses Ops.hpp and fails if this drifts.
    """
    return (set(_PRIMARY) - _ARCHIVE_ONLY_SPELLINGS) | set(HOST_ONLY_OPS)
