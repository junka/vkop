"""Shader-key manifest for onnx2vkop's per-model archive selection.

Mirrors core/ShaderKeys.hpp on the C++ side. The Python copy is what a
per-model vkop_shaders.bin is filtered by; the C++ copy is what a running
Runtime looks each op up in. They MUST agree — a mismatch means the archive
either lacks a shader the graph will ask for (Runtime falls back to an
embedded pointer, defeating strip mode) or ships shaders the graph never
touches (wasted bytes). model/pypi/tests/test_shader_map_agrees.py parses
both sides and fails on drift.

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
  · Secondary pipelines (Matmul's coop / coop16 / coop_i8 / pack, Conv2d's
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
    "Floor": ("floor", "floor", False, False),
    "FusedElemwise": (None, "fused_elemwise", False, True),
    "Gather": (None, "gather", False, True),
    "Gemm": (None, "gemm", False, False),
    "GlobalAveragePool": ("globalaveragepool", None, False, False),
    "GridSample": ("gridsample", None, False, False),
    "LayerNorm": ("layernorm", "layernorm", False, True),
    "LayerNormalization": ("layernorm", "layernorm", False, True),
    "MatMul": ("matmul", "matmul", False, True),
    "Max": (None, "max", False, True),
    "MaxPool": ("maxpool2d", None, True, False),
    "MaxPool2d": ("maxpool2d", None, True, False),
    "Maxpool2d": ("maxpool2d", None, True, False),
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
    "Relu": ("relu", "relu", False, False),
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
        "buffer_matmul_coop_i8",
        "buffer_matmul_pack",
    ],
    ("MatMul", "image"): [
        "image_matmul_nv",
        "image_matmul_coop",
    ],
    # Conv2d's fp16 pack pass reuses buffer_matmul_pack (same shader), but
    # ship it explicitly so removing that shader from MatMul would trip a
    # loud test failure here rather than a silent miss at load time.
    ("Conv2d", "buffer"): ["buffer_matmul_pack"],
    ("Conv", "buffer"): ["buffer_matmul_pack"],
    # int64 shape-meta chains built by Expand/Concat/Gather/Equal/Where on
    # an int64 input; their set_runtime_device() creates a second pipeline
    # keyed by *_int64. Only present on the buffer backend -- the image
    # backend has no int64 path.
    ("Expand", "buffer"): ["buffer_expand_int64"],
    ("Concat", "buffer"): ["buffer_concat_int64"],
    ("Gather", "buffer"): ["buffer_gather_int64"],
    ("Equal", "buffer"): ["buffer_equal_int64"],
    ("Where", "buffer"): ["buffer_where_int64"],
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

    `backend` picks image or buffer primary; the secondary table is emitted
    regardless because those pipelines are selected on device features the
    converter cannot know. `emit_fp16` broadens primary keys to include the
    fp16 dual build; False keeps the archive minimal when the target is
    known to be fp32.
    """
    if backend not in ("buffer", "image"):
        raise ValueError(f"unknown backend {backend!r}, want buffer|image")
    keys: Set[str] = set()
    for op in set(op_types):
        keys.update(_primary_keys(op, backend, emit_fp16))
        keys.update(_SECONDARY.get((op, backend), ()))
    return keys


def known_ops() -> Set[str]:
    """Every OpType string the map knows about. Tests use this to catch
    drift when a new operator lands on one side but not the other."""
    return set(_PRIMARY.keys()) | set(_SECONDARY.keys())
