"""Standalone tests for fold_cast_into_rotary.

The vision tower exports HF's apply_rotary_pos_emb_vision verbatim, and that
function's dtype juggling

    orig_dtype = tensor.dtype
    tensor = tensor.float()
    ...
    return tensor.to(orig_dtype)

became a real Cast pair in the exported graph, around every one of the 24
blocks' Q and K rotations (48 pairs, 96 dispatches):

    Squeeze [f16] -> Cast(to=f32) -> RotaryEmbedding -> Cast(to=f16)

Dropping the pair is what makes the op fp16 again, but the part worth pinning
is *why it is safe*, because the obvious "it's just a dtype round trip" reading
is wrong: only X is promoted — the rotation runs in fp32 either way, and cos/sin
stay fp32 across the fold. So the tests below check both halves of the claim —

  * the fold moves RE's read point onto the fp16 tensor the up-Cast promoted
    and takes over the fp16 tensor the down-Cast produced (Y), leaving cos/sin
    exactly where they were;
  * and it declines whenever the tensor it would make fp16 has another reader,
    because that op still declares the old dtype and would read half2-packed
    words as fp32.

The runtime half of the contract (a fp16 X next to fp32 cos/sin must load each
at its own stride) is `BufferRankTest.RotaryEmbeddingFp16WithFp32CosSin`.
"""
import numpy as np
import pytest
from onnx import numpy_helper

from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.optimizer import FusionOptimizer

F16, F32 = 10, 1
HD = 64
H = 16
S = 196


def tdict(name, dims=None, dtype=F32):
    return {"name": name, "shape": list(dims or []), "dtype": dtype}


def count(dag, op_type):
    return sum(1 for n in dag.nodes.values() if n.op_type == op_type)


def build(src_dtype=F16, x_dtype=F32, down_to=F16, cos_dtype=F32,
          re_extra_reader=False, down_extra_reader=False, extra_re_output=False):
    """One vision-tower rotation: fp16 source, the Cast pair, RE, and cos/sin.

    `src_dtype` is what the up-Cast reads (fp16 in a real export). The three
    `*_reader` flags each add a second consumer of a tensor the fold touches.
    """
    m = DAGBasedModel()

    # The fp16 X source: rank-3 [S,H,D] (the tower squeezes the batch axis away
    # before RoPE), hung off a Squeeze so it has a producer.
    m.nodes["SQ"] = Node("Squeeze", "SQ", {},
                         [tdict("sq_src", [1, S, H, HD], src_dtype),
                          tdict("sq_ax", [1], 7)],
                         [tdict("sq_x", [S, H, HD], src_dtype)])
    m.initializers["sq_ax"] = numpy_helper.from_array(
        np.array([0], dtype=np.int64), "sq_ax")

    m.nodes["UP"] = Node("Cast", "UP", {"to": F32},
                         [tdict("sq_x", [S, H, HD], src_dtype)],
                         [tdict("up_x", [S, H, HD], x_dtype)])
    m.nodes["RE"] = Node("RotaryEmbedding", "RE", {},
                         [tdict("up_x", [S, H, HD], x_dtype),
                          tdict("cos", [S, HD], cos_dtype),
                          tdict("sin", [S, HD], cos_dtype)],
                         [tdict("re_y", [S, H, HD], x_dtype)])
    m.nodes["DOWN"] = Node("Cast", "DOWN", {"to": down_to},
                           [tdict("re_y", [S, H, HD], x_dtype)],
                           [tdict("down_y", [S, H, HD], F16)])

    if re_extra_reader:
        m.nodes["SINK1"] = Node("Relu", "SINK1", {},
                                [tdict("re_y", [S, H, HD], x_dtype)],
                                [tdict("sink1", [S, H, HD], x_dtype)])
    if down_extra_reader:
        m.nodes["SINK2"] = Node("Relu", "SINK2", {},
                                [tdict("down_y", [S, H, HD], F16)],
                                [tdict("sink2", [S, H, HD], F16)])
    if extra_re_output:
        m.nodes["RE"].outputs.append(tdict("re_y2", [S, H, HD], x_dtype))
        m.nodes["SINK3"] = Node("Relu", "SINK3", {},
                                [tdict("re_y2", [S, H, HD], x_dtype)],
                                [tdict("sink3", [S, H, HD], x_dtype)])
    return m


def matches_of(dag):
    return FusionOptimizer.match_cast_into_rotary(dag)


# ---------------------------------------------------------------- positives

def test_vision_pair_folds():
    """The real export shape: both Casts go, RE reads the fp16 tensor the
    up-Cast promoted and writes the fp16 tensor the down-Cast produced."""
    m = build()
    ms = matches_of(m)
    assert len(ms) == 1, ms
    assert FusionOptimizer.fold_cast_into_rotary(m, ms[0])

    re = m.nodes["RE"]
    assert [i["name"] for i in re.inputs] == ["sq_x", "cos", "sin"]
    assert re.inputs[0]["dtype"] == F16
    # cos/sin are untouched — still the exporter's fp32. The op forwards that as
    # pc.cs_fp32 so the fp16 shader reads them at the fp32 stride.
    assert [i["dtype"] for i in re.inputs[1:]] == [F32, F32]
    # Y takes over the down-Cast's tensor so its consumers still resolve.
    assert [o["name"] for o in re.outputs] == ["down_y"]
    assert re.outputs[0]["dtype"] == F16

    assert "UP" not in m.nodes
    assert "DOWN" not in m.nodes
    # What produced the fp16 X stays: the fold moves the read point, it does not
    # rewire the tensor that already existed.
    assert "SQ" in m.nodes
    assert count(m, "Cast") == 0


def test_cos_sin_fp16_also_folds():
    """Nothing in the fold depends on cos/sin being fp32 — the op reads them at
    whatever dtype they are; only X's read point moves."""
    m = build(cos_dtype=F16)
    ms = matches_of(m)
    assert len(ms) == 1, ms
    assert FusionOptimizer.fold_cast_into_rotary(m, ms[0])
    assert m.nodes["RE"].inputs[0]["dtype"] == F16


def test_shared_down_output_folds():
    """Two readers of the down-Cast's fp16 output are fine: both already expect
    fp16, and after the fold RE is their producer. Nothing about them changes."""
    m = build(down_extra_reader=True)
    ms = matches_of(m)
    assert len(ms) == 1, ms
    assert FusionOptimizer.fold_cast_into_rotary(m, ms[0])
    assert m.nodes["SINK2"].inputs[0]["name"] == "down_y"


# ---------------------------------------------------------------- negatives

def test_shared_upcast_declined():
    """Another op reads the up-Cast's fp32 output: the fold would make that
    tensor fp16 under an op still declaring fp32, which reads half2-packed words
    as fp32 ([[fp32-buffer-op-built-with-fp16-shader]]). Decline."""
    m = build(re_extra_reader=True)
    assert matches_of(m) == []


def test_second_re_output_declined():
    """RE with a second output is not the single-tensor form the fold rewires —
    Y is replaced wholesale, so a second output would be dropped."""
    m = build(extra_re_output=True)
    assert matches_of(m) == []


def test_redundant_up_cast_declined():
    """A Cast(to=f32) over an already-fp32 tensor is a no-op that
    eliminate_redundant_cast (priority 100, ahead of this pass) removes; this
    pass only claims the fp16->fp32 promotion, so it never has to decide what a
    fp32 RE should do."""
    m = build(src_dtype=F32, x_dtype=F32)
    assert matches_of(m) == []


def test_wrong_down_cast_dtype_declined():
    """The wrap must end back on fp16 (to=10). A Cast to something else means
    the pair is not the fp16 round trip, and taking its output would leave RE
    writing a dtype it was not asked for."""
    m = build(down_to=F32)
    assert matches_of(m) == []


def test_bare_re_declined():
    """No Cast pair at all: nothing to fold."""
    m = build()
    del m.nodes["UP"]
    del m.nodes["DOWN"]
    m.nodes["RE"].inputs[0] = tdict("sq_x", [S, H, HD], F16)
    m.nodes["RE"].outputs[0] = tdict("down_y", [S, H, HD], F16)
    assert matches_of(m) == []
