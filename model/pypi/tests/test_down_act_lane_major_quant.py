"""The byte fold must accept the lane-major 3-D gate/up layout.

fold_swiglu_down_proj rewrites Wg/Wu from [K_out, N_up] to
[K_out, ceil(N_up/8), 8]. _quantize_down_act_weights used to bail out on any
weight whose rank was not 2, so all three fold weights stayed fp16 (+1.05 GB
of traffic per decode token) while the host still routed the node to the byte
kernel. These cases pin the converter half of that ABI:

  * the 3-D layout is quantized, and its payload and scale table are
    byte-identical to the 2-D layout's -- the reorder is a reshape, so the
    runtime must see the same values;
  * K_out stays the leading dim, which is what ops/Matmul.hpp now reads
    (shape[rank-2] would have read the band count, N_up/8);
  * a layout whose middle axis does not cover N_up stays fp16 instead of
    producing a wrong answer.
"""
import contextlib
import io

import numpy as np
from onnx import TensorProto, numpy_helper

from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.optimizer import Quantizer

FLOAT, FLOAT16, INT8 = (TensorProto.FLOAT, TensorProto.FLOAT16,
                        TensorProto.INT8)
K_OUT, N_UP, N = 16, 32, 8


def tdict(name, dims=None, dtype=FLOAT):
    return {"name": name, "shape": list(dims or []), "dtype": dtype}


def build(band_pad=0):
    """One fold node [x, Wd, Wg, Wu] with lane-major Wg/Wu.

    `band_pad` widens the middle axis so bands*8 no longer covers N_up.
    """
    rng = np.random.default_rng(7)
    wd = rng.normal(size=(N_UP, N)).astype(np.float32)
    wg = rng.normal(size=(K_OUT, N_UP)).astype(np.float32)
    wu = rng.normal(size=(K_OUT, N_UP)).astype(np.float32)
    bands = N_UP // 8 + band_pad
    gate = wg.reshape(K_OUT, N_UP // 8, 8)
    up = wu.reshape(K_OUT, N_UP // 8, 8)
    if band_pad:
        gate = np.concatenate(
            [gate, np.zeros((K_OUT, band_pad, 8), dtype=np.float32)], axis=1)
        up = np.concatenate(
            [up, np.zeros((K_OUT, band_pad, 8), dtype=np.float32)], axis=1)
    m = DAGBasedModel()
    m.initializers["Wd"] = numpy_helper.from_array(wd, "Wd")
    m.initializers["Wg"] = numpy_helper.from_array(gate, "Wg")
    m.initializers["Wu"] = numpy_helper.from_array(up, "Wu")
    m.nodes["MM"] = Node(
        "MatMul", "MM", {"down_act": [1], "transB": 0},
        [tdict("x", [1, K_OUT]), tdict("Wd", [N_UP, N]),
         tdict("Wg", [K_OUT, bands, 8]), tdict("Wu", [K_OUT, bands, 8])],
        [tdict("y", [1, N])])
    return m, wd, wg, wu


def quantize(m):
    with contextlib.redirect_stdout(io.StringIO()):
        return Quantizer._quantize_down_act_weights(m)


def payload(m, name):
    return numpy_helper.to_array(m.initializers[name])


def scale_table(m):
    return numpy_helper.to_array(m.initializers["Wd_down_act_scale"])


def test_lane_major_3d_is_quantized():
    m, _, wg, wu = build()
    assert quantize(m) == 1
    node = m.nodes["MM"]
    assert len(node.inputs) == 5
    for name in ("Wd", "Wg", "Wu"):
        assert m.initializers[name].data_type == INT8
        assert m.initializers[name].dims == ([K_OUT, N_UP // 8, 8]
                                            if name != "Wd" else [N_UP, N])
    # K_out is the leading dim -- the host reads shape[0], not shape[rank-2].
    assert m.initializers["Wg"].dims[0] == K_OUT
    assert m.initializers["Wu"].dims[0] == K_OUT
    assert int(scale_table(m).size) == N + 2 * N_UP


def test_lane_major_matches_row_major_bytes():
    m3, _, _, _ = build()
    m2, _, wg, wu = build()
    # Same values in the plain [K_out, N_up] layout: the fold's reorder is a
    # reshape, so a correct quantizer must land on identical bytes.
    m2.initializers["Wg"] = numpy_helper.from_array(wg, "Wg")
    m2.initializers["Wu"] = numpy_helper.from_array(wu, "Wu")
    assert quantize(m3) == 1
    assert quantize(m2) == 1
    assert payload(m3, "Wg").reshape(-1).tobytes() == \
        payload(m2, "Wg").reshape(-1).tobytes()
    assert payload(m3, "Wu").reshape(-1).tobytes() == \
        payload(m2, "Wu").reshape(-1).tobytes()
    # Per-column scales must also be in n order, not band order.
    assert np.array_equal(scale_table(m3), scale_table(m2))


def test_lane_major_with_short_bands_stays_fp16():
    m, _, _, _ = build(band_pad=1)
    assert quantize(m) == 0
    assert len(m.nodes["MM"].inputs) == 4
    for name in ("Wd", "Wg", "Wu"):
        assert m.initializers[name].data_type == FLOAT
