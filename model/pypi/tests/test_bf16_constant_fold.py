"""ConstantFolder folds Cast(to=BFLOAT16) to the bf16 VALUE.

`_eval`'s Cast arm looked `to` up in a dtype map that has FLOAT16 but no
BFLOAT16, and fell back to float32 -- so a constant chain through
Cast(to=BFLOAT16) (the usual way a bf16 export spells a float literal) folded
to the UNROUNDED fp32 value, while the same chain through Cast(to=FLOAT16) got
fp16's rounding. The folded constant is now rounded to bf16 (RNE, as ONNX
casts) and held as float32, the one width every target takes it from without a
second rounding (under --act fp16 converter._downcast_bfloat16 retargets the
Cast to FLOAT16 first, so only --act bf16 folds through bf16).
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto as T
from onnx import helper as h
from onnx import numpy_helper as nh

from onnx2vkop.optimizer import ConstantFolder

PYPI = str(Path(__file__).resolve().parents[1])

# 1 + 2**-8 is a bf16 tie (rounds to even: 1.0) but exact in fp16; -(2 + 2**-7)
# rounds to -2.0 in bf16 and is exact in fp16; 3 and 0.5 are exact in both.
VALUES = np.array([1.00390625, 3.0, -2.0078125, 0.5] * 4,
                  np.float32).reshape(1, 16)
BF16 = np.array([1.0, 3.0, -2.0, 0.5] * 4, np.float32).reshape(1, 16)


def cast_chain(to):
    nodes = [h.make_node("Constant", [], ["c"], value=nh.from_array(VALUES, "cv")),
             h.make_node("Cast", ["c"], ["cb"], to=to),
             h.make_node("Cast", ["cb"], ["cf"], to=T.FLOAT),
             h.make_node("Add", ["x", "cf"], ["y"])]
    g = h.make_graph(nodes, "g",
                     [h.make_tensor_value_info("x", T.FLOAT, [1, 16])],
                     [h.make_tensor_value_info("y", T.FLOAT, [1, 16])])
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m


def folded(model):
    model = ConstantFolder.fold(model, verbose=False)
    assert [n.op_type for n in model.graph.node] == ["Add"]
    inits = {i.name: i for i in model.graph.initializer}
    return inits[model.graph.node[0].input[1]]


def test_cast_to_bf16_folds_to_the_rounded_value():
    init = folded(cast_chain(T.BFLOAT16))
    assert init.data_type == T.FLOAT
    np.testing.assert_array_equal(nh.to_array(init), BF16)


def test_cast_to_fp16_is_unchanged():
    init = folded(cast_chain(T.FLOAT16))
    assert init.data_type == T.FLOAT
    np.testing.assert_array_equal(nh.to_array(init), VALUES)


@pytest.mark.parametrize("act,dtype,want", [
    # --act fp16: converter._downcast_bfloat16 retargets Cast(to=BFLOAT16) to
    # FLOAT16 before folding (the same width a runtime Cast would produce), and
    # every VALUE is exact in fp16.
    ("fp16", "float16", VALUES),
    # --act bf16: the Cast keeps to=BFLOAT16, so the fold rounds through bf16.
    ("bf16", "bfloat16", BF16),
], ids=["act-fp16", "act-bf16"])
def test_cli_artifact_carries_the_cast_width_value(tmp_path, act, dtype, want):
    """End to end: the folded constant reaches the artifact rounded to the
    width the (normalized) Cast names, not left at the source fp32."""
    from onnx2vkop.generated.vkop.model import Model as _Model
    from onnx2vkop.inspect import ModelView

    src, out = str(tmp_path / "m.onnx"), tmp_path / "m.vkopbin"
    onnx.save(cast_chain(T.BFLOAT16), src)
    p = subprocess.run([sys.executable, "-m", "onnx2vkop.cli", "-i", src,
                        "-o", str(out), "--act", act], capture_output=True,
                       text=True, cwd=PYPI, env={**os.environ, "PYTHONPATH": PYPI})
    assert p.returncode == 0, (p.stdout + p.stderr)[-2000:]
    view = ModelView(str(out))
    rows = {r["name"]: r for r in view.initializers()}
    assert set(rows) == {"cf"}, rows
    row = rows["cf"]
    assert row["dtype"] == dtype, row
    data = out.read_bytes()
    if view.external_blob_trailer is not None:
        raw = data[view.external_blob_trailer + row["offset"]:][:row["size"]]
    else:
        blob = _Model.Model.GetRootAs(data, 0).InitializerBlobAsNumpy().tobytes()
        raw = blob[row["offset"]: row["offset"] + row["size"]]
    if dtype == "float16":
        got = np.frombuffer(raw, np.float16).astype(np.float32)
    else:
        got = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
    np.testing.assert_array_equal(got.reshape(1, 16), want)
