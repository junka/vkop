"""Regressions found converting a real ModelOpt 0.48.0 NVFP4 export.

1. NVFP4QuantExporter.post_process (`_cast_input_dtypes`) puts a Cast(to=FLOAT16)
   on the ACTIVATION of every fp4 Gemm/MatMul, also in an fp32 model whose GEMM
   output stays FLOAT. Under -q fp32 that mixed-width GEMM converted and then
   produced all-NaN output. PreQuantizedWeights.strip_fp4_operand_casts bypasses
   that Cast on the proto before fusion.
2. -q fp32 now refuses a node that declares FP16 and FP32 operands side by side
   (GraphFp16.mixed_width_conflicts) instead of shipping it.
3. A fold that deletes the node producing a GRAPH OUTPUT (redundant Cast,
   Identity, ...) used to re-point node inputs only; the prune then emptied the
   graph and a 0-node .vkopbin was written with exit code 0.
4. The W4A4 refusal names the op producing the DequantizeLinear payload
   (trt::TRT_FP4DynamicQuantize for ModelOpt's NVFP4_DEFAULT_CFG).
"""
import contextlib
import io
import json
import os
import subprocess
import sys

import numpy
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.optimizer import FusionOptimizer, PreQuantizedWeights

PYPI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(PYPI))
BLOCK = 16
E2M1 = numpy.array([0, 0.5, 1, 1.5, 2, 3, 4, 6,
                    -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=numpy.float32)


def e4m3(byte):
    s = -1.0 if byte & 0x80 else 1.0
    e, m = (byte >> 3) & 0xF, byte & 7
    return s * (m / 8 * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))


E4M3 = numpy.array([e4m3(b) for b in range(256)], dtype=numpy.float32)


def quiet(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args)


def pack(codes):
    flat = numpy.asarray(codes, dtype=numpy.uint8).reshape(-1) & 0x0F
    return (flat[0::2] | (flat[1::2] << 4)).astype(numpy.uint8).tobytes()


# --------------------------------------------------------------------------
# a ModelOpt-0.48-shaped fp32 MLP: Gemm(up, bias) -> SiLU -> MatMul(down)
# --------------------------------------------------------------------------

def weight_chain(prefix, n, k, rng, nodes, inits):
    """[N, K] e2m1 payload behind ModelOpt's two-DQ chain; returns the DQ
    output name and the dequantized [N, K] fp32 weight."""
    codes = rng.randint(0, 16, size=(n, k)).astype(numpy.uint8)
    sb = rng.randint(0x20, 0x48, size=(n, k // BLOCK)).astype(numpy.uint8)
    g = numpy.float32(0.0173)
    inits += [
        helper.make_tensor(f"{prefix}.weight_f4", TensorProto.FLOAT4E2M1,
                           [n, k], pack(codes), raw=True),
        helper.make_tensor(f"{prefix}.weight_f8_scale",
                           TensorProto.FLOAT8E4M3FN, [n, k // BLOCK],
                           sb.tobytes(), raw=True),
        numpy_helper.from_array(numpy.array([g], numpy.float32),
                                f"{prefix}.weight_f8_scale_f32_scale"),
    ]
    nodes += [
        helper.make_node("DequantizeLinear",
                         [f"{prefix}.weight_f8_scale",
                          f"{prefix}.weight_f8_scale_f32_scale"],
                         [f"{prefix}.weight_f32_scale"],
                         name=f"{prefix}.weight_DequantizeLinear"),
        helper.make_node("DequantizeLinear",
                         [f"{prefix}.weight_f4", f"{prefix}.weight_f32_scale"],
                         [f"{prefix}_deq"], axis=-1, block_size=BLOCK,
                         name=f"{prefix}.weight_DequantizeLinear_1"),
    ]
    w = E2M1[codes] * numpy.repeat(E4M3[sb] * g, BLOCK, axis=1)
    return f"{prefix}_deq", w.astype(numpy.float32)


def modelopt_mlp(path, m_rows=4, k=64, h=128, seed=3, value_info=True,
                 act_cast=True):
    """What ModelOpt 0.48's NVFP4 W4A16 export of an fp32 Linear-SiLU-Linear
    looks like (out/w4a16_fp32.onnx of the investigation, scaled down):
    Cast(to=fp16) on BOTH GEMM operands, FLOAT GEMM outputs and bias."""
    rng = numpy.random.RandomState(seed)
    nodes, inits = [], []
    up, w_up = weight_chain("up", h, k, rng, nodes, inits)
    down, w_down = weight_chain("down", k, h, rng, nodes, inits)
    bias = (rng.randn(h) * 0.1).astype(numpy.float32)
    inits.append(numpy_helper.from_array(bias, "up.bias"))
    a_up = "x"
    if act_cast:
        nodes.append(helper.make_node("Cast", ["x"], ["x_f16"],
                                      to=TensorProto.FLOAT16))
        a_up = "x_f16"
    nodes += [
        helper.make_node("Cast", [up], [up + "_f16"], to=TensorProto.FLOAT16),
        helper.make_node("Gemm", [a_up, up + "_f16", "up.bias"], ["h"],
                         transB=1, name="/up/Gemm"),
        helper.make_node("Sigmoid", ["h"], ["sg"]),
        helper.make_node("Mul", ["h", "sg"], ["a"]),
        helper.make_node("Transpose", [down], ["down_t"], perm=[1, 0]),
    ]
    a_down = "a"
    if act_cast:
        nodes.append(helper.make_node("Cast", ["a"], ["a_f16"],
                                      to=TensorProto.FLOAT16))
        a_down = "a_f16"
    nodes += [
        helper.make_node("Cast", ["down_t"], ["down_t_f16"],
                         to=TensorProto.FLOAT16),
        helper.make_node("MatMul", [a_down, "down_t_f16"], ["out"],
                         name="/down/MatMul"),
    ]
    vi = []
    if value_info:
        vi = [helper.make_tensor_value_info(n, TensorProto.FLOAT, None)
              for n in ("h", "sg", "a", up, down, "down_t")]
    graph = helper.make_graph(
        nodes, "mlp",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [m_rows, k])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [m_rows, k])],
        inits, value_info=vi)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)])
    model.ir_version = 10
    if path:
        onnx.save(model, path)
    return model, w_up, bias, w_down


def by_output(model):
    return {o: n for n in model.graph.node for o in n.output}


# --------------------------------------------------------------------------
# 1. strip_fp4_operand_casts on the proto
# --------------------------------------------------------------------------

def test_activation_casts_are_bypassed_weight_casts_kept():
    model, *_ = modelopt_mlp(None)
    assert quiet(PreQuantizedWeights.strip_fp4_operand_casts, model) == 2
    prod = by_output(model)
    assert "x_f16" not in prod and "a_f16" not in prod
    assert prod["h"].input[0] == "x"
    assert prod["out"].input[0] == "a"
    # The weight-side Casts are views the nvfp4 fold absorbs: untouched.
    assert prod["h"].input[1] == "up_deq_f16"
    assert prod["out"].input[1] == "down_t_f16"


def test_without_value_info_the_gemm_bias_decides():
    """No value_info: the Gemm's FLOAT bias marks it as an fp32 GEMM (its
    Cast reads the FLOAT graph input). The down MatMul's Cast reads an
    undeclared intermediate, so its source width is unknown and it is left
    alone -- conservatively; under -q fp32 the width check then refuses it
    instead of shipping NaNs (test_fp32_refuses_a_mixed_width_node)."""
    model, *_ = modelopt_mlp(None, value_info=False)
    assert quiet(PreQuantizedWeights.strip_fp4_operand_casts, model) == 1
    prod = by_output(model)
    assert prod["h"].input[0] == "x"
    assert prod["out"].input[0] == "a_f16"


def test_an_fp16_island_keeps_its_cast():
    """A GEMM declared FLOAT16 is a real domain boundary, not an artifact."""
    model, *_ = modelopt_mlp(None)
    for vi in model.graph.value_info:
        if vi.name == "h":
            vi.type.tensor_type.elem_type = TensorProto.FLOAT16
    for o in model.graph.output:
        o.type.tensor_type.elem_type = TensorProto.FLOAT16
    assert quiet(PreQuantizedWeights.strip_fp4_operand_casts, model) == 0


def test_a_float_weight_gemm_is_not_touched():
    """Only a GEMM whose B traces to an E2M1 DequantizeLinear is ModelOpt's."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])
    w = numpy_helper.from_array(numpy.ones((8, 4), numpy.float32), "W")
    nodes = [helper.make_node("Cast", ["x"], ["xh"], to=TensorProto.FLOAT16),
             helper.make_node("MatMul", ["xh", "W"], ["y"])]
    model = helper.make_model(helper.make_graph(nodes, "g", [x], [y], [w]))
    assert quiet(PreQuantizedWeights.strip_fp4_operand_casts, model) == 0
    assert len(model.graph.node) == 2


def test_a_cast_with_another_reader_is_rerouted_but_kept():
    model, *_ = modelopt_mlp(None)
    model.graph.node.append(helper.make_node("Relu", ["x_f16"], ["side"]))
    model.graph.output.append(
        helper.make_tensor_value_info("side", TensorProto.FLOAT16, None))
    assert quiet(PreQuantizedWeights.strip_fp4_operand_casts, model) == 2
    prod = by_output(model)
    assert "x_f16" in prod and prod["h"].input[0] == "x"


# --------------------------------------------------------------------------
# 1+2 through the CLI
# --------------------------------------------------------------------------

def convert(src, out, *extra, ok=True):
    p = subprocess.run(
        [sys.executable, "-m", "onnx2vkop.cli", "-i", src, "-o", out,
         "--target-backend", "buffer", *extra],
        capture_output=True, text=True, cwd=PYPI)
    log = p.stdout + p.stderr
    if ok:
        assert p.returncode == 0, log[-2000:]
    else:
        assert p.returncode != 0, log[-2000:]
    return log


def nodes_of(path):
    p = subprocess.run(
        [sys.executable, "-m", "onnx2vkop.inspect", path, "--json"],
        capture_output=True, text=True, env={**os.environ, "PYTHONPATH": PYPI})
    assert p.returncode == 0, p.stderr[-400:]
    return json.loads(p.stdout)


@pytest.mark.parametrize("quant", [[], ["-q", "fp32"]], ids=["default", "fp32"])
def test_modelopt_fp32_export_converts_with_fp32_activations(tmp_path, quant):
    src = str(tmp_path / "m.onnx")
    modelopt_mlp(src)
    out = str(tmp_path / "m.vkopbin")
    log = convert(src, out, *quant)
    assert "bypassed 2 fp16 activation Cast(s)" in log
    d = nodes_of(out)
    mms = [n for n in d["nodes"] if n["op_type"] == "MatMul"]
    assert len(mms) == 2
    if quant:
        # -q fp32: both GEMMs read an fp32 A and write fp32 -- one width.
        for n in mms:
            assert n["inputs"][0]["dtype"] == "float32", n
            assert n["outputs"][0]["dtype"] == "float32", n
        assert d["outputs"][0]["dtype"] == "float32"


def mixed_width_graph(path):
    """An fp32 graph whose MatMul reads a Cast(to=fp16) activation but writes
    a FLOAT output -- the shape ModelOpt's cast leaves when nothing strips it."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 16])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 8])
    w = numpy_helper.from_array(
        numpy.random.RandomState(0).randn(16, 8).astype(numpy.float32), "W")
    nodes = [helper.make_node("Cast", ["x"], ["xh"], to=TensorProto.FLOAT16),
             helper.make_node("MatMul", ["xh", "W"], ["y"], name="mm")]
    model = helper.make_model(helper.make_graph(nodes, "g", [x], [y], [w]),
                              opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)


def test_fp32_refuses_a_mixed_width_node(tmp_path):
    src = str(tmp_path / "mix.onnx")
    mixed_width_graph(src)
    log = convert(src, str(tmp_path / "mix.vkopbin"), "-q", "fp32", ok=False)
    assert "-q fp32 keeps the export's widths" in log
    assert "MatMul mm" in log


def test_default_domain_still_converts_the_mixed_graph(tmp_path):
    """The FP16 domain pass reconciles the widths; only -q fp32 refuses."""
    src = str(tmp_path / "mix.onnx")
    mixed_width_graph(src)
    convert(src, str(tmp_path / "mix.vkopbin"))


def test_a_coherent_fp32_graph_passes_the_fp32_check(tmp_path):
    src = str(tmp_path / "ok.onnx")
    modelopt_mlp(src, act_cast=False)
    convert(src, str(tmp_path / "ok.vkopbin"), "-q", "fp32")


# --------------------------------------------------------------------------
# 3. folds that delete a graph output's producer
# --------------------------------------------------------------------------

def t(name, dtype=TensorProto.FLOAT):
    return {"name": name, "shape": [1, 8], "dtype": dtype}


def relu_cast_dag(cast_src="r"):
    m = DAGBasedModel()
    m.inputs = [{"name": "x", "shape": [1, 8], "dtype": "float32"}]
    m.outputs = [{"name": "y", "shape": [1, 8], "dtype": "float32"}]
    if cast_src == "r":
        m.nodes["relu"] = Node("Relu", "relu", {}, [t("x")], [t("r")])
    m.nodes["cast"] = Node("Cast", "cast", {"to": TensorProto.FLOAT},
                           [t(cast_src)], [t("y")])
    return m


def test_redundant_cast_on_a_graph_output_hands_its_name_back():
    m = relu_cast_dag()
    matches = quiet(FusionOptimizer.match_redundant_cast, m)
    single = [mt for mt in matches if mt["type"] == "redundant_single"]
    assert len(single) == 1
    assert FusionOptimizer.fold_redundant_cast(m, single[0])
    assert list(m.nodes) == ["relu"]
    assert m.nodes["relu"].outputs[0]["name"] == "y"
    assert m.outputs == [{"name": "y", "shape": [1, 8], "dtype": "float32"}]
    assert m.dangling_outputs() == []


def test_a_cast_from_a_graph_input_to_a_graph_output_stays():
    """Neither name can move (both are the I/O contract): keep the node."""
    m = relu_cast_dag(cast_src="x")
    matches = quiet(FusionOptimizer.match_redundant_cast, m)
    single = [mt for mt in matches if mt["type"] == "redundant_single"]
    assert len(single) == 1
    assert not FusionOptimizer.fold_redundant_cast(m, single[0])
    assert list(m.nodes) == ["cast"]


def test_identity_on_a_graph_output_hands_its_name_back():
    m = relu_cast_dag()
    m.nodes["cast"].op_type = "Identity"
    assert FusionOptimizer.fold_identity(m, {"node": m.nodes["cast"]})
    assert m.nodes["relu"].outputs[0]["name"] == "y"


def test_hollow_graph_is_refused_at_write():
    m = relu_cast_dag()
    del m.nodes["cast"]
    assert m.dangling_outputs() == ["y"]
    with pytest.raises(RuntimeError, match="no producer"):
        m.check_graph_not_hollow()
    empty = DAGBasedModel()
    empty.outputs = [{"name": "x", "dtype": "float32"}]
    empty.inputs = [{"name": "x", "dtype": "float32"}]
    with pytest.raises(RuntimeError, match="no nodes"):
        empty.check_graph_not_hollow()


@pytest.mark.parametrize("quant", [[], ["-q", "fp32"]], ids=["default", "fp32"])
def test_cli_keeps_the_graph_when_its_output_is_a_redundant_cast(tmp_path, quant):
    """Relu -> Cast(to=FLOAT) -> y used to convert to a 0-node artifact."""
    nodes = [helper.make_node("Relu", ["x"], ["r"]),
             helper.make_node("Cast", ["r"], ["y"], to=TensorProto.FLOAT)]
    g = helper.make_graph(
        nodes, "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8])])
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    src = str(tmp_path / "rc.onnx")
    onnx.save(model, src)
    out = str(tmp_path / "rc.vkopbin")
    convert(src, out, *quant)
    d = nodes_of(out)
    assert [n["op_type"] for n in d["nodes"]] == ["Relu"]
    assert d["nodes"][0]["outputs"][0]["name"] == "y"
    assert d["outputs"][0]["name"] == "y"
    if quant:
        assert d["outputs"][0]["dtype"] == "float32"


# --------------------------------------------------------------------------
# 4. the W4A4 refusal names the producer
# --------------------------------------------------------------------------

def test_w4a4_refusal_names_trt_fp4_dynamic_quantize():
    m = DAGBasedModel()
    m.inputs = [{"name": "x", "shape": [2, 32], "dtype": "float32"}]
    m.initializers["s"] = numpy_helper.from_array(
        numpy.array(1.0, numpy.float32), "s")
    m.nodes["dq_in"] = Node(
        "TRT_FP4DynamicQuantize", "/up/input_quantizer/TRT_FP4DynamicQuantize",
        {"axis": -1, "block_size": 16}, [t("x")], [t("q"), t("q_scale")])
    m.nodes["dq"] = Node("DequantizeLinear", "/up/input_quantizer/DequantizeLinear",
                         {}, [t("q_scale"), t("s")], [t("q_scale_f")])
    with pytest.raises(ValueError) as e:
        quiet(PreQuantizedWeights.fold, m)
    msg = str(e.value)
    assert "TRT_FP4DynamicQuantize '/up/input_quantizer/TRT_FP4DynamicQuantize'" in msg
    assert "W4A4" in msg


def test_bare_value_refusal_names_a_graph_input():
    m = DAGBasedModel()
    m.inputs = [{"name": "x", "shape": [2, 32], "dtype": "int8"}]
    m.initializers["s"] = numpy_helper.from_array(
        numpy.array(1.0, numpy.float32), "s")
    m.nodes["dq"] = Node("DequantizeLinear", "dq", {}, [t("x"), t("s")],
                         [t("xf")])
    with pytest.raises(ValueError, match="it is a graph input"):
        quiet(PreQuantizedWeights.fold, m)


# --------------------------------------------------------------------------
# end to end on the GPU (opt-in: needs vkop-bench)
# --------------------------------------------------------------------------

BENCH = os.environ.get("VKOP_BENCH")
ARCHIVE = os.environ.get("VKOP_MASTER_ARCHIVE",
                         os.path.join(REPO, "build", "vkop_shaders.bin"))


@pytest.mark.skipif(not BENCH or not os.path.exists(ARCHIVE),
                    reason="set VKOP_BENCH (a vkop-bench, or a wrapper that "
                           "exports the Vulkan loader env) and a shader archive")
@pytest.mark.parametrize("m_rows", [1, 16], ids=["gemv", "tile"])
@pytest.mark.parametrize("quant", [[], ["-q", "fp32"]], ids=["default", "fp32"])
def test_modelopt_fp32_export_matches_onnxruntime(tmp_path, m_rows, quant):
    pytest.importorskip("onnxruntime")
    k, h = 128, 256
    src = str(tmp_path / "m.onnx")
    _, w_up, bias, w_down = modelopt_mlp(src, m_rows=m_rows, k=k, h=h)
    # The fp32 reference ModelOpt's fake-quant describes: dequantized weights,
    # no fp16 operand casts.
    nodes = [helper.make_node("Gemm", ["x", "Wu", "bu"], ["h"], transB=1),
             helper.make_node("Sigmoid", ["h"], ["sg"]),
             helper.make_node("Mul", ["h", "sg"], ["a"]),
             helper.make_node("MatMul", ["a", "Wd"], ["out"])]
    ref = helper.make_model(helper.make_graph(
        nodes, "ref",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [m_rows, k])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [m_rows, k])],
        [numpy_helper.from_array(w_up, "Wu"), numpy_helper.from_array(bias, "bu"),
         numpy_helper.from_array(numpy.ascontiguousarray(w_down.T), "Wd")]),
        opset_imports=[helper.make_opsetid("", 17)])
    ref.ir_version = 8
    ref_path = str(tmp_path / "ref.onnx")
    onnx.save(ref, ref_path)
    out = str(tmp_path / "m.vkopbin")
    convert(src, out, "--master-archive", ARCHIVE, *quant)
    p = subprocess.run(
        [sys.executable, os.path.join(REPO, "tools", "vkop_vs_ort.py"), out,
         "--onnx", ref_path, "--bench", BENCH, "--random-fill"],
        capture_output=True, text=True)
    report = p.stdout + p.stderr
    assert p.returncode == 0, report[-2000:]
    assert "RESULT: 1/1" in report, report[-2000:]
    assert "NaN" not in report, report[-2000:]
