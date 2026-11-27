"""Conservative H3 attention fusion: positive and fail-closed graph cases."""

import copy

import pytest

from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.h3_attention import apply


def tensor(name, shape, dtype=16):
    return {"name": name, "shape": list(shape), "dtype": dtype,
            "value_dynamic": True}


def build():
    dag = DAGBasedModel()
    qshape, ktshape, scores = [-1, -1, -1, 128], [-1, -1, 128, -1], [-1] * 4
    q = tensor("q", qshape)
    kt = tensor("kt", ktshape)
    v = tensor("v", qshape)
    score = tensor("score", scores)
    prob = tensor("prob", scores)
    out = tensor("out", qshape)
    dag.inputs = [tensor("q0", qshape), tensor("k0", qshape), v]
    dag.nodes = {
        "qt": Node("Transpose", "qt", {"perm": [0, 2, 1, 3]},
                   [dag.inputs[0]], [tensor("qt", qshape)]),
        "kt": Node("Transpose", "kt", {"perm": [0, 2, 3, 1]},
                   [dag.inputs[1]], [tensor("kt0", ktshape)]),
        "qm": Node("Mul", "qm", {},
                   [tensor("qt", qshape), tensor("scale", [1])], [q]),
        "km": Node("Mul", "km", {},
                   [tensor("kt0", ktshape), tensor("scale", [1])], [kt]),
        "qk": Node("MatMul", "qk", {}, [q, kt], [score]),
        "sx": Node("Softmax", "sx", {"axis": -1}, [score], [prob]),
        "pv": Node("MatMul", "pv", {}, [prob, v], [out]),
    }
    dag.outputs = [out]
    dag.build_dependencies()
    return dag


def test_fold_keeps_scaled_inputs_and_builds_dependencies():
    dag = build()
    assert apply(dag) == 1
    assert set(dag.nodes) == {"qt", "kt", "qm", "km",
                              "H3_Attention_K_Transpose_pv", "H3_Attention_pv"}
    tp = dag.nodes["H3_Attention_K_Transpose_pv"]
    fused = dag.nodes["H3_Attention_pv"]
    assert tp.attributes == {"perm": [0, 1, 3, 2]}
    assert tp.inputs[0]["name"] == "kt"
    assert tp.outputs[0]["shape"] == [-1, -1, -1, 128]
    assert [x["name"] for x in fused.inputs] == ["q", tp.outputs[0]["name"], "v"]
    assert fused.outputs[0]["name"] == "out"
    assert fused.attributes == {"causal": 0, "h3_self_attention": 1}
    assert fused.dependencies == {"qm", tp.name}
    assert tp.dependencies == {"km"}
    assert len(dag.topological_sort()) == len(dag.nodes)
    assert apply(dag) == 0


@pytest.mark.parametrize("change", [
    "axis", "implicit_axis", "score_graph_output", "prob_graph_output",
    "score_reader", "prob_reader", "swapped_pv", "qk_transpose", "pv_transpose",
    "qk_epilogue", "fp16", "head_dim", "batch_broadcast", "head_broadcast",
    "seq_mismatch", "not_h3_transpose", "not_scaled", "zero_dim",
])
def test_unsafe_patterns_are_not_folded(change):
    dag = build()
    if change == "axis":
        dag.nodes["sx"].attributes["axis"] = 1
    elif change == "implicit_axis":
        dag.nodes["sx"].attributes.clear()
    elif change.endswith("_graph_output"):
        name = change.removesuffix("_graph_output")
        dag.outputs.append(copy.deepcopy(dag.nodes["qk" if name == "score" else "sx"].outputs[0]))
    elif change.endswith("_reader"):
        name = change.removesuffix("_reader")
        dag.nodes["reader"] = Node("Identity", "reader", {},
                                   [tensor(name, [-1] * 4)], [tensor("side", [-1] * 4)])
    elif change == "swapped_pv":
        dag.nodes["pv"].inputs.reverse()
    elif change == "qk_transpose":
        dag.nodes["qk"].attributes["transB"] = 1
    elif change == "pv_transpose":
        dag.nodes["pv"].attributes["transA"] = 1
    elif change == "qk_epilogue":
        dag.nodes["qk"].attributes["epi_bias"] = 1
    elif change == "fp16":
        dag.nodes["pv"].inputs[1]["dtype"] = 10
    elif change == "head_dim":
        dag.nodes["qk"].inputs[0]["shape"][3] = 64
    elif change == "batch_broadcast":
        dag.nodes["qk"].inputs[0]["shape"][0] = 2
        dag.nodes["qk"].inputs[1]["shape"][0] = 1
    elif change == "head_broadcast":
        dag.nodes["qk"].inputs[0]["shape"][1] = 56
        dag.nodes["pv"].inputs[1]["shape"][1] = 1
    elif change == "seq_mismatch":
        dag.nodes["qk"].inputs[0]["shape"][2] = 1018
        dag.nodes["qk"].inputs[1]["shape"][3] = 1017
    elif change == "not_h3_transpose":
        dag.nodes["kt"].attributes["perm"] = [0, 1, 3, 2]
    elif change == "not_scaled":
        dag.nodes["qm"].op_type = "Identity"
    elif change == "zero_dim":
        dag.nodes["qk"].inputs[0]["shape"][0] = 0
    original = set(dag.nodes)
    assert apply(dag) == 0
    assert set(dag.nodes) == original


def test_unique_names_do_not_overwrite_graph_values():
    dag = build()
    dag.inputs.append(tensor("kt__h3_attention_k", [1]))
    dag.nodes["H3_Attention_pv"] = Node("Identity", "H3_Attention_pv", {},
                                       [dag.inputs[-1]], [tensor("other", [1])])
    assert apply(dag) == 1
    fused = dag.nodes["H3_Attention_pv_1"]
    assert fused.inputs[1]["name"] == "kt__h3_attention_k_1"


def test_cli_opt_in_runs_after_bf16_domain(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    import numpy as np
    import onnx
    from onnx import TensorProto as T, helper as h, numpy_helper
    from onnx2vkop.inspect import ModelView

    shape = [1, "seq", 4, 128]
    graph = h.make_graph([
        h.make_node("Transpose", ["q0"], ["qt"], name="qt", perm=[0, 2, 1, 3]),
        h.make_node("Transpose", ["k0"], ["kt"], name="kt", perm=[0, 2, 3, 1]),
        h.make_node("Mul", ["qt", "scale"], ["q"], name="qm"),
        h.make_node("Mul", ["kt", "scale"], ["k"], name="km"),
        h.make_node("MatMul", ["q", "k"], ["score"], name="qk"),
        h.make_node("Softmax", ["score"], ["prob"], name="sx", axis=-1),
        h.make_node("MatMul", ["prob", "v"], ["out"], name="pv"),
    ], "h3", [h.make_tensor_value_info("q0", T.FLOAT, shape),
               h.make_tensor_value_info("k0", T.FLOAT, shape),
               h.make_tensor_value_info("v", T.FLOAT, [1, 4, "seq", 128])],
       [h.make_tensor_value_info("out", T.FLOAT, [1, 4, "seq", 128])],
       [numpy_helper.from_array(np.array([0.3], dtype=np.float32), "scale")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 17)])
    model.ir_version = 8
    src = tmp_path / "h3.onnx"
    onnx.save(model, str(src))
    pypi = str(Path(__file__).resolve().parents[1])
    base_env = dict(os.environ, PYTHONPATH=pypi)
    base_env.pop("VKOP_NO_FUSIONS", None)
    for switch in (None, "0", "1"):
        enabled = switch == "1"
        out = tmp_path / f"h3_{switch}.vkopbin"
        env = dict(base_env)
        env.pop("VKOP_H3_FUSED_ATTENTION", None)
        if switch is not None:
            env["VKOP_H3_FUSED_ATTENTION"] = switch
        run = subprocess.run(
            [sys.executable, "-m", "onnx2vkop.cli", "-i", str(src), "-o",
             str(out), "--act", "bf16", "--target-backend", "buffer"],
            env=env, capture_output=True, text=True)
        assert run.returncode == 0, run.stdout + run.stderr
        nodes = ModelView(str(out)).nodes()
        assert sum(n["op_type"] == "Attention" for n in nodes) == int(enabled)
        assert sum(n["op_type"] == "Softmax" for n in nodes) == int(not enabled)
        if enabled:
            assert "[h3-attention] fused 1 " in run.stdout

    run = subprocess.run(
        [sys.executable, "-m", "onnx2vkop.cli", "-i", str(src), "-o",
         str(tmp_path / "bad.vkopbin"), "--act", "fp16",
         "--target-backend", "buffer"],
        env=dict(base_env, VKOP_H3_FUSED_ATTENTION="1"),
        capture_output=True, text=True)
    assert run.returncode != 0
    assert "requires --act bf16" in run.stdout + run.stderr


def test_real_h3_metadata_if_available():
    """Optional read-only check against the saved real-model metadata."""
    import os
    from onnx2vkop.inspect import ModelView

    path = os.environ.get("VKOP_H3_GRAPH_METADATA")
    if not path:
        pytest.skip("set VKOP_H3_GRAPH_METADATA to a saved model metadata file")
    view = ModelView(path)
    dag = DAGBasedModel()

    def convert(t):
        return dict(t, shape=t["dims"])

    dag.inputs = [convert(t) for t in view.shapes()]
    dag.outputs = [convert(t) for t in view.outputs()]
    for n in view.nodes():
        dag.nodes[n["name"]] = Node(n["op_type"], n["name"], n["attributes"],
                                    [convert(t) for t in n["inputs"]],
                                    [convert(t) for t in n["outputs"]])
    assert apply(dag) == 52
    assert len(dag.topological_sort()) == len(dag.nodes)
