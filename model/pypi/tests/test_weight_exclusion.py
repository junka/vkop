"""Node/scope-level precision exclusion (--exclude-node / --exclude-scope /
--exclude-downstream-of), from the pattern to the byte that stays wide.

An accuracy loss in a quantized model is localized by handing one layer back its
original dtype and watching the metric, so the knob has to name NODES: "this
MatMul's weight stays fp16". Not op types -- sensitivity is a property of this
weight on this model, not of MatMul in general. The contract these cases pin is
therefore in three parts, and a bug in any one of them silently returns the
un-excluded model with a clean log:

  * the pattern picks the nodes the user meant (globs, the '/'-and-'.' token
    boundary that keeps 'layers.0' off 'layers.01', the transitive downstream
    closure), and a pattern that picks nothing FAILS rather than reports success
    on the wrong model;
  * the node maps to the weight it quantizes -- including through a Transpose or
    a DequantizeLinear, because the exporter puts the reshape between them and
    the user names the MatMul;
  * the byte passes then leave that initializer exactly as it was: no payload
    rewrite, no appended scale, and for a SwiGLU fold or AWQ the whole set or the
    rescale goes wide too, since a partial one is a graph the runtime rejects or
    a weight that lost its 1/s.

The `-q fp16` and pre-quantized-import boundaries are reported, not honoured, and
there is a case for each: both are places where a lone wide weight is a Cast
node or a dead DequantizeLinear rather than a precision knob, so the honest
behaviour is to say the flag changed nothing.
"""
import contextlib
import io
from types import SimpleNamespace

import numpy
import pytest
from onnx import TensorProto, numpy_helper

from onnx2vkop import exclusion
from onnx2vkop.converter import ModelConverter
from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.optimizer import Quantizer


def tref(name):
    return {"name": name, "shape": [-1]}


def add_node(m, op_type, name, inputs, outputs=None, attrs=None):
    m.nodes[name] = Node(op_type, name, dict(attrs or {}),
                         [tref(i) for i in inputs],
                         [tref(o) for o in (outputs or [name + "_out"])])
    return m


def weight(m, name, shape=(16, 16), dtype=numpy.float32, seed=0):
    rng = numpy.random.default_rng(seed)
    arr = rng.standard_normal(shape).astype(dtype)
    m.initializers[name] = numpy_helper.from_array(arr, name)
    return arr


def two_layer_model():
    """'/model/layers.0/*' and '/model/layers.1/*' MatMuls, one weight each.

    The names are the PyTorch exporter's spelling -- slash per scope, dot for the
    block index -- because that is what a user copies out of Netron, and the
    leading '/' is what makes the variant handling observable.
    """
    m = DAGBasedModel()
    for layer in (0, 1):
        for proj in ("q_proj", "down_proj"):
            weight(m, f"model.layers.{layer}.{proj}.weight", seed=layer * 7)
            add_node(m, "MatMul",
                     f"/model/layers.{layer}/{proj}/MatMul",
                     [f"act{layer}.{proj}",
                      f"model.layers.{layer}.{proj}.weight"])
    m.outputs = [tref(n + "_out") for n in m.nodes]
    return m


def run(fn, *a, **kw):
    """A pass call, with its report captured instead of printed."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*a, **kw)
    return result, buf.getvalue()


def dtype_of(m, name):
    return m.initializers[name].data_type


def input_names(node):
    return [i["name"] for i in node.inputs]


# --------------------------------------------------------------------------
# the resolver: pattern -> nodes -> weights
# --------------------------------------------------------------------------

def test_exact_node_name_selects_its_weight():
    m = two_layer_model()
    got = run(exclusion.resolve, m, ["/model/layers.0/q_proj/MatMul"])[0]
    assert got == {"model.layers.0.q_proj.weight"}


def test_glob_selects_every_matching_node():
    m = two_layer_model()
    got = run(exclusion.resolve, m, ["*/layers.1/*/MatMul"])[0]
    assert got == {f"model.layers.1.{p}.weight" for p in ("q_proj", "down_proj")}


def test_a_leading_slash_in_the_pattern_is_optional():
    """Users write the module path, exporters prefix it with '/'. Both spellings
    have to select the same node, or the flag reads as broken on half the
    models."""
    m = two_layer_model()
    with_slash = run(exclusion.resolve, m, ["/model/layers.0/down_proj/MatMul"])[0]
    without = run(exclusion.resolve, m, ["model/layers.0/down_proj/MatMul"])[0]
    assert with_slash == without == {"model.layers.0.down_proj.weight"}


def test_one_value_can_carry_a_comma_separated_list():
    m = two_layer_model()
    got = run(exclusion.resolve,
              m, ["*/layers.0/q_proj/MatMul, */layers.1/down_proj/MatMul"])[0]
    assert got == {"model.layers.0.q_proj.weight",
                   "model.layers.1.down_proj.weight"}


def test_scope_is_a_token_prefix_not_a_string_prefix():
    """'layers.0' must cover layers.0's subtree and not reach into 'layers.01',
    which a plain startswith cannot tell apart."""
    m = DAGBasedModel()
    for name, weight_name in (("/model/layers.0/attn/MatMul", "W0"),
                              ("/model/layers.01/attn/MatMul", "W01"),
                              ("/model/layers.0/attn/o/MatMul", "W0o")):
        weight(m, weight_name)
        add_node(m, "MatMul", name, ["x", weight_name])
    got = run(exclusion.resolve, m, [], ["model.layers.0"])[0]
    assert got == {"W0", "W0o"}


def test_scope_split_accepts_both_separators():
    """The same dotted index reaches the node name through a slash from one
    exporter and a dot from another, so neither spelling may be the odd one
    out."""
    m = DAGBasedModel()
    weight(m, "Wa")
    weight(m, "Wb")
    add_node(m, "MatMul", "/enc/blocks.3/attn/MatMul", ["x", "Wa"])
    add_node(m, "MatMul", "enc.blocks.3.attn.MatMul", ["x", "Wb"])
    assert run(exclusion.resolve, m, [], ["enc.blocks.3"])[0] == {"Wa", "Wb"}


def test_downstream_of_is_the_transitive_closure():
    m = DAGBasedModel()
    weight(m, "Wemb")
    weight(m, "W1")
    weight(m, "W2")
    weight(m, "Wfar")
    add_node(m, "Gather", "/embed", ["ids", "Wemb"], ["emb_out"])
    add_node(m, "MatMul", "/layers.0", ["emb_out", "W1"], ["a0"])
    add_node(m, "Relu", "/relu", ["a0"], ["r0"])
    add_node(m, "MatMul", "/layers.1", ["r0", "W2"], ["a1"])
    add_node(m, "MatMul", "/unrelated", ["other", "Wfar"], ["a9"])
    got = run(exclusion.resolve, m, [], [], ["/embed"])[0]
    assert got == {"Wemb", "W1", "W2"}


def test_a_pattern_that_matches_no_node_fails_the_conversion():
    """The one failure mode this module exists to prevent: a typo'd flag selects
    nothing, the run reports success, and the measured model is the un-excluded
    one."""
    m = two_layer_model()
    with pytest.raises(exclusion.ExclusionError) as ei:
        run(exclusion.resolve, m, ["/model/layers.7/q_proj/MatMul"])
    assert "matched no node" in str(ei.value)


def test_an_empty_scope_fails_instead_of_naming_everything():
    """'/' and '.' tokenize to nothing, and the empty list is a prefix of every
    node name -- so the alternative to refusing is silently excluding the whole
    model."""
    m = two_layer_model()
    for pattern in ("/", ".."):
        with pytest.raises(exclusion.ExclusionError) as ei:
            run(exclusion.resolve, m, [], [pattern])
        assert "no path tokens" in str(ei.value)


def test_a_flag_with_no_pattern_in_it_fails_too():
    """--exclude-node ',' is not 'no flag': a run that reports success on the
    un-excluded model is the failure this module is built to make impossible."""
    m = two_layer_model()
    with pytest.raises(exclusion.ExclusionError) as ei:
        run(exclusion.resolve, m, [" "])
    assert "contains no pattern" in str(ei.value)


def test_no_flags_resolve_to_nothing_and_say_nothing():
    m = two_layer_model()
    got, log = run(exclusion.resolve, m)
    assert got == set() and log == ""


def test_a_weight_reached_through_a_transpose_is_the_one_excluded():
    """The user names the MatMul; the weight is behind the Transpose the
    exporter put in front of it (and which a fusion pass later folds away)."""
    m = DAGBasedModel()
    weight(m, "W")
    add_node(m, "Transpose", "/attn/Transpose", ["W"], ["Wt"])
    add_node(m, "MatMul", "/attn/MatMul", ["x", "Wt"])
    assert run(exclusion.resolve, m, ["/attn/MatMul"])[0] == {"W"}


def test_a_pre_quantized_weight_resolves_to_its_payload_and_says_so(capsys):
    """A QDQ weight is the same MatMul slot one node further on, so the pattern
    still names it -- but the import that runs next is not excluded, and the log
    has to say that instead of letting the run read as fp16."""
    m = DAGBasedModel()
    payload = numpy_helper.from_array(
        numpy.zeros((16, 16), dtype=numpy.int8), "W_q")
    m.initializers["W_q"] = payload
    m.initializers["W_s"] = numpy_helper.from_array(
        numpy.ones(16, dtype=numpy.float32), "W_s")
    add_node(m, "DequantizeLinear", "/W/dequantize", ["W_q", "W_s"], ["W_deq"])
    add_node(m, "MatMul", "/attn/MatMul", ["x", "W_deq"])
    got = run(exclusion.resolve, m, ["/attn/MatMul"])[0]
    assert got == {"W_q"}
    err = capsys.readouterr().err
    assert "quantization grid" in err and "W_q" in err


def test_a_weight_shared_outside_the_exclusion_is_reported(capsys):
    m = DAGBasedModel()
    weight(m, "W")
    add_node(m, "MatMul", "/keep/MatMul", ["x", "W"])
    add_node(m, "MatMul", "/gone/MatMul", ["y", "W"])
    got = run(exclusion.resolve, m, ["/gone/MatMul"])[0]
    assert got == {"W"}
    err = capsys.readouterr().err
    assert "outside the exclusion" in err and "/keep/MatMul" in err


def test_nodes_with_no_initializer_warn_that_nothing_changed(capsys):
    m = DAGBasedModel()
    add_node(m, "Relu", "/act/Relu", ["x"])
    run(exclusion.resolve, m, ["/act/Relu"])
    assert "identical to the one without the flag" in capsys.readouterr().err


# --------------------------------------------------------------------------
# the byte passes: an excluded name keeps its dtype and gets no scale
# --------------------------------------------------------------------------

def matmul_pair(dtype=numpy.float32):
    """Two GEMM MatMuls, so one exclusion leaves the other still quantizing --
    which is the half of the contract that a blanket skip would also pass."""
    m = DAGBasedModel()
    keep = weight(m, "W_keep", seed=1, dtype=dtype)
    wide = weight(m, "W_wide", seed=2, dtype=dtype)
    add_node(m, "MatMul", "/a/MatMul", ["x", "W_keep"], ["a_out"])
    add_node(m, "MatMul", "/b/MatMul", ["y", "W_wide"], ["b_out"])
    return m, keep, wide


def test_int8_excludes_by_name_and_still_quantizes_its_sibling():
    m, _, _ = matmul_pair()
    _, log = run(Quantizer.quantize_to_int8_weight_only, m,
                 None, {"W_wide"})
    assert dtype_of(m, "W_wide") == TensorProto.FLOAT
    assert dtype_of(m, "W_keep") == TensorProto.INT8
    assert input_names(m.nodes["/b/MatMul"]) == ["y", "W_wide"]
    assert input_names(m.nodes["/a/MatMul"]) == ["x", "W_keep", "W_keep_scale"]
    assert "Excluded 'W_wide'" in log


def test_fp8_excluded_weight_stays_float():
    m, _, _ = matmul_pair(numpy.float16)
    run(Quantizer.quantize_to_fp8_weight_only, m, "fp8e4m3", None, {"W_wide"})
    assert dtype_of(m, "W_wide") == TensorProto.FLOAT16
    assert dtype_of(m, "W_keep") == TensorProto.FLOAT8E4M3FN
    assert input_names(m.nodes["/b/MatMul"]) == ["y", "W_wide"]


def test_int4_excluded_weight_stays_float():
    m, _, _ = matmul_pair()
    run(Quantizer.quantize_to_4bit_weight_only, m, 8, "int4", {"W_wide"})
    assert dtype_of(m, "W_wide") == TensorProto.FLOAT
    assert dtype_of(m, "W_keep") == TensorProto.INT4
    assert input_names(m.nodes["/b/MatMul"]) == ["y", "W_wide"]


def test_nvfp4_excluded_weight_gets_neither_scale_input():
    m, _, _ = matmul_pair()
    run(Quantizer.quantize_to_nvfp4_weight_only, m, 16, None, {"W_wide"})
    assert dtype_of(m, "W_wide") == TensorProto.FLOAT
    assert input_names(m.nodes["/b/MatMul"]) == ["y", "W_wide"]
    # The two-level format appends BOTH of its scales, so the wide sibling
    # reads as a plain two-input MatMul and the quantized one as a four.
    assert len(input_names(m.nodes["/a/MatMul"])) == 4


def test_exclusion_does_not_perturb_the_weight_it_leaves_alone():
    """The A/B is only honest if the quantized half is the same bytes it would
    be without the flag: a pass that changed the sibling's scale, its layout or
    the node set would move the metric for a reason the exclusion did not
    cause."""
    a, _, _ = matmul_pair()
    b, _, _ = matmul_pair()
    run(Quantizer.quantize_to_int8_weight_only, a)
    run(Quantizer.quantize_to_int8_weight_only, b, None, {"W_wide"})
    assert sorted(a.nodes) == sorted(b.nodes)
    assert input_names(a.nodes["/a/MatMul"]) == input_names(b.nodes["/a/MatMul"])
    assert numpy.array_equal(
        numpy_helper.to_array(a.initializers["W_keep"]),
        numpy_helper.to_array(b.initializers["W_keep"]))
    assert numpy.array_equal(
        numpy_helper.to_array(a.initializers["W_keep_scale"]),
        numpy_helper.to_array(b.initializers["W_keep_scale"]))
    # and the difference is exactly the excluded weight plus the scale it did
    # not get.
    assert dtype_of(a, "W_wide") == TensorProto.INT8
    assert dtype_of(b, "W_wide") == TensorProto.FLOAT
    assert "W_wide_scale" in a.initializers
    assert "W_wide_scale" not in b.initializers


# --------------------------------------------------------------------------
# the passes whose unit is bigger than one initializer
# --------------------------------------------------------------------------

def swiglu_model():
    """A folded down_proj node: [x, Wd, Wg, Wu], one concatenated scale table.

    Built already folded, because that is the shape _quantize_down_act_weights
    reads; a single excluded member has to hold all three wide, since the
    runtime rejects a partially converted fold outright.
    """
    m = DAGBasedModel()
    for name in ("Wd", "Wg", "Wu"):
        weight(m, name, shape=(16, 16))
    add_node(m, "MatMul", "/mlp/down/MatMul", ["x", "Wd", "Wg", "Wu"],
             ["down_out"], attrs={"down_act": 1})
    return m


def test_swiglu_fold_excludes_as_a_set():
    m = swiglu_model()
    run(Quantizer.quantize_to_int8_weight_only, m, None, {"Wg"})
    assert [dtype_of(m, n) for n in ("Wd", "Wg", "Wu")] == [TensorProto.FLOAT] * 3
    assert input_names(m.nodes["/mlp/down/MatMul"]) == ["x", "Wd", "Wg", "Wu"]


def test_swiglu_fold_still_converts_when_nothing_is_excluded():
    m = swiglu_model()
    run(Quantizer.quantize_to_int8_weight_only, m)
    assert len(input_names(m.nodes["/mlp/down/MatMul"])) == 5
    assert dtype_of(m, "Wd") == TensorProto.INT8


# --------------------------------------------------------------------------
# AWQ: the exclusion has to land on the rescale, not on its result
# --------------------------------------------------------------------------

def awq_model():
    """Two MatMuls whose A is a graph input, so AWQ folds 1/s as a constant Mul.

    An excluded weight must not be multiplied at all: filtering the returned set
    would leave W scaled and its 1/s nowhere, which is a wrong model rather than
    a less-compressed one.
    """
    m = DAGBasedModel()
    keep = weight(m, "W_keep", seed=3)
    wide = weight(m, "W_wide", seed=4)
    add_node(m, "MatMul", "/a/MatMul", ["x", "W_keep"], ["a_out"])
    add_node(m, "MatMul", "/b/MatMul", ["y", "W_wide"], ["b_out"])
    return m, keep, wide


def test_awq_neither_rescales_nor_quantizes_an_excluded_weight():
    m, _, wide = awq_model()
    scaled, log = run(Quantizer.quantize_to_awq_weight_only, m, 8, "int4",
                      None, 0.5, 20, {"W_wide"})
    assert scaled == {"W_keep"}
    assert "leaving 'W_wide' alone" in log
    assert dtype_of(m, "W_wide") == TensorProto.FLOAT
    assert numpy.array_equal(
        numpy_helper.to_array(m.initializers["W_wide"]), wide)
    assert "/b/MatMul" in m.nodes
    assert input_names(m.nodes["/b/MatMul"]) == ["y", "W_wide"]
    # The scale Mul exists for the weight that took the transform, and not for
    # the one that did not.
    assert not any(n.op_type == "Mul" and "W_wide" in input_names(n)
                   for n in m.nodes.values())


def test_awq_scaled_weight_still_lands_on_the_byte_grid():
    m, _, _ = awq_model()
    scaled, _ = run(Quantizer.quantize_to_awq_weight_only, m, 8, "int4",
                    None, 0.5, 20, {"W_wide"})
    assert dtype_of(m, "W_keep") == TensorProto.INT4
    assert len(input_names(m.nodes["/a/MatMul"])) == 3


# --------------------------------------------------------------------------
# the pipeline: the flags reach the quantizer from the exporter's names
# --------------------------------------------------------------------------

def _opts(**kw):
    base = dict(quant="int8", unify=False, rgba=False, group_size=64,
                target_backend="buffer", exclude_nodes=None,
                exclude_scopes=None, exclude_downstream_of=None)
    base.update(kw)
    return SimpleNamespace(**base)


def pipeline_model():
    """The two-layer DAG, with the shapes a liveness prune keeps.

    apply_optimizations ends in prune_and_materialize, whose liveness is seeded
    from the graph outputs -- a node that produces nothing reachable gets
    deleted, and then the case would not be testing what it looks like.
    """
    m = two_layer_model()
    m.inputs = [tref("act0.q_proj"), tref("act0.down_proj"),
                tref("act1.q_proj"), tref("act1.down_proj")]
    return m


def test_pipeline_excludes_by_the_exporters_scope():
    m = pipeline_model()
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ModelConverter().apply_optimizations(
            m, _opts(exclude_scopes=["model.layers.0"]))
    for proj in ("q_proj", "down_proj"):
        name = f"model.layers.0.{proj}.weight"
        assert dtype_of(m, name) == TensorProto.FLOAT, name
    for layer in (1,):
        for proj in ("q_proj", "down_proj"):
            name = f"model.layers.{layer}.{proj}.weight"
            assert dtype_of(m, name) == TensorProto.INT8, name


def test_pipeline_reports_the_exclusion_it_applied():
    m = pipeline_model()
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ModelConverter().apply_optimizations(
            m, _opts(exclude_nodes=["/model/layers.1/q_proj/MatMul"]))
    log = out.getvalue()
    assert "--exclude-node '/model/layers.1/q_proj/MatMul': 1 node(s)" in log
    assert "1 node(s), 1 weight(s) kept at their source precision" in log


def test_pipeline_without_the_flags_quantizes_every_weight():
    m = pipeline_model()
    with contextlib.redirect_stdout(io.StringIO()):
        ModelConverter().apply_optimizations(m, _opts())
    for name in m.initializers:
        if name.endswith(".weight"):
            assert dtype_of(m, name) == TensorProto.INT8, name


def test_fp16_says_the_flags_changed_nothing(capsys):
    """-q fp16 has no per-node exclusion; the warning is the difference between
    a user learning that now and learning it from a flat accuracy curve."""
    m = pipeline_model()
    with contextlib.redirect_stdout(io.StringIO()):
        ModelConverter().apply_optimizations(m, _opts(quant="fp16", exclude_scopes=["model.layers.0"]))
    assert "-q fp16 ignores the exclusion" in capsys.readouterr().err


def test_no_quant_warns_that_there_was_nothing_to_exclude(capsys):
    m = pipeline_model()
    with contextlib.redirect_stdout(io.StringIO()):
        ModelConverter().apply_optimizations(m, _opts(quant=None, exclude_scopes=["model.layers.0"]))
    assert "runs no weight quantizer" in capsys.readouterr().err
