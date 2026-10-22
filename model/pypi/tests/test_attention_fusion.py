"""Standalone tests for the rewritten fuse_into_attention matcher/folder.

The pass folds the post-fold HF attention subgraph

    MatMul_qk(Q, K) -> [Mul(scale)] -> [Add(att_mask)] -> Softmax -> MatMul_v(., V)

into one ``Attention(Q, K, V[, mask])`` node. Two things are being pinned
here, both of which the version this replaced got wrong:

  * it *matches* the shape real exports actually have — scale applied to QK^T
    after the MatMul, and K's transpose already folded into ``transB=1`` by
    fuse_transpose_into_matmul (priority 85, which runs before this pass at
    75). The old matcher required a scale-only Q path *and* a scale+transpose
    K path, so it never fired on a modern graph;
  * it declines anything the runtime kernel cannot serve — the whole-row
    kernel takes only rank-4 [B,H,s,D] inputs with a *static* kv <= 2048 and an
    even head_dim <= 256, and the symbolic (kv = -1) decoder graph must keep
    its existing Plan A/B path. It also declines when a node it would delete
    has more than one consumer.

The op itself (ops/Attention.hpp + shaders/buffer/attention.comp) is covered
numerically by BufferRankTest.AttentionFp32AndFp16.
"""
import numpy as np
import pytest

from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.optimizer import FusionOptimizer


def tdict(name, dims=None, dtype=10):
    return {"name": name, "shape": list(dims or []), "dtype": dtype}


def count(dag, op_type):
    return sum(1 for n in dag.nodes.values() if n.op_type == op_type)


def matches_of(dag):
    return FusionOptimizer.match_attention(dag)


def build(scale=True, mask=True, folded=True, qh=4, kvh=4, seq=5, kv=5, hd=64,
          q_dims=None, k_dims=None, v_dims=None, extra_softmax_consumer=False):
    """Assemble the post-fold attention subgraph and return the DAG.

    folded=True uses ``transB=1`` (the fuse_transpose_into_matmul form);
    folded=False leaves an explicit 4-D last-two-dims Transpose on K.
    """
    m = DAGBasedModel()
    q_dims = q_dims or [1, qh, seq, hd]
    k_dims = k_dims or [1, kvh, kv, hd]
    v_dims = v_dims or [1, kvh, kv, hd]

    mm_qk = Node("MatMul", "MM_QK",
                 {"transB": 1} if folded else {},
                 [tdict("q", q_dims), tdict("kT" if not folded else "k",
                                            k_dims if folded else [1, kvh, hd, kv])],
                 [tdict("mm_out", [1, qh, seq, kv])])
    m.nodes["MM_QK"] = mm_qk

    cur = "mm_out"
    if not folded:
        tp = Node("Transpose", "TP_K", {"perm": [0, 1, 3, 2]},
                  [tdict("k", k_dims)], [tdict("kT", [1, kvh, hd, kv])])
        m.nodes["TP_K"] = tp

    if scale:
        # A TensorProto, not a bare numpy scalar: the loader folds Constants
        # into TensorProto initializers, and that is the only form the writer
        # can serialize (dag._init_byte_len reads .dims). The matcher reads
        # both, which _to_float is there to cover.
        from onnx import numpy_helper

        m.initializers["scale_const"] = numpy_helper.from_array(
            np.float32(0.125), "scale_const")
        mul = Node("Mul", "MUL", {},
                   [tdict(cur, [1, qh, seq, kv]),
                    tdict("scale_const", [])],
                   [tdict("scaled", [1, qh, seq, kv])])
        m.nodes["MUL"] = mul
        cur = "scaled"

    if mask:
        add = Node("Add", "ADD", {},
                   [tdict(cur, [1, qh, seq, kv]),
                    tdict("att_mask", [1, 1, seq, kv])],
                   [tdict("biased", [1, qh, seq, kv])])
        m.nodes["ADD"] = add
        cur = "biased"

    sx = Node("Softmax", "SX", {}, [tdict(cur, [1, qh, seq, kv])],
              [tdict("sx_out", [1, qh, seq, kv])])
    m.nodes["SX"] = sx

    mm_v = Node("MatMul", "MM_V", {},
                [tdict("sx_out", [1, qh, seq, kv]), tdict("v", v_dims)],
                [tdict("out", [1, qh, seq, hd])])
    m.nodes["MM_V"] = mm_v

    if extra_softmax_consumer:
        # e.g. a dtype cast on the softmax output: deleting the Softmax would
        # strand this consumer on a tensor nobody writes.
        m.nodes["SX_CAST"] = Node("Cast", "SX_CAST", {},
                                  [tdict("sx_out", [1, qh, seq, kv])],
                                  [tdict("sx_cast", [1, qh, seq, kv])])

    m.outputs = [mm_v.outputs[0]]
    return m


# ---------------------------------------------------------------- positives

def test_transb_scale_mask_folds():
    """The canonical HF form: transB=1 K, scale Mul, mask Add."""
    m = build()
    ms = matches_of(m)
    assert len(ms) == 1, ms
    assert FusionOptimizer.fold_attention(m, ms[0])

    assert count(m, "Attention") == 1
    for gone in ("MM_QK", "MUL", "ADD", "SX", "MM_V"):
        assert gone not in m.nodes, gone
    fused = m.nodes["Attention_fused_MM_V"]
    assert [i["name"] for i in fused.inputs] == ["q", "k", "v", "att_mask"]
    assert fused.attributes["scale"] == 0.125
    assert fused.attributes["num_heads"] == 4
    assert fused.attributes["num_kv_heads"] == 4
    assert fused.attributes["causal"] == 0
    # output tensor is reused so downstream consumers still resolve
    assert fused.outputs[0]["name"] == "out"


# ------------------------------------------------- mask_causal declaration

def test_declared_causal_mask_sets_attribute():
    """A graph that declares its bias *is* the standard causal mask gets
    mask_causal=1, which lets the kernel stop a row's KV sweep at its
    diagonal. The declaration is the only source: nothing in the graph shows
    it, because the bias rides in as a graph input."""
    m = build()
    m.causal_mask_inputs = {"att_mask"}
    ms = matches_of(m)
    assert FusionOptimizer.fold_attention(m, ms[0])
    fused = m.nodes["Attention_fused_MM_V"]
    assert fused.attributes["mask_causal"] == 1
    # causality is still the mask's job; the kernel does not synthesize it
    assert fused.attributes["causal"] == 0


def test_undeclared_mask_omits_attribute():
    """Silence means "not declared", not "declared false": the op must then
    sweep the whole row. Absence of the attribute is what carries that, so a
    model converted before the declaration existed is bit-identical."""
    m = build()
    assert not m.causal_mask_inputs
    ms = matches_of(m)
    assert FusionOptimizer.fold_attention(m, ms[0])
    assert "mask_causal" not in m.nodes["Attention_fused_MM_V"].attributes


def test_declaration_for_other_tensor_ignored():
    m = build()
    m.causal_mask_inputs = {"some_other_mask"}
    ms = matches_of(m)
    assert FusionOptimizer.fold_attention(m, ms[0])
    assert "mask_causal" not in m.nodes["Attention_fused_MM_V"].attributes


def test_declared_but_no_mask_still_omits_attribute():
    """A declaration without a bound mask is inert — the kernel's tile bound
    keys off the *op's* causal terms, and neither is set, so the op behaves
    exactly as an undeclared one."""
    m = build(mask=False)
    m.causal_mask_inputs = {"att_mask"}
    ms = matches_of(m)
    assert FusionOptimizer.fold_attention(m, ms[0])
    assert "mask_causal" not in m.nodes["Attention_fused_MM_V"].attributes


def test_explicit_transpose_form_folds():
    """fuse_transpose_into_matmul did not run: K's Transpose is still there."""
    m = build(folded=False)
    ms = matches_of(m)
    assert len(ms) == 1, ms
    assert FusionOptimizer.fold_attention(m, ms[0])
    assert count(m, "Transpose") == 0  # the K transpose is consumed
    assert count(m, "Attention") == 1
    fused = m.nodes[[n for n in m.nodes if n.startswith("Attention_fused")][0]]
    assert [i["name"] for i in fused.inputs][:3] == ["q", "k", "v"]


def test_no_scale_no_mask_folds():
    """Bare MatMul -> Softmax -> MatMul (a ViT-style export without bias)."""
    m = build(scale=False, mask=False)
    ms = matches_of(m)
    assert len(ms) == 1, ms
    assert FusionOptimizer.fold_attention(m, ms[0])
    fused = m.nodes[[n for n in m.nodes if n.startswith("Attention_fused")][0]]
    assert [i["name"] for i in fused.inputs] == ["q", "k", "v"]
    assert "scale" not in fused.attributes  # 1.0 is the runtime default


def test_scale_outside_mask_add_declined():
    """MatMul -> Add(mask) -> Mul(scale) -> Softmax must NOT fold.

    That is softmax((QK^T + mask) * scale): the scale applies to the mask as
    well, and the kernel scales QK^T only. Folding it would silently change the
    result, so the matcher declines and the MatMul epilogue keeps the form.
    """
    m = DAGBasedModel()
    m.initializers["scale_const"] = np.float32(0.05)
    m.nodes["MM_QK"] = Node("MatMul", "MM_QK", {"transB": 1},
                            [tdict("q", [1, 4, 5, 64]), tdict("k", [1, 4, 5, 64])],
                            [tdict("mm_out", [1, 4, 5, 64])])
    m.nodes["ADD"] = Node("Add", "ADD", {},
                          [tdict("mm_out", [1, 4, 5, 64]),
                           tdict("att_mask", [1, 1, 5, 5])],
                          [tdict("biased", [1, 4, 5, 64])])
    m.nodes["MUL"] = Node("Mul", "MUL", {},
                          [tdict("biased", [1, 4, 5, 64]), tdict("scale_const", [])],
                          [tdict("scaled", [1, 4, 5, 64])])
    m.nodes["SX"] = Node("Softmax", "SX", {}, [tdict("scaled", [1, 4, 5, 64])],
                         [tdict("sx_out", [1, 4, 5, 64])])
    m.nodes["MM_V"] = Node("MatMul", "MM_V", {},
                           [tdict("sx_out", [1, 4, 5, 64]), tdict("v", [1, 4, 5, 64])],
                           [tdict("out", [1, 4, 5, 64])])
    m.outputs = [m.nodes["MM_V"].outputs[0]]
    assert matches_of(m) == []


def test_add_around_div_folds():
    """softmax((QK^T / sqrt(d)) + mask): scale inside the Add, via Div."""
    m = DAGBasedModel()
    m.initializers["sqrt_d"] = np.float32(8.0)
    m.nodes["MM_QK"] = Node("MatMul", "MM_QK", {"transB": 1},
                            [tdict("q", [1, 4, 5, 64]), tdict("k", [1, 4, 5, 64])],
                            [tdict("mm_out", [1, 4, 5, 64])])
    m.nodes["DIV"] = Node("Div", "DIV", {},
                          [tdict("mm_out", [1, 4, 5, 64]), tdict("sqrt_d", [])],
                          [tdict("scaled", [1, 4, 5, 64])])
    # [B,1,S,kv] — kv is the KV length (5), not the head dim (64). The kernel
    # reads the bias as one flat [B,S,kv] block, so the last axis has to be kv.
    m.nodes["ADD"] = Node("Add", "ADD", {},
                          [tdict("scaled", [1, 4, 5, 64]),
                           tdict("att_mask", [1, 1, 5, 5])],
                          [tdict("biased", [1, 4, 5, 64])])
    m.nodes["SX"] = Node("Softmax", "SX", {}, [tdict("biased", [1, 4, 5, 64])],
                         [tdict("sx_out", [1, 4, 5, 64])])
    m.nodes["MM_V"] = Node("MatMul", "MM_V", {},
                           [tdict("sx_out", [1, 4, 5, 64]), tdict("v", [1, 4, 5, 64])],
                           [tdict("out", [1, 4, 5, 64])])
    m.outputs = [m.nodes["MM_V"].outputs[0]]

    ms = matches_of(m)
    assert len(ms) == 1, ms
    assert ms[0]["scale_val"] == pytest.approx(0.125)
    assert ms[0]["mask_input"]["name"] == "att_mask"


# ---------------------------------------------- mask shape (fail-closed)
# attention.comp reads the additive bias as ONE flat [B,S,kv] block, indexed
# (b*S+s)*kv+j. Anything else — a per-head [B,H,S,kv] bias, a single broadcast
# row, a scalar, a rank-3 carry-over — reads the wrong element or past the end,
# and nothing downstream can tell. The matcher declines the whole fusion
# instead, which keeps the unfused (correct) path.

def test_mask_shapes_the_kernel_cannot_read_are_declined():
    for bad in ([1, 4, 5, 5],   # per-head: the head axis is not folded out
                [1, 1, 1, 5],   # one broadcast row, not S rows
                [1, 1, 5, 64],  # head_dim in the kv slot (the shape these
                                # fixtures used to carry by copy-paste)
                [1, 1, 5],      # rank 3
                []):            # scalar
        m = build()
        m.nodes["ADD"].inputs[1] = tdict("att_mask", bad)
        assert matches_of(m) == [], bad


def test_rank2_mask_accepted_for_batch_one():
    """[S,kv] with batch 1 is the same flat block, so it stays foldable."""
    m = build()
    m.nodes["ADD"].inputs[1] = tdict("att_mask", [5, 5])
    assert len(matches_of(m)) == 1


def test_gqa_head_map_attributes():
    m = build(qh=8, kvh=2)
    ms = matches_of(m)
    assert len(ms) == 1, ms
    assert ms[0]["num_heads"] == 8
    assert ms[0]["num_kv_heads"] == 2


def test_div_scale_folds():
    """softmax(QK^T / sqrt(d)) — Div by a scalar initializer."""
    m = DAGBasedModel()
    m.initializers["sqrt_d"] = np.float32(8.0)
    m.nodes["MM_QK"] = Node("MatMul", "MM_QK", {"transB": 1},
                            [tdict("q", [1, 4, 5, 64]), tdict("k", [1, 4, 5, 64])],
                            [tdict("mm_out", [1, 4, 5, 64])])
    m.nodes["DIV"] = Node("Div", "DIV", {},
                          [tdict("mm_out", [1, 4, 5, 64]), tdict("sqrt_d", [])],
                          [tdict("scaled", [1, 4, 5, 64])])
    m.nodes["SX"] = Node("Softmax", "SX", {}, [tdict("scaled", [1, 4, 5, 64])],
                         [tdict("sx_out", [1, 4, 5, 64])])
    m.nodes["MM_V"] = Node("MatMul", "MM_V", {},
                           [tdict("sx_out", [1, 4, 5, 64]), tdict("v", [1, 4, 5, 64])],
                           [tdict("out", [1, 4, 5, 64])])
    m.outputs = [m.nodes["MM_V"].outputs[0]]

    ms = matches_of(m)
    assert len(ms) == 1, ms
    assert ms[0]["scale_val"] == 1.0 / 8.0


# ---------------------------------------------------------------- negatives

def test_symbolic_batch_seq_kv_accepted():
    """The decoder graph: batch/seq/kv recorded as -1, head_dim static.

    This is the actual llm.onnx signature — Q [-1,16,-1,128], K/V
    [-1,8,-1,128], transB=1 (rep=2). The kernel reads batch/seq/kv from the
    bound tensors at execute() time, so only head_dim and the head counts have
    to be statically known.
    """
    m = build(q_dims=[-1, 16, -1, 128], k_dims=[-1, 8, -1, 128],
              v_dims=[-1, 8, -1, 128], qh=16, kvh=8)
    ms = matches_of(m)
    assert len(ms) == 1, ms
    assert ms[0]["num_heads"] == 16
    assert ms[0]["num_kv_heads"] == 8


def test_large_kv_accepted():
    """kv is unbounded now (tiled kernel): no 2048 cap."""
    m = build(seq=1, kv=4096, k_dims=[1, 4, 4096, 64], v_dims=[1, 4, 4096, 64])
    assert len(matches_of(m)) == 1


def test_symbolic_head_dim_declined():
    """head_dim must be static: the shader's MAX_HD envelope depends on it."""
    m = build(q_dims=[1, 4, 5, -1], k_dims=[1, 4, 5, -1], v_dims=[1, 4, 5, -1])
    assert matches_of(m) == []


def test_odd_head_dim_declined():
    m = build(hd=65, k_dims=[1, 4, 5, 65], v_dims=[1, 4, 5, 65])
    assert matches_of(m) == []


def test_head_dim_over_cap_declined():
    m = build(hd=512, k_dims=[1, 4, 5, 512], v_dims=[1, 4, 5, 512])
    assert matches_of(m) == []


def test_non_multiple_heads_declined():
    m = build(qh=6, kvh=4, k_dims=[1, 4, 5, 64], v_dims=[1, 4, 5, 64])
    assert matches_of(m) == []


def test_extra_softmax_consumer_declined():
    """Deleting the Softmax would strand the second consumer."""
    m = build(extra_softmax_consumer=True)
    assert matches_of(m) == []


def test_rank3_inputs_declined():
    """The kernel requires rank-4 [B,H,s,D]; a rank-3 export is not served."""
    m = build(q_dims=[4, 5, 64], k_dims=[4, 5, 64], v_dims=[4, 5, 64])
    assert matches_of(m) == []


def test_wrong_transpose_perm_declined():
    """A Transpose that is not the 4-D last-two-dims swap is not K's K^T."""
    m = build(folded=False)
    m.nodes["TP_K"].attributes["perm"] = [3, 2, 1, 0]
    assert matches_of(m) == []


def test_multi_consumer_scale_declined():
    """scale Mul feeding anything but the Add is not safe to delete."""
    m = build(mask=False)
    # a second consumer of the scaled tensor
    m.nodes["SCALED_CAST"] = Node("Cast", "SCALED_CAST", {},
                                  [tdict("scaled", [1, 4, 5, 64])],
                                  [tdict("scaled_cast", [1, 4, 5, 64])])
    assert matches_of(m) == []


def test_transposed_matmul_v_declined():
    """P @ V^T is not attention."""
    m = build()
    m.nodes["MM_V"].attributes["transB"] = 1
    assert matches_of(m) == []


def test_transposed_qk_declined():
    """transA would hand the kernel Q already transposed, and the head_dim
    check would then be reading the seq axis."""
    m = build()
    m.nodes["MM_QK"].attributes["transA"] = 1
    assert matches_of(m) == []


# ------------------------------------- the declaration's converter plumbing

def _tiny_onnx(tmp_path, metadata=(), env=None, monkeypatch=None):
    """A convertible 1-node ONNX whose graph input `attention_bias` stands in
    for the real model's additive causal bias. Small enough that a real
    onnxoptimizer + checker pass runs on it."""
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    b = helper.make_tensor_value_info("attention_bias", TensorProto.FLOAT16,
                                      [1, 1, 1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Identity", ["x"], ["y"], name="ID")
    graph = helper.make_graph([node], "g", [x, b], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    for key, val in metadata:
        model.metadata_props.add(key=key, value=val)
    path = str(tmp_path / "tiny.onnx")
    onnx.save_model(model, path)
    if monkeypatch is not None:
        for k, v in (env or {}).items():
            monkeypatch.setenv(k, v)
    return path


def test_model_metadata_declares_causal_mask(tmp_path, monkeypatch):
    """The export-side channel: a model that names its causal bias in
    metadata_props gets its name into the DAG, where the matcher picks it up.
    Model-level (not ValueInfoProto) because onnxoptimizer.optimize() rebuilds
    the graph and drops the per-value-info props — the same reason
    vkop_io_dtype is model-level."""
    from onnx2vkop.converter import ModelConverter

    path = _tiny_onnx(
        tmp_path,
        metadata=[("vkop_causal_mask:attention_bias", "1")],
        monkeypatch=monkeypatch,
    )
    dag = ModelConverter().parse_onnx_model(path, 1)
    assert dag.causal_mask_inputs == {"attention_bias"}


def test_env_declares_causal_mask_for_an_existing_model(tmp_path, monkeypatch):
    """The bridge for a model exported before the declaration existed: the
    same names can be supplied at conversion time instead of re-exporting."""
    from onnx2vkop.converter import ModelConverter

    path = _tiny_onnx(tmp_path, env={"VKOP_CAUSAL_MASK": "attention_bias, other"},
                      monkeypatch=monkeypatch)
    dag = ModelConverter().parse_onnx_model(path, 1)
    assert dag.causal_mask_inputs == {"attention_bias", "other"}


def test_no_declaration_means_no_causal_mask(tmp_path, monkeypatch):
    """Silence stays silent: a model converted with neither the metadata nor
    the env var declares nothing, so the op sweeps the full row exactly as it
    did before the flag existed."""
    from onnx2vkop.converter import ModelConverter

    monkeypatch.delenv("VKOP_CAUSAL_MASK", raising=False)
    path = _tiny_onnx(tmp_path, monkeypatch=monkeypatch)
    dag = ModelConverter().parse_onnx_model(path, 1)
    assert dag.causal_mask_inputs == set()


def test_written_artifact_carries_the_declaration():
    """The runtime reads the op's surface off the .vkopbin, not off the
    attribute dict, so the declaration has to survive serialization: a writer
    that dropped mask_causal would silently fall back to the full sweep (the
    speed fix quietly gone), and one that *invented* it for an undeclared
    model would be the dangerous direction. Both halves are read back here."""
    import mmap
    import os
    import sys
    import tempfile

    _gen = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "onnx2vkop", "generated")
    if _gen not in sys.path:
        sys.path.insert(0, _gen)
    from onnx2vkop.generated.vkop.model.AttrType import AttrType
    from onnx2vkop.generated.vkop.model.Model import Model

    def read_back(dag):
        dag.build_dependencies()
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "attn.vkopbin")
            dag.save_to_binary(out)
            with open(out, "rb") as fh:
                mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
                model = Model.GetRootAsModel(mm, 0)
                attrs = None
                for i in range(model.NodesLength()):
                    node = model.Nodes(i)
                    if node.OpType() != b"Attention":
                        continue
                    attrs = {}
                    for k in range(node.AttributesLength()):
                        a = node.Attributes(k)
                        if a.Type() == AttrType.Int64:
                            attrs[a.Key().decode()] = a.Ival()
                    break
                mm.close()
        assert attrs is not None, "Attention missing from the artifact"
        return attrs

    declared = build()
    declared.causal_mask_inputs = {"att_mask"}
    assert FusionOptimizer.fold_attention(declared, matches_of(declared)[0])
    assert read_back(declared)["mask_causal"] == 1

    undeclared = build()
    assert FusionOptimizer.fold_attention(undeclared, matches_of(undeclared)[0])
    assert "mask_causal" not in read_back(undeclared)
