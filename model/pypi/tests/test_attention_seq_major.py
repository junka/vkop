"""Standalone tests for fuse_attention_seq_major.

The pass drops the Q/K/V head-reorder that exists only to hand the fused
Attention op its native [B,H,S,D] order, plus the Transpose that undoes it, and
declares ``seq_major=1`` instead. On the 24-block ViT vision tower that is 96
Transpose + 72 Unsqueeze dispatches, measured at -12.7 ms of Transpose GPU time
with the graph's four outputs bit-identical to the unfolded graph.

The structure it has to match is the one the vision encoder really exports —
this is the part worth pinning, because a plausible-looking reconstruction of it
gets the pass wrong:

    qkv reshape -> Transpose([1,0,2,3]) -> Split(axis=0) -> [1,S,H,D]
      Squeeze -> [S,H,D]                       (the batch axis dropped)
      fp16->fp32 Cast, RotaryEmbedding, fp32->fp16 Cast   <-- Q/K ONLY
      Transpose([1,0,2]) -> [H,S,D]            (a real permutation)
      Unsqueeze(0) -> [1,H,S,D]
      Attention -> [1,H,S,D]
    Transpose([0,2,1,3]) -> [1,S,H,D]          (the same permutation, back)

Two consequences fall out of that shape and are what these tests exist for:

  * the RotaryEmbedding (and its Cast pair) sits *between* the Squeeze and the
    Transpose, so the fold cannot rewire Q/K back to the Split output — it moves
    the op's read point onto the Transpose's own input, the rank-3 [S,H,D], and
    leaves the whole RoPE path wired to the tensor it already had;
  * Y must come out rank-4 [1,S,H,D] even though its operands are rank-3, since
    the tensor it takes over is what the dropped undo-Transpose used to write.

Negatives cover the ways a partial or wrong fold could still "look" fine.
"""
import numpy as np
import pytest
from onnx import numpy_helper

from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.optimizer import FusionOptimizer

HD = 64
H = 16
S = 196
KV = 196


def tdict(name, dims=None, dtype=10):
    return {"name": name, "shape": list(dims or []), "dtype": dtype}


def count(dag, op_type):
    return sum(1 for n in dag.nodes.values() if n.op_type == op_type)


def axis_init(m, name, axis):
    """A Squeeze/Unsqueeze axes initializer (int64) — _axes_as_list reads the
    op's second input out of dag.initializers, exactly as a real export has it."""
    m.initializers[name] = numpy_helper.from_array(
        np.array([axis], dtype=np.int64), name)


def reorder_chain(m, tag, with_rope, unsqueeze=True):
    """Build one Q/K/V head-reorder: [1,S,H,D] -> ... -> [1,H,S,D].

    `with_rope` inserts the fp16->fp32 Cast / RotaryEmbedding / fp32->fp16 Cast
    that the vision tower puts on Q and K (but not V) between the Squeeze and
    the Transpose. `unsqueeze=False` leaves the Transpose's rank-3 output as the
    Attention's operand, i.e. an operand that was never turned head-major.

    Returns the tensor *dict* the fold would read if it fired — the Transpose's
    input — so tests can mutate it to describe a different graph.
    """
    sq = Node("Squeeze", f"SQ_{tag}", {},
              [tdict(f"{tag}_src", [1, S, H, HD]),
               tdict(f"sq_ax_{tag}", [1], 7)],
              [tdict(f"sq_{tag}", [S, H, HD])])
    m.nodes[sq.name] = sq
    axis_init(m, f"sq_ax_{tag}", 0)

    cur = f"sq_{tag}"
    if with_rope:
        up = Node("Cast", f"CAST_UP_{tag}", {},
                  [tdict(cur, [S, H, HD])],
                  [tdict(f"cast_up_{tag}", [S, H, HD], 1)])
        m.nodes[up.name] = up
        rope = Node("RotaryEmbedding", f"ROPE_{tag}", {},
                    [tdict(f"cast_up_{tag}", [S, H, HD], 1),
                     tdict(f"cos_{tag}", [S, HD // 2], 1),
                     tdict(f"sin_{tag}", [S, HD // 2], 1)],
                    [tdict(f"rope_{tag}", [S, H, HD], 1)])
        m.nodes[rope.name] = rope
        dn = Node("Cast", f"CAST_DN_{tag}", {},
                  [tdict(f"rope_{tag}", [S, H, HD], 1)],
                  [tdict(f"cast_dn_{tag}", [S, H, HD])])
        m.nodes[dn.name] = dn
        cur = f"cast_dn_{tag}"

    acc = tdict(cur, [S, H, HD])
    tp = Node("Transpose", f"TP_{tag}", {"perm": [1, 0, 2]},
              [acc],
              [tdict(f"tp_{tag}", [H, S, HD])])
    m.nodes[tp.name] = tp
    if not unsqueeze:
        return acc
    unsq = Node("Unsqueeze", f"UNSQ_{tag}", {},
                [tdict(f"tp_{tag}", [H, S, HD]),
                 tdict(f"unsq_ax_{tag}", [1], 7)],
                [tdict(f"unsq_{tag}", [1, H, S, HD])])
    m.nodes[unsq.name] = unsq
    axis_init(m, f"unsq_ax_{tag}", 0)
    return acc


def build(v_no_unsqueeze=False, v_dims=None, q_dims=None,
          out_perm=(0, 2, 1, 3), has_cache=False, drop_out_transpose=False,
          rope_on=("q", "k")):
    """Assemble the vision tower's post-fusion attention subgraph."""
    m = DAGBasedModel()

    q_acc = reorder_chain(m, "q", "q" in rope_on)
    k_acc = reorder_chain(m, "k", "k" in rope_on)
    v_acc = reorder_chain(m, "v", "v" in rope_on, unsqueeze=not v_no_unsqueeze)
    # Mutating the Transpose's own input dict is how a test describes a graph
    # where the fold would read a different tensor than [S,H,D].
    if q_dims is not None:
        q_acc["shape"] = list(q_dims)
    if v_dims is not None:
        v_acc["shape"] = list(v_dims)

    attrs = {"num_heads": H, "num_kv_heads": H, "causal": 0, "scale": 0.125}
    if has_cache:
        attrs["has_cache"] = 1
    v_in = tdict("tp_v" if v_no_unsqueeze else "unsq_v",
                 [H, S, HD] if v_no_unsqueeze else [1, H, S, HD])
    att = Node("Attention", "ATT", attrs,
               [tdict("unsq_q", [1, H, S, HD]),
                tdict("unsq_k", [1, H, KV, HD]),
                v_in],
               [tdict("att_out", [1, H, S, HD])])
    m.nodes["ATT"] = att

    cur = "att_out"
    if not drop_out_transpose:
        otr = Node("Transpose", "OTR", {"perm": list(out_perm)},
                   [tdict("att_out", [1, H, S, HD])],
                   [tdict("att_bshd", [1, S, H, HD])])
        m.nodes["OTR"] = otr
        cur = "att_bshd"

    m.nodes["RS"] = Node("Reshape", "RS", {},
                         [tdict(cur, [1, S, H, HD])],
                         [tdict("flat", [S, H * HD])])
    m.outputs = [m.nodes["RS"].outputs[0]]
    return m


def matches_of(dag):
    return FusionOptimizer.match_attention_seq_major(dag)


# ---------------------------------------------------------------- positives

def test_vision_chain_folds():
    """The real export shape, RoPE and all: the op reads the rank-3 pre-Transpose
    operands and writes the rank-4 tensor the undo Transpose used to produce."""
    m = build()
    ms = matches_of(m)
    assert len(ms) == 1, ms
    assert FusionOptimizer.fold_attention_seq_major(m, ms[0])

    att = m.nodes["ATT"]
    assert att.attributes["seq_major"] == 1
    # Q/K keep their RoPE path; only the read point moved.
    assert [i["name"] for i in att.inputs] == ["cast_dn_q", "cast_dn_k", "sq_v"]
    assert all(len(i["shape"]) == 3 for i in att.inputs)
    # Y takes over the undo Transpose's tensor so its consumers still resolve.
    assert [o["name"] for o in att.outputs] == ["att_bshd"]
    assert att.outputs[0]["shape"] == [1, S, H, HD]
    assert m.nodes["RS"].inputs[0]["name"] == "att_bshd"

    # The pair and the undo Transpose are gone...
    for gone in ("TP_q", "TP_k", "TP_v", "UNSQ_q", "UNSQ_k", "UNSQ_v", "OTR"):
        assert gone not in m.nodes, gone
    # ...and the RoPE path is untouched: it is what keeps Q/K correct, so a
    # fold that removed any of these would be a wrong-number bug, not a win.
    for kept in ("SQ_q", "CAST_UP_q", "ROPE_q", "CAST_DN_q", "SQ_v"):
        assert kept in m.nodes, kept


def test_no_rope_on_v_still_folds():
    """V has no RoPE in the vision tower — its Transpose reads the Squeeze
    output directly, so the fold lands one hop further back."""
    m = build(rope_on=("q", "k"))
    ms = matches_of(m)
    assert FusionOptimizer.fold_attention_seq_major(m, ms[0])
    assert m.nodes["ATT"].inputs[2]["name"] == "sq_v"


def test_rope_missing_entirely_still_folds():
    """A graph with no RotaryEmbedding at all (bare Squeeze/Transpose/Unsqueeze)
    is the simplest form of the same chain."""
    m = build(rope_on=())
    ms = matches_of(m)
    assert len(ms) == 1, ms
    assert FusionOptimizer.fold_attention_seq_major(m, ms[0])
    assert [i["name"] for i in m.nodes["ATT"].inputs] == ["sq_q", "sq_k", "sq_v"]


# ---------------------------------------------------------------- negatives

def test_partial_fold_declined():
    """All of Q/K/V must match. Here V was never turned head-major — its
    Transpose feeds the op directly — so it is already seq-major while Q/K are
    not. Folding Q/K alone would leave the op reading one operand in each order
    under a single flag, which is silently wrong rather than merely slower."""
    m = build(v_no_unsqueeze=True)
    assert matches_of(m) == []


def test_rank4_transpose_input_declined():
    """A Transpose that declares a rank-4 input cannot be the [1,0,2] head/seq
    swap — the perm is rank-3 — so the graph is not the chain this pass knows
    how to reason about, and it declines rather than guess."""
    m = build(v_dims=[1, S, H, HD])
    assert matches_of(m) == []


def test_rank4_q_source_declined():
    m = build(q_dims=[1, S, H, HD])
    assert matches_of(m) == []


def test_kv_cache_form_declined():
    """The cache form reads its fresh rows at the native pitch; the op rejects
    seq_major together with has_cache, so the pass must not emit it."""
    m = build(has_cache=True)
    assert matches_of(m) == []


def test_wrong_out_perm_declined():
    """The undo Transpose must be the exact inverse. Anything else means the
    output is not the [B,S,H,D] the fold claims, and keeping it would swap the
    result twice."""
    m = build(out_perm=(1, 0, 2, 3))
    assert matches_of(m) == []


def test_missing_out_transpose_declined():
    m = build(drop_out_transpose=True)
    assert matches_of(m) == []


def test_head_count_mismatch_declined():
    """num_heads is the anchor tying the shapes to the graph: if the operands'
    head axis disagrees, the pairing is wrong and folding would mis-read every
    row."""
    m = build()
    m.nodes["ATT"].attributes["num_heads"] = 8
    assert matches_of(m) == []


def test_shared_transpose_declined():
    """A Transpose feeding two Unsqueezes cannot be deleted."""
    m = build()
    dup = Node("Unsqueeze", "UNSQ_dup", {},
               [tdict("tp_q", [H, S, HD]), tdict("unsq_dup_ax", [1], 7)],
               [tdict("unsq_dup", [1, H, S, HD])])
    m.nodes[dup.name] = dup
    axis_init(m, "unsq_dup_ax", 0)
    assert matches_of(m) == []
