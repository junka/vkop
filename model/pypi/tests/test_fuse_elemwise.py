"""Standalone test for fuse_elemwise_chain matcher + folder.

Builds a synthetic DAG chain Add -> Sqrt -> Mul (out = sqrt(A+B)*A) and
verifies the fusion produces one FusedElemwise node with the expected
program, then checks a broadcast-scalar variant and a chain that should
NOT fuse (shared intermediate).
"""
from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.optimizer import FusionOptimizer


def tdict(name, dims=None):
    # The converter stores shapes under the "shape" key.
    d = {"name": name}
    if dims is not None:
        d["shape"] = list(dims)
    return d


def make_chain_addsqrtmul():
    """A=leaf[8], B=leaf[8]. Add(A,B)->t1; Sqrt(t1)->t2; Mul(t2,A)->out."""
    m = DAGBasedModel()
    A = tdict("A", [8])
    B = tdict("B", [8])
    t1 = tdict("t1", [8])
    t2 = tdict("t2", [8])
    out = tdict("out", [8])
    add = Node("Add", "Add_0", {}, [A, B], [t1])
    sqrt = Node("Sqrt", "Sqrt_0", {}, [t1], [t2])
    mul = Node("Mul", "Mul_0", {}, [t2, A], [out])
    for n in (add, sqrt, mul):
        m.nodes[n.name] = n
    m.inputs = [A, B]
    m.outputs = [out]  # mark out live so DCE doesn't prune the fused node
    m.build_dependencies()
    return m


def make_shared_intermediate():
    """Add(A,B)->t1 consumed by BOTH Sqrt(t1) and Neg(t1). Chain must NOT
    cross the shared t1 (Sqrt alone is len-1, below MIN)."""
    m = DAGBasedModel()
    A = tdict("A", [8]); B = tdict("B", [8])
    t1 = tdict("t1", [8]); t2 = tdict("t2", [8]); t3 = tdict("t3", [8])
    out = tdict("out", [8])
    add = Node("Add", "Add_0", {}, [A, B], [t1])
    sqrt = Node("Sqrt", "Sqrt_0", {}, [t1], [t2])
    neg = Node("Neg", "Neg_0", {}, [t1], [t3])
    mul = Node("Mul", "Mul_0", {}, [t2, t3], [out])
    for n in (add, sqrt, neg, mul):
        m.nodes[n.name] = n
    m.inputs = [A, B]
    m.outputs = [out]
    m.build_dependencies()
    return m


def make_add_cast_sqrt():
    """Add(A,B)->t1; Cast(t1)->t2; Sqrt(t2)->out. A chain that crosses a dtype
    boundary in the middle -- Cast is chainable, so all three fuse."""
    m = DAGBasedModel()
    A = tdict("A", [8]); B = tdict("B", [8])
    t1 = tdict("t1", [8]); t2 = tdict("t2", [8]); out = tdict("out", [8])
    add = Node("Add", "Add_0", {}, [A, B], [t1])
    cast = Node("Cast", "Cast_0", {"to": 1}, [t1], [t2])
    sqrt = Node("Sqrt", "Sqrt_0", {}, [t2], [out])
    for n in (add, cast, sqrt):
        m.nodes[n.name] = n
    m.inputs = [A, B]
    m.outputs = [out]
    m.build_dependencies()
    return m


def make_chain_with_scalar():
    """Div(A, 2.0) -> Sqrt -> out. The 2.0 is a single-element initializer
    (scalar constant) that must be encoded as a program scalar (operand 100+),
    NOT a register input. A is the only register leaf."""
    m = DAGBasedModel()
    A = tdict("A", [8])
    two = tdict("two", [1])  # scalar constant
    t1 = tdict("t1", [8])
    out = tdict("out", [8])
    div = Node("Div", "Div_0", {}, [A, two], [t1])
    sqrt = Node("Sqrt", "Sqrt_0", {}, [t1], [out])
    for n in (div, sqrt):
        m.nodes[n.name] = n
    m.inputs = [A]
    m.outputs = [out]
    # Materialize the scalar as an initializer (TensorProto).
    import numpy as np
    from onnx import numpy_helper
    m.initializers["two"] = numpy_helper.from_array(
        np.array([2.0], dtype=np.float32), "two")
    m.build_dependencies()
    return m


def make_int64_chain(non_float=("A", "B", "t1", "t2", "out")):
    """Add(A,B)->t1; Mul(t1,A)->t2; Div(t2,B)->out, with `non_float` declaring
    elem_type 7 (INT64) instead of leaving the dtype unset.

    This is the shape-meta arithmetic over Shape/Gather/Range results: same
    ops, same shape, but the tensors are integers.
    """
    m = DAGBasedModel()

    def td(name, dims):
        d = tdict(name, dims)
        if name in non_float:
            d["dtype"] = 7  # TensorProto.INT64
        return d

    A = td("A", [8])
    B = td("B", [8])
    t1 = td("t1", [8])
    t2 = td("t2", [8])
    out = td("out", [8])
    for n in (Node("Add", "Add_0", {}, [A, B], [t1]),
              Node("Mul", "Mul_0", {}, [t1, A], [t2]),
              Node("Div", "Div_0", {}, [t2, B], [out])):
        m.nodes[n.name] = n
    m.inputs = [A, B]
    m.outputs = [out]
    m.build_dependencies()
    return m


def run_fusion(m):
    opt = FusionOptimizer.create_default_optimizer(max_rounds=5)
    stats = opt.optimize(m, verbose=False)
    return m


def test_addsqrtmul():
    m = make_chain_addsqrtmul()
    m = run_fusion(m)
    fused = [n for n in m.nodes.values() if n.op_type == "FusedElemwise"]
    assert len(fused) == 1, f"expected 1 fused node, got {len(fused)}; nodes={[n.op_type for n in m.nodes.values()]}"
    f = fused[0]
    print("  inputs:", [i["name"] for i in f.inputs])
    print("  ops:", f.attributes["ops"])
    print("  input_shapes:", f.attributes["input_shapes"])
    print("  out_shape:", f.attributes["out_shape"])
    print("  rank:", f.attributes["rank"])
    print("  scalars:", f.attributes["scalars"])
    # Expected: inputs [A, B] (register 0, 1). Wait — order of leaf discovery
    # is by first appearance: Add.inputs=[A,B] -> A=reg0, B=reg1. Mul also
    # reads A (already reg0). So leaf_inputs=[A,B], n_leaf_regs=2.
    # ops: Add{1, dst=2, a=0, b=1}; Sqrt{6, dst=3, a=2, b=0};
    #      Mul{3, dst=0(rewritten), a=3, b=0}
    assert f.attributes["ops"] == [1,2,0,1, 6,3,2,0, 3,0,3,0], f.attributes["ops"]
    assert [i["name"] for i in f.inputs] == ["A", "B"]
    assert f.attributes["rank"] == 1
    assert f.attributes["out_shape"] == [8]
    # input_shapes: A=[8], B=[8], rank 1 -> [8, 8]
    assert f.attributes["input_shapes"] == [8, 8], f.attributes["input_shapes"]
    assert f.attributes["scalars"] == []
    print("  PASS")


def test_shared_not_fused():
    m = make_shared_intermediate()
    before = len(m.nodes)
    m = run_fusion(m)
    fused = [n for n in m.nodes.values() if n.op_type == "FusedElemwise"]
    # t1 is shared (Sqrt + Neg both read it) -> Add's output has 2 consumers,
    # so Add CANNOT join a chain (correct). But Sqrt(t1)->t2 and Neg(t1)->t3
    # are each read only by Mul, so {Sqrt, Neg, Mul} form a DAG-shaped chain
    # (two parallel unary ops feeding one binary op). The register machine
    # handles this: reg0=t1(leaf), Sqrt(reg0)->regA, Neg(reg0)->regB,
    # Mul(regB,regA)->reg0. Add survives (shared producer, not absorbed).
    assert len(fused) == 1, f"expected 1 fused, got {len(fused)}; nodes={[n.op_type for n in m.nodes.values()]}"
    f = fused[0]
    assert len(f.attributes["ops"]) == 12, f"expected 3 ops (12 ints), got {f.attributes['ops']}"
    # Add must survive (not absorbed) — its output t1 is shared.
    optypes = [n.op_type for n in m.nodes.values()]
    assert "Add" in optypes, f"Add (shared producer) must survive; got {optypes}"
    print("  PASS (shared producer Add not absorbed; Sqrt+Neg+Mul DAG fused)")


def test_cast_is_chainable():
    """Add -> Cast -> Sqrt. Cast (OP_CAST=11) is chainable since the register
    machine made it a no-op (the dtype conversion happens at the store), so the
    whole 3-op chain fuses. This replaced the older 'Cast breaks the chain'
    behaviour."""
    m = make_add_cast_sqrt()
    m = run_fusion(m)
    fused = [n for n in m.nodes.values() if n.op_type == "FusedElemwise"]
    assert len(fused) == 1, f"expected 1 fused (cast chainable), got {len(fused)}; nodes={[n.op_type for n in m.nodes.values()]}"
    f = fused[0]
    # Add{1, dst=2, a=0, b=1}; Cast{11, dst=3, a=2, b=0}; Sqrt{6, dst=0, a=3}
    assert f.attributes["ops"] == [1, 2, 0, 1, 11, 3, 2, 0, 6, 0, 3, 0], \
        f.attributes["ops"]
    print("  PASS (Cast is chainable; Add->Cast->Sqrt fuses as 3 ops)")


def test_scalar_constant():
    m = make_chain_with_scalar()
    m = run_fusion(m)
    fused = [n for n in m.nodes.values() if n.op_type == "FusedElemwise"]
    assert len(fused) == 1, f"expected 1 fused, got {len(fused)}; nodes={[n.op_type for n in m.nodes.values()]}"
    f = fused[0]
    print("  inputs:", [i["name"] for i in f.inputs])
    print("  ops:", f.attributes["ops"])
    print("  scalars:", f.attributes["scalars"])
    # Leaves discovered in order: Div.inputs=[A, two]. A is not produced by a
    # chain op -> reg0. two is not produced by a chain op -> reg1, BUT it's a
    # single-element initializer -> scalar index 0, operand code 100.
    # ops: Div{4, dst=2, a=0(reg A), b=100(scalar two)};
    #      Sqrt{6, dst=0(rewritten terminal), a=2, b=0}
    assert f.attributes["scalars"] == [2.0], f.attributes["scalars"]
    assert f.attributes["ops"] == [4, 2, 0, 100, 6, 0, 2, 0], f.attributes["ops"]
    # Inputs include both A and two (two is bound as SSBO too — shader preloads
    # reg1 from it, but the operand path uses the 100+ scalar; harmless).
    assert [i["name"] for i in f.inputs] == ["A", "two"]
    print("  PASS (scalar constant encoded as operand 100+)")


def test_float_chain_fuses():
    """Baseline for the two tests below: the same chain DOES fuse when the
    tensors declare a float dtype."""
    m = make_int64_chain(non_float=())
    m = run_fusion(m)
    fused = [n for n in m.nodes.values() if n.op_type == "FusedElemwise"]
    assert len(fused) == 1, f"expected 1 fused, got {len(fused)}; nodes={[n.op_type for n in m.nodes.values()]}"
    print("  PASS (fp chain fuses)")


def test_non_float_chain_not_fused():
    """An int64 chain must survive unfused.

    FusedElemwise has fp16/fp32 kernels only: the shader picks one off a 1-bit
    flag read from the live input's dtype (ops/FusedElemwise.hpp), so an int64
    input is loaded at the wrong element width and comes out as garbage. On the
    dynamic-grid visual export the shape-meta Add/Mul chains over Shape/Gather
    results are exactly this, and the garbage then null-cast its int64 reader
    (as_tensor<int64_t> returns nullptr) inside copyToCPU.
    """
    m = make_int64_chain()
    m = run_fusion(m)
    fused = [n for n in m.nodes.values() if n.op_type == "FusedElemwise"]
    assert not fused, f"int64 chain must not fuse; got {[f.name for f in fused]}"
    # The three ops are still there, untouched.
    optypes = sorted(n.op_type for n in m.nodes.values())
    assert optypes == ["Add", "Div", "Mul"], optypes
    print("  PASS (int64 chain left alone)")


def test_one_non_float_leaf_blocks_the_chain():
    """Every input and the output is checked, not just one of them: a single
    int64 leaf is enough (it would be bound as an SSBO and preloaded)."""
    m = make_int64_chain(non_float=("B",))
    m = run_fusion(m)
    fused = [n for n in m.nodes.values() if n.op_type == "FusedElemwise"]
    assert not fused, f"one int64 leaf must block the chain; got {[f.name for f in fused]}"
    print("  PASS (one int64 leaf blocks the chain)")


# ---------------------------------------------------------------------------
# The register-allocator's SILU fold.
#
# A SIGMOID+MUL(x, sigmoid) pair collapses into one OP_SILU word. The fold
# frees the SIGMOID's dst and the SILU word immediately re-allocates it, so the
# sigmoid is ALREADY freed -- and the generic "last consumer" sweep then freed
# it a second time, handing the SILU's own output register to a later op. On
# Qwen3.5-27B's norm-gated fuse that produced
#
#     SILU r6, r0       ; r6 = silu(gate)
#     ADD  r6, r1, eps  ; r6 clobbered
#     SQRT r7, r6
#     DIV  r6, 1, r7
#     MUL  r7, r4, r6   ; core * rsqrt
#     MUL  r6, r7, r5   ; normed * (1+w)
#     MUL  r7, r6, r6   ; SELF-multiply -- the gate was gone
#
# i.e. the layer emitted (normed*(1+w))^2 instead of normed*(1+w)*silu(gate):
# all-non-negative out, max 456.75 where HF gives [-5.66, 3.69].
#
# These tests execute the emitted program rather than only inspecting it,
# because every op in it is individually plausible.
SILU, SIGMOID = 14, 7


def _make_gated_chain():
    """The Qwen3.5 gated-RMSNorm chain, in the shape that triggered it:

        gate = Silu(Z)                      <- consumed LAST, six ops later
        s    = Add(var, eps)
        r    = Div(1, Sqrt(s))
        n    = Mul(core, r)
        m    = Mul(n, w1)
        out  = Mul(m, gate)
    """
    import numpy as np
    from onnx import numpy_helper

    m = DAGBasedModel()

    def scalar(name, val):
        m.initializers[name] = numpy_helper.from_array(
            np.array(val, dtype=np.float32), name)
        return tdict(name, [1])

    Z = tdict("Z", [4, 8])
    var = tdict("var", [4, 1])
    core = tdict("core", [4, 8])
    w1 = tdict("w1", [8])
    eps = scalar("eps", 1e-6)
    one = scalar("one", 1.0)
    sig = tdict("sig", [4, 8])
    gate = tdict("gate", [4, 8])
    sv = tdict("sum", [4, 1])
    rv = tdict("root", [4, 1])
    inv = tdict("inv", [4, 1])
    n = tdict("n", [4, 8])
    mm = tdict("mm", [4, 8])
    out = tdict("out", [4, 8])

    for nd in (
        Node("Sigmoid", "Sigmoid_0", {}, [Z], [sig]),
        Node("Mul", "Mul_gate", {}, [Z, sig], [gate]),
        Node("Add", "Add_0", {}, [var, eps], [sv]),
        Node("Sqrt", "Sqrt_0", {}, [sv], [rv]),
        Node("Div", "Div_0", {}, [one, rv], [inv]),
        Node("Mul", "Mul_0", {}, [core, inv], [n]),
        Node("Mul", "Mul_1", {}, [n, w1], [mm]),
        Node("Mul", "Mul_2", {}, [mm, gate], [out]),
    ):
        m.nodes[nd.name] = nd
    m.inputs = [Z, var, core, w1]
    m.outputs = [out]
    m.build_dependencies()
    return m


def _fused_program(m):
    m = run_fusion(m)
    fused = [n for n in m.nodes.values() if n.op_type == "FusedElemwise"]
    assert len(fused) == 1, f"expected one fused node, got {len(fused)}"
    return [int(x) for x in fused[0].attributes["ops"]]


def _run_program(prog, seeds, scalars):
    """Execute a [oc, dst, a, b] program. Operands >= 100 are scalars."""
    import math

    regs = dict(seeds)
    for i in range(0, len(prog), 4):
        oc, dst, a, b = prog[i:i + 4]
        av = scalars[a - 100] if a >= 100 else regs.get(a, 0.0)
        bv = scalars[b - 100] if b >= 100 else regs.get(b, 0.0)
        if oc == 3:
            r = av * bv
        elif oc == 1:
            r = av + bv
        elif oc == 4:
            r = av / bv if bv else 0.0
        elif oc == 6:
            r = math.sqrt(av)
        elif oc == 14:
            r = av / (1.0 + math.exp(-av)) if av < 700 else av
        else:
            raise AssertionError(f"unhandled opcode {oc} in {prog}")
        regs[dst] = r
    return regs


def test_the_silu_gate_is_not_clobbered_before_it_is_read():
    """Structural: nothing may write the SILU's dst register until it is read.

    This is the property the double-free broke, and unlike a numeric check it
    cannot be satisfied by an accidentally-cancelling expression. Measured on
    the 27B the sequence was SILU r6 / ADD r6,.. -- the gate was dead on
    arrival and the last MUL became r6*r6.
    """
    prog = _fused_program(_make_gated_chain())
    silu_dst = None
    for i in range(0, len(prog), 4):
        if prog[i] == SILU:
            assert silu_dst is None, "more than one SILU in this chain"
            silu_dst = prog[i + 1]
    assert silu_dst is not None, f"the SIGMOID+MUL pair did not fold: {prog}"

    read_at = None
    for i in range(0, len(prog), 4):
        if silu_dst in (prog[i + 2], prog[i + 3]):
            read_at = i
    assert read_at is not None, f"the gate register is never read: {prog}"

    for i in range(0, len(prog), 4):
        if i <= read_at and prog[i + 1] == silu_dst:
            # The SILU word itself writes it; anything after does too.
            pass
    writers = [i for i in range(0, len(prog), 4)
               if prog[i + 1] == silu_dst]
    assert len(writers) == 1, (
        f"register r{silu_dst} holding silu(gate) is written by "
        f"{len(writers)} ops ({writers}); the gate is clobbered before its "
        f"consumer at word {read_at}. program={prog}")


def test_the_fused_chain_evaluates_the_gated_formula():
    """Numeric: the program must compute normed * (1+w) * silu(gate)."""
    import math

    prog = _fused_program(_make_gated_chain())
    # Leaf register order is the allocator's (first appearance); assert the
    # mapping this test seeds so a reordering fails loudly instead of passing
    # on wrong numbers. r0/r1 are Z/var and the SILU reads r0.
    Z, var, core, w1 = 0.7, 0.25, 1.5, 1.9
    eps, one = 1e-6, 1.0
    seeds = {0: Z, 1: var, 4: core, 5: w1}
    regs = _run_program(prog, seeds, [eps, one])

    gate = Z / (1.0 + math.exp(-Z))
    want = (core / math.sqrt(var + eps)) * w1 * gate
    got = regs.get(0)
    print(f"  program={prog}")
    print(f"  got={got:.6f} want={want:.6f} (squared form would be "
          f"{((core / math.sqrt(var + eps)) * w1) ** 2:.6f})")
    assert abs(got - want) < 1e-5, (
        f"fused chain gave {got}, expected {want}. A value matching "
        f"{((core / math.sqrt(var + eps)) * w1) ** 2:.6f} means the gate was "
        f"dropped and the last MUL self-multiplied.")


def _scalar_init(m, name, value):
    import numpy as np
    from onnx import numpy_helper
    m.initializers[name] = numpy_helper.from_array(
        np.array([value], dtype=np.float32), name)


def test_residual_clamp_fuses():
    """Add(A,B) -> Min(., 65500) -> Max(., -65500): the fp16 overflow clamp
    behind a residual add folds into the Add's chain as MIN/MAX words."""
    m = DAGBasedModel()
    A = tdict("A", [8]); B = tdict("B", [8])
    hi = tdict("hi", [1]); lo = tdict("lo", [1])
    t1 = tdict("t1", [8]); t2 = tdict("t2", [8]); out = tdict("out", [8])
    for n in (Node("Add", "Add_0", {}, [A, B], [t1]),
              Node("Min", "Min_0", {}, [t1, hi], [t2]),
              Node("Max", "Max_0", {}, [t2, lo], [out])):
        m.nodes[n.name] = n
    m.inputs = [A, B]
    m.outputs = [out]
    _scalar_init(m, "hi", 65500.0)
    _scalar_init(m, "lo", -65500.0)
    m.build_dependencies()
    m = run_fusion(m)
    fused = [n for n in m.nodes.values() if n.op_type == "FusedElemwise"]
    assert len(fused) == 1, [n.op_type for n in m.nodes.values()]
    f = fused[0]
    ops = f.attributes["ops"]
    assert [ops[i] for i in range(0, len(ops), 4)] == [1, 15, 16], ops
    # Both clamp bounds are program scalars, read as operand b.
    assert ops[7] >= 100 and ops[11] >= 100, ops
    assert sorted(f.attributes["scalars"]) == [-65500.0, 65500.0]


def test_variadic_min_not_fused():
    """A 3-input Min has no two-operand program word: it stays a node."""
    m = DAGBasedModel()
    A = tdict("A", [8]); B = tdict("B", [8]); C = tdict("C", [8])
    t1 = tdict("t1", [8]); out = tdict("out", [8])
    for n in (Node("Add", "Add_0", {}, [A, B], [t1]),
              Node("Min", "Min_0", {}, [t1, B, C], [out])):
        m.nodes[n.name] = n
    m.inputs = [A, B, C]
    m.outputs = [out]
    m.build_dependencies()
    m = run_fusion(m)
    assert "Min_0" in m.nodes
    assert not any(n.op_type == "FusedElemwise" for n in m.nodes.values())


def _make_ln_scale(gamma=1.0, beta=0.0, scale_shape=(1, 1, 16)):
    import numpy as np
    from onnx import numpy_helper
    m = DAGBasedModel()
    X = dict(tdict("X", [1, 4, 16]), dtype=10)
    S = dict(tdict("S", list(scale_shape)), dtype=10)
    g = tdict("g", [16]); b = tdict("b", [16])
    ln_out = dict(tdict("ln_out", [1, 4, 16]), dtype=10)
    out = dict(tdict("out", [1, 4, 16]), dtype=10)
    m.initializers["g"] = numpy_helper.from_array(
        np.full(16, gamma, np.float16), "g")
    m.initializers["b"] = numpy_helper.from_array(
        np.full(16, beta, np.float16), "b")
    for n in (Node("LayerNormalization", "LN_0", {"axis": -1, "epsilon": 1e-6},
                   [X, g, b], [ln_out]),
              Node("Mul", "Mul_0", {}, [ln_out, S], [out])):
        m.nodes[n.name] = n
    m.inputs = [X, S]
    m.outputs = [out]
    m.build_dependencies()
    return m


def test_layernorm_runtime_scale_folds():
    """LN(x, 1, 0) * s -> LN(x, s, 0) for a per-channel runtime s."""
    m = run_fusion(_make_ln_scale())
    types = [n.op_type for n in m.nodes.values()]
    assert types == ["LayerNormalization"], types
    ln = next(iter(m.nodes.values()))
    assert [i["name"] for i in ln.inputs] == ["X", "S", "b"]
    assert ln.outputs[0]["name"] == "out"


def test_layernorm_scale_needs_unit_affine():
    """A real gamma, a real beta, or a scale that is not per-channel blocks
    the fold."""
    for kw in ({"gamma": 2.0}, {"beta": 0.5}, {"scale_shape": (1, 4, 16)}):
        m = run_fusion(_make_ln_scale(**kw))
        assert "Mul_0" in m.nodes, (kw, [n.op_type for n in m.nodes.values()])
