"""A fusion pass that deletes a producer without rebinding its consumer must
be visible, not silent: the runtime binds an unresolvable input name to an empty
buffer and the kernel reads zeros.

dangling_inputs() is the audit; the writer calls it at the one point where every
pass has already run, and MultiRoundOptimizer calls it after each pass so the
damage is attributed to the pass that did it.
"""
from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.optimizer import MultiRoundOptimizer, OptimizationPass


def td(name):
    return {"name": name, "shape": [2, 2], "dtype": 1}


def _model(with_stranded):
    m = DAGBasedModel()
    m.inputs = [td("x")]
    m.outputs = [td("out")]
    m.initializers = {}
    m.nodes["mul"] = Node("Mul", "mul", {}, [td("x")], [td("m")])
    if with_stranded:
        m.nodes["add"] = Node("Add", "add", {}, [td("m"), td("gone")], [td("out")])
    else:
        m.nodes["add"] = Node("Add", "add", {}, [td("m"), td("x")], [td("out")])
    return m


def test_finds_the_input_nothing_produces():
    assert _model(True).dangling_inputs() == [("add", "gone")]
    assert _model(False).dangling_inputs() == []


def test_an_initializer_counts_as_a_producer():
    m = _model(False)
    m.initializers["b"] = None
    m.nodes["add"].inputs[1]["name"] = "b"
    assert m.dangling_inputs() == []


def test_writer_reports_stranding_at_write_time(tmp_path, capsys):
    _model(True).save_to_binary(str(tmp_path / "s.vkopbin"))
    out = capsys.readouterr().out
    assert "[integrity] 1 dangling input(s)" in out
    assert "add <- gone" in out


def test_writer_tolerates_an_input_ref_with_no_dims(tmp_path):
    """A name-only input ref must serialize as an empty ShapeRef.

    "No dims recorded" is a legitimate state the reader already understands —
    the parser writes "shape": [] for a tensor it cannot resolve. A hard
    subscript on the key instead made one such ref fatal: KeyError('shape') in
    dag.build_shape_vec, raised at *write* time, after the whole conversion, with
    no .vkopbin produced. The Gemm -> MatMul(+Add) lowering was the caller that
    hit it (104 refs on visual.onnx).
    """
    m = _model(False)
    m.nodes["add"].inputs[0] = {"name": "m"}  # no shape/dtype/value_dynamic
    m.build_dependencies()
    out = tmp_path / "nodims.vkopbin"
    m.save_to_binary(str(out))
    assert out.stat().st_size > 0


def test_the_pass_that_strands_it_is_named(capsys):
    class _StrandPass(OptimizationPass):
        """Deletes a producer and forgets to rebind the consumer that reads it."""

        def __init__(self):
            super().__init__("strand_producer", priority=1)

        def apply(self, dag_model):
            if "producer" not in dag_model.nodes:
                return 0
            del dag_model.nodes["producer"]
            return 1

    m = _model(False)
    m.nodes["producer"] = Node("Identity", "producer", {}, [td("x")], [td("p")])
    m.nodes["add"].inputs[1] = td("p")  # add = mul(x) + producer(x)
    m.build_dependencies()
    assert m.dangling_inputs() == []

    opt = MultiRoundOptimizer(max_rounds=3)
    opt.register_pass(_StrandPass())
    capsys.readouterr()
    opt.optimize(m, verbose=True)
    out = capsys.readouterr().out
    assert "dangling input(s) after strand_producer" in out
    assert "add <- p" in out
