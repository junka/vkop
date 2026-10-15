"""MultiRoundOptimizer must report which fusion fired and how often.

PatternBasedFusionPass.apply returns the number of matches fused rather than a
flag, so the report can name the passes that actually did work on a given model
-- the only way to tell a load-bearing pass from one that never fires without
reading the whole convert log.
"""
from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.optimizer import MultiRoundOptimizer, OptimizationPass


def _model():
    m = DAGBasedModel()
    m.inputs = [{"name": "x", "shape": [2, 2], "dtype": 1}]
    m.outputs = [{"name": "out", "shape": [2, 2], "dtype": 1}]
    m.nodes["mul"] = Node("Mul", "mul", {},
                          [{"name": "x", "shape": [2, 2], "dtype": 1}],
                          [{"name": "out", "shape": [2, 2], "dtype": 1}])
    m.initializers = {}
    return m


class _CountingPass(OptimizationPass):
    def __init__(self, hits):
        super().__init__("fake_fuse", priority=1)
        self.calls = 0
        self.hits = hits

    def apply(self, dag_model):
        self.calls += 1
        # Fire on the first round only: the second round must report no change.
        return self.hits if self.calls == 1 else 0


def test_report_counts_fused_matches_per_pass():
    m = _model()
    opt = MultiRoundOptimizer(max_rounds=5)
    opt.register_pass(_CountingPass(hits=3))
    stats = opt.optimize(m, verbose=False)
    assert stats.patterns_fused == {"fake_fuse": 3}
    assert stats.rounds_executed == 2


def test_pass_that_never_fires_is_absent_from_the_report():
    m = _model()
    opt = MultiRoundOptimizer(max_rounds=5)
    opt.register_pass(_CountingPass(hits=0))
    stats = opt.optimize(m, verbose=False)
    assert stats.patterns_fused == {}
    assert stats.rounds_executed == 1
    assert stats.nodes_before == stats.nodes_after
