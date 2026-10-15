"""The writer must emit the same bytes regardless of PYTHONHASHSEED.

Four places iterate a node's dependency/dependent SETS: topological_sort and
find_concurrent_nodes (they decide the emitted node order and the order inside
each concurrent level), and the dependencies/dependents vectors of the node
table. Set iteration order follows the per-process string hash seed, so the same
input model produced byte-different .vkopbin files from run to run -- which
makes any byte-compare gate on the converter useless (the level *membership* is
order-independent, so the artifacts were always equivalent, just not
reproducible).

The check has to spawn processes: PYTHONHASHSEED only takes effect at
interpreter start.
"""
import json
import os
import subprocess
import sys
import textwrap

_GRAPH = textwrap.dedent(
    """
    import sys
    from onnx2vkop.dag import DAGBasedModel, Node

    def t(name):
        return {"name": name, "shape": [2, 2], "dtype": 1}

    m = DAGBasedModel()
    m.inputs = [t("x")]
    m.outputs = [t("out")]

    hub = Node("Mul", "hub", {}, [t("x")], [t("h")])
    m.nodes[hub.name] = hub
    for i in range(8):
        b = Node("Add", "branch%02d" % i, {}, [t("h")], [t("b%d" % i)])
        m.nodes[b.name] = b
    fan = Node("Concat", "fan", {}, [t("b%d" % i) for i in range(8)], [t("f")])
    m.nodes[fan.name] = fan
    m.nodes["tail"] = Node("Relu", "tail", {}, [t("f")], [t("out")])

    m.build_dependencies()
    m.save_to_binary(sys.argv[1])
    """
)


def _run(code, path, seed):
    env = dict(os.environ, PYTHONHASHSEED=str(seed))
    return subprocess.run([sys.executable, "-c", code, path], env=env,
                          check=True, capture_output=True, text=True)


def test_binary_is_seed_invariant(tmp_path):
    blobs = {}
    for seed in (0, 1, 7):
        out = tmp_path / ("s%d.vkopbin" % seed)
        _run(_GRAPH, str(out), seed)
        blobs[seed] = out.read_bytes()
    assert len(blobs[0]) > 0
    assert blobs[0] == blobs[1] == blobs[7]


def test_level_membership_does_not_depend_on_the_seed(tmp_path):
    """Sorting must reorder, not reshape: every level keeps the same nodes."""
    code = _GRAPH + "print(json.dumps([sorted(lv) for lv in m.find_concurrent_nodes()]))"
    code = code.replace("import sys", "import sys, json")
    outs = []
    for seed in (0, 3):
        r = _run(code, str(tmp_path / ("l%d.vkopbin" % seed)), seed)
        outs.append(json.loads(r.stdout.strip().splitlines()[-1]))
    expected = [["hub"], sorted("branch%02d" % i for i in range(8)), ["fan"], ["tail"]]
    assert outs[0] == outs[1] == expected
