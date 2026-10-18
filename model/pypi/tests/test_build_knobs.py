"""The five build-time knobs added in 3f077d5 must reach the stage they name.

`--external-blob-mb`, `--max-fold-rounds`, `--max-fuse-rounds`, `--no-passes`
and `--no-fusions` are not threaded through signatures -- the CLI writes
`VKOP_*` env vars (`cli.py:_apply_build_knobs`) and each reader looks the name
up when it runs. That is one config source instead of two, but it means a knob
is live only if (a) the reader still defaults to `None` rather than a literal,
(b) it reads at call time, and (c) the value it accepts is the value the CLI
emits. Until now nothing in the tree pinned any of that: restoring
`fold(max_rounds: int = 25)` would have turned `--max-fold-rounds` into a no-op
that still printed its `[config]` line, and an ablation list containing a typo
printed back as if it had disabled something -- which makes the A/B number
behind it measure the same pipeline twice.

Scope note: `VKOP_MAX_VIEW_ROUNDS` (`optimizer.py:390`) is deliberately not in
this list. It is read once at module import into `_MAX_VIEW_ROUNDS`, so it is a
shell-env-only knob and cannot be made to follow a CLI flag without moving the
read to the use site. A test here would freeze that asymmetry in place.
"""
import inspect
import os
from types import SimpleNamespace

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from onnx2vkop.cli import _apply_build_knobs
from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.inspect import ModelView
from onnx2vkop.optimizer import (
    ConstantFolder,
    FusionOptimizer,
    MultiRoundOptimizer,
    ONNXOptimizer,
    _disabled_split,
    _env_int,
)

_KNOB_ENVS = (
    "VKOP_EXTERNAL_BLOB_MB",
    "VKOP_MAX_FOLD_ROUNDS",
    "VKOP_MAX_FUSE_ROUNDS",
    "VKOP_NO_PASSES",
    "VKOP_NO_FUSIONS",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """No knob may inherit a value from the developer's shell."""
    for name in _KNOB_ENVS:
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------- _env_int --


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, 25),      # unset
        ("", 25),        # VKOP_MAX_FOLD_ROUNDS= is a no-op, not a crash
        ("7", 7),
        ("0", 0),        # zero must survive: it is how you disable a loop
        ("abc", 25),     # unparsable falls back rather than raising
        ("-1", -1),      # int() accepts negatives, unlike the .isdigit() path
    ],
)
def test_env_int_falls_back_instead_of_raising(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("VKOP_KNOB_UNDER_TEST", raising=False)
    else:
        monkeypatch.setenv("VKOP_KNOB_UNDER_TEST", raw)
    assert _env_int("VKOP_KNOB_UNDER_TEST", 25) == expected


# ------------------------------------------------- signatures stay env-open --


@pytest.mark.parametrize(
    "reader",
    [ConstantFolder.fold, MultiRoundOptimizer.__init__,
     FusionOptimizer.create_default_optimizer, FusionOptimizer.optimize],
)
def test_round_readers_default_to_none_so_the_flag_can_win(reader):
    """A literal default here silently disowns --max-fold/fuse-rounds.

    `fold(model, max_rounds=25)` reads the env only when max_rounds is None, so
    reverting the default makes the CLI flag dead while its [config] line keeps
    printing.
    """
    param = inspect.signature(reader).parameters["max_rounds"]
    assert param.default is None, (
        f"{getattr(reader, '__qualname__', reader)} still hardcodes "
        f"max_rounds={param.default!r}; the CLI/env knob cannot reach it"
    )


# ------------------------------------------------------ call-time resolution --


def test_fuse_rounds_is_read_when_the_optimizer_is_built(monkeypatch):
    monkeypatch.setenv("VKOP_MAX_FUSE_ROUNDS", "7")
    assert MultiRoundOptimizer().max_rounds == 7
    # The env is a default, not a override: an explicit integer still wins.
    assert MultiRoundOptimizer(max_rounds=3).max_rounds == 3


def test_fold_rounds_cap_is_applied_from_the_env(monkeypatch):
    """max_rounds=0 must fold nothing, proving fold() uses the resolved value."""
    monkeypatch.setenv("VKOP_MAX_FOLD_ROUNDS", "0")
    blocked = _count_constants(ConstantFolder.fold(_const_chain(), verbose=False))
    monkeypatch.setenv("VKOP_MAX_FOLD_ROUNDS", "25")
    folded = _count_constants(ConstantFolder.fold(_const_chain(), verbose=False))
    assert blocked > folded, (
        f"VKOP_MAX_FOLD_ROUNDS=0 left {blocked} constants, 25 left {folded} -- "
        "the cap is not coming from the env"
    )


def _const_chain():
    """x + (C0 + C1 + C2): a constant subtree a dynamic input cannot erase."""
    consts = [
        helper.make_node(
            "Constant", [], [f"c{i}"], name=f"c{i}",
            value=helper.make_tensor(f"v{i}", TensorProto.FLOAT, [1], [float(i + 1)]),
        )
        for i in range(3)
    ]
    nodes = list(consts)
    acc = "c0"
    for i in range(1, 3):
        nodes.append(helper.make_node("Add", [acc, f"c{i}"], [f"s{i}"], name=f"add{i}"))
        acc = f"s{i}"
    nodes.append(helper.make_node("Add", [acc, "x"], ["out"], name="addx"))
    graph = helper.make_graph(
        nodes, "chain",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1])],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _count_constants(model):
    return sum(1 for n in model.graph.node if n.op_type == "Constant")


# ----------------------------------------------------------- the CLI mapping --


def test_cli_writes_only_the_knobs_it_was_given(monkeypatch):
    monkeypatch.setenv("VKOP_MAX_FOLD_ROUNDS", "9")  # pre-existing shell value
    _apply_build_knobs(SimpleNamespace(
        external_blob_mb=1, max_fold_rounds=None, max_fuse_rounds=3,
        no_passes="a,b", no_fusions=None,
    ))
    assert os.environ["VKOP_EXTERNAL_BLOB_MB"] == "1"
    assert os.environ["VKOP_MAX_FUSE_ROUNDS"] == "3"
    assert os.environ["VKOP_NO_PASSES"] == "a,b"
    # An omitted flag leaves the shell's value alone instead of clearing it.
    assert os.environ["VKOP_MAX_FOLD_ROUNDS"] == "9"
    assert "VKOP_NO_FUSIONS" not in os.environ


# ------------------------------------------------- the writer's blob switch --


def _dag_with_2p4mb_weight():
    """A graph whose only bulk is one 2.4 MB initializer, kept live by a Mul."""
    def t(name, dims):
        return {"name": name, "shape": dims, "dtype": 1}

    m = DAGBasedModel()
    m.inputs = [t("x", [2, 2])]
    m.outputs = [t("out", [2, 2])]
    m.initializers = {
        "W": numpy_helper.from_array(
            np.zeros(600_000, dtype=np.float32), name="W"),
    }
    m.nodes["mul"] = Node("Mul", "mul", {}, [t("x", [2, 2]), t("W", [600_000])],
                          [t("out", [2, 2])])
    m.build_dependencies()
    return m


@pytest.mark.parametrize(
    "env_value,external",
    [
        (None, False),      # default 800 MB keeps 2.4 MB inline
        ("1", True),        # forced external: blob moves to the file trailer
        ("0", True),        # zero is a real threshold, not "unset"
        ("abc", False),     # dag.py gates on .isdigit(), so this falls back to 800
        ("", False),
    ],
)
def test_external_blob_mb_switches_the_writer(tmp_path, monkeypatch, env_value,
                                              external):
    if env_value is not None:
        monkeypatch.setenv("VKOP_EXTERNAL_BLOB_MB", env_value)
    out = tmp_path / "knob.vkopbin"
    _dag_with_2p4mb_weight().save_to_binary(str(out))
    view = ModelView(str(out))
    assert bool(view.external_blob_trailer) is external


# ------------------------------------------------------- ablation name checks --


def test_disabled_split_separates_typos(monkeypatch):
    monkeypatch.setenv("VKOP_NO_PASSES", "eliminate_identity, NOT_A_REAL_PASS ")
    matching, stray = _disabled_split("VKOP_NO_PASSES", {"eliminate_identity"})
    assert matching == ["eliminate_identity"]
    assert stray == ["NOT_A_REAL_PASS"]


def test_no_passes_typo_is_named_on_the_real_path(monkeypatch, capsys):
    monkeypatch.setenv("VKOP_NO_PASSES", "eliminate_identity,NOT_A_REAL_PASS")
    ONNXOptimizer.optimize_model(_const_chain(), batch_size=1)
    printed = capsys.readouterr().out
    assert "names no onnxoptimizer pass: ['NOT_A_REAL_PASS']" in printed
    # The valid name must not be swept into the typo notice.
    assert "disabled: ['eliminate_identity']" in printed


def test_no_fusions_typo_is_caught_once_the_registry_is_complete(monkeypatch,
                                                                  capsys):
    monkeypatch.setenv("VKOP_NO_FUSIONS", "fuse_into_attention,eliminate_identty")
    opt = FusionOptimizer.create_default_optimizer(max_rounds=1)
    printed = capsys.readouterr().out
    assert opt.warn_unknown_disabled() == ["eliminate_identty"]
    assert "names no registered pass" in printed
    # A pass that really was disabled is known, absent from the run list, and
    # never reported as a typo -- known_passes exists to keep those two straight.
    assert "fuse_into_attention" in opt.known_passes
    assert "fuse_into_attention" not in {p.name for p in opt.passes}
    assert "fuse_into_attention" not in printed.split("names no registered pass")[1]
