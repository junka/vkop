"""Node/scope-level precision exclusion for the weight quantizers.

The way an accuracy loss gets localized in a quantized model is by handing the
same layer back to full precision and watching the metric move. DL4AGX does
exactly that and names it an exclusion list; what it does NOT do is keep a list
of op types, because "MatMul is sensitive" is not a property of MatMul -- it is
a property of this MatMul's weights on this model. So the knobs here select
nodes, not formats:

    --exclude-node 'model.layers.*/mlp/down_proj/MatMul'   glob on the node name
    --exclude-scope  model.layers.0                        whole subtree
    --exclude-downstream-of /encoder/embed                 everything that reads
                                                          its output, transitively

Each resolves to a set of weight initializer names, and the byte quantizers
(int8, fp8e4m3/e5m2, int4, nf4, nvfp4, awq) refuse exactly those: the weight
keeps the dtype it arrived with, no scale input is appended, and the node runs
the fp32/fp16 kernel its wide weight always ran. Nothing else about the graph
changes, which is what makes the A/B honest -- the same shaders, the same node
count, one precision difference at a named place.

Two things this deliberately is not, and both are the same reason: a lone wide
weight in an otherwise byte model is not a no-op, it is a decision with
consequences the pass below cannot own.

  * It does not apply to `-q fp16`. `quantize_to_fp16_selective` propagates the
    fp32 domain through the ops that can only be right at full width
    (`_FP32_PROPAGATING`) precisely because a single fp32 weight next to fp16
    activations needs a Cast node -- an island is a correctness problem, not a
    size saving. A per-node "stay fp32" verdict there would have to run the same
    propagation, and that is a different feature.
  * It does not apply to a weight that arrives already on somebody else's
    quantization grid. `PreQuantizedWeights.fold` imports those; it is not
    quantizing them, and excluding its node would leave a live DequantizeLinear
    the import was there to remove. Such a name is reported instead, so a run
    can never read as "this layer is fp16" while its bytes say int8.

One boundary the pass below cannot keep by itself: an excluded weight has to be
in memory to be left alone. Under `-q int8/fp8e4m3/fp8e5m2`, a large weight is
normally deferred to the external-data file and quantized at trailer time, where
the node it belongs to is no longer in hand -- so the CLI runs `--stream-weights
off` for those formats when any exclusion flag is set (and refuses an explicit
`on` rather than silently ignoring it). 4-bit, nvfp4 and AWQ already refuse
deferral because their scale tables span K, so for those this changes nothing.

A pattern is matched case-sensitively with `fnmatch` against the node name (so
`*`, `?` and `[seq]` work), and a name with a leading '/' matches with or
without it. Every flag is repeatable and also takes a comma-separated list; a
scope is a token prefix, where the tokens are the name split on '/' and '.' --
that is what makes 'layers.0' a scope of 'layers.0.attn' and not of 'layers.01'.

Resolution happens BEFORE the fusion passes, for two reasons that point the same
way: the names in the flags are the exporter's names, which is what a user
copies out of a Netron screenshot or a per-layer error table, and fusion
rewrites them (`fuse_into_attention` replaces several MatMuls with one node).
The weights, though, survive fusion under the same name -- a fused node re-points
its input slots at the same initializers -- so resolving nodes early and mapping
them to weight names is stable across the passes that follow.
"""

import fnmatch
import re
import sys
from collections import defaultdict, deque

import onnx

# The ops that only reshape or re-decode a weight on the way to its consumer. A
# MatMul that reads Transpose(W) is quantizing W, and an exclusion that names
# that MatMul means W, not the transpose's output tensor. A QDQ weight is read
# through DequantizeLinear by the graph but lands in the same slot post-fold.
_PAST_THROUGH = ("DequantizeLinear", "Cast", "Transpose", "Reshape", "Identity")

# Module paths reach a node name through either separator, and different
# exporters disagree about which: PyTorch writes '/model/layers.0/attn/MatMul'
# (slash per scope, dot for the numeric index), and an all-dots spelling is
# common enough that both have to count as boundaries.
_SEPARATORS = re.compile(r"[/.]+")


class ExclusionError(ValueError):
    """A pattern matched no node.

    Refusing is the whole point: an exclusion flag that silently selects nothing
    reports a clean run and yields a model that is byte-identical to the one
    without it, which is the worst possible outcome for the accuracy experiment
    it was supposed to serve.
    """


def _tokens(name):
    return [t for t in _SEPARATORS.split(name) if t]


def _patterns(values):
    """The flag values, with each comma-separated group spread out.

    Whitespace between the items is dropped so a shell-quoted
    'a, b, c' is the same three patterns as three flags. An item that is only
    whitespace would match everything as a glob and exclude the whole model,
    so it goes away rather than joining the list.
    """
    out = []
    for value in values or ():
        out.extend(part.strip() for part in str(value).split(","))
    return [p for p in out if p]


def _name_variants(node_name):
    """The spellings a pattern may be written against.

    A leading '/' is how the exporter scopes an absolute path, not part of the
    module name, so both the raw name and it without the slash are matched. That
    way '*down_proj/MatMul' and 'down_proj/MatMul' behave the same whichever way
    the model was exported.
    """
    yield node_name
    stripped = node_name.lstrip("/")
    if stripped != node_name:
        yield stripped


def _glob_match(node_name, pattern):
    return any(fnmatch.fnmatchcase(n, pattern)
               for n in _name_variants(node_name))


def _scope_match(node_name, scope):
    toks = _tokens(scope)
    return any(_tokens(n)[:len(toks)] == toks
               for n in _name_variants(node_name))


def _consumer_index(nodes):
    """output tensor name -> the node names that read it."""
    idx = defaultdict(set)
    for node in nodes.values():
        for inp in node.inputs:
            if inp["name"]:
                idx[inp["name"]].add(node.name)
    return idx


def _producer_index(nodes):
    out = {}
    for node in nodes.values():
        for o in node.outputs:
            name = o["name"] if isinstance(o, dict) else o
            out[name] = node
    return out


def _downstream_closure(nodes, seeds):
    """seeds plus every node that transitively reads their values."""
    consumers = _consumer_index(nodes)
    seen = set(seeds)
    queue = deque(seeds)
    while queue:
        node = nodes[queue.popleft()]
        for o in node.outputs:
            name = o["name"] if isinstance(o, dict) else o
            for nxt in consumers.get(name, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
    return seen


def _weight_inputs(dag_model, node, producers):
    """The initializers this node quantizes, one reshape hop away if needed.

    Slot 0 is included when it IS an initializer: a folded or transposed graph
    can put the weight there, and what matters is the tensor, not its position.
    """
    inits = dag_model.initializers
    found = set()
    for inp in node.inputs:
        name = inp["name"]
        if name in inits:
            found.add(name)
            continue
        prod = producers.get(name)
        if prod is None or prod.op_type not in _PAST_THROUGH:
            continue
        if prod.inputs and prod.inputs[0]["name"] in inits:
            found.add(prod.inputs[0]["name"])
    return found


def resolve(dag_model, node_patterns=None, scopes=None, downstream_of=None):
    """Turn the CLI patterns into the initializer names the quantizers skip.

    Returns the set of names, empty when no flag was given. Every pattern that
    WAS given must hit at least one node or this raises; each hit is reported
    with its counts, because the line a user reads off this run is the line
    their next experiment is keyed on.
    """
    given = (
        ("--exclude-node", node_patterns, _glob_match),
        ("--exclude-scope", scopes, _scope_match),
        ("--exclude-downstream-of", downstream_of, _glob_match),
    )
    flags = []
    for label, raw, matcher in given:
        patterns = _patterns(raw)
        if raw and not patterns:
            # A flag whose value has nothing usable in it ('--exclude-node ,')
            # is the same trap as a typo'd pattern: a clean run on the
            # un-excluded model. Refuse rather than read it as 'no flag'.
            raise ExclusionError(f"{label} {raw!r} contains no pattern")
        for pattern in patterns:
            if label == "--exclude-scope" and not _tokens(pattern):
                raise ExclusionError(
                    f"{label} {pattern!r} has no path tokens. An empty scope "
                    "is a prefix of every node name, so it would read as 'all "
                    "of them' -- name the subtree you mean, e.g. "
                    "model.layers.0."
                )
            flags.append((label, pattern, matcher))
    if not flags:
        return set()

    nodes = dag_model.nodes
    producers = _producer_index(nodes)
    hits = []
    for label, pattern, matcher in flags:
        selected = {n.name for n in nodes.values() if matcher(n.name, pattern)}
        if not selected:
            raise ExclusionError(
                f"{label} {pattern!r} matched no node. The names here are "
                f"the exporter's, so check them against the graph as it "
                f"stands (Netron, or the operator census this conversion "
                f"prints); a pattern that selects nothing would leave the "
                f"model byte-identical to the un-excluded one."
            )
        if label == "--exclude-downstream-of":
            selected = _downstream_closure(nodes, selected)
        hits.append((label, pattern, selected))

    excluded_nodes = set().union(*(s for _, _, s in hits))
    weights = set()
    for name in sorted(excluded_nodes):
        weights |= _weight_inputs(dag_model, nodes[name], producers)

    for label, pattern, selected in hits:
        print(f"[exclude] {label} {pattern!r}: {len(selected)} node(s)")
    print(f"[exclude] {len(excluded_nodes)} node(s), {len(weights)} "
          f"weight(s) kept at their source precision")
    if not weights:
        print("[exclude] WARNING: the matched nodes read no initializer, "
              "so nothing is held back -- this conversion is identical to "
              "the one without the flag.", file=sys.stderr)
    # A weight shared with a node outside the exclusion stays wide for that
    # consumer too: the format is a property of the stored tensor, and this
    # pass does not copy a weight to give one consumer its own dtype.
    readers = _consumer_index(nodes)
    for weight in sorted(weights):
        outside = sorted(readers[weight] - excluded_nodes)
        if outside:
            print(f"[exclude] '{weight}' is also read by "
                  f"{', '.join(outside)} outside the exclusion; it stays "
                  "wide there too", file=sys.stderr)
    _note_imported(dag_model, weights)
    return weights


def _note_imported(dag_model, weights):
    """Say so when an excluded weight is one the QDQ fold imports anyway.

    A payload initializer of integer or fp8 element type is not a weight this
    convertor would quantize -- it is somebody else's grid, and the fold that
    runs next imports it onto the byte path whatever this flag says (see the
    module docstring). Silence here would let an exclusion list read as "these
    layers are fp16" while the report under it says int8.
    """
    byte_types = {v for v in (getattr(onnx.TensorProto, n, None) for n in
                              ("INT8", "UINT8", "INT4", "UINT4",
                               "FLOAT8E4M3FN", "FLOAT8E5M2", "FLOAT4E2M1"))
                  if v is not None}
    hit = sorted(n for n in weights
                 if dag_model.initializers[n].data_type in byte_types)
    if hit:
        print("[exclude] NOTE: these arrive on a quantization grid already; "
              "the pre-quantized import is not excluded, so they are imported "
              f"as byte weights anyway: {', '.join(hit)}", file=sys.stderr)
