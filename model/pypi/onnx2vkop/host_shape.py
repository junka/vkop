"""Host-side shape-expression extraction (Phase 4-r2 形状链 host 化).

The decode graph spends ~214 dispatches per round on an int64 shape-meta
subgraph — Shape(x) -> Gather(Shape(x), i) -> Concat(...) — whose ONLY
terminal consumers are the 118 Reshape nodes' target-shape inputs (inputs[1]).
Every element of every one of those int64 vectors is either a constant or a
single dim of some live tensor, so the whole subgraph can be evaluated on the
host from tensor metadata (getShape()) with zero GPU work and zero
synchronisation.

This pass rewrites each convertible Reshape to a SINGLE input (the data
tensor) plus a string attribute `vkop_host_shape` that encodes the target
shape as a term list:

    c<value>          -> constant dim
    i<dim_idx>:<name> -> runtime dim: the dim_idx-th dim of tensor <name>
                         (graph input, initializer or intermediate — anything
                         the runtime holds as a tensor)

e.g. `c1,i1:inputs_embeds,c2048` = [1, seq, 2048]. core/runtime.cpp parses the
attribute at LoadModel, resolves each term's tensor pointer once, and
evaluates the term list lazily right before the Reshape executes each round —
lazy because a term may reference a tensor produced earlier in the SAME round
(e.g. the layer's residual stream), whose dims are only final once its
producer has run.

Safety rules (fail closed -> the Reshape keeps its old two-input form):
  * the target chain must be built only from Shape / Gather / Concat / Cast
    (int64->int64) over initializers, with Gather indices and Concat inputs
    being constants;
  * every REF source tensor's producer must be a topological ancestor of the
    Reshape (or the source is a graph input / initializer), so the lazy
    evaluation reads an already-final shape;
  * no consumer of a REF source may record a reinterpretation reshape_view
    (Unsqueeze-fold pattern) — the live dims must always equal the shape the
    deleted Shape op would have read;
  * a target vector consumed anywhere else than a Reshape[1] stays alive via
    normal liveness (save_to_binary pruning), so partial conversion degrades
    to the old dispatch path, never to wrong shapes.
"""

from onnx import numpy_helper

# The attribute name the C++ loader keys on (core/runtime.cpp LoadModel).
HOST_SHAPE_ATTR = "vkop_host_shape"

# Op types allowed inside a Reshape target-shape chain.
_CHAIN_OPS = {"Shape", "Gather", "Concat", "Cast"}


def _term_str(term):
    if term[0] == "c":
        return "c%d" % (term[1],)
    return "i%d:%s" % (term[2], term[1])


class _Resolver:
    """Backward resolver: tensor name -> list of terms, or None (bail)."""

    def __init__(self, dag_model):
        self.dag = dag_model
        self.producer = {}
        self.tensor_shape = {}
        for n in dag_model.nodes.values():
            for o in n.outputs:
                self.producer[o["name"]] = n
                self.tensor_shape[o["name"]] = list(o.get("shape") or [])
        for i in dag_model.inputs:
            self.tensor_shape[i["name"]] = list(i.get("shape") or [])
        for name, init in dag_model.initializers.items():
            self.tensor_shape[name] = list(init.dims)
        self.memo = {}

    def _init_int_values(self, tname):
        """A small int64/int32 initializer as a python int list, else None."""
        init = self.dag.initializers.get(tname)
        if init is None:
            return None
        if init.data_type not in (7, 6):  # int64, int32
            return None
        if len(init.dims) > 1:
            return None
        if init.raw_data and len(init.raw_data) > 256:
            return None
        try:
            arr = numpy_helper.to_array(init).reshape(-1)
        except Exception:
            return None
        return [int(v) for v in arr]

    def _const_scalar(self, tname):
        vals = self._init_int_values(tname)
        if vals is None or len(vals) != 1:
            return None
        return vals[0]

    def resolve(self, tname):
        """(terms, chain_nodes) for the 1-D int64 vector `tname`, or None.

        chain_nodes: names of the chain ops the terms were built from — the
        caller excludes them from the post-rewrite reachability check, because
        once the Reshape stops consuming the chain those nodes are pruned and
        can no longer guarantee ordering for the REF sources.
        """
        if tname in self.memo:
            return self.memo[tname]
        self.memo[tname] = None  # cycle guard
        result = self._resolve_impl(tname, set())
        self.memo[tname] = result
        return result

    def _resolve_impl(self, tname, chain):
        # A plain initializer IS the constant vector.
        if tname in self.dag.initializers:
            vals = self._init_int_values(tname)
            if vals is None:
                return None
            return [("c", v) for v in vals], chain

        node = self.producer.get(tname)
        if node is None or node.op_type not in _CHAIN_OPS:
            return None
        chain = chain | {node.name}

        if node.op_type == "Shape":
            src = node.inputs[0]["name"]
            shape = self.tensor_shape.get(src)
            if shape is None or len(shape) == 0:
                return None  # unknown rank or rank-0: fail closed
            return [("r", src, i) for i in range(len(shape))], chain

        if node.op_type == "Gather":
            axis = node.attributes.get("axis", 0)
            if axis != 0:
                return None
            up = self._resolve_impl(node.inputs[0]["name"], chain)
            if up is None:
                return None
            data, chain = up
            idx = self._const_scalar(node.inputs[1]["name"])
            if idx is None:
                return None
            if idx < 0:
                idx += len(data)
            if idx < 0 or idx >= len(data):
                return None
            return [data[idx]], chain

        if node.op_type == "Concat":
            axis = node.attributes.get("axis", 0)
            if axis not in (0, -1):
                return None
            terms = []
            for inp in node.inputs:
                up = self._resolve_impl(inp["name"], chain)
                if up is None:
                    return None
                part, chain = up
                terms.extend(part)
            return terms, chain

        # Cast: int64 -> int64 is an identity on the vector.
        if node.op_type == "Cast":
            if node.attributes.get("to") != 7:
                return None
            in_dtype = node.inputs[0].get("dtype")
            if in_dtype not in (7, None):
                return None
            return self._resolve_impl(node.inputs[0]["name"], chain)

        return None


def _view_consistent(consumers_of, src, canon):
    """True when no consumer of `src` records a reshape_view that
    reinterprets the tensor. The runtime evaluates REF terms from the LIVE
    tensor dims (getShape()), but the deleted Shape op read the producer's
    shape. A consumer in between that records a different view (e.g. an
    Unsqueeze fold recording a 5-D view on a 4-D tensor — same element
    count, so Tensor::reshape_view applies it) would corrupt the
    host-evaluated dims. Records containing -1 are skipped by the runtime
    (dynamic sentinel → no view applied) and empty records carry no
    reinterpretation, so both are always consistent."""
    if canon is None:
        return False
    for rec in consumers_of.get(src, ()):
        if not rec or -1 in rec:
            continue
        if rec != canon:
            return False
    return True


def _data_reachable_producers(node, producer_of_tensor, dependencies, exclude):
    """Producers reachable from `node`'s DATA inputs without passing through
    `exclude` nodes. This mirrors the graph AFTER the rewrite (chain nodes
    pruned, Reshape bound to its data input only): a REF source whose producer
    is not here could execute after the Reshape in the recomputed levels, and
    the lazy per-round evaluation would read a stale dim."""
    seen = set()
    stack = []
    # Post-rewrite the Reshape's ONLY input is the data tensor (inputs[0]);
    # the target input is gone, so ordering can only flow from here.
    prod = producer_of_tensor.get(node.inputs[0]["name"])
    if prod:
        stack.append(prod)
    while stack:
        cur = stack.pop()
        if cur in seen or cur in exclude:
            continue
        seen.add(cur)
        stack.extend(dependencies[cur])
    return seen


def extract_host_shape_expressions(dag_model):
    """Rewrite Reshape nodes to host-evaluated target shapes.

    Returns a stats dict {reshapes, converted} for logging. Chain nodes whose
    outputs lose their last consumer are removed later by the standard
    liveness pruning inside save_to_binary().
    """
    # Fresh dependency edges: fusion/quant passes may have left stale ones.
    dag_model.build_dependencies()
    nodes = dag_model.nodes
    resolver = _Resolver(dag_model)
    graph_input_names = {i["name"] for i in dag_model.inputs}
    producer_of_tensor = {}
    for n in nodes.values():
        for o in n.outputs:
            producer_of_tensor[o["name"]] = n.name
    # Recorded edge shapes per tensor name, for the view-consistency check.
    consumers_of = {}
    for n in nodes.values():
        for inp in n.inputs:
            consumers_of.setdefault(inp["name"], []).append(
                list(inp.get("shape") or []))

    converted = 0
    total = 0
    for node in list(nodes.values()):
        if node.op_type != "Reshape" or len(node.inputs) != 2:
            continue
        total += 1
        resolved = resolver.resolve(node.inputs[1]["name"])
        if resolved is None:
            continue
        terms, chain = resolved

        # Post-rewrite ordering guard: every REF source must be a graph
        # input, an initializer, or produced by a node reachable from the
        # Reshape's data inputs WITHOUT crossing the chain being deleted.
        reachable = None
        ok = True
        for term in terms:
            if term[0] != "r":
                continue
            src = term[1]
            # Every REF source must present the SAME live shape the deleted
            # Shape op would have read: no consumer may record a
            # reinterpretation view on it (fail closed otherwise).
            if not _view_consistent(consumers_of, src,
                                    resolver.tensor_shape.get(src)):
                ok = False
                break
            if src in graph_input_names or src in dag_model.initializers:
                continue
            prod = producer_of_tensor.get(src)
            if prod is None:
                ok = False
                break
            if reachable is None:
                reachable = _data_reachable_producers(
                    node, producer_of_tensor,
                    {n: v.dependencies for n, v in nodes.items()}, chain)
            if prod not in reachable and prod != node.name:
                ok = False
                break
        if not ok or not terms:
            continue

        node.inputs = [node.inputs[0]]
        node.attributes[HOST_SHAPE_ATTR] = ",".join(
            _term_str(t) for t in terms)
        converted += 1

    return {"reshapes": total, "converted": converted}
