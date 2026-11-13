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

# Op types allowed inside a target-shape chain.
#
# The vector being resolved is a 1-D int tensor whose entries are either
# constants or single dims of live tensors. Besides the original
# Shape/Gather/Concat/Cast spine, real graphs build the same vector with
# rank shuffles (Unsqueeze/Squeeze/Reshape), constant-vector broadcasts
# (ConstantOfShape) and element-wise select/arith (Equal/Where/Mul/Div/
# Add/Sub/Slice). Those are all resolvable as long as every element stays
# expressible as "constant" or "dim of a named tensor" -- see _Sym.
_CHAIN_OPS = {
    "Shape", "Gather", "Concat", "Cast",
    "Unsqueeze", "Squeeze", "Reshape", "Slice",
    "ConstantOfShape",
    "Equal", "Where", "Mul", "Div", "Add", "Sub",
}


def _term_str(term):
    if term[0] == "c":
        return "c%d" % (term[1],)
    # ("d", tensor, dim_idx) -> the runtime's `i<dim>:<tensor>` term.
    return "i%d:%s" % (term[2], term[1])


# A symbolic element of a shape vector. Two forms:
#   ("c", value)             a compile-time constant
#   ("d", tensor, dim_idx)   the dim_idx-th dim of a live tensor, read at run time
# Everything the resolver returns is a list of these, so an arbitrary chain of
# rank shuffles / broadcasts / element-wise select folds back to the same
# term language the runtime already understands. Values that cannot be
# expressed this way bail the whole chain (fail closed).
def _C(v):
    return ("c", int(v))


def _D(t, i):
    return ("d", t, int(i))


class _Bail(Exception):
    """The chain cannot be expressed in the term language; fail closed."""


def _as_const(e):
    """The constant value of `e`, or None when it is a runtime dim."""
    return e[1] if e[0] == "c" else None


def _elemwise(op, a, b):
    """Apply a binary op element-wise to two equally-shaped sym vectors."""
    if len(a) != len(b):
        raise _Bail("broadcast of unequal sym vector lengths")
    out = []
    for x, y in zip(a, b):
        cx, cy = _as_const(x), _as_const(y)
        if cx is not None and cy is not None:
            if op == "mul":
                out.append(_C(cx * cy))
            elif op == "div":
                if cy == 0:
                    raise _Bail("division by zero")
                out.append(_C(int(cx / cy)))
            elif op == "add":
                out.append(_C(cx + cy))
            elif op == "sub":
                out.append(_C(cx - cy))
            elif op == "eq":
                out.append(_C(1 if cx == cy else 0))
            else:
                raise _Bail("unknown element-wise op " + op)
            continue
        # At least one side is a live dim. Only the folds below are provable.
        if op == "mul" and (cx == 1 or cy == 1):
            out.append(y if cx == 1 else x)
        elif op == "add" and (cx == 0 or cy == 0):
            out.append(y if cx == 0 else x)
        elif op == "sub" and cy == 0:
            out.append(x)
        elif op == "eq":
            # A tensor dim is never negative, so `dim == -1` (and any other
            # negative constant) is statically false. This is exactly the
            # Where(x == -1, 1, x) idiom the LM head uses to turn the Reshape
            # sentinel back into a real size.
            other = cy if cx is None else cx
            if other is not None and other < 0:
                out.append(_C(0))
            elif other is not None and other == 0:
                # dim == 0 is possible (an empty tensor), so this is NOT
                # statically decidable. Bail rather than guess.
                raise _Bail("dim == 0 is not statically decidable")
            else:
                raise _Bail("dim == non-negative is not statically decidable")
        else:
            raise _Bail("cannot fold %s with a live dim" % op)
    return out


def _where(cond, x, y):
    """Element-wise select; folds whenever the condition is a constant."""
    if len(cond) != len(x) or len(cond) != len(y):
        raise _Bail("Where operand length mismatch")
    out = []
    for c, a, b in zip(cond, x, y):
        cc = _as_const(c)
        if cc is None:
            raise _Bail("Where condition is a live dim")
        out.append(a if cc != 0 else b)
    return out


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

    def _shape_operand_dims(self, tname):
        """When `tname` is Shape(t) and t's declared shape is fully concrete,
        return that shape; else None. The element count of Shape(t) is then a
        compile-time constant even though the values are runtime dims."""
        node = self.producer.get(tname)
        if node is None or node.op_type != "Shape":
            return None
        src = node.inputs[0]["name"]
        shape = self.tensor_shape.get(src)
        if not shape:
            return None
        if any(d is None or d <= 0 for d in shape):
            return None
        return list(shape)

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
            return [_C(v) for v in vals], chain

        node = self.producer.get(tname)
        if node is None or node.op_type not in _CHAIN_OPS:
            return None
        chain = chain | {node.name}

        try:
            return self._resolve_node(node, chain)
        except _Bail:
            return None

    def _sub(self, name, chain):
        """Resolve one operand, propagating _Bail."""
        up = self._resolve_impl(name, chain)
        if up is None:
            raise _Bail("operand %s is not resolvable" % name)
        return up

    def _const_list(self, name, chain):
        """Resolve an operand that must be a pure constant vector."""
        vals, chain = self._sub(name, chain)
        out = []
        for e in vals:
            c = _as_const(e)
            if c is None:
                raise _Bail("operand %s is not a constant" % name)
            out.append(c)
        return out, chain

    def _resolve_node(self, node, chain):
        op = node.op_type

        if op == "Shape":
            src = node.inputs[0]["name"]
            shape = self.tensor_shape.get(src)
            if shape is None or len(shape) == 0:
                raise _Bail("unknown rank")  # unknown rank or rank-0
            return [_D(src, i) for i in range(len(shape))], chain

        if op == "Gather":
            axis = node.attributes.get("axis", 0)
            if axis != 0:
                raise _Bail("Gather axis != 0")
            data, chain = self._sub(node.inputs[0]["name"], chain)
            idx = self._const_scalar(node.inputs[1]["name"])
            if idx is None:
                raise _Bail("Gather index is not a scalar constant")
            if idx < 0:
                idx += len(data)
            if idx < 0 or idx >= len(data):
                raise _Bail("Gather index out of range")
            return [data[idx]], chain

        if op == "Concat":
            ax = node.attributes.get("axis", 0)
            rank = len(self.tensor_shape.get(node.outputs[0]["name"], []) or [])
            # Concatenation along the only axis of a 1-D vector is the only
            # case the term list can express. A rank-1 output concatenated on
            # axis 0 (or -1, which is the same axis) is it.
            if ax not in (0, -1) or rank > 1:
                raise _Bail("Concat not along a 1-D axis")
            terms = []
            for inp in node.inputs:
                part, chain = self._sub(inp["name"], chain)
                terms.extend(part)
            return terms, chain

        # Cast: int64 -> int64 is an identity on the vector.
        if op == "Cast":
            if node.attributes.get("to") != 7:
                raise _Bail("Cast to a non-int64 dtype")
            in_dtype = node.inputs[0].get("dtype")
            if in_dtype not in (7, None):
                raise _Bail("Cast from a non-int64 dtype")
            return self._sub(node.inputs[0]["name"], chain)

        # -- rank shuffles: a 1-D shape vector passes through unchanged -----
        # Unsqueeze adds a leading/trailing unit dim to the CONTAINER, not to
        # the vector's contents; the term list describes the contents.
        if op in ("Unsqueeze", "Squeeze"):
            return self._sub(node.inputs[0]["name"], chain)

        if op == "Reshape":
            # Only a reshape that keeps the element count and rank 1 is
            # representable. A literal -1 in the target is resolved by ONNX to
            # the remaining count, so accept it when the total is known.
            data, chain = self._sub(node.inputs[0]["name"], chain)
            target, chain = self._const_list(node.inputs[1]["name"], chain)
            prod = len(data)
            if len(target) == 1 and target[0] == -1:
                return data, chain
            known = 1
            neg = 0
            for d in target:
                if d == -1:
                    neg += 1
                else:
                    known *= d
            if neg > 1 or (neg == 1 and known == 0):
                raise _Bail("ambiguous Reshape target")
            if neg == 1:
                if prod % known != 0:
                    raise _Bail("Reshape target does not divide")
                target = [prod // known if d == -1 else d for d in target]
            if len(target) != 1 or target[0] != prod:
                raise _Bail("Reshape changes the vector length")
            return data, chain

        if op == "Slice":
            data, chain = self._sub(node.inputs[0]["name"], chain)
            n = len(data)
            starts, chain = self._const_list(node.inputs[1]["name"], chain)
            ends, chain = self._const_list(node.inputs[2]["name"], chain)
            if len(starts) != 1 or len(ends) != 1:
                raise _Bail("multi-axis Slice")
            steps = [1]
            if len(node.inputs) > 3 and node.inputs[3]["name"]:
                steps, chain = self._const_list(node.inputs[3]["name"], chain)
            if len(steps) != 1 or steps[0] == 0:
                raise _Bail("unsupported Slice step")
            st, en, sp = starts[0], ends[0], steps[0]
            if sp < 0:
                # A negative step reverses; the term list has no way to say
                # that, so bail.
                raise _Bail("negative Slice step")
            st = max(0, st + n if st < 0 else st)
            en = max(0, en + n if en < 0 else en)
            en = min(en, n)
            return data[st:en:sp], chain

        if op == "ConstantOfShape":
            # The output is `fill` repeated once per element of the shape
            # input. Two ways that count is known statically:
            #   * the shape input is a constant vector -> multiply it; and
            #   * the shape input is Shape(t) for a tensor whose declared shape
            #     is fully concrete -> the count is t's element count.
            # The second case is what the LM head's target chain hits:
            # ConstantOfShape(Shape(rv)) with rv declared [2]. Without it the
            # whole Where(x == -1, 1, x) select below bails, and with it the
            # select folds to a plain dim vector.
            # Otherwise the vector LENGTH is a runtime dim, which the term
            # language cannot express (it has no repeat) -- fail closed.
            src_shape = self._shape_operand_dims(node.inputs[0]["name"])
            if src_shape is not None:
                count = 1
                for d in src_shape:
                    count *= d
                chain = chain | {node.inputs[0]["name"]}
            else:
                shape_vals, chain = self._const_list(
                    node.inputs[0]["name"], chain)
                count = 1
                for d in shape_vals:
                    count *= d
            fill = self._const_of_shape_fill(node)
            if fill is None:
                raise _Bail("ConstantOfShape fill is not an int")
            return [_C(fill)] * count, chain

        # -- element-wise arithmetic / select ------------------------------
        if op in ("Mul", "Div", "Add", "Sub"):
            a, chain = self._sub(node.inputs[0]["name"], chain)
            b, chain = self._sub(node.inputs[1]["name"], chain)
            # A scalar operand broadcasts over the vector.
            if len(a) == 1 and len(b) > 1:
                a = a * len(b)
            elif len(b) == 1 and len(a) > 1:
                b = b * len(a)
            key = {"Mul": "mul", "Div": "div", "Add": "add",
                   "Sub": "sub"}[op]
            return _elemwise(key, a, b), chain

        if op == "Equal":
            a, chain = self._sub(node.inputs[0]["name"], chain)
            b, chain = self._sub(node.inputs[1]["name"], chain)
            if len(a) == 1 and len(b) > 1:
                a = a * len(b)
            elif len(b) == 1 and len(a) > 1:
                b = b * len(a)
            return _elemwise("eq", a, b), chain

        if op == "Where":
            c, chain = self._sub(node.inputs[0]["name"], chain)
            x, chain = self._sub(node.inputs[1]["name"], chain)
            y, chain = self._sub(node.inputs[2]["name"], chain)
            if len(x) == 1 and len(c) > 1:
                x = x * len(c)
            if len(y) == 1 and len(c) > 1:
                y = y * len(c)
            return _where(c, x, y), chain

        raise _Bail("unhandled chain op " + op)

    def _const_of_shape_fill(self, node):
        """The integer fill value of a ConstantOfShape, or None."""
        attrs = getattr(node, "attributes", None) or {}
        val = attrs.get("value")
        if val is None:
            return 0  # ONNX default fill is 0
        try:
            import numpy as np

            arr = np.asarray(val).reshape(-1)
            if arr.size != 1:
                return None
            return int(arr[0])
        except Exception:
            return None

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


# Which input slot of which op holds a target shape the term language can
# express, per ONNX. Each entry is (op_type, shape_input_index). Only ops whose
# runtime implementation consumes `set_host_target_shape` may appear: the
# rewrite drops the shape input entirely, so an op that does not override that
# hook would lose its only source of the target dims.
_HOST_TARGET_CONSUMERS = {
    "Reshape": 1,
    "Expand": 1,
}


def extract_host_shape_expressions(dag_model):
    """Rewrite shape-consuming nodes to host-evaluated target shapes.

    Returns a stats dict {candidates, converted} for logging. Chain nodes whose
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
    per_op = {}
    for node in list(nodes.values()):
        slot = _HOST_TARGET_CONSUMERS.get(node.op_type)
        if slot is None or len(node.inputs) != slot + 1:
            continue
        if not node.inputs[slot].get("name"):
            continue
        total += 1
        resolved = resolver.resolve(node.inputs[slot]["name"])
        if resolved is None:
            continue
        terms, chain = resolved

        # Post-rewrite ordering guard: every REF source must be a graph
        # input, an initializer, or produced by a node reachable from this
        # node's DATA inputs WITHOUT crossing the chain being deleted.
        reachable = None
        ok = True
        for term in terms:
            if term[0] != "d":
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
        per_op[node.op_type] = per_op.get(node.op_type, 0) + 1
        converted += 1

    return {"candidates": total, "converted": converted, "per_op": per_op}
