"""Opt-in H3 BF16 attention fold, after activation dtype propagation.

Keep the already-rounded Q/K Mul outputs. K is transposed back into the
existing Attention kernel's [B,H,S,D] layout rather than changing its shader.
The fused kernel does not round the score and softmax intermediates to BF16,
so this is an approximate transformation and must be validated on the model.
"""

from .dag import Node
from .optimizer import FusionOptimizer


def _bf16(tensor):
    return tensor.get("dtype") in (16, "bfloat16")


def _compatible(*dims):
    known = {d for d in dims if isinstance(d, int) and d > 0}
    return len(known) <= 1 and all(isinstance(d, int) and d != 0 for d in dims)


def _plain_matmul(node):
    return (node is not None and node.op_type == "MatMul"
            and len(node.inputs) == 2 and len(node.outputs) == 1
            and all(k in ("transA", "transB") and int(v or 0) == 0
                    for k, v in node.attributes.items()))


def apply(dag):
    """Fold only the H3-shaped BF16 chain; return the number of folds."""
    producer, consumers = FusionOptimizer.get_producer_consumer_from_dag(dag)
    graph_outputs = {t["name"] for t in dag.outputs}
    used_names = ({t["name"] for n in dag.nodes.values()
                   for t in n.inputs + n.outputs}
                  | {t["name"] for t in dag.inputs + dag.outputs}
                  | set(dag.initializers))

    def unique(base, used):
        name = base
        suffix = 0
        while name in used:
            suffix += 1
            name = f"{base}_{suffix}"
        used.add(name)
        return name

    def only_consumer(tensor, node, index):
        return (tensor["name"] not in graph_outputs
                and consumers.get(tensor["name"]) == [(node, index)])

    count = 0
    for softmax in list(dag.nodes.values()):
        if (softmax.op_type != "Softmax" or len(softmax.inputs) != 1
                or len(softmax.outputs) != 1
                or set(softmax.attributes) != {"axis"}
                or softmax.attributes["axis"] not in (-1, 3)):
            continue
        score, prob = softmax.inputs[0], softmax.outputs[0]
        qk = producer.get(score["name"])
        readers = consumers.get(prob["name"], [])
        if len(readers) != 1:
            continue
        pv, index = readers[0]
        if (index != 0 or not _plain_matmul(qk) or not _plain_matmul(pv)
                or not only_consumer(qk.outputs[0], softmax, 0)
                or not only_consumer(prob, pv, 0)):
            continue
        q, kt = qk.inputs
        v = pv.inputs[1]
        q_mul, k_mul = producer.get(q["name"]), producer.get(kt["name"])
        if (q_mul is None or q_mul.op_type != "Mul"
                or k_mul is None or k_mul.op_type != "Mul"):
            continue
        # H3 has a [B,S,H,D] -> [B,H,D,S] transpose before the K Mul.
        if not any(
            (p := producer.get(t["name"])) is not None
            and p.op_type == "Transpose"
            and list(p.attributes.get("perm", [])) == [0, 2, 3, 1]
            for t in k_mul.inputs
        ):
            continue
        tensors = (q, kt, v, score, qk.outputs[0], prob, pv.inputs[0],
                   pv.outputs[0])
        if not all(_bf16(t) for t in tensors):
            continue
        shapes = [list(t.get("shape") or []) for t in (q, kt, v, score, prob,
                                                        pv.outputs[0])]
        if any(len(s) != 4 for s in shapes):
            continue
        qs, ks, vs, ss, ps, ys = shapes
        if qs[3] != 128 or ks[2] != 128 or vs[3] != 128 or ys[3] != 128:
            continue
        if not all(_compatible(*dims) for dims in (
            (qs[0], ks[0], vs[0], ss[0], ps[0], ys[0]),
            (qs[1], ks[1], vs[1], ss[1], ps[1], ys[1]),
            (qs[2], ks[3], vs[2], ss[2], ss[3], ps[2], ps[3], ys[2]),
        )):
            continue

        k_name = unique(f"{kt['name']}__h3_attention_k", used_names)
        transpose_name = unique(f"H3_Attention_K_Transpose_{pv.name}",
                                set(dag.nodes))
        attention_name = unique(f"H3_Attention_{pv.name}",
                                set(dag.nodes) | {transpose_name})
        k = dict(kt, name=k_name, shape=[ks[0], ks[1], ks[3], ks[2]])
        dag.nodes[transpose_name] = Node(
            "Transpose", transpose_name, {"perm": [0, 1, 3, 2]}, [kt], [k])
        dag.nodes[attention_name] = Node(
            "Attention", attention_name,
            {"causal": 0, "h3_self_attention": 1},
            [q, k, v], pv.outputs[:])
        for node in (qk, softmax, pv):
            del dag.nodes[node.name]
        count += 1

    if count:
        dag.build_dependencies()
    return count
