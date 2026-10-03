#!/usr/bin/env python3
"""Staticize the DiT graphs and back-fill every intermediate's shape into value_info.

onnx.shape_inference gives up on these graphs: data_prop does not chain through
computed intermediates, and the symbolic dims (target_len / prefix_len / kv_len)
make the whole shape-metadata chain (Shape -> Gather -> Mod -> Slice -> Concat
-> Reshape) statically unsolvable. So we (1) freeze the graph inputs to the
concrete sizes the C++ driver actually uses, then (2) walk the nodes ourselves,
evaluating the small shape-chain tensors with numpy and deriving dims for the
dataflow ops from them. Everything stays metadata-only — the graph structure and
the external weight file are untouched, we only add value_info entries so the
converter records real shapes and the runtime shape pool can recycle buffers.

    python propagate_shapes.py prefill decode

Writes dit_{mode}_static.onnx next to dit_{mode}_si.onnx.
"""

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).resolve().parent

STATIC_SHAPES = {
    # C++ image_gen 驱动的固定尺寸：prefix_len=64, target 512x512 -> 32x32 latent
    "prefill": {
        "prompt_embeds": [1, 64, 4096],
        "cos": [64, 128],
        "sin": [64, 128],
        "timestep_zero": [1],
        "attention_bias": [1, 1, 64, 64],
    },
    "decode": {
        "target_latents": [1, 1024, 64],
        "timestep": [1],
        "cos": [1024, 128],
        "sin": [1024, 128],
        "attention_bias": [1, 1, 1024, 1088],
        **{f"past_kv_{i}": [1, 2, 32, 64, 128] for i in range(32)},
    },
}

_ELEMTYPE_NP = {
    1: np.float32, 10: np.float16, 7: np.int64, 6: np.int32,
    9: np.bool_, 4: np.uint64, 3: np.int8, 2: np.uint8,
}
_NP_ELEMTYPE = {v: k for k, v in _ELEMTYPE_NP.items()}


def set_shape(proto, dims):
    ns = onnx.TensorShapeProto()
    for d in dims:
        ns.dim.add().dim_value = int(d)
    proto.type.tensor_type.ClearField("shape")
    proto.type.tensor_type.shape.CopyFrom(ns)


def staticize(m, shapes):
    for inp in m.graph.input:
        if inp.name in shapes:
            set_shape(inp, shapes[inp.name])


def freeze_outputs(m, t):
    """把 graph.output 的符号维换成具体值。

    导出器留下的输出形状是 ['MatMulsample_dim_0', 'target_len', 64] 这类符号，
    转换器查表时记成 -1，运行时就把它当动态维、既不进形状池也读不到真实长度。
    """
    frozen = 0
    for out in m.graph.output:
        dims = t.dims.get(out.name)
        if dims is None or any(d is None for d in dims):
            continue
        if all(d.HasField("dim_value") for d in out.type.tensor_type.shape.dim) and \
                [d.dim_value for d in out.type.tensor_type.shape.dim] == dims:
            continue
        set_shape(out, dims)
        frozen += 1
    return frozen


def broadcast(a, b):
    # 任一侧未知就必须返回未知：广播只会让秩变大，拿已知那侧冒充结果会把
    # 一个高秩激活记成低秩形状，转换器据此分配的 buffer 直接小于实际占用。
    if a is None or b is None:
        return None
    rank = max(len(a), len(b))
    a2 = [1] * (rank - len(a)) + list(a)
    b2 = [1] * (rank - len(b)) + list(b)
    out = []
    for x, y in zip(a2, b2):
        if x == y:
            out.append(x)
        elif x == 1:
            out.append(y)
        elif y == 1:
            out.append(x)
        else:
            return None  # incompatible — give up on this node
    return out


def matmul_dims(a, b):
    if a is None or b is None or len(a) < 2 or len(b) < 2:
        return None
    if a[-1] != b[-2]:
        return None
    batch = broadcast(a[:-2], b[:-2])
    if batch is None:
        return None
    return batch + [a[-2], b[-1]]


class ShapeTable:
    """dims/elem_type per tensor name, seeded from io + initializers."""

    def __init__(self, g):
        self.dims = {}
        self.etype = {}
        for vi in list(g.value_info) + list(g.input) + list(g.output):
            tt = vi.type.tensor_type
            if tt.HasField("shape") and all(d.HasField("dim_value") for d in tt.shape.dim):
                self.dims[vi.name] = [d.dim_value for d in tt.shape.dim]
                self.etype[vi.name] = tt.elem_type
        for init in g.initializer:
            self.dims[init.name] = list(init.dims)
            self.etype[init.name] = init.data_type
        self.inits = {i.name: i for i in g.initializer}
        self.init_arrays = {}
        for i in g.initializer:
            if i.data_location == onnx.TensorProto.EXTERNAL or len(i.raw_data) > (1 << 20):
                continue
            try:
                self.init_arrays[i.name] = numpy_helper.to_array(i)
            except Exception:
                pass
        # 图里现成的 Constant 节点：不删、不求值，但它的 dims 必须能用。
        for n in g.node:
            if n.op_type != "Constant" or not n.output:
                continue
            for a in n.attribute:
                if a.name == "value" and a.type == onnx.AttributeProto.TENSOR:
                    t = a.t
                    if t.data_location == onnx.TensorProto.EXTERNAL:
                        continue
                    self.dims.setdefault(n.output[0], list(t.dims))
                    self.etype.setdefault(n.output[0], t.data_type)
                    if len(t.raw_data) <= (1 << 20):
                        try:
                            self.init_arrays[n.output[0]] = numpy_helper.to_array(t)
                        except Exception:
                            pass

    def arr(self, name):
        """小张量的具体值：initializer/Constant 原值或传播过程中算出来的值。"""
        if name in self.init_arrays:
            return self.init_arrays[name]
        return getattr(self, "values", {}).get(name)

    def record(self, name, dims, etype):
        if dims is None:
            return
        self.dims[name] = list(dims)
        if etype:
            self.etype[name] = etype


def compute(node, t):
    op = node.op_type
    ins = [t.dims.get(i) for i in node.input]

    def arr(i):
        return t.arr(node.input[i]) if len(node.input) > i else None

    def etype_of(i):
        return t.etype.get(node.input[i]) if len(node.input) > i else None

    def attr_i(name, default=None):
        for a in node.attribute:
            if a.name == name:
                return a.i
        return default

    def attr_ints(name, default=None):
        for a in node.attribute:
            if a.name == name:
                return list(a.ints)
        return default

    if op == "MatMul":
        return [(matmul_dims(ins[0], ins[1]), etype_of(0))]
    if op in ("Mul", "Add", "Sub", "Div", "Min", "Max", "Pow", "Mod", "Where"):
        if op == "Where":
            d = broadcast(broadcast(ins[0], ins[1]), ins[2])
            et = etype_of(1) or etype_of(2)
        else:
            d = broadcast(ins[0], ins[1])
            et = etype_of(0) or etype_of(1)
        return [(d, et)]
    if op in ("Neg", "Sqrt", "Floor", "Ceil", "Erf", "Tanh", "Sigmoid",
              "HardSwish", "Softmax", "LayerNormalization", "Identity",
              "Gelu", "FastGelu", "LeakyRelu", "Relu", "Mish", "Clip",
              "Cos", "Sin", "Exp", "Log", "Abs", "Reciprocal"):
        return [(ins[0], etype_of(0))]
    if op == "Cast":
        return [(ins[0], attr_i("to"))]
    if op == "Reshape":
        shp = arr(1)
        if shp is None or ins[0] is None:
            return [(None, etype_of(0))]
        allowzero = attr_i("allowzero", 0)
        shp = shp.astype(np.int64).tolist()
        in_elems = int(np.prod(ins[0])) if ins[0] else 0
        out = []
        for i, s in enumerate(shp):
            if s == 0 and not allowzero:
                out.append(ins[0][i] if i < len(ins[0]) else 1)
            else:
                out.append(int(s))
        if -1 in out:
            pos = out.index(-1)
            rest = int(np.prod([d for j, d in enumerate(out) if j != pos])) or 1
            out[pos] = in_elems // rest
        return [(out, etype_of(0))]
    if op == "Transpose":
        perm = attr_ints("perm")
        if perm is None:
            perm = list(range(len(ins[0])))[::-1]
        return [([ins[0][p] for p in perm], etype_of(0))]
    if op == "Unsqueeze":
        axes = (arr(1).tolist() if len(node.input) > 1 else attr_ints("axes", []))
        out = list(ins[0])
        for ax in sorted(int(a) for a in axes):
            ax = ax if ax >= 0 else ax + len(out) + 1
            out.insert(ax, 1)
        return [(out, etype_of(0))]
    if op == "Squeeze":
        axes = (arr(1).tolist() if len(node.input) > 1 else attr_ints("axes", None))
        if axes is None:
            return [([d for d in ins[0] if d != 1], etype_of(0))]
        axes = set(int(a) if a >= 0 else a + len(ins[0]) for a in axes)
        return [([d for i, d in enumerate(ins[0]) if i not in axes], etype_of(0))]
    if op == "Concat":
        if ins[0] is None:
            return [(None, None)]
        axis = attr_i("axis", 0)
        axis = axis if axis >= 0 else axis + len(ins[0])
        if any(d is None for d in ins):
            return [(None, etype_of(0))]
        out = list(ins[0])
        out[axis] = sum(d[axis] for d in ins)
        return [(out, etype_of(0))]
    if op == "Slice":
        x = ins[0]
        starts, ends = arr(1), arr(2)
        if x is None or starts is None or ends is None:
            return [(None, etype_of(0))]
        axes = arr(3).tolist() if len(node.input) > 3 and arr(3) is not None else list(range(len(x)))
        steps = arr(4).tolist() if len(node.input) > 4 and arr(4) is not None else [1] * len(axes)
        out = list(x)
        for a, s, e, st in zip(axes, starts.tolist(), ends.tolist(), steps):
            a = a if a >= 0 else a + len(x)
            n = x[a]
            s = s + n if s < 0 else s
            e = e + n if e < 0 else min(e, n)
            out[a] = (e - s + st - 1) // st if st > 0 else max(0, (s - e - st - 1) // (-st))
        return [(out, etype_of(0))]
    if op == "Gather":
        if ins[0] is None:
            return [(None, None)]
        axis = attr_i("axis", 0)
        axis = axis if axis >= 0 else axis + len(ins[0])
        idx_dims = ins[1]
        if idx_dims is None:
            return [(None, etype_of(0))]
        out = ins[0][:axis] + list(idx_dims) + ins[0][axis + 1:]
        return [(out, etype_of(0))]
    if op == "Shape":
        return [([len(ins[0])], TensorProto.INT64)]
    if op == "Split":
        n = len(node.output)
        x = ins[0]
        if x is None:
            return [(None, etype_of(0))] * n
        axis = attr_i("axis", 1) % len(x)
        parts = attr_ints("split")
        if parts is None:
            p = arr(1)
            parts = None if p is None else p.astype(np.int64).tolist()
        if parts is None:
            if x[axis] % n:
                return [(None, etype_of(0))] * n
            parts = [x[axis] // n] * n
        # 0 表示"由总量反推"、负数表示未知剩余，两种都可能出错，直接放弃。
        if len(parts) != n or any(p <= 0 for p in parts) or sum(parts) != x[axis]:
            return [(None, etype_of(0))] * n
        outs = []
        for p in parts:
            d = list(x)
            d[axis] = int(p)
            outs.append((d, etype_of(0)))
        return outs
    if op in ("ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin", "ReduceProd"):
        keepdims = attr_i("keepdims", 1)
        x = ins[0]
        axes = attr_ints("axes")
        if axes is None and len(node.input) > 1 and arr(1) is not None:
            axes = arr(1).tolist()
        if axes is None:
            axes = list(range(len(x)))
        axes = set(a if a >= 0 else a + len(x) for a in axes)
        if keepdims:
            out = [1 if i in axes else d for i, d in enumerate(x)]
        else:
            out = [d for i, d in enumerate(x) if i not in axes]
        return [(out, etype_of(0))]
    if op == "Expand":
        shp = arr(1)
        if shp is None:
            return [(None, etype_of(0))]
        return [(broadcast(ins[0], shp.astype(np.int64).tolist()), etype_of(0))]
    if op == "ConstantOfShape":
        val = next((a.t for a in node.attribute if a.name == "value"), None)
        et = val.data_type if val is not None else TensorProto.FLOAT
        shp = arr(0) if node.input and node.input[0] else None
        if shp is None:
            return [(None, et)]
        return [(shp.astype(np.int64).tolist(), et)]
    if op == "Constant":
        for a in node.attribute:
            if a.name == "value":
                tt = a.t
                return [(list(tt.dims), tt.data_type)]
        return [(None, None)]
    if op == "Range":
        return [([None], etype_of(0))]  # dynamic length — leave unset
    if op in ("GatherElements", "ScatterND", "ScatterElements",
              "TopK", "NonZero", "Einsum", "Attention", "Compress", "Pad",
              "Resize", "Tile", "Flatten", "SimplifiedLayerNormalization",
              "SkipLayerNormalization", "ReduceL2", "ArgMax", "ArgMin",
              "Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual",
              "Not", "And", "Or", "Xor", "IsInf", "IsNaN", "Sign", "Round",
              "Shrink", "CumSum", "Trilu", "Dropout", "LSTM", "If", "Loop"):
        # 少见算子：保守跳过（保持 unknown），不出错
        return [(None, None)]
    return [(None, None)]


VALUE_CAP = 1 << 20  # 只为 <=1MB 的张量算具体值（shape 元数据链），权重跳过


def eval_value(node, t):
    """元数据算子的 numpy 求值；输入不全是小常量就返回 None。"""
    op = node.op_type
    names = [i for i in node.input if i]
    # Constant/Split 自带取值条件；Shape/Size 只读输入的秩与形状，不需要张量值
    # （否则任何大激活的 /Shape 都会被"输入不在 values 里"的门禁挡掉）。
    if op in ("Shape", "Size"):
        dims = t.dims.get(node.input[0])
        if dims is None:
            return None
        return (np.array(dims, dtype=np.int64) if op == "Shape"
                else np.array(int(np.prod(dims)), dtype=np.int64))
    if op not in ("Constant", "Split") and (
            not names or any(i not in t.values for i in names)):
        return None
    ins = [t.values[i] for i in names if i in t.values]

    def attr_i(name, default=None):
        for a in node.attribute:
            if a.name == name:
                return a.i
        return default

    def attr_ints(name, default=None):
        for a in node.attribute:
            if a.name == name:
                return list(a.ints)
        return default

    if op == "Constant":
        for a in node.attribute:
            if a.name == "value":
                if a.t.data_location == TensorProto.EXTERNAL:
                    return None
                arr = numpy_helper.to_array(a.t)
                return arr if arr.nbytes <= VALUE_CAP else None
        return None
    if op == "Split":
        x = t.values.get(node.input[0])
        if x is None:
            return None
        axis = attr_i("axis", 1)
        parts = attr_ints("split")
        if parts is None:
            p = t.values.get(node.input[1]) if len(node.input) > 1 else None
            parts = None if p is None else p.astype(np.int64).tolist()
        if parts is None:
            if x.shape[axis] % len(node.output):
                return None
            parts = [x.shape[axis] // len(node.output)] * len(node.output)
        if any(p <= 0 for p in parts) or sum(parts) != x.shape[axis]:
            return None
        return np.split(x, np.cumsum(parts)[:-1], axis=axis)
    if op == "Gather":
        return np.take(ins[0], ins[1], axis=attr_i("axis", 0))
    if op == "Concat":
        return np.concatenate(ins, axis=attr_i("axis", 0))
    if op == "Reshape":
        return ins[0].reshape(ins[1].astype(np.int64).tolist())
    if op == "Transpose":
        return np.transpose(ins[0], attr_ints("perm", None))
    if op == "Unsqueeze":
        x = ins[0]
        for ax in sorted(int(a) for a in ins[1].tolist()):
            x = np.expand_dims(x, ax if ax >= 0 else ax + x.ndim + 1)
        return x
    if op == "Squeeze":
        x = ins[0]
        axes = ins[1].tolist() if len(ins) > 1 else None
        return np.squeeze(x, axis=tuple(axes) if axes else None)
    if op == "Cast":
        to = attr_i("to")
        return ins[0].astype(_ELEMTYPE_NP.get(to, np.float32))
    if op == "Slice":
        x, starts, ends = ins[0], ins[1], ins[2]
        axes = ins[3].tolist() if len(ins) > 3 else list(range(x.ndim))
        steps = ins[4].tolist() if len(ins) > 4 else [1] * len(axes)
        idx = [slice(None)] * x.ndim
        for a, s, e, st in zip(axes, starts.tolist(), ends.tolist(), steps):
            idx[a] = slice(int(s), int(e), int(st))
        return x[tuple(idx)]
    if op == "Mul":
        return ins[0] * ins[1]
    if op == "Add":
        return ins[0] + ins[1]
    if op == "Sub":
        return ins[0] - ins[1]
    if op == "Div":
        return np.trunc(ins[0] / ins[1]) if np.issubdtype(ins[0].dtype, np.integer) else ins[0] / ins[1]
    if op == "Mod":
        return np.fmod(ins[0], ins[1]) if attr_i("fmod", 0) else np.mod(ins[0], ins[1])
    if op == "Neg":
        return -ins[0]
    if op == "Floor":
        return np.floor(ins[0])
    if op == "Where":
        return np.where(ins[0], ins[1], ins[2])
    if op == "Range":
        return np.arange(ins[0].item(), ins[1].item(), ins[2].item())
    if op == "ConstantOfShape":
        val = next((a.t for a in node.attribute if a.name == "value"), None)
        fill = numpy_helper.to_array(val).item() if val is not None else 0
        return np.full(ins[0].astype(np.int64).tolist(), fill)
    if op == "Expand":
        return np.broadcast_to(ins[0], ins[1].astype(np.int64).tolist()).copy()
    if op == "Tile":
        return np.tile(ins[0], ins[1].astype(int).tolist())
    return None


def propagate(g):
    """同时算出 dims/elem_type 与 shape 链小张量的具体值。

    图未必严格拓扑序，故多轮推进直到不再变化：DiT 的元数据链最长约八九层
    （Mod→Reshape→Slice→Shape→Slice→Concat→Reshape，再串上 Split→Unsqueeze→
    Add→Mul→MatMul→Shape），8 轮足以收敛。
    """
    t = ShapeTable(g)
    t.values = dict(t.init_arrays)
    for _ in range(8):
        changed = False
        for node in g.node:
            outs = [o for o in node.output if o]
            if not outs:
                continue
            if any(o not in t.values for o in outs):
                try:
                    val = eval_value(node, t)
                except Exception:
                    val = None
                if val is not None:
                    # 多输出算子（Split）返回等长的数组列表，其余只有一份值。
                    vals = list(val) if isinstance(val, (list, tuple)) else [val]
                    for nm, v in zip(outs, vals):
                        if v is None or v.nbytes > VALUE_CAP or nm in t.values:
                            continue
                        et = _NP_ELEMTYPE.get(v.dtype.type, None)
                        t.values[nm] = v
                        t.record(nm, list(v.shape), et)
                        changed = True
            if any(o not in t.dims for o in outs):
                try:
                    results = compute(node, t)
                except Exception:
                    results = [(None, None)]
                for nm, (dims, et) in zip(outs, results):
                    if dims is None or nm in t.dims:
                        continue
                    t.record(nm, dims, et)
                    changed = True
        if not changed:
            break
    return t


def write_value_info(g, t):
    existing = {vi.name for vi in g.value_info}
    added = 0
    for node in g.node:
        for o in node.output:
            if not o or o in existing:
                continue
            dims = t.dims.get(o)
            et = t.etype.get(o)
            if dims is None or et is None or any(d is None for d in dims):
                continue
            g.value_info.append(helper.make_tensor_value_info(o, et, list(dims)))
            existing.add(o)
            added += 1
    return added


def run(mode):
    src = HERE / f"dit_{mode}_si.onnx"
    dst = HERE / f"dit_{mode}_static.onnx"

    m = onnx.load(str(src), load_external_data=False)
    staticize(m, STATIC_SHAPES[mode])
    del m.graph.value_info[:]      # 全部重算，不留推断链的残留
    t = propagate(m.graph)
    added = write_value_info(m.graph, t)
    frozen = freeze_outputs(m, t)

    g = m.graph
    inits = {x.name for x in g.initializer}
    gios = {x.name for x in list(g.input) + list(g.output)}
    total = unknown = known = rank0 = 0
    unknown_ops = {}
    for n in g.node:
        for o in n.output:
            if not o or o in inits or o in gios:
                continue
            total += 1
            dims = t.dims.get(o)
            if dims is None or any(d is None for d in dims):
                unknown += 1
                unknown_ops[n.op_type] = unknown_ops.get(n.op_type, 0) + 1
            elif len(dims) == 0:
                rank0 += 1
            else:
                known += 1
    print(f"[{mode}] value_info added: {added} graph outputs frozen: {frozen}")
    print(f"[{mode}] node outputs: {total} known: {known} rank0: {rank0} unknown: {unknown}")
    print(f"[{mode}] unknown by op:", sorted(unknown_ops.items(), key=lambda x: -x[1])[:12])
    # 只序列化 proto 本体：onnx.save 会顺带重写 external data 文件，
    # 对 offset=0 的首张量必然报 "must be between current file size ..."。
    with open(dst, "wb") as f:
        f.write(m.SerializeToString())
    print(f"[{mode}] saved {dst.name} ({dst.stat().st_size:,} bytes)")


if __name__ == "__main__":
    for arg in sys.argv[1:] or ["prefill", "decode"]:
        run(arg)
