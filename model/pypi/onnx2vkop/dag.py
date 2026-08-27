"""DAG-based model representation."""

from collections import deque
from typing import Any, Dict, List, Set

import numpy as np
import flatbuffers
from onnx import numpy_helper

# FlatBuffers 的 Builder 最大 2GB（内部用 32-bit offset 寻址）。LLM 权重达
# 数 GB 时无法把 initializer_blob 塞进 FlatBuffer 向量，改为把 blob 以「外部
# 数据」形式追加在文件末尾：vkopbin = FlatBuffer(root) || external_blob，
# 文件尾部固定 8 字节 LE uint64 记录 external_blob 的起始偏移（为 0 表示没有
# external region，兼容旧文件）。C++ 加载器照常 mmap 整个文件后按该偏移定位
# blob。schema 无需改动（外部区域对 FlatBuffer 完全透明）。
_EXTERNAL_BLOB_MAGIC = 8  # 8 字节 offset 尾巴

from .generated.vkop.model import (
    Attribute,
    ConcurrentLevel,
    InitializerEntry,
    Model,
    Node as NodeTable,
    RGBAConversionMeta,
    ShapeRef,
    TensorData,
    UnifiedMeta,
)
from .generated.vkop.model.AttrType import AttrType

# ONNX TensorProto data_type int -> dtype string. Mirrors the C++ reader's
# datatyep_map[] so writer and reader agree on the spelling.
_DATA_TYPE_MAP = {
    1: "float32",
    2: "uint8",
    3: "int8",
    4: "uint16",
    5: "int16",
    6: "int32",
    7: "int64",
    8: "string",
    9: "bool",
    10: "float16",
    11: "float64",
    12: "uint32",
    13: "uint64",
    14: "complex64",
    15: "complex128",
    16: "bfloat16",
}

# 64-byte alignment for the compact initializer blob, matching the legacy
# C++ two-pass scan (load.cpp alignment constant).
_BLOB_ALIGNMENT = 64


class Node:
    """Represents a node in the computation graph."""

    def __init__(
        self, op_type: str, name: str, attributes: Dict, inputs: List[Dict], outputs: List[Dict]
    ):
        self.op_type = op_type
        self.name = name
        self.attributes = attributes
        self.inputs = inputs
        self.outputs = outputs
        self.dependencies: Set[str] = set()
        self.dependents: Set[str] = set()


class DAGBasedModel:
    """Represents a DAG-based neural network model."""

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.inputs: List[Dict] = []
        self.outputs: List[Dict] = []
        self.initializers: Dict[str, Any] = {}
        # Unified-tensor / RGBA sub-allocation metadata. Populated by the
        # optimizer when --unify / --rgba are wired (Phase 3b/3c); the writer
        # emits them as first-class FlatBuffers tables/structs so the C++
        # reader no longer needs the magic-initializer-name hack.
        self.unified: bool = False
        self.rgba: bool = False
        self.unified_meta: List[Dict] = []
        self.unified_names: str = ""
        self.rgba_meta: List[Dict] = []
        self.rgba_names: str = ""
        self.unified_blob_offset: int = 0

    def add_node(self, node: Node):
        """Add a node to the model."""
        self.nodes[node.name] = node

    def build_dependencies(self):
        """Build dependency relationships between nodes."""
        for node in self.nodes.values():
            node.dependencies.clear()
            node.dependents.clear()

        tensor_producer = {}
        for node in self.nodes.values():
            for output_name in node.outputs:
                tensor_producer[output_name["name"]] = node.name

        for node in self.nodes.values():
            for input_name in node.inputs:
                producer_name = tensor_producer.get(input_name["name"])
                if producer_name and producer_name != node.name:
                    producer_node = self.nodes[producer_name]
                    node.dependencies.add(producer_name)
                    producer_node.dependents.add(node.name)

    def topological_sort(self) -> List[Node]:
        """Perform topological sort to determine execution order."""
        in_degree = {name: len(node.dependencies) for name, node in self.nodes.items()}
        queue = deque([name for name, degree in in_degree.items() if degree == 0])
        sorted_nodes = []

        while queue:
            current = queue.popleft()
            sorted_nodes.append(current)
            for dependent_name in self.nodes[current].dependents:
                in_degree[dependent_name] -= 1
                if in_degree[dependent_name] == 0:
                    queue.append(dependent_name)

        if len(sorted_nodes) != len(self.nodes):
            raise ValueError("Cycle detected in computation graph!")

        return [self.nodes[name] for name in sorted_nodes]

    def find_concurrent_nodes(self) -> List[List[str]]:
        """Find nodes that can be executed in parallel."""
        levels = []
        in_degree = {name: len(node.dependencies) for name, node in self.nodes.items()}

        queue = deque([name for name, degree in in_degree.items() if degree == 0])

        while queue:
            current_level = []
            level_size = len(queue)

            for _ in range(level_size):
                node_name = queue.popleft()
                current_level.append(node_name)

                for dependent_name in self.nodes[node_name].dependents:
                    in_degree[dependent_name] -= 1
                    if in_degree[dependent_name] == 0:
                        queue.append(dependent_name)

            if current_level:
                levels.append(current_level)

        return levels

    def prune_dead_and_materialize_constants(self):
        """写盘前清理运行时图（幂等，重复调用安全）。

        两件事：
          1. 后向 liveness 剪枝：从 graph outputs 出发，把到不了输出端的节点
             整棵删掉（大模型跳过 onnxoptimizer 死代码清除时的悬空 Constant，
             以及 ConvertToDAG 阶段才暴露的 Identity 等）。
          2. 物化活着的 Constant：凡带 value 张量属性的 Constant 节点，值整体
             搬进 initializers（C++ 加载器原生支持 int64/int32/fp32/fp16/int8），
             删节点。这样 runtime 端只需实现 Shape/Unsqueeze/Cast/Squeeze/
             ScatterND，不必实现 Constant/Identity。标量统一 np.atleast_1d 升到
             1 维——0 维初始器在 C++ 端 Tensor<T> 里 size_=0，数据会丢。

        剪枝只删 PRUNEABLE 节点：Constant/Identity/Shape/Unsqueeze/Squeeze/Cast/
        ScatterND 全是 runtime 不支持或不必要的元算子；真正有副作用的
        （Conv/MatMul/输出）一律不碰，避免误删数据依赖。
        """
        if not getattr(self, "_prune_materialize_done", False):
            self._prune_dead_and_materialize_constants_impl()
            self._prune_materialize_done = True

    def _prune_dead_and_materialize_constants_impl(self):
        # All vkop ops are pure (inputs -> outputs, no side effects / control
        # flow), so ANY node whose output can't reach a graph output or a live
        # initializer is dead and can be pruned — not just shape-meta ops.
        # (Previously only {Constant,Identity,Shape,Unsqueeze,Squeeze,Cast,
        # ScatterND} were pruned, which left dead Gather/Div/Mul/etc. orphaned
        # when their pruneable shape-meta producers got deleted — e.g. the
        # rotary half-boundary chain /Shape_2 -> /Gather_2 -> /Div -> /Cast ->
        # /Unsqueeze_8 becomes dead after fuse_rotary_embedding removes the
        # Slice consumers, but /Gather_2 and /Div were non-pruneable and stayed
        # as orphans referencing now-missing tensors -> runtime lookup crash.)
        nodes = list(self.nodes.values())
        tensor_producer = {}
        for n in nodes:
            for out in n.outputs:
                tensor_producer[out["name"]] = n

        # 1) 后向 liveness：graph outputs 及 initializer 名视为 live 种子。
        seed = set()
        seed.update(o["name"] for o in self.outputs)
        seed.update(self.initializers.keys())
        live_tensors = set()  # tensor name
        live_nodes = set()    # node name
        stack = list(seed)
        while stack:
            tname = stack.pop()
            if tname in live_tensors:
                continue
            live_tensors.add(tname)
            n = tensor_producer.get(tname)
            if n is None or n.name in live_nodes:
                continue
            live_nodes.add(n.name)
            for inp in n.inputs:
                stack.append(inp["name"])

        dead_nodes = []
        live_constants = []
        for n in nodes:
            if n.name in live_nodes:
                if n.op_type == "Constant":
                    live_constants.append(n)
                continue
            dead_nodes.append(n)

        # 2) 物化活着的 Constant：值搬进 initializers，删节点。
        #    ONNX Constant 节点可能用任意一种属性携带值：
        #      value (Tensor) / value_int / value_float / value_ints /
        #      value_floats / sparse_value。converter 只把 Tensor 属性读成
        #    ndarray；标量形式 (value_int=2 等) 是 INT/FLOAT/INTS 属性，被
        #    读成 Python int/float/list。这里把所有形式统一物化成至少 1-D
        #    的 ndarray initializer，否则 RMSNorm 的 Pow(x,2) 指数在运行时
        #    绑定到一个空 SSBO，GPU 读到 0，pow(x,0)=1 污染整层。
        materialized_count = 0
        for n in live_constants:
            name = n.outputs[0]["name"]
            value = n.attributes.get("value")
            if isinstance(value, np.ndarray):
                arr = np.atleast_1d(np.ascontiguousarray(value))
            elif "value_float" in n.attributes:
                arr = np.array([float(n.attributes["value_float"])],
                               dtype=np.float32)
            elif "value_floats" in n.attributes:
                arr = np.array(n.attributes["value_floats"], dtype=np.float32)
            elif "value_int" in n.attributes:
                # ONNX value_int is int64; runtime stores int initializers as
                # int32, so downcast here.
                arr = np.array([int(n.attributes["value_int"])], dtype=np.int64)
            elif "value_ints" in n.attributes:
                arr = np.array(n.attributes["value_ints"], dtype=np.int64)
            else:
                # value_string / sparse_value / no value: nothing to materialize.
                continue
            arr = np.atleast_1d(np.ascontiguousarray(arr))
            self.initializers[name] = numpy_helper.from_array(arr, name)
            self.nodes.pop(n.name, None)
            materialized_count += 1

        for n in dead_nodes:
            self.nodes.pop(n.name, None)

        if dead_nodes or live_constants:
            print(f"[prune+materialize] removed {len(dead_nodes)} dead nodes, "
                  f"materialized {materialized_count}/{len(live_constants)} "
                  f"Constant(s) as initializers")
        self.build_dependencies()

    def save_to_binary(self, file_path: str):
        """Save model to a FlatBuffers binary file.

        The initializer data is laid out as a single contiguous, 64-byte-aligned
        blob (the "compact memory" scheme the C++ runtime expects) with a side
        table of (name, dtype, dims, offset, size) entries. The Python writer
        computes the offsets once; the C++ reader memmaps the file and points
        straight at the blob — zero-copy, no two-pass scan.
        """
        # 写盘前先清理运行时图：剪死代码 + 物化 Constant（见上）。幂等。
        self.prune_dead_and_materialize_constants()

        builder = flatbuffers.Builder(1 << 16)

        # --- Build the 64-byte-aligned initializer blob + side table. ---
        # The unified sub-region (if --unify was used) is appended after the
        # regular initializers so its absolute base is unified_blob_offset;
        # UnifiedMeta offsets are relative to that base, kept relative here and
        # made absolute by the writer only for the on-disk side table.
        # 大 blob（LLM 权重数 GB）塞不进 FlatBuffer（2GB 上限），走外部数据：
        # blob 整体写到文件末尾，FlatBuffer 内不存字节。判断标准与 2GB 上限
        # 留安全余量——超过 ~800MB 就外置。
        #
        # 字节数用 dtype+dims 直接算，不调 to_array/tobytes——对 3.4GB 的 LLM
        # 权重，预扫描时把每个 initializer 解成 ndarray 再 tobytes 会瞬间分配
        # 数 GB 临时内存，叠加后面 blob 累积导致 OOM。
        _DTYPE_BYTES = {
            1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8, 9: 1, 10: 2,
            11: 8, 12: 4, 13: 8, 16: 2,
        }

        def _init_byte_len(arr) -> int:
            nbytes = _DTYPE_BYTES.get(arr.data_type, 0)
            n = 1
            for d in arr.dims:
                n *= int(d)
            return n * nbytes

        external = False
        try:
            total_blob = 0
            for _name, arr in self.initializers.items():
                total_blob += _init_byte_len(arr)
            if getattr(self, "_unified_bytes", b""):
                total_blob += len(self._unified_bytes)
            if total_blob > (800 << 20):
                external = True
        except Exception:
            external = False

        # external 模式：blob 直接流式写盘（见文件写入段），这里只建侧表 +
        # 记录 (offset, byte_len)，绝不把整个 blob 累积进内存。非 external
        # （小模型）仍走 bytearray 路径，blob 要塞进 FlatBuffer 向量。
        blob = bytearray() if not external else None
        cur_blob_len = 0  # running blob length (external: virtual offset)
        init_entries = []  # (name_off, dtype_off, dims_off, offset, size)
        # name -> (blob_offset, size) for later offset resolution (rgba meta).
        blob_offsets: Dict[str, tuple] = {}
        # external-only: ordered (name, arr) list to replay at write time.
        external_init_order = [] if external else None
        for name, arr in self.initializers.items():
            data_size = _init_byte_len(arr)
            aligned_offset = (cur_blob_len + _BLOB_ALIGNMENT - 1) & ~(_BLOB_ALIGNMENT - 1)
            pad = aligned_offset - cur_blob_len
            if external:
                external_init_order.append((name, arr, aligned_offset, data_size, pad))
            else:
                if pad:
                    blob.extend(b"\x00" * pad)
                arr_np = np.ascontiguousarray(numpy_helper.to_array(arr))
                blob.extend(arr_np.tobytes())
            cur_blob_len = aligned_offset + data_size
            blob_offsets[name] = (aligned_offset, data_size)

            dtype_str = _DATA_TYPE_MAP.get(arr.data_type, "UNDEFINED")
            dims = list(arr.dims)

            name_off = builder.CreateString(name)
            dtype_off = builder.CreateString(dtype_str)
            InitializerEntry.StartDimsVector(builder, len(dims))
            for d in reversed(dims):
                builder.PrependUint32(int(d))
            dims_off = builder.EndVector()

            InitializerEntry.Start(builder)
            InitializerEntry.AddName(builder, name_off)
            InitializerEntry.AddDtype(builder, dtype_off)
            InitializerEntry.AddDims(builder, dims_off)
            InitializerEntry.AddOffset(builder, aligned_offset)
            InitializerEntry.AddSize(builder, data_size)
            init_entries.append(InitializerEntry.End(builder))

        # --- Append the unified sub-region (if any) and record its base. ---
        unified_bytes = getattr(self, "_unified_bytes", b"")
        if unified_bytes:
            # Rewrite unified_meta offsets to be absolute into the full blob.
            unified_base = (cur_blob_len + _BLOB_ALIGNMENT - 1) & ~(_BLOB_ALIGNMENT - 1)
            unified_pad = unified_base - cur_blob_len
            if external:
                # deferred to stream write; just track length
                cur_blob_len = unified_base + len(unified_bytes)
            else:
                if unified_pad:
                    blob.extend(b"\x00" * unified_pad)
                blob.extend(unified_bytes)
                cur_blob_len = len(blob)
            self.unified_blob_offset = unified_base
            for m in self.unified_meta:
                m["offset"] = m["offset"] + unified_base
        else:
            self.unified_blob_offset = 0

        Model.StartInitializersVector(builder, len(init_entries))
        for off in reversed(init_entries):
            builder.PrependUOffsetTRelative(off)
        initializers_vec = builder.EndVector()

        if external:
            # 外部数据模式：blob 不写入 FlatBuffer 向量，直接以裸字节追加到
            # 文件末尾（见 save_to_binary）。blob 字段留空（0 offset），
            # 布局与加载器行为见 save_to_binary 注释。
            blob_vec = 0
        else:
            Model.StartInitializerBlobVector(builder, len(blob))
            # FlatBuffers writes vectors back-to-front; emit bytes in reverse.
            for b in reversed(blob):
                builder.PrependByte(b)
            blob_vec = builder.EndVector()

        # --- inputs / outputs (ShapeRef vectors) ---
        # ONNX elem_type int -> canonical dtype name. Per-node shapes in the
        # converter carry elem_type ints (the optimizer's cast-folding reads
        # them as ints), while graph inputs/outputs already carry string names.
        # The flatbuffer field is a string, so normalize both forms here.
        _ELEM_TYPE_NAME = {
            1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16",
            6: "int32", 7: "int64", 8: "string", 9: "bool", 10: "float16",
            11: "float64", 12: "uint32", 13: "uint64", 16: "bfloat16",
        }

        def _dtype_to_str(dtype):
            if isinstance(dtype, str):
                return dtype
            if isinstance(dtype, (int, np.integer)):
                return _ELEM_TYPE_NAME.get(int(dtype), "")
            return ""

        def build_shape_vec(shapes):
            offs = []
            for item in shapes:
                name = item["name"] if isinstance(item, dict) else str(item)
                dims = item["shape"] if isinstance(item, dict) else []
                dtype = item.get("dtype", "") if isinstance(item, dict) else ""
                dtype = _dtype_to_str(dtype)
                name_off = builder.CreateString(name)
                ShapeRef.StartDimsVector(builder, len(dims))
                for d in reversed(dims):
                    builder.PrependInt32(int(d))
                dims_off = builder.EndVector()
                dtype_off = builder.CreateString(dtype) if dtype else 0
                ShapeRef.Start(builder)
                ShapeRef.AddName(builder, name_off)
                ShapeRef.AddDims(builder, dims_off)
                if dtype_off:
                    ShapeRef.AddDtype(builder, dtype_off)
                offs.append(ShapeRef.End(builder))
            Model.StartInputsVector(builder, len(offs))  # any Start*Vector works
            for off in reversed(offs):
                builder.PrependUOffsetTRelative(off)
            return builder.EndVector()

        inputs_vec = build_shape_vec(self.inputs)
        outputs_vec = build_shape_vec(self.outputs)

        # --- nodes ---
        node_offs = []
        for node in self.nodes.values():
            op_type_off = builder.CreateString(node.op_type)
            name_off = builder.CreateString(node.name)

            attr_offs = []
            for key, value in node.attributes.items():
                key_off = builder.CreateString(key)
                # Pre-create all sub-objects (strings/vectors/tables) before
                # starting the Attribute table — FlatBuffers forbids building
                # nested objects while a table is open.
                sval_off = 0
                ints_off = 0
                floats_off = 0
                tval_off = 0
                attr_type = None
                # bool must be checked before int (bool is an int subclass).
                if isinstance(value, bool):
                    attr_type = AttrType.Bool
                elif isinstance(value, str):
                    attr_type = AttrType.String
                    sval_off = builder.CreateString(value)
                elif isinstance(value, int):
                    attr_type = AttrType.Int64
                elif isinstance(value, float):
                    attr_type = AttrType.Float32
                elif isinstance(value, list) and all(isinstance(v, int) for v in value):
                    attr_type = AttrType.Ints
                    Attribute.StartIntsVector(builder, len(value))
                    for v in reversed(value):
                        builder.PrependUint32(int(v))
                    ints_off = builder.EndVector()
                elif isinstance(value, list) and all(isinstance(v, float) for v in value):
                    attr_type = AttrType.Floats
                    Attribute.StartFloatsVector(builder, len(value))
                    for v in reversed(value):
                        builder.PrependFloat32(float(v))
                    floats_off = builder.EndVector()
                elif isinstance(value, np.ndarray):
                    attr_type = AttrType.Tensor
                    t_arr = numpy_helper.from_array(value, key)
                    t_dtype = builder.CreateString(
                        _DATA_TYPE_MAP.get(t_arr.data_type, "UNDEFINED"))
                    t_dims = list(t_arr.dims)
                    TensorData.StartDimsVector(builder, len(t_dims))
                    for d in reversed(t_dims):
                        builder.PrependUint32(int(d))
                    t_dims_off = builder.EndVector()
                    t_data = np.ascontiguousarray(numpy_helper.to_array(t_arr)).tobytes()
                    TensorData.StartDataVector(builder, len(t_data))
                    for b in reversed(t_data):
                        builder.PrependByte(b)
                    t_data_off = builder.EndVector()
                    TensorData.Start(builder)
                    TensorData.AddDtype(builder, t_dtype)
                    TensorData.AddDims(builder, t_dims_off)
                    TensorData.AddData(builder, t_data_off)
                    tval_off = TensorData.End(builder)
                else:
                    raise ValueError(f"Unsupported attribute type for: {key}")

                Attribute.Start(builder)
                Attribute.AddKey(builder, key_off)
                Attribute.AddType(builder, attr_type)
                if sval_off:
                    Attribute.AddSval(builder, sval_off)
                elif attr_type == AttrType.Int64:
                    Attribute.AddIval(builder, int(value))
                elif attr_type == AttrType.Float32:
                    Attribute.AddFval(builder, float(value))
                elif attr_type == AttrType.Bool:
                    Attribute.AddBval(builder, bool(value))
                if ints_off:
                    Attribute.AddInts(builder, ints_off)
                if floats_off:
                    Attribute.AddFloats(builder, floats_off)
                if tval_off:
                    Attribute.AddTval(builder, tval_off)
                attr_offs.append(Attribute.End(builder))
            NodeTable.StartAttributesVector(builder, len(attr_offs))
            for off in reversed(attr_offs):
                builder.PrependUOffsetTRelative(off)
            attrs_vec = builder.EndVector()

            node_inputs_vec = build_shape_vec(node.inputs)
            node_outputs_vec = build_shape_vec(node.outputs)

            def build_str_vec(strings):
                offs = [builder.CreateString(s) for s in strings]
                NodeTable.StartDependenciesVector(builder, len(offs))
                for off in reversed(offs):
                    builder.PrependUOffsetTRelative(off)
                return builder.EndVector()

            deps_vec = build_str_vec(list(node.dependencies))
            dependents_vec = build_str_vec(list(node.dependents))

            NodeTable.Start(builder)
            NodeTable.AddOpType(builder, op_type_off)
            NodeTable.AddName(builder, name_off)
            NodeTable.AddAttributes(builder, attrs_vec)
            NodeTable.AddInputs(builder, node_inputs_vec)
            NodeTable.AddOutputs(builder, node_outputs_vec)
            NodeTable.AddDependencies(builder, deps_vec)
            NodeTable.AddDependents(builder, dependents_vec)
            node_offs.append(NodeTable.End(builder))

        Model.StartNodesVector(builder, len(node_offs))
        for off in reversed(node_offs):
            builder.PrependUOffsetTRelative(off)
        nodes_vec = builder.EndVector()

        # --- unified / rgba metadata (struct vectors + name tables) ---
        unified_meta_vec = 0
        if self.unified_meta:
            Model.StartUnifiedMetaVector(builder, len(self.unified_meta))
            for m in reversed(self.unified_meta):
                dims = (m.get("dims", [0, 0, 0, 0]) + [0, 0, 0, 0])[:4]
                UnifiedMeta.CreateUnifiedMeta(
                    builder, int(m["dtype"]), int(m["name_len"]),
                    int(m["offset"]), int(m["size"]), [int(d) for d in dims])
            unified_meta_vec = builder.EndVector()
        unified_names_off = builder.CreateString(self.unified_names) if self.unified_names else 0

        name_to_blob_offset = {n: off for n, (off, _sz) in blob_offsets.items()}
        rgba_meta_vec = 0
        if self.rgba_meta:
            Model.StartRgbaMetaVector(builder, len(self.rgba_meta))
            for m in reversed(self.rgba_meta):
                dims = (m.get("dims", [0, 0, 0, 0]) + [0, 0, 0, 0])[:4]
                name = m.get("_name", "")
                off = name_to_blob_offset.get(name, m.get("offset", 0))
                RGBAConversionMeta.CreateRgbaconversionMeta(
                    builder, int(m["dtype"]), int(m["name_len"]),
                    int(off), int(m["size"]), [int(d) for d in dims])
            rgba_meta_vec = builder.EndVector()
        rgba_names_off = builder.CreateString(self.rgba_names) if self.rgba_names else 0

        # --- concurrent levels ---
        concurrent_levels = self.find_concurrent_nodes()
        level_offs = []
        for level in concurrent_levels:
            offs = [builder.CreateString(n) for n in level]
            ConcurrentLevel.StartNodesVector(builder, len(offs))
            for off in reversed(offs):
                builder.PrependUOffsetTRelative(off)
            nodes_in_level = builder.EndVector()
            ConcurrentLevel.Start(builder)
            ConcurrentLevel.AddNodes(builder, nodes_in_level)
            level_offs.append(ConcurrentLevel.End(builder))
        Model.StartConcurrentLevelsVector(builder, len(level_offs))
        for off in reversed(level_offs):
            builder.PrependUOffsetTRelative(off)
        concurrent_vec = builder.EndVector()

        # --- root Model table ---
        Model.Start(builder)
        Model.AddMagic(builder, 0x504F4B56)
        Model.AddVersion(builder, 2)
        Model.AddInputs(builder, inputs_vec)
        Model.AddOutputs(builder, outputs_vec)
        Model.AddNodes(builder, nodes_vec)
        Model.AddInitializers(builder, initializers_vec)
        Model.AddInitializerBlob(builder, blob_vec)
        Model.AddUnified(builder, self.unified)
        Model.AddRgba(builder, self.rgba)
        if unified_meta_vec:
            Model.AddUnifiedMeta(builder, unified_meta_vec)
        if unified_names_off:
            Model.AddUnifiedNames(builder, unified_names_off)
        if rgba_meta_vec:
            Model.AddRgbaMeta(builder, rgba_meta_vec)
        if rgba_names_off:
            Model.AddRgbaNames(builder, rgba_names_off)
        Model.AddUnifiedBlobOffset(builder, self.unified_blob_offset)
        Model.AddConcurrentLevels(builder, concurrent_vec)
        root = Model.End(builder)
        builder.Finish(root, file_identifier=b"VKOP")

        with open(file_path, "wb") as f:
            f.write(builder.Output())
            if external:
                # 外部数据：blob（含 unified 区域）以 64 字节对齐追加在 FlatBuffer
                # 之后；文件末尾 8 字节 LE uint64 记录 blob 起始偏移，以便 C++
                # 加载器按此定位。FlatBuffer 的 blob 字段此时为空，加载器逻辑不变。
                #
                # 流式写：逐个 initializer to_array→tobytes→write，写完即释放。
                # 旧实现把整个 blob（3.4GB）累积进一个 bytearray 再 bytes() 写，
                # 峰值 ~2×blob 叠加 proto 原始副本导致 OOM。
                align = _BLOB_ALIGNMENT
                pad = (align - (f.tell() % align)) % align
                if pad:
                    f.write(b"\x00" * pad)
                blob_offset = f.tell()
                written = 0
                for _name, arr, aligned_offset, data_size, ipad in external_init_order:
                    if ipad:
                        f.write(b"\x00" * ipad)
                    arr_np = np.ascontiguousarray(numpy_helper.to_array(arr))
                    f.write(arr_np.tobytes())
                    written = aligned_offset + data_size
                # unified sub-region (if any)
                unified_bytes = getattr(self, "_unified_bytes", b"")
                if unified_bytes:
                    unified_base = self.unified_blob_offset
                    upad = unified_base - written
                    if upad:
                        f.write(b"\x00" * upad)
                    f.write(unified_bytes)
                f.write(blob_offset.to_bytes(_EXTERNAL_BLOB_MAGIC, "little"))

