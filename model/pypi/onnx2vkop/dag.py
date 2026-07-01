"""DAG-based model representation."""

from collections import deque
from typing import Any, Dict, List, Set

import numpy as np
import flatbuffers
from onnx import numpy_helper

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

    def save_to_binary(self, file_path: str):
        """Save model to a FlatBuffers binary file.

        The initializer data is laid out as a single contiguous, 64-byte-aligned
        blob (the "compact memory" scheme the C++ runtime expects) with a side
        table of (name, dtype, dims, offset, size) entries. The Python writer
        computes the offsets once; the C++ reader memmaps the file and points
        straight at the blob — zero-copy, no two-pass scan.
        """
        builder = flatbuffers.Builder(1 << 16)

        # --- Build the 64-byte-aligned initializer blob + side table. ---
        # The unified sub-region (if --unify was used) is appended after the
        # regular initializers so its absolute base is unified_blob_offset;
        # UnifiedMeta offsets are relative to that base, kept relative here and
        # made absolute by the writer only for the on-disk side table.
        blob = bytearray()
        init_entries = []  # (name_off, dtype_off, dims_off, offset, size)
        # name -> (blob_offset, size) for later offset resolution (rgba meta).
        blob_offsets: Dict[str, tuple] = {}
        for name, arr in self.initializers.items():
            arr_np = np.ascontiguousarray(numpy_helper.to_array(arr))
            data = arr_np.tobytes()
            data_size = len(data)
            aligned_offset = (len(blob) + _BLOB_ALIGNMENT - 1) & ~(_BLOB_ALIGNMENT - 1)
            if aligned_offset > len(blob):
                blob.extend(b"\x00" * (aligned_offset - len(blob)))
            blob.extend(data)
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
            unified_base = (len(blob) + _BLOB_ALIGNMENT - 1) & ~(_BLOB_ALIGNMENT - 1)
            if unified_base > len(blob):
                blob.extend(b"\x00" * (unified_base - len(blob)))
            blob.extend(unified_bytes)
            self.unified_blob_offset = unified_base
            for m in self.unified_meta:
                m["offset"] = m["offset"] + unified_base
        else:
            self.unified_blob_offset = 0

        Model.StartInitializersVector(builder, len(init_entries))
        for off in reversed(init_entries):
            builder.PrependUOffsetTRelative(off)
        initializers_vec = builder.EndVector()

        Model.StartInitializerBlobVector(builder, len(blob))
        # FlatBuffers writes vectors back-to-front; emit bytes in reverse.
        for b in reversed(blob):
            builder.PrependByte(b)
        blob_vec = builder.EndVector()

        # --- inputs / outputs (ShapeRef vectors) ---
        def build_shape_vec(shapes):
            offs = []
            for item in shapes:
                name = item["name"] if isinstance(item, dict) else str(item)
                dims = item["shape"] if isinstance(item, dict) else []
                name_off = builder.CreateString(name)
                ShapeRef.StartDimsVector(builder, len(dims))
                for d in reversed(dims):
                    builder.PrependUint32(int(d))
                dims_off = builder.EndVector()
                ShapeRef.Start(builder)
                ShapeRef.AddName(builder, name_off)
                ShapeRef.AddDims(builder, dims_off)
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
        Model.AddVersion(builder, 1)
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

