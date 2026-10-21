"""DAG-based model representation."""

import os
from collections import deque
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import flatbuffers
from onnx import numpy_helper

from . import weight_stream

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
    KernelPlan,
    Model,
    Node as NodeTable,
    RGBAConversionMeta,
    ShapeRef,
    TensorData,
    UnifiedMeta,
)
from .generated.vkop.model.AttrType import AttrType
from . import shader_map

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
    # ONNX's fp8 element types. vkop writes them only for a weight-only
    # quantized MatMul payload: one byte per value, decoded to float by the
    # buffer kernel (see f8_val in shaders/buffer/matmul.comp).
    17: "float8e4m3fn",
    19: "float8e5m2",
    # ONNX's 4-bit element types. vkop writes them only for a weight-only
    # quantized payload; the C++ loader has a kernel for "int4", for "nf4"
    # (which is a UINT4 payload whose nibble indexes the NF4 codebook — the name
    # comes from the tensor's vkop_dtype metadata, see _init_dtype_name), and for
    # "float4e2m1fn" (NVFP4's E2M1 nibble, scaled per 16-value block by a
    # float8e4m3fn table plus one fp32 factor per tensor).
    21: "uint4",
    22: "int4",
    23: "float4e2m1fn",
}

# Sub-byte element types: two values per byte, the even element in the low
# nibble, in the array's row-major order. ONNX's raw_data for these is already
# that packed byte string, while a numpy 4-bit array's tobytes() is one byte per
# value — so a packed initializer's blob bytes are its raw_data verbatim, and
# its size comes from the dims and the bit width, not from either dtype's
# itemsize (see _init_byte_len / _init_bytes).
_PACKED_DATA_TYPES = frozenset((21, 22, 23))
_PACKED_BITS = 4

# Element types numpy cannot round-trip: onnx's numpy_helper decodes the fp8 ones
# only if ml_dtypes is installed, which the converter must not depend on. Their
# raw_data is already the byte string the blob wants, so they go the same
# pass-through route as the packed types and only get length-checked.
_RAW_ONLY_DATA_TYPES = _PACKED_DATA_TYPES | frozenset((17, 19))

# The extension channel a quantizer uses to name a payload whose ONNX element
# type is only its storage (nf4 rides on UINT4).
_VKOP_DTYPE_META_KEY = "vkop_dtype"

# The extension channel a *graph* uses to declare a property the converter
# cannot check by looking at the graph. Today that is `vkop_causal_mask:<tensor>`
# = 1: the tensor rides in as an additive attention bias whose untouched
# entries are exactly the standard causal mask (0 at j <= kv - q + i, fp16 min
# above). Only whoever *builds* the bias knows that — in this model it is a
# graph input filled by the driver (llm_chat's causal_bias) — so the fact has
# to travel with the model. It reaches the Attention op as `mask_causal` and
# lets the kernel stop a causal row's KV sweep at its diagonal (attention.comp
# t_end), which halves a causal prefill. A model that says nothing gets the
# full sweep, so the declaration can only add speed, never change a result.
_VKOP_CAUSAL_MASK_META_PREFIX = "vkop_causal_mask:"

# 字节数用 dtype+dims 直接算，不调 to_array/tobytes——对 3.4GB 的 LLM
# 权重，预扫描时把每个 initializer 解成 ndarray 再 tobytes 会瞬间分配
# 数 GB 临时内存，叠加后面 blob 累积导致 OOM。
_DTYPE_BYTES = {
    1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8, 9: 1, 10: 2,
    11: 8, 12: 4, 13: 8, 16: 2, 17: 1, 19: 1,
}


def _init_byte_len(arr) -> int:
    """The bytes this initializer occupies in the blob.

    Computed from the dtype and dims instead of by decoding the payload: a
    prescan that ran to_array() + tobytes() on every initializer would allocate
    several GB of temporaries for a 3.4GB LLM weight set, on top of the blob it
    is only measuring."""
    n = 1
    for d in arr.dims:
        n *= int(d)
    if arr.data_type in _PACKED_DATA_TYPES:
        return (n * _PACKED_BITS + 7) // 8
    nbytes = _DTYPE_BYTES.get(arr.data_type, 0)
    return n * nbytes


def _init_bytes(arr) -> bytes:
    """The initializer's payload exactly as the blob records it. A packed
    sub-byte tensor's raw_data IS the packed nibbles, and re-serializing it
    through numpy would write one byte per value instead; fp8 lands here too
    because numpy can only decode it through an optional dependency. Everything
    else goes through the ndarray so a tensor stored in int64_data/int32_data
    lands correctly."""
    if arr.data_type in _RAW_ONLY_DATA_TYPES:
        data = arr.raw_data
        expected = _init_byte_len(arr)
        if len(data) != expected:
            raise ValueError(
                f"raw-only initializer '{arr.name}' carries {len(data)} bytes, "
                f"but its dims declare {expected}"
            )
        return data
    return np.ascontiguousarray(numpy_helper.to_array(arr)).tobytes()


def _init_dtype_name(arr) -> str:
    """The dtype string the C++ loader reads: vkop's own name when the tensor
    carries one (nf4), else the ONNX element type's spelling."""
    for entry in arr.metadata_props:
        if entry.key == _VKOP_DTYPE_META_KEY:
            return entry.value
    return _DATA_TYPE_MAP.get(arr.data_type, "UNDEFINED")


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
        # Weights whose payload never entered RAM (see weight_stream): name ->
        # {location, offset, length, data_type, dims}. The writer copies them
        # from the source ONNX data file straight into the external trailer.
        self.deferred_weights: Dict[str, Any] = {}
        self.weight_base_dir: str = ""
        # Tensor names the model declared (model-level metadata_props) to be the
        # standard causal additive mask. Read by the attention matcher; empty
        # for a model that declares nothing.
        self.causal_mask_inputs: set = set()
        # Graph outputs to keep only at their last position (VKOP_LOGITS_LAST_ROW
        # at conversion time). "*" means every graph output a MatMul produces.
        # Empty — the default — leaves the graph exactly as exported.
        self.logits_last_row: set = set()

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
            for dependent_name in sorted(self.nodes[current].dependents):
                in_degree[dependent_name] -= 1
                if in_degree[dependent_name] == 0:
                    queue.append(dependent_name)

        if len(sorted_nodes) != len(self.nodes):
            raise ValueError("Cycle detected in computation graph!")

        return [self.nodes[name] for name in sorted_nodes]

    def dangling_inputs(self) -> List[Tuple[str, str]]:
        """Return (node name, input name) pairs whose producer is gone.

        A fusion pass may delete or rename a producer and forget to rebind a
        consumer; the name then resolves to nothing in the manifest, and the
        runtime binds that input to an empty buffer -- the kernel reads zeros
        instead of failing. Terminal pruning drops unreferenced things, but a
        consumer pointing at a missing producer is the other half of that
        problem and nothing checked it, so this is the audit the fusion stage
        runs against itself.
        """
        produced = {inp["name"] for inp in self.inputs}
        produced |= set(self.initializers)
        for node in self.nodes.values():
            for out in node.outputs:
                produced.add(out["name"])
        return [
            (node.name, name)
            for node in self.nodes.values()
            for name in (inp["name"] for inp in node.inputs)
            if name and name not in produced
        ]

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

                for dependent_name in sorted(self.nodes[node_name].dependents):
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

    def terminal_liveness_prune(self) -> None:
        """Delete nodes nothing consumes, then initializers no LIVE node reads.

        The runtime uploads every InitializerEntry it finds, one staging copy
        plus one submit/fence-wait each -- measured ~85 us per dead entry, ~190
        us per suppressed wait -- so an entry or a node nothing reads is pure
        load-time cost. prune_dead... runs once, BEFORE the fusion passes, and
        those passes rebind or delete consumers without touching either
        self.nodes or self.initializers, so whatever they orphan stays in the
        file (llm.vkopbin shipped 2134 unreferenced initializers, 19,324 bytes
        of folded shape-chain constants). Writing is the one point where
        nothing can orphan anything afterwards, so this sweep runs there rather
        than as another optimizer pass -- and it has to be a node sweep too,
        because an initializer read only by a dead node is as dead as the node.

        Unified / RGBA sub-allocation members are kept even when unreferenced:
        their bytes already live in the shared region, dropping one would mean
        repacking every offset in it, and they cost a BufferView each (0.18 us),
        not an upload.
        """
        producer = {}
        for node in self.nodes.values():
            for out in node.outputs:
                producer[out["name"]] = node
        live_nodes = set()
        stack = [o["name"] for o in self.outputs]
        while stack:
            n = producer.get(stack.pop())
            if n is None or n.name in live_nodes:
                continue
            live_nodes.add(n.name)
            stack.extend(inp["name"] for inp in n.inputs)
        dead_nodes = [name for name in self.nodes if name not in live_nodes]
        for name in dead_nodes:
            del self.nodes[name]

        live = {o["name"] for o in self.outputs}
        live.update(i["name"] for i in self.inputs)
        for m in self.unified_meta:
            live.add(m["_name"])
        for m in self.rgba_meta:
            live.add(m.get("_name", ""))
        for name in live_nodes:
            live.update(inp["name"] for inp in self.nodes[name].inputs)
        dead_inits = [name for name in self.initializers if name not in live]
        freed = sum(_init_byte_len(self.initializers[name])
                    for name in dead_inits)
        for name in dead_inits:
            del self.initializers[name]
        if dead_nodes or dead_inits:
            print(f"[terminal-prune] dropped {len(dead_nodes)} dead node(s), "
                  f"{len(dead_inits)} unreferenced initializer(s) "
                  f"({freed} bytes)")
        if dead_nodes:
            self.build_dependencies()

    def save_to_binary(self, file_path: str, shader_archive: bytes = None):
        """Save model to a FlatBuffers binary file.

        The initializer data is laid out as a single contiguous, 64-byte-aligned
        blob (the "compact memory" scheme the C++ runtime expects) with a side
        table of (name, dtype, dims, offset, size) entries. The Python writer
        computes the offsets once; the C++ reader memmaps the file and points
        straight at the blob — zero-copy, no two-pass scan.

        `shader_archive`, when given, is a VKSP archive (tools/filter_shaders.py
        serialize()) stored in the Model's `shader_archive` field. The runtime
        then takes every op's SPIR-V from the .vkopbin itself, so the model is
        a single deployable file with nothing to configure at run time.
        onnx2vkop always passes one; None is the writer-level "bare model" case,
        which no libvkop can run — libvkop carries no built-in SPIR-V, so its
        shaders can only come from a model that has them.

        The kernel selection that archive was filtered by is written into the
        same file (Model.kernel_plans, indexed by Node.plan), so the runtime
        reads the converter's answer instead of re-deriving it.
        """
        # 写盘前先清理运行时图：剪死代码 + 物化 Constant（见上）。幂等。
        self.prune_dead_and_materialize_constants()
        # 终局 liveness：所有 pass 跑完后，先删没有任何消费者指向的节点，再删
        # 没有活节点引用的初始值，两者都不写进产物（见 terminal_liveness_prune）。
        self.terminal_liveness_prune()

        # 最后一道完整性检查：融合 pass 删了生产者却忘了改绑消费者时，那个输入名
        # 在产物里谁都不指向，runtime 会把它绑到空 buffer 读出 0（静默错值而不是
        # 报错）。写盘点是唯一能看到全部 pass 结果的地方，所以在这里查一次。
        stranded = self.dangling_inputs()
        if stranded:
            print(f"[integrity] {len(stranded)} dangling input(s), runtime reads "
                  f"zeros for them: "
                  + ", ".join(f"{n} <- {i}" for n, i in stranded[:5]))

        # Same place to hold the streaming contract to account: a weight whose
        # bytes were left in the source file must still be the tensor we
        # promised to copy. See weight_stream.verify_deferred.
        weight_stream.verify_deferred(self.deferred_weights, self)

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
        # 字节数用 dtype+dims 直接算，不调 to_array/tobytes——见 _init_byte_len。

        # Members of the unified sub-region: their bytes live in _unified_bytes
        # rather than the regular blob, but they still need an InitializerEntry
        # row -- the runtime looks each one up by name for dtype/dims before
        # sub-allocating it (core/runtime.cpp:419).
        unified_names = {m["_name"] for m in self.unified_meta}

        external = False
        try:
            total_blob = 0
            for _name, arr in self.initializers.items():
                if _name in unified_names:
                    continue
                total_blob += _init_byte_len(arr)
            if getattr(self, "_unified_bytes", b""):
                total_blob += len(self._unified_bytes)
            # VKOP_EXTERNAL_BLOB_MB (default 800) is the switch point; a
            # smaller value forces external mode on models that would
            # otherwise inline (useful for testing the trailer path without
            # a multi-GB file). onnx2vkop --external-blob-mb writes this env.
            threshold_mb = 800
            _e = os.environ.get("VKOP_EXTERNAL_BLOB_MB")
            if _e and _e.isdigit():
                threshold_mb = int(_e)
            if total_blob > (threshold_mb << 20):
                external = True
        except Exception:
            external = False

        # A deferred weight has no bytes in this process to put in a FlatBuffer
        # vector, so the inline layout cannot carry one. The threshold that
        # selects the layout is a CLI knob, so say what to change instead of
        # letting the trailer writer meet a tensor with no payload.
        if self.deferred_weights and not external:
            raise weight_stream.DeferredWeightError(
                f"{len(self.deferred_weights)} weight(s) were left in their "
                f"external file, but the blob ({total_blob >> 20} MiB) fits a "
                f"FlatBuffer vector (threshold {threshold_mb} MiB) and that "
                f"path needs the bytes in memory. Reconvert with "
                f"--external-blob-mb 1 to force the trailer, or "
                f"VKOP_STREAM_WEIGHTS=0 to hydrate everything.")

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

        def add_entry(name, arr, offset, size):
            """One InitializerEntry side-table row for `name` at blob `offset`."""
            name_off = builder.CreateString(name)
            dtype_off = builder.CreateString(_init_dtype_name(arr))
            dims = list(arr.dims)
            InitializerEntry.StartDimsVector(builder, len(dims))
            for d in reversed(dims):
                builder.PrependUint32(int(d))
            dims_off = builder.EndVector()

            InitializerEntry.Start(builder)
            InitializerEntry.AddName(builder, name_off)
            InitializerEntry.AddDtype(builder, dtype_off)
            InitializerEntry.AddDims(builder, dims_off)
            InitializerEntry.AddOffset(builder, offset)
            InitializerEntry.AddSize(builder, size)
            init_entries.append(InitializerEntry.End(builder))

        for name, arr in self.initializers.items():
            if name in unified_names:
                continue
            data_size = _init_byte_len(arr)
            aligned_offset = (cur_blob_len + _BLOB_ALIGNMENT - 1) & ~(_BLOB_ALIGNMENT - 1)
            pad = aligned_offset - cur_blob_len
            if external:
                external_init_order.append((name, arr, aligned_offset, data_size, pad))
            else:
                if pad:
                    blob.extend(b"\x00" * pad)
                blob.extend(_init_bytes(arr))
            cur_blob_len = aligned_offset + data_size
            blob_offsets[name] = (aligned_offset, data_size)
            add_entry(name, arr, aligned_offset, data_size)

        # --- Append the unified sub-region (if any) and record its base. ---
        unified_bytes = getattr(self, "_unified_bytes", b"")
        if unified_bytes:
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
            # UnifiedMeta offsets stay relative to the region: LoadModel bases the
            # region's source pointer at unified_blob_offset and then indexes the
            # shared buffer by meta.offset() (core/runtime.cpp:384-394), and a
            # relative offset is the only thing that fits UnifiedMeta's int32 for
            # a multi-GB blob. The side-table rows keep the schema's rule of
            # absolute-into-blob, so add the base back for those.
            for m in self.unified_meta:
                name = m["_name"]
                abs_offset = m["offset"] + unified_base
                blob_offsets[name] = (abs_offset, m["size"])
                add_entry(name, self.initializers[name], abs_offset, m["size"])
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

        # --- embedded SPIR-V archive (VKSP) ---
        # 单文件部署：C++ 的 Runtime::LoadModel 只从这里取 shader（默认构建
        # 的 libvkop 只剩桩，运行期没有任何环境变量能替换），字节数很小（只含
        # 本图能 dispatch 的 key，llm 旋转形态 ~933 KB），所以直接放进 FlatBuffer，
        # 不走 initializer blob 那套外置 trailer——那套是为几 GB 权重准备的，而
        # 2 GiB 的向量上限对归档来说完全够用。
        if shader_archive:
            shader_archive_vec = builder.CreateByteVector(bytes(shader_archive))
        else:
            shader_archive_vec = 0

        # --- inputs / outputs (ShapeRef vectors) ---
        # ONNX elem_type int -> canonical dtype name. Per-node shapes in the
        # converter carry elem_type ints (the optimizer's cast-folding reads
        # them as ints), while graph inputs/outputs already carry string names.
        # The flatbuffer field is a string, so normalize both forms here.
        _ELEM_TYPE_NAME = {
            1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16",
            6: "int32", 7: "int64", 8: "string", 9: "bool", 10: "float16",
            11: "float64", 12: "uint32", 13: "uint64", 16: "bfloat16",
            # Quantized captions: a KV cache rewritten to a low-bit format
            # carries its per-node dtype here (the same spellings the graph
            # I/O and initializers use). Missing, these came out as "" and the
            # runtime fell back to "historical fp16" for the Q/DQ chain that
            # feeds the cache.
            17: "float8e4m3fn", 19: "float8e5m2",
            21: "uint4", 22: "int4", 23: "float4e2m1fn",
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
                # .get(), not []: a tensor reference may legitimately carry no
                # dims (the parser writes "shape": [] for an input tensor it
                # cannot find), and dtype/value_dynamic above are already
                # read tolerantly. A hard subscript here turned one name-only
                # reference into an unsavable model.
                dims = (item.get("shape") or []) if isinstance(item, dict) else []
                dtype = item.get("dtype", "") if isinstance(item, dict) else ""
                value_dynamic = bool(item.get("value_dynamic", False)) if isinstance(item, dict) else False
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
                if value_dynamic:
                    ShapeRef.AddValueDynamic(builder, True)
                offs.append(ShapeRef.End(builder))
            Model.StartInputsVector(builder, len(offs))  # any Start*Vector works
            for off in reversed(offs):
                builder.PrependUOffsetTRelative(off)
            return builder.EndVector()

        inputs_vec = build_shape_vec(self.inputs)
        outputs_vec = build_shape_vec(self.outputs)

        # --- kernel plan (the converter's kernel selection, frozen in-file) ---
        # One row per distinct op_type that has a GPU shader, referenced by
        # Node.plan. shader_map.kernel_plans_for_model is the same table the
        # embedded archive is filtered by (cli.build_shader_subset), so the
        # runtime's lookup key and the bytes shipped for it come from one
        # decision instead of two. Rows are emitted in first-appearance order
        # so reconverting an unchanged graph yields an identical table.
        plan_rows, plan_index = shader_map.kernel_plans_for_model(
            [n.op_type for n in self.nodes.values()])
        plan_offs = []
        for row in plan_rows:
            op_type_off = builder.CreateString(row["op_type"])
            image_off = builder.CreateString(row["image"]) if row["image"] else 0
            image16_off = (builder.CreateString(row["image_fp16"])
                           if row["image_fp16"] else 0)
            buffer_off = builder.CreateString(row["buffer"]) if row["buffer"] else 0
            buffer16_off = (builder.CreateString(row["buffer_fp16"])
                            if row["buffer_fp16"] else 0)
            var_offs = [builder.CreateString(v) for v in row["variants"]]
            KernelPlan.StartVariantsVector(builder, len(var_offs))
            for off in reversed(var_offs):
                builder.PrependUOffsetTRelative(off)
            variants_vec = builder.EndVector()
            KernelPlan.Start(builder)
            KernelPlan.AddOpType(builder, op_type_off)
            if image_off:
                KernelPlan.AddImage(builder, image_off)
            if image16_off:
                KernelPlan.AddImageFp16(builder, image16_off)
            if buffer_off:
                KernelPlan.AddBuffer(builder, buffer_off)
            if buffer16_off:
                KernelPlan.AddBufferFp16(builder, buffer16_off)
            KernelPlan.AddVariants(builder, variants_vec)
            plan_offs.append(KernelPlan.End(builder))
        if plan_offs:
            Model.StartKernelPlansVector(builder, len(plan_offs))
            for off in reversed(plan_offs):
                builder.PrependUOffsetTRelative(off)
            kernel_plans_vec = builder.EndVector()
        else:
            kernel_plans_vec = 0

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
                        # Schema is [uint]; store signed ints as two's
                        # complement — the C++ loader (attrValueToString)
                        # reinterprets back to int32 (e.g. axes=[-1]).
                        builder.PrependUint32(int(v) & 0xFFFFFFFF)
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

            deps_vec = build_str_vec(sorted(node.dependencies))
            dependents_vec = build_str_vec(sorted(node.dependents))

            NodeTable.Start(builder)
            NodeTable.AddOpType(builder, op_type_off)
            NodeTable.AddName(builder, name_off)
            NodeTable.AddAttributes(builder, attrs_vec)
            NodeTable.AddInputs(builder, node_inputs_vec)
            NodeTable.AddOutputs(builder, node_outputs_vec)
            NodeTable.AddDependencies(builder, deps_vec)
            NodeTable.AddDependents(builder, dependents_vec)
            # -1 (the field default, and what a planless op_type gets) keeps
            # the runtime on its built-in key table.
            plan_idx = plan_index.get(node.op_type, -1)
            if plan_idx >= 0:
                NodeTable.AddPlan(builder, plan_idx)
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
        # VKOP_MERGE_LEVELS=N merges every N consecutive Kahn levels into a
        # single chunk level, reducing vkQueueSubmit calls from L to ceil(L/N).
        # N=1: no change (original behavior, 815 submits).
        # N>1: chunk merge. Each chunk is one submit; timeline waits *across*
        # different submits are honored by the driver (unlike the broken-in-
        # Apple single-submit multi-VkSubmitInfo case). Intra-chunk op-to-op
        # timeline waits *inside* one command buffer ARE honored by Apple's
        # driver. Empirically, N=30 collapses 815→27 submits, cutting the
        # 198ms host-side submit floor to ~7ms. Default 1 (OFF).
        import os as _os
        raw_levels = self.find_concurrent_nodes()
        merge_n = int(_os.environ.get("VKOP_MERGE_LEVELS", "1"))
        if merge_n <= 1:
            concurrent_levels = raw_levels
        else:
            concurrent_levels = []
            chunk = []
            for lv in raw_levels:
                chunk.extend(lv)
                if len(chunk) >= merge_n * 2:   # *2 heuristic: keep chunks ~balanced
                    concurrent_levels.append(chunk)
                    chunk = []
            if chunk:
                concurrent_levels.append(chunk)
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
        Model.AddVersion(builder, 3)
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
        if shader_archive_vec:
            Model.AddShaderArchive(builder, shader_archive_vec)
        if kernel_plans_vec:
            Model.AddKernelPlans(builder, kernel_plans_vec)
        root = Model.End(builder)
        builder.Finish(root, file_identifier=b"VKOP")

        with open(file_path, "wb") as f:
            f.write(builder.Output())
            if external:
                # 外部数据：blob（含 unified 区域）以 64 字节对齐追加在 FlatBuffer
                # 之后；文件末尾 8 字节 LE uint64 记录 blob 起始偏移，以便 C++
                # 加载器按此定位。FlatBuffer 的 blob 字段此时为空，加载器逻辑不变。
                #
                # 流式写：逐个 initializer 取 payload 字节（_init_bytes）→write，
                # 写完即释放。旧实现把整个 blob（3.4GB）累积进一个 bytearray 再
                # bytes() 写，峰值 ~2×blob 叠加 proto 原始副本导致 OOM。
                align = _BLOB_ALIGNMENT
                pad = (align - (f.tell() % align)) % align
                if pad:
                    f.write(b"\x00" * pad)
                blob_offset = f.tell()
                written = 0
                for _name, arr, aligned_offset, data_size, ipad in external_init_order:
                    if ipad:
                        f.write(b"\x00" * ipad)
                    entry = self.deferred_weights.get(_name)
                    if entry is not None:
                        # Never held in RAM: copy it from the source file, at
                        # the offset the side table already promised.
                        if entry["length"] != data_size:
                            raise weight_stream.DeferredWeightError(
                                f"deferred weight '{_name}': side table says "
                                f"{data_size} bytes, the source says "
                                f"{entry['length']}")
                        weight_stream.write_payload(entry, f, self.weight_base_dir)
                    else:
                        f.write(_init_bytes(arr))
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

