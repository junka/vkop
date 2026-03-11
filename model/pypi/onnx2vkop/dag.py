"""DAG-based model representation."""

import struct
from collections import deque
from typing import Any, Dict, List, Set

import numpy as np
from onnx import numpy_helper


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
        """Save model to binary file with DAG information."""
        with open(file_path, "wb") as f:
            self._write_list_with_shapes(f, self.inputs)
            self._write_list_with_shapes(f, self.outputs)

            f.write(struct.pack("I", len(self.nodes)))
            for node_name, node in self.nodes.items():
                self._write_string(f, node.op_type)
                self._write_string(f, node.name)
                self._write_dict(f, node.attributes)
                self._write_list_with_shapes(f, node.inputs)
                self._write_list_with_shapes(f, node.outputs)

                f.write(struct.pack("I", len(node.dependencies)))
                for dep_name in node.dependencies:
                    self._write_string(f, dep_name)

                f.write(struct.pack("I", len(node.dependents)))
                for dep_name in node.dependents:
                    self._write_string(f, dep_name)

            f.write(struct.pack("I", len(self.initializers)))
            for name, arr in self.initializers.items():
                self._write_string(f, name)
                self._write_array(f, arr)

            concurrent_levels = self.find_concurrent_nodes()
            f.write(struct.pack("I", len(concurrent_levels)))
            for level in concurrent_levels:
                f.write(struct.pack("I", len(level)))
                for node_name in level:
                    self._write_string(f, node_name)

    @staticmethod
    def _write_string(f, s: str):
        b = s.encode("utf-8")
        f.write(struct.pack("I", len(b)))
        f.write(b)

    @staticmethod
    def _write_list_with_shapes(f, lst):
        f.write(struct.pack("I", len(lst)))
        for item in lst:
            if isinstance(item, dict) and "name" in item and "shape" in item:
                DAGBasedModel._write_string(f, item["name"])
                DAGBasedModel._write_list(f, item["shape"])
            else:
                DAGBasedModel._write_string(f, str(item))

    @staticmethod
    def _write_list(f, lst):
        f.write(struct.pack("I", len(lst)))
        for item in lst:
            if isinstance(item, int):
                f.write(struct.pack("I", item))
            elif isinstance(item, float):
                f.write(struct.pack("f", item))
            else:
                DAGBasedModel._write_string(f, str(item))

    @staticmethod
    def _write_dict(f, d: dict):
        f.write(struct.pack("I", len(d)))
        for key, value in d.items():
            DAGBasedModel._write_string(f, key)
            if isinstance(value, str):
                f.write(b"\x00")
                DAGBasedModel._write_string(f, value)
            elif isinstance(value, int):
                f.write(b"\x01")
                f.write(struct.pack("q", value))
            elif isinstance(value, float):
                f.write(b"\x02")
                f.write(struct.pack("f", value))
            elif isinstance(value, list):
                if all(isinstance(v, int) for v in value):
                    f.write(b"\x03")
                    DAGBasedModel._write_list(f, value)
                elif all(isinstance(v, float) for v in value):
                    f.write(b"\x04")
                    DAGBasedModel._write_list(f, value)
                else:
                    raise ValueError(f"Unsupported list type in attribute: {key}")
            elif isinstance(value, np.ndarray):
                f.write(b"\x05")
                DAGBasedModel._write_array(f, numpy_helper.from_array(value, key))

    @staticmethod
    def _write_array(f, arr):
        data_type_map = {
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
        data_type = data_type_map.get(arr.data_type, "UNDEFINED")
        DAGBasedModel._write_string(f, str(data_type))
        DAGBasedModel._write_list(f, list(arr.dims))

        arr_np = np.ascontiguousarray(numpy_helper.to_array(arr))
        data = arr_np.tobytes()
        f.write(struct.pack("Q", len(data)))
        f.write(data)
