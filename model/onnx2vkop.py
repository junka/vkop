#!/bin/env python3

import os
import sys
import struct
import onnx
import numpy as np
from onnx import numpy_helper
import onnxoptimizer as optimizer
from collections import defaultdict, deque


class VkModel:
    def __init__(self):
        """
        Initializes the class with the following attributes:

        Attributes:
            inputs (list): A list to store the input nodes of the graph.
            outputs (list): A list to store the output nodes of the graph.
            nodes (list): A list to store all the nodes in the graph.
            initializers (dict): A dictionary to store the initial values of the graph's parameters.
        """
        self.inputs = []
        self.outputs = []
        self.nodes = []
        self.initializers = {}

    def save_to_binary(self, file_path):
        with open(file_path, 'wb') as f:
            # Save inputs with shapes
            self._write_list_with_shapes(f, self.inputs)

            # Save outputs with shapes
            self._write_list_with_shapes(f, self.outputs)

            # Save nodes including attributes and input/output shapes
            f.write(struct.pack('I', len(self.nodes)))
            for node in self.nodes:
                self._write_string(f, node['op_type'])
                self._write_string(f, node['name'])
                self._write_dict(f, node['attributes'])  # Attributes
                self._write_list_with_shapes(f, node['inputs'])
                self._write_list_with_shapes(f, node['outputs'])

            # Save initializers (name -> numpy array)
            f.write(struct.pack('I', len(self.initializers)))
            for name, arr in self.initializers.items():
                self._write_string(f, name)
                self._write_array(f, arr)

    @staticmethod
    def _write_string(f, s):
        b = s.encode('utf-8')
        f.write(struct.pack('I', len(b)))
        f.write(b)

    @staticmethod
    def _write_list_with_shapes(f, lst):
        f.write(struct.pack('I', len(lst)))
        for item in lst:
            VkModel._write_string(f, item['name'])
            VkModel._write_list(f, item['shape'])

    @staticmethod
    def _write_list(f, lst):
        f.write(struct.pack('I', len(lst)))
        for item in lst:
            if isinstance(item, int):
                f.write(struct.pack('I', item))  # For dimensions
            elif isinstance(item, float):
                f.write(struct.pack('d', item))  # For float attributes
            else:
                VkModel._write_string(f, item)  # For names

    @staticmethod
    def _write_dict(f, d):
        f.write(struct.pack('I', len(d)))
        for key, value in d.items():
            VkModel._write_string(f, key)
            if isinstance(value, str):
                f.write(b'\x00')  # Tag for string
                VkModel._write_string(f, value)
            elif isinstance(value, int):
                f.write(b'\x01')  # Tag for int
                f.write(struct.pack('q', value))
            elif isinstance(value, float):
                f.write(b'\x02')  # Tag for float
                f.write(struct.pack('d', value))
            elif isinstance(value, list):
                if all(isinstance(v, int) for v in value):
                    f.write(b'\x03')  # Tag for list of ints
                    VkModel._write_list(f, value)
                elif all(isinstance(v, float) for v in value):
                    f.write(b'\x04')  # Tag for list of floats
                    VkModel._write_list(f, value)
                else:
                    raise ValueError(f"Unsupported list type in attribute: {key}")

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
            16: "bfloat16"
        }
        data_type = data_type_map.get(arr.data_type, onnx.TensorProto.UNDEFINED)
        VkModel._write_string(f, data_type)
        VkModel._write_list(f, list(arr.dims))
        total_elements = np.prod(arr.dims)
        print("Array shape:", arr.dims, "Data type:", data_type, "Total elements:", total_elements)
        arr = np.ascontiguousarray(numpy_helper.to_array(arr))
        data = arr.tobytes()
        f.write(struct.pack('Q', len(data)))
        f.write(data)


def optimize_onnx_model(onnx_model):
    """
    Optimize the ONNX model using ONNX's built-in optimizer.
    """
    passes = [
        "eliminate_deadend",  # Remove unused nodes
        "eliminate_identity",  # Remove identity nodes
        "eliminate_shape_op",  # Remove shape nodes
        "eliminate_nop_dropout",  # Remove no-op dropout
        "eliminate_nop_monotone_argmax",  # Remove no-op argmax
        "eliminate_nop_pad",  # Remove no-op padding
        "eliminate_nop_transpose",  # Remove no-op transpose
        "eliminate_unused_initializer",  # Remove unused initializers
        "fuse_consecutive_squeezes",  # Fuse consecutive squeeze operations
        "fuse_consecutive_unsqueezes",  # Fuse consecutive unsqueeze operations
        "fuse_consecutive_transposes",  # Fuse consecutive transpose operations
        "fuse_add_bias_into_conv",  # Fuse add bias into convolution
        "fuse_bn_into_conv",  # Fuse batch normalization into convolution
    ]
    optimized_model = optimizer.optimize(onnx_model, passes)
    return optimized_model


def is_topologically_sortable(graph):
    nodes = list(graph.node)
    n = len(nodes)

    # Step 1: 构建 tensor -> producing node index 映射
    produced_by = {}
    for idx, node in enumerate(nodes):
        for out in node.output:
            if out == "":  # 跳过空输出（虽然罕见）
                continue
            if out in produced_by:
                raise ValueError(f"Tensor '{out}' is produced by multiple nodes!")
            produced_by[out] = idx  # 记录生产者索引

    # 初始可用张量：inputs + initializers
    initial_tensors = {inp.name for inp in graph.input}
    initial_tensors.update(init.name for init in graph.initializer)

    # Step 2: 计算每个节点的入度（依赖的未满足输入数）
    in_degree = [0] * n
    dependents = defaultdict(list)  # idx -> list of dependent node indices

    for idx, node in enumerate(nodes):
        unmet = 0
        for inp in node.input:
            if inp == "":
                continue
            if inp in initial_tensors:
                continue
            elif inp in produced_by:
                producer_idx = produced_by[inp]
                dependents[producer_idx].append(idx)
                unmet += 1
            else:
                raise ValueError(f"Input tensor '{inp}' is not defined in the graph!")
        in_degree[idx] = unmet

    # Step 3: Kahn's algorithm using indices
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)

    executed = 0
    while queue:
        u = queue.popleft()
        executed += 1
        for v in dependents[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if executed != n:
        raise ValueError("Graph has a cycle or unresolved dependencies!")
    return True


def parse_onnx_model(onnx_path):
    model = onnx.load(onnx_path)

    # Optimize the ONNX model
    print("Optimizing ONNX model...")
    model = optimize_onnx_model(model)

    vk_model = VkModel()
    graph = model.graph
    try:
        onnx.checker.check_model(model, full_check=True)
        print("ONNX model passed full validation.")
    except onnx.checker.ValidationError as e:
        print(f"ONNX model full validation failed: {e}")

    # check if graph is topologically sorted
    if is_topologically_sortable(graph) == False:
        print("Graph is not topologically sorted. Please sort it before proceeding.")
        return 


    # Inputs with shapes
    for inp in graph.input:
        tensor_type = inp.type.tensor_type
        shape_dims = [
            dim.dim_value if dim.HasField("dim_value") else 1
            for dim in tensor_type.shape.dim
        ]
        print("Graph input:", inp.name, "of shape:", shape_dims)
        vk_model.inputs.append({'name': inp.name, 'shape': shape_dims})

    # Outputs with shapes
    for out in graph.output:
        tensor_type = out.type.tensor_type
        shape_dims = [
            dim.dim_value if dim.HasField("dim_value") else 1
            for dim in tensor_type.shape.dim
        ]
        print("Graph output:", out.name, "of shape:", shape_dims)
        vk_model.outputs.append({'name': out.name, 'shape': shape_dims})

    # Nodes with attributes and shapes of inputs/outputs
    for node in graph.node:
        print("Processing node:", node.name, "of type:", node.op_type)
        attributes = {}
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.INT:
                attributes[attr.name] = attr.i
            elif attr.type == onnx.AttributeProto.FLOAT:
                attributes[attr.name] = attr.f
            elif attr.type == onnx.AttributeProto.STRING:
                attributes[attr.name] = attr.s.decode('utf-8')
            elif attr.type == onnx.AttributeProto.TENSOR:
                attributes[attr.name] = numpy_helper.to_array(attr.t)
            elif attr.type == onnx.AttributeProto.INTS:
                attributes[attr.name] = list(attr.ints)
            elif attr.type == onnx.AttributeProto.FLOATS:
                attributes[attr.name] = list(attr.floats)
            else:
                print(f"Warning: Unsupported attribute type {attr.type} for attribute {attr.name}")

        inputs_with_shape = []
        for input_name in node.input:
            print("  Input:", input_name)
            # Find the input tensor to get its shape
            input_tensor = None
            for value_info in graph.value_info:
                if value_info.name == input_name:
                    input_tensor = value_info
                    break
            if input_tensor is None:
                for inp in graph.input:
                    if inp.name == input_name:
                        input_tensor = inp
                        break
            if input_tensor is None:
                for out in graph.output:
                    if out.name == input_name:
                        input_tensor = out
                        break
            if input_tensor is None:
                for initializer in graph.initializer:
                    if initializer.name == input_name:
                        data_type_map = {
                            1: onnx.TensorProto.FLOAT,
                            2: onnx.TensorProto.UINT8,
                            3: onnx.TensorProto.INT8,
                            4: onnx.TensorProto.UINT16,
                            5: onnx.TensorProto.INT16,
                            6: onnx.TensorProto.INT32,
                            7: onnx.TensorProto.INT64,
                            8: onnx.TensorProto.STRING,
                            9: onnx.TensorProto.BOOL,
                            10: onnx.TensorProto.FLOAT16,
                            11: onnx.TensorProto.DOUBLE,
                            12: onnx.TensorProto.UINT32,
                            13: onnx.TensorProto.UINT64,
                            14: onnx.TensorProto.COMPLEX64,
                            15: onnx.TensorProto.COMPLEX128,
                            16: onnx.TensorProto.BFLOAT16
                        }
                        data_type = data_type_map.get(initializer.data_type, onnx.TensorProto.UNDEFINED)
                        input_tensor = onnx.helper.make_tensor_value_info(
                            initializer.name, data_type, initializer.dims
                        )
                        break

            if input_tensor is None:
                print(f"Warning: Input tensor \"{input_name}\" not found in graph.")
                inputs_with_shape.append(
                    {'name': input_name, 'shape': []}
                )
                continue
            tensor_type = input_tensor.type.tensor_type
            shape_dims = [
                dim.dim_value if dim.HasField("dim_value") else 1
                for dim in tensor_type.shape.dim
            ]
            inputs_with_shape.append(
                {'name': input_name, 'shape': shape_dims}
            )

        outputs_with_shape = []
        for output_name in node.output:
            print("  Output:", output_name)
            # Find the output tensor to get its shape
            output_tensor = None
            for value_info in graph.value_info:
                if value_info.name == output_name:
                    output_tensor = value_info
                    break
            if output_tensor is None:
                for out in graph.output:
                    if out.name == output_name:
                        output_tensor = out
                        break
            if output_tensor is None:
                print(f"Warning: Output tensor {output_name} not found in graph.")
                outputs_with_shape.append({'name': output_name, 'shape':[]})
                continue
            tensor_type = output_tensor.type.tensor_type
            shape_dims = [
                dim.dim_value if dim.HasField("dim_value") else 1
                for dim in tensor_type.shape.dim
            ]
            outputs_with_shape.append(
                {'name': output_name, 'shape': shape_dims}
            )

        vk_model.nodes.append({
            'op_type': node.op_type,
            'name': node.name,
            'attributes': attributes,
            'inputs': inputs_with_shape,
            'outputs': outputs_with_shape
        })

    # Initializers (parameters)
    for initializer in graph.initializer:
        name = initializer.name
        vk_model.initializers[name] = initializer

    return vk_model


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python onnx2vkop.py <onnx_model_path>")
        sys.exit(1)

    onnx_model_path = sys.argv[1]
    output_bin_path = os.path.splitext(onnx_model_path)[0] + ".bin"

    print("Parsing ONNX model...")
    onnxmodel = parse_onnx_model(onnx_model_path)

    print("Saving to binary format...")
    onnxmodel.save_to_binary(output_bin_path)

    print("Model saved to:", output_bin_path)
