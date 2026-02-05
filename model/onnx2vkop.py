#!/bin/env python3

import os
import sys
import struct
import onnx
import numpy as np
from onnx import numpy_helper, shape_inference
import onnxoptimizer as optimizer
from onnxsim import simplify
from collections import defaultdict, deque
from typing import List, Dict, Set, Tuple, Optional
import argparse


class Node:
    def __init__(self, op_type, name, attributes, inputs, outputs):
        self.op_type = op_type
        self.name = name
        self.attributes = attributes
        self.inputs = inputs  # List of input node names
        self.outputs = outputs  # List of output node names
        self.dependencies = set()  # Node names this node depends on
        self.dependents = set()  # Node names that depend on this node


class DAGBasedModel:
    def __init__(self):
        self.nodes = {}  # Map from node name to Node object
        self.inputs = []  # Input nodes
        self.outputs = []  # Output nodes
        self.initializers = {}  # Initializers (name -> numpy array)

    def add_node(self, node):
        self.nodes[node.name] = node

    def build_dependencies(self):
        """Build dependency relationships between nodes."""
        for node in self.nodes.values():
            node.dependencies.clear()
            node.dependents.clear()

        # Build tensor -> producer mapping
        tensor_producer = {}
        for node in self.nodes.values():
            for output_name in node.outputs:
                tensor_producer[output_name['name']] = node.name

        # Build dependencies
        for node in self.nodes.values():
            for input_name in node.inputs:
                producer_name = tensor_producer.get(input_name['name'])
                if producer_name and producer_name != node.name:
                    producer_node = self.nodes[producer_name]
                    node.dependencies.add(producer_name)
                    producer_node.dependents.add(node.name)

    def topological_sort(self):
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

    def find_concurrent_nodes(self):
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

                # Update in-degrees of dependents
                for dependent_name in self.nodes[node_name].dependents:
                    in_degree[dependent_name] -= 1
                    if in_degree[dependent_name] == 0:
                        queue.append(dependent_name)

            if current_level:
                levels.append(current_level)
        
        return levels

    def save_to_binary(self, file_path):
        """Save model to binary file with DAG information."""
        with open(file_path, 'wb') as f:
            # Save inputs with shapes
            self._write_list_with_shapes(f, self.inputs)

            # Save outputs with shapes
            self._write_list_with_shapes(f, self.outputs)

            # Save nodes including attributes and input/output shapes
            f.write(struct.pack('I', len(self.nodes)))
            for node_name, node in self.nodes.items():
                self._write_string(f, node.op_type)
                self._write_string(f, node.name)
                self._write_dict(f, node.attributes)  # Attributes
                self._write_list_with_shapes(f, node.inputs)
                self._write_list_with_shapes(f, node.outputs)

                # Write dependencies and dependents for DAG structure
                f.write(struct.pack('I', len(node.dependencies)))
                for dep_name in node.dependencies:
                    self._write_string(f, dep_name)

                f.write(struct.pack('I', len(node.dependents)))
                for dep_name in node.dependents:
                    self._write_string(f, dep_name)

            # Save initializers (name -> numpy array)
            f.write(struct.pack('I', len(self.initializers)))
            for name, arr in self.initializers.items():
                self._write_string(f, name)
                self._write_array(f, arr)

            # Save concurrent execution levels
            concurrent_levels = self.find_concurrent_nodes()
            f.write(struct.pack('I', len(concurrent_levels)))
            for level in concurrent_levels:
                f.write(struct.pack('I', len(level)))
                for node_name in level:
                    self._write_string(f, node_name)

    @staticmethod
    def _write_string(f, s):
        b = s.encode('utf-8')
        f.write(struct.pack('I', len(b)))
        f.write(b)

    @staticmethod
    def _write_list_with_shapes(f, lst):
        f.write(struct.pack('I', len(lst)))
        for item in lst:
            if isinstance(item, dict) and 'name' in item and 'shape' in item:
                DAGBasedModel._write_string(f, item['name'])
                DAGBasedModel._write_list(f, item['shape'])
            else:
                DAGBasedModel._write_string(f, str(item))

    @staticmethod
    def _write_list(f, lst):
        f.write(struct.pack('I', len(lst)))
        for item in lst:
            if isinstance(item, int):
                f.write(struct.pack('I', item))  # For dimensions
            elif isinstance(item, float):
                f.write(struct.pack('f', item))  # For float attributes
            else:
                DAGBasedModel._write_string(f, str(item))  # For names

    @staticmethod
    def _write_dict(f, d):
        f.write(struct.pack('I', len(d)))
        for key, value in d.items():
            DAGBasedModel._write_string(f, key)
            if isinstance(value, str):
                f.write(b'\x00')  # Tag for string
                DAGBasedModel._write_string(f, value)
            elif isinstance(value, int):
                f.write(b'\x01')  # Tag for int
                f.write(struct.pack('q', value))
            elif isinstance(value, float):
                f.write(b'\x02')  # Tag for float
                f.write(struct.pack('f', value))
            elif isinstance(value, list):
                if all(isinstance(v, int) for v in value):
                    f.write(b'\x03')  # Tag for list of ints
                    DAGBasedModel._write_list(f, value)
                elif all(isinstance(v, float) for v in value):
                    f.write(b'\x04')  # Tag for list of floats
                    DAGBasedModel._write_list(f, value)
                else:
                    raise ValueError(f"Unsupported list type in attribute: {key}")
            elif isinstance(value, np.ndarray):
                f.write(b'\x05')  # Tag for numpy array
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
            16: "bfloat16"
        }
        data_type = data_type_map.get(arr.data_type, onnx.TensorProto.UNDEFINED)
        DAGBasedModel._write_string(f, str(data_type))
        DAGBasedModel._write_list(f, list(arr.dims))
        total_elements = np.prod(arr.dims)
        print("Array ",  arr.name, "shape:", arr.dims, "Data type:", data_type, "Total elements:", total_elements)
        arr_np = np.ascontiguousarray(numpy_helper.to_array(arr))
        data = arr_np.tobytes()
        f.write(struct.pack('Q', len(data)))
        f.write(data)

def optimize_onnx_model(onnx_model, batch_size = 1):
    """
    Optimize the ONNX model using ONNX's built-in optimizer.
    """
    passes = [
        "eliminate_deadend",  # Remove unused nodes
        "eliminate_identity",  # Remove identity nodes
        # "eliminate_shape_op",  # Remove shape nodes
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
    initializer_names = {init.name for init in onnx_model.graph.initializer}
    actual_inputs = [inp for inp in onnx_model.graph.input if inp.name not in initializer_names]
    print("Actual model inputs:", [inp.name for inp in actual_inputs])
    print("Parameters count:", len(initializer_names))

    input_shapes = {}
    for inp in actual_inputs:
        name = inp.name
        tensor_type = inp.type.tensor_type
        dim = tensor_type.shape.dim
        if len(dim) != 4:
            assert len(dim) == 4, "Input shape must be 4D"
        fixed_shape = []
        for d in dim:
            if d.HasField("dim_value"):
                fixed_shape.append(d.dim_value)
            elif d.HasField("dim_param"):
                # 动态维度（如 "N", "batch"） 固定为 1
                fixed_shape.append(batch_size)
            else:
                # 未知维度（理论上不应出现），保守设为 1
                fixed_shape.append(1)
        print(f"Guess input: {name} of shape: {fixed_shape}")
        input_shapes[name] = fixed_shape
    optimized_model, check = simplify(onnx_model, overwrite_input_shapes=input_shapes)
    assert check, "Simplified ONNX model could not be validated"
    optimized_model = optimizer.optimize(optimized_model, passes)
    optimized_model = shape_inference.infer_shapes(optimized_model, strict_mode=True)

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

    # 初始可用张量: inputs + initializers
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

def get_producer_consumer_from_dag(dag_model):
    """
    从已构建的DAG中提取producer和consumer映射 避免重复计算
    """
    # tensor_name -> producing node dictionary
    producer = {}
    # tensor_name -> list of (consumer_node_dict, input_index)
    consumers = defaultdict(list)
    
    # 构建tensor到producer的映射
    for node in dag_model.nodes.values():
        node_dict = {
            'op_type': node.op_type,
            'name': node.name,
            'attributes': node.attributes,
            'inputs': node.inputs,
            'outputs': node.outputs
        }
        for output in node.outputs:
            producer[output['name']] = node_dict
    
    # 构建tensor到consumers的映射
    for node in dag_model.nodes.values():
        node_dict = {
            'op_type': node.op_type,
            'name': node.name,
            'attributes': node.attributes,
            'inputs': node.inputs,
            'outputs': node.outputs
        }
        for idx, input_tensor in enumerate(node.inputs):
            consumers[input_tensor['name']].append((node_dict, idx))
    
    return producer, consumers


def fuse_gated_conv(dag_model):
    producer, consumers = get_producer_consumer_from_dag(dag_model)
    nodes_dict = {node.name: node for node in dag_model.nodes.values()}

    for node in list(dag_model.nodes.values()):  # 使用list()避免迭代时修改
        if node.op_type != 'Mul':
            continue

        if len(node.inputs) != 2:
            continue

        inp0, inp1 = node.inputs[0]['name'], node.inputs[1]['name']

        # 尝试两种组合: inp0 是 conv, inp1 是 sigmoid(conv)
        candidates = [
            (inp0, inp1),
            (inp1, inp0)
        ]

        matched = False
        for conv_out, sig_out in candidates:
            if conv_out not in producer or sig_out not in producer:
                continue

            conv_node_dict = producer[conv_out]
            sig_node_dict = producer[sig_out]

            if conv_node_dict['op_type'] != 'Conv':
                continue
            if sig_node_dict['op_type'] != 'Sigmoid':
                continue
            if len(sig_node_dict['inputs']) != 1:
                continue
            if sig_node_dict['inputs'][0]['name'] != conv_out:
                continue

            conv_consumers = consumers[conv_out]
            if len(conv_consumers) != 2:
                continue
            consumer_names = {c[0]['op_type'] for c in conv_consumers}
            if not (consumer_names == {'Sigmoid', 'Mul'}):
                continue

            # 查找对应的Node对象
            conv_node_obj = nodes_dict.get(conv_node_dict['name'])
            sig_node_obj = nodes_dict.get(sig_node_dict['name'])
            mul_node_obj = nodes_dict.get(node.name)

            if conv_node_obj and sig_node_obj and mul_node_obj:
                # 创建融合节点
                fused = Node(
                    op_type=conv_node_obj.op_type,
                    name=conv_node_obj.name,
                    attributes=dict(conv_node_obj.attributes),
                    inputs=conv_node_obj.inputs,
                    outputs=mul_node_obj.outputs
                )
                fused.attributes['activation'] = 'Swish'

                # 从dag_model中移除原节点
                del dag_model.nodes[conv_node_obj.name]
                del dag_model.nodes[sig_node_obj.name]
                del dag_model.nodes[mul_node_obj.name]

                # 添加融合节点
                dag_model.nodes[fused.name] = fused
                matched = True
                break

        if matched:
            continue


def fuse_conv_bn(dag_model):
    """
    Fuse Conv + BatchNormalization patterns into a single Conv node.
    This optimization reduces the number of operations and can improve performance.
    Works for both Conv+BN and Conv+BN+ReLU patterns.
    """
    producer, consumers = get_producer_consumer_from_dag(dag_model)

    nodes_dict = {node.name: node for node in dag_model.nodes.values()}

    for node in list(dag_model.nodes.values()):
        if node.op_type != 'Conv':
            continue

        conv_out = node.outputs[0]['name']

        conv_consumers = consumers.get(conv_out, [])
        if len(conv_consumers) != 1:
            continue

        bn_node_dict, _ = conv_consumers[0]
        if bn_node_dict['op_type'] != 'BatchNormalization':
            continue

        # inputs[1] are merged scale, bias, mean, variance respectively
        tensor_name = bn_node_dict['inputs'][1]['name']
        if tensor_name not in dag_model.initializers:
            print(f"Warning: BatchNormalization parameters {tensor_name} not found, skipping fusion")
            continue

        tensor_array = numpy_helper.to_array(dag_model.initializers[tensor_name])
        total_elements = len(tensor_array)
        padded_N = total_elements // 4

        scale_array = np.zeros(padded_N, dtype=np.float32)
        bias_array = np.zeros(padded_N, dtype=np.float32)
        mean_array = np.zeros(padded_N, dtype=np.float32)
        var_array = np.zeros(padded_N, dtype=np.float32)
        for i in range(padded_N // 4):
            base_idx = i * 16
            for j in range(4):
                scale_array[i * 4 + j] = tensor_array[base_idx + j]           # scale
                bias_array[i * 4 + j] = tensor_array[base_idx + 4 + j]        # bias
                mean_array[i * 4 + j] = tensor_array[base_idx + 8 + j]        # mean
                var_array[i * 4 + j] = tensor_array[base_idx + 12 + j]

        eps = float(bn_node_dict['attributes'].get('epsilon', 1e-5))

        has_conv_bias = len(node.inputs) > 2  # 输入0是data, 输入1是weights, 输入2是bias(如果有)

        conv_weight_name = node.inputs[1]['name']
        if conv_weight_name not in dag_model.initializers:
            print(f"Warning: Conv weight {conv_weight_name} not found, skipping fusion")
            continue

        conv_weight = numpy_helper.to_array(dag_model.initializers[conv_weight_name])

        if has_conv_bias:
            conv_bias_name = node.inputs[2]['name']
            if conv_bias_name not in dag_model.initializers:
                print(f"Warning: Conv bias {conv_bias_name} not found, skipping fusion")
                continue
            conv_bias = numpy_helper.to_array(dag_model.initializers[conv_bias_name])
        else:
            conv_bias = np.zeros(scale_array.shape, dtype=scale_array.dtype)

        print("Fusing Conv node", node.name, "with BN node", bn_node_dict['name'])
        # 执行参数融合
        # 计算: gamma / sqrt(var + eps)
        inv_std = scale_array / np.sqrt(var_array + eps)

        # 新权重 = 旧权重 * (gamma / sqrt(var + eps))
        # 对于卷积，我们需要正确处理维度
        fused_weight = conv_weight * inv_std.reshape(-1, 1, 1, 1)  # reshape to match conv weight shape

        # 新偏置 = (旧偏置 - mean) * (gamma / sqrt(var + eps)) + beta
        fused_bias = (conv_bias - mean_array) * inv_std + bias_array

        # 直接更新dag_model.initializers中的权重和偏置
        # 更新权重为融合后的权重
        fused_weight_tensor = numpy_helper.from_array(fused_weight, conv_weight_name)
        dag_model.initializers[conv_weight_name] = fused_weight_tensor

        # 为偏置创建名称（如果原来没有bias，现在添加）
        if has_conv_bias:
            conv_bias_name = node.inputs[2]['name']
        else:
            conv_bias_name = f"{node.name}_fused_bias"

        fused_bias_tensor = numpy_helper.from_array(fused_bias, conv_bias_name)
        dag_model.initializers[conv_bias_name] = fused_bias_tensor

        fused = Node(
            op_type=node.op_type,
            name=node.name,
            attributes=dict(node.attributes),
            inputs=[
                node.inputs[0],  # 输入数据
                node.inputs[1],  # 权重（名称不变，但数值已更新）
                {'name': conv_bias_name, 'shape': list(fused_bias.shape)}
            ],
            outputs=bn_node_dict['outputs']
        )

        # 替换节点
        del dag_model.nodes[node.name]
        del dag_model.nodes[bn_node_dict['name']]
        dag_model.nodes[fused.name] = fused

    print("Fusing Conv+BN patterns...")


def fuse_conv_simple_activation(dag_model):
    producer, consumers = get_producer_consumer_from_dag(dag_model)
    ACTIVATIONS = {"Relu", "Sigmoid", "Tanh", "HardSwish", "Mish"}

    nodes_dict = {node.name: node for node in dag_model.nodes.values()}

    for node in list(dag_model.nodes.values()):
        if node.op_type in ['Conv', 'BatchNormalization', 'Add', 'Gemm'] and len(node.outputs) == 1:
            out_name = node.outputs[0]['name']
            outs = consumers.get(out_name, [])
            if len(outs) == 1:
                next_node_dict, _ = outs[0]
                if next_node_dict['op_type'] == 'Clip':
                    if len(next_node_dict['inputs']) > 1:
                        min_input_name = next_node_dict['inputs'][1]['name']
                        max_input_name = next_node_dict['inputs'][2]['name']

                        if min_input_name in dag_model.initializers:
                            min_array = numpy_helper.to_array(dag_model.initializers[min_input_name])
                            if min_array.size == 1:
                                min_val = float(min_array.item())
                            del dag_model.initializers[min_input_name]

                        if max_input_name in dag_model.initializers:
                            max_array = numpy_helper.to_array(dag_model.initializers[max_input_name])
                            if max_array.size == 1:
                                max_val = float(max_array.item())
                            del dag_model.initializers[max_input_name]
                    elif 'min' in next_node_dict['attributes'] or 'max' in next_node_dict['attributes']:
                        min_val = next_node_dict['attributes'].get('min', -1.0)
                        max_val = next_node_dict['attributes'].get('max', 1.0)
                    if min_val == 0.0 and max_val == 6.0:
                        fused = Node(
                            op_type=node.op_type,
                            name=node.name,
                            attributes=dict(node.attributes),
                            inputs=node.inputs,
                            outputs=next_node_dict['outputs']
                        )
                        fused.attributes['activation'] = 'Relu6'

                        del dag_model.nodes[node.name]
                        del dag_model.nodes[next_node_dict['name']]
                        dag_model.nodes[fused.name] = fused
                elif (next_node_dict['op_type'] in ACTIVATIONS and
                    len(next_node_dict['inputs']) == 1 and
                    next_node_dict['inputs'][0]['name'] == out_name):

                    fused = Node(
                        op_type=node.op_type,
                        name=node.name,
                        attributes=dict(node.attributes),
                        inputs=node.inputs,
                        outputs=next_node_dict['outputs']
                    )
                    fused.attributes['activation'] = next_node_dict['op_type']

                    del dag_model.nodes[node.name]
                    del dag_model.nodes[next_node_dict['name']]
                    dag_model.nodes[fused.name] = fused
                    continue


def broadcast_shapes(shape_a, shape_b):
    """Compute broadcasted shape per ONNX rules."""
    if not shape_a and not shape_b:
        return []
    if not shape_a:
        return shape_b
    if not shape_b:
        return shape_a

    # Make copies and align from right
    a, b = list(shape_a), list(shape_b)
    len_a, len_b = len(a), len(b)
    max_len = max(len_a, len_b)

    # Left-pad with 1s
    a = [1] * (max_len - len_a) + a
    b = [1] * (max_len - len_b) + b

    result = []
    for da, db in zip(a, b):
        if da == db:
            result.append(da)
        elif da == 1:
            result.append(db)
        elif db == 1:
            result.append(da)
        else:
            raise ValueError(f"Shapes {shape_a} and {shape_b} are not broadcastable")
    return result


def merge_initializers(dag_model):
    """
    Merge batch normalization parameters into a single tensor [4, N]
    where N is the number of channels.

    Layout:
    Row 0: scale/weight (default: 1.0)
    Row 1: bias (default: 0.0)
    Row 2: mean (required)
    Row 3: variance (required)

    This modifies the dag_model by:
    1. Finding batch normalization nodes
    2. Merging their 4 input parameters into one
    3. Updating the nodes and initializers accordingly
    """

    # Find all batch normalization nodes
    bn_nodes = []
    for node_name, node in dag_model.nodes.items():
        if node.op_type == 'BatchNormalization':
            bn_nodes.append((node_name, node))

    # Keep track of which initializers have been merged
    merged_initializers = set()

    for node_name, node in bn_nodes:
        # BatchNormalization typically has 5 inputs:
        # input, scale, bias, mean, variance
        if len(node.inputs) < 5:
            print(f"Warning: BatchNormalization node {node.name} has less than 5 inputs")
            continue

        # Get the names of the parameters
        # input[0] is the actual input data
        # inputs[1-4] are scale, bias, mean, variance respectively
        input_data = node.inputs[0]    # input data
        scale_name = node.inputs[1]['name']  # scale/weight
        bias_name = node.inputs[2]['name']   # bias
        mean_name = node.inputs[3]['name']   # mean
        var_name = node.inputs[4]['name']    # variance

        # Check if all required parameters exist
        required_params = [mean_name, var_name]
        for param_name in required_params:
            if param_name not in dag_model.initializers:
                print(f"Error: Required parameter {param_name} not found in initializers")
                continue

        # Get the parameter arrays
        try:
            mean_array = numpy_helper.to_array(dag_model.initializers[mean_name])
            var_array = numpy_helper.to_array(dag_model.initializers[var_name])

            # Validate that mean and variance have the same shape
            if mean_array.shape != var_array.shape:
                print(f"Error: Mean {mean_array.shape} and variance {var_array.shape} have different shapes")
                continue

            # Handle optional parameters with defaults
            if scale_name in dag_model.initializers:
                scale_array = numpy_helper.to_array(dag_model.initializers[scale_name])
            else:
                scale_array = np.ones_like(mean_array, dtype=mean_array.dtype)

            if bias_name in dag_model.initializers:
                bias_array = numpy_helper.to_array(dag_model.initializers[bias_name])
            else:
                bias_array = np.zeros_like(mean_array, dtype=mean_array.dtype)

            # where N is the number of elements in each parameter
            N = mean_array.size
            padded_N = ((N+3)//4) * 4
            merged_data = np.zeros((4 * padded_N), dtype=mean_array.dtype)

            # Reshape all arrays to 1D for consistent indexing
            scale_flat = scale_array.flatten()
            bias_flat = bias_array.flatten()
            mean_flat = mean_array.flatten()
            var_flat = var_array.flatten()
            if scale_flat.size < padded_N:
                scale_flat = np.pad(scale_flat, (0, padded_N - scale_flat.size), constant_values=1.0)
                bias_flat = np.pad(bias_flat, (0, padded_N - bias_flat.size), constant_values=0.0)
                mean_flat = np.pad(mean_flat, (0, padded_N - mean_flat.size), constant_values=0.0)
                var_flat = np.pad(var_flat, (0, padded_N - var_flat.size), constant_values=1.0)

            # Fill the merged tensor
            for i in range(padded_N // 4):
                base_idx = i * 4
                # Reorganize data to match C++ implementation (interleaved format)
                # Each group of 16 elements contains 4 vec4: scale, bias, mean, variance
                for j in range(4):
                    if base_idx + j < N:
                        merged_data[i * 16 + j] = scale_flat[base_idx + j]           # scale
                        merged_data[i * 16 + 4 + j] = bias_flat[base_idx + j]        # bias
                        merged_data[i * 16 + 8 + j] = mean_flat[base_idx + j]        # mean
                        merged_data[i * 16 + 12 + j] = var_flat[base_idx + j]        # variance
                    else:
                        merged_data[i * 16 + j] = 1.0                               # scale default
                        merged_data[i * 16 + 4 + j] = 0.0                           # bias default
                        merged_data[i * 16 + 8 + j] = 0.0                           # mean default
                        merged_data[i * 16 + 12 + j] = 1.0                          # variance default
            # Create a new initializer name
            merged_name = f"{node.name}_bn_params"

            # Convert back to ONNX tensor
            merged_tensor = numpy_helper.from_array(merged_data, merged_name)

            # Add the merged tensor to initializers
            dag_model.initializers[merged_name] = merged_tensor
            
            # Update the node to use the merged parameter
            # Change inputs from 5 to 2: [input, merged_params]
            new_inputs = [
                input_data,  # Original input data (index 0)
                {'name': merged_name, 'shape': [4 * padded_N]}  # Merged parameters
            ]

            node.inputs = new_inputs

            # Mark original initializers for removal
            merged_initializers.update([scale_name, bias_name, mean_name, var_name])

            print(f"Merged batchnorm for {node.name}: "
                  f"scale({scale_name}), bias({bias_name}), mean({mean_name}), var({var_name}) "
                  f"-> merged({merged_name})")

        except Exception as e:
            print(f"Error processing BatchNormalization node {node.name}: {e}")
            continue

    # Remove the original individual initializers
    for initializer_name in merged_initializers:
        if initializer_name in dag_model.initializers:
            del dag_model.initializers[initializer_name]

    print(f"Merged {len(bn_nodes)} BatchNormalization nodes, removed {len(merged_initializers)} initializers")


def convert_flat_to_reshape(dag_model):
    """
    Convert Flat nodes to Reshape nodes with explicit shapes.

    Flatten operation flattens the input tensor into a 2D tensor, keeping dimensions
    up to axis-1 and flattening the rest into the second dimension.
    """
    nodes_to_update = []

    for node_name, node in dag_model.nodes.items():
        if node.op_type == 'Flatten':
            # Get input shape
            if len(node.inputs) > 0 and len(node.inputs[0]['shape']) > 0:
                input_shape = node.inputs[0]['shape']
                print(f"Flatten node {node.name} input shape: {input_shape}")

                # Get axis attribute (default is 1 according to ONNX spec)
                axis = node.attributes.get('axis', 1)

                # Calculate output shape for flatten:
                # First part: product of dimensions from 0 to axis-1
                # Second part: product of dimensions from axis to end
                if axis == 0:
                    first_part = 1
                else:
                    first_part = 1
                    for i in range(axis):
                        first_part *= input_shape[i]

                second_part = 1
                for i in range(axis, len(input_shape)):
                    second_part *= input_shape[i]

                output_shape = [first_part, second_part]

                # Create new reshape node
                reshape_node = Node(
                    op_type='Reshape',
                    name=node.name,
                    attributes={},
                    inputs=node.inputs[:],  # Copy original inputs
                    outputs=node.outputs[:]  # Copy original outputs
                )

                # Add shape tensor as second input
                shape_tensor_name = node.name + '_shape'
                shape_tensor = np.array(output_shape, dtype=np.int64)
                shape_initializer = numpy_helper.from_array(shape_tensor, shape_tensor_name)
                dag_model.initializers[shape_tensor_name] = shape_initializer

                # Add the shape tensor as the second input to reshape
                reshape_node.inputs.append({
                    'name': shape_tensor_name,
                    'shape': list(shape_tensor.shape)
                })

                nodes_to_update.append((node_name, reshape_node))
                print(f"Converted Flatten node '{node.name}' to Reshape with shape {output_shape} (axis={axis})")
            else:
                # If we can't determine the shape, keep the original node
                print(f"Warning: Could not convert Flatten node '{node.name}' - missing shape info")

    # Apply updates
    for old_name, new_node in nodes_to_update:
        del dag_model.nodes[old_name]
        dag_model.nodes[new_node.name] = new_node


def remove_redundant_reshape(dag_model):
    """
    Remove redundant reshape nodes where input and output shapes are the same.
    Updates connections so that the reshape's input becomes the next node's input.
    Also cleans up unused initializers.
    """
    # Build mapping of output names to producing nodes
    producer = {}
    for node in dag_model.nodes.values():
        for out in node.outputs:
            producer[out['name']] = node

    # Track which nodes to remove
    to_remove = []
    # Map from reshape output names to their input names
    reshape_remap = {}
    # Track initializers used by redundant reshapes
    redundant_initializer_names = set()

    # First pass: identify redundant reshapes and build remapping
    for node_name, node in dag_model.nodes.items():
        if node.op_type == 'Reshape':
            # Check if input and output shapes are the same
            if (len(node.inputs) >= 1 and len(node.outputs) >= 1 and
                node.inputs[0]['shape'] == node.outputs[0]['shape']):

                # This is a redundant reshape node
                input_name = node.inputs[0]['name']
                output_name = node.outputs[0]['name']

                # Record the mapping for remapping
                reshape_remap[output_name] = input_name
                # Mark this reshape node for removal
                to_remove.append(node_name)

                # Collect initializers used by this reshape node (typically the shape tensor)
                for inp in node.inputs[1:]:  # Skip the first input (data), consider the shape input
                    if inp['name'] in dag_model.initializers:
                        redundant_initializer_names.add(inp['name'])

                print(f"Identified redundant Reshape node: {node.name}")

    # Check if the collected initializers are used by any other nodes
    # If not, they should be removed
    initializers_to_remove = set()
    if redundant_initializer_names:
        # Build a set of all tensor names used by all nodes (except the ones we're removing)
        used_tensors = set()
        for node_name, node in dag_model.nodes.items():
            if node_name not in to_remove:  # Skip nodes we're going to remove
                for inp in node.inputs:
                    used_tensors.add(inp['name'])
                for out in node.outputs:
                    used_tensors.add(out['name'])

        # Check if any of our redundant initializers are actually used elsewhere
        for initializer_name in redundant_initializer_names:
            if initializer_name not in used_tensors:
                initializers_to_remove.add(initializer_name)

    # Second pass: update all nodes that reference the removed reshape outputs
    for node in dag_model.nodes.values():
        if node.name in to_remove:
            continue  # Skip the nodes we're removing

        # Update inputs that reference removed reshape outputs
        for inp in node.inputs:
            if inp['name'] in reshape_remap:
                old_name = inp['name']
                inp['name'] = reshape_remap[old_name]
                # Also update the shape if needed (should be the same)
                # Find the source node/input to get the correct shape
                print(f"Remapped input {old_name} to {inp['name']} in node {node.name}")

    # Remove the marked nodes
    if to_remove:
        for node_name in to_remove:
            if node_name in dag_model.nodes:
                del dag_model.nodes[node_name]
        print(f"Removed {len(to_remove)} redundant reshape nodes")

    # Remove unused initializers
    if initializers_to_remove:
        for initializer_name in initializers_to_remove:
            if initializer_name in dag_model.initializers:
                del dag_model.initializers[initializer_name]
        print(f"Removed {len(initializers_to_remove)} unused initializers: {initializers_to_remove}")


def quantize_to_fp16_selective(dag_model):
    """
    Selectively quantize model weights to FP16 based on operator type and parameter sensitivity.

    This function converts only appropriate FP32 initializers to FP16, considering:
    1. Which operator uses the initializer
    2. What role the initializer plays (weights vs. batch norm parameters)
    3. Sensitivity of different parameter types to quantization

    Generally safe to convert:
    - Convolution weights
    - Gemm/Linear weights
    - Recurrent weights

    Usually NOT safe to convert:
    - BatchNorm parameters (scale, bias, mean, var)
    - Small embedding tables
    """
    print("Selectively quantizing model to FP16...")

    converted_count = 0
    skipped_count = 0

    # Build mapping from initializer names to their consumers
    initializer_consumers = defaultdict(list)
    for node in dag_model.nodes.values():
        for inp in node.inputs:
            initializer_consumers[inp['name']].append(node)

    # Data types that should remain unchanged
    preserve_types = {
        onnx.TensorProto.UINT8,
        onnx.TensorProto.INT8,
        onnx.TensorProto.UINT16,
        onnx.TensorProto.INT16,
        onnx.TensorProto.INT32,
        onnx.TensorProto.INT64,
        onnx.TensorProto.UINT32,
        onnx.TensorProto.UINT64,
        onnx.TensorProto.BOOL
    }

    # Operators whose weights are usually safe to quantize
    safe_weight_operators = {
        'Conv', 'Gemm', 'MatMul', 'ConvTranspose',
        'LSTM', 'GRU', 'RNN'
    }

    # Parameters that are usually sensitive to FP16 quantization
    sensitive_parameters = {
        'BatchNormalization'
    }

    for name, initializer in dag_model.initializers.items():
        # Skip non-FP32 tensors
        if initializer.data_type != onnx.TensorProto.FLOAT:
            if initializer.data_type in preserve_types:
                print(f"Preserving {onnx.TensorProto.DataType.Name(initializer.data_type)} tensor '{name}'")
            elif initializer.data_type == onnx.TensorProto.FLOAT16:
                print(f"Skipping already FP16 tensor '{name}'")
            else:
                data_type_name = onnx.TensorProto.DataType.Name(initializer.data_type) if initializer.data_type <= 16 else "UNKNOWN"
                print(f"Preserving {data_type_name} tensor '{name}' (type: {initializer.data_type})")
            skipped_count += 1
            continue

        # Check who consumes this initializer
        consumers = initializer_consumers.get(name, [])
        consumer_ops = {node.op_type for node in consumers}

        # Determine if this initializer should be quantized
        should_quantize = False
        reason = ""

        arr = numpy_helper.to_array(initializer)
        if any(op in safe_weight_operators for op in consumer_ops):
            # This initializer is consumed by operators known to be safe for FP16
            should_quantize = True
            reason = f"consumed by safe operators {consumer_ops}"
        elif not consumers:
            # Orphaned initializer - better to preserve
            should_quantize = False
            reason = "no consumers"
        elif any(op in sensitive_parameters for op in consumer_ops):
            # Consumed by sensitive operators like BatchNormalization
            should_quantize = False
            reason = f"consumed by sensitive operators {consumer_ops}"
        else:
            # Default behavior - check size (small tensors might be sensitive)
            should_quantize = False
            reason = f"small tensor ({arr.size} elements)"

        if should_quantize:
            # Convert to numpy array
            arr = numpy_helper.to_array(initializer)

            # Convert to FP16
            arr_fp16 = arr.astype(np.float16)

            # Convert back to ONNX tensor with FP16 data type
            fp16_initializer = numpy_helper.from_array(arr_fp16, name)

            # Update the initializer in the model
            dag_model.initializers[name] = fp16_initializer
            print(f"Converted FP32 tensor '{name}' to FP16 ({reason})")
            print(f"New shape: {fp16_initializer.dims}")
            converted_count += 1
        else:
            skipped_count += 1

    print(f"Converted {converted_count} FP32 tensors to FP16")
    print(f"Preserved {skipped_count} tensors")


def quantize_to_int8_weight_only(dag_model):
    """
    Quantize model weights to INT8 using weight-only quantization.
    This function converts appropriate FP32 initializers to INT8, preserving scale information.

    For each tensor that is quantized:
    - The original FP32 weights are quantized to INT8
    - A scale tensor is created and stored as a separate initializer
    - The scale is calculated per-channel or per-tensor depending on tensor size

    Generally safe to convert:
    - Convolution weights
    - Gemm/Linear weights
    - Recurrent weights
    """
    print("Applying weight-only INT8 quantization...")

    converted_count = 0
    skipped_count = 0

    # Build mapping from initializer names to their consumers
    initializer_consumers = defaultdict(list)
    for node in dag_model.nodes.values():
        for inp in node.inputs:
            initializer_consumers[inp['name']].append(node)

    # Data types that should remain unchanged
    preserve_types = {
        onnx.TensorProto.UINT8,
        onnx.TensorProto.INT8,
        onnx.TensorProto.UINT16,
        onnx.TensorProto.INT16,
        onnx.TensorProto.INT32,
        onnx.TensorProto.INT64,
        onnx.TensorProto.UINT32,
        onnx.TensorProto.UINT64,
        onnx.TensorProto.BOOL
    }

    # Operators whose weights are usually safe to quantize to INT8
    safe_weight_operators = {
        'Conv', 'MatMul', 'ConvTranspose',
        'LSTM', 'GRU', 'RNN'
    }

    # Parameters that are usually sensitive to INT8 quantization
    sensitive_parameters = {
        'BatchNormalization', 'LayerNormalization', 'GroupNormalization'
    }

    # Get a list of keys to iterate over, to avoid modifying the dict during iteration
    initializers_keys = list(dag_model.initializers.keys())

    for name in initializers_keys:
        initializer = dag_model.initializers[name]

        # Skip non-FP32 tensors
        if initializer.data_type != onnx.TensorProto.FLOAT:
            if initializer.data_type in preserve_types:
                print(f"Preserving {onnx.TensorProto.DataType.Name(initializer.data_type)} tensor '{name}'")
            elif initializer.data_type == onnx.TensorProto.FLOAT16:
                print(f"Skipping already FP16 tensor '{name}'")
            else:
                data_type_name = onnx.TensorProto.DataType.Name(initializer.data_type) if initializer.data_type <= 16 else "UNKNOWN"
                print(f"Preserving {data_type_name} tensor '{name}' (type: {initializer.data_type})")
            skipped_count += 1
            continue

        # Check who consumes this initializer
        consumers = initializer_consumers.get(name, [])
        consumer_ops = {node.op_type for node in consumers}

        # Determine if this initializer should be quantized
        should_quantize = False
        reason = ""

        if any(op in safe_weight_operators for op in consumer_ops):
            # This initializer is consumed by operators known to be safe for INT8 quantization
            should_quantize = True
            reason = f"consumed by safe operators {consumer_ops}"
        elif not consumers:
            # Orphaned initializer - better to preserve
            should_quantize = False
            reason = "no consumers"
        elif any(op in sensitive_parameters for op in consumer_ops):
            # Consumed by sensitive operators like BatchNormalization
            should_quantize = False
            reason = f"consumed by sensitive operators {consumer_ops}"
        else:
            # Default behavior - check size (small tensors might be sensitive)
            should_quantize = False
        # Additional check: skip bias tensors
        # Bias tensors are typically 1D and have small sizes
        arr = numpy_helper.to_array(initializer)
        if len(arr.shape) == 1:  # Common bias tensor characteristics
            # Check if it's connected to a bias input in operations like Conv, Gemm, etc.
            is_bias = False
            for node in consumers:
                op_type = node.op_type
                if op_type in ['Conv', 'ConvTranspose']:
                    # Check if this tensor is the bias input (usually the 3rd input for Conv, 2nd for Gemm)
                    for idx, inp in enumerate(node.inputs):
                        if inp['name'] == name:
                            # For Conv: 0=inputs, 1=weights, 2=bias
                            if (op_type == 'Conv' and idx == 2) or \
                               (op_type == 'ConvTranspose' and idx == 2):
                                is_bias = True
                                break
            if is_bias:
                print(f"Preserving bias tensor '{name}' as FP32")
                skipped_count += 1
                should_quantize = False
        if should_quantize:
            # Convert to numpy array
            arr = numpy_helper.to_array(initializer)
            original_fp32 = arr.copy()

            # Determine quantization axis based on operator type and tensor shape
            axis = None
            if len(arr.shape) >= 2:
                # For multi-dimensional weights, use operator-specific quantization axis
                for node in consumers:
                    op_type = node.op_type
                    if op_type == 'Conv':
                        # Conv weights: [C_out, C_in, K, K] - quantize per output channel
                        # Reduce along (C_in, K, K) dimensions -> axis=(1, 2, 3)
                        if len(arr.shape) == 4:
                            axis = (1, 2, 3)
                        elif len(arr.shape) == 3:
                            axis = (1, 2)
                        else:
                            axis = 0
                        break
                    elif op_type == 'ConvTranspose':
                        # ConvTranspose weights: [C_in, C_out, K, K] - quantize per output channel
                        # Reduce along (C_in, K, K) dimensions -> axis=(0, 2, 3)
                        if len(arr.shape) == 4:
                            axis = (0, 2, 3)
                        else:
                            # For other shapes, default to axis=1
                            axis = 1
                        break
                    elif op_type == 'Gemm':
                        # Gemm weights: [out, in] - quantize per output dimension
                        # Reduce along in dimension -> axis=1
                        if len(arr.shape) == 2:
                            axis = 1
                        else:
                            # For other shapes, default to axis=0
                            axis = 0
                        break
                    elif op_type == 'MatMul':
                        # MatMul weights: typically [in, out] - quantize per output dimension
                        # Reduce along in dimension -> axis=0
                        if len(arr.shape) == 2:
                            axis = 0
                        else:
                            # For other shapes, default to axis=1
                            axis = 1
                        break
                    elif op_type in ['LSTM', 'GRU']:
                        # LSTM/GRU weights: [D, 4H, I] or [D, 4H, H] - quantize per output dimension
                        # Reduce along I or H dimension -> axis=2
                        if len(arr.shape) == 3:
                            axis = 2
                        else:
                            # For other shapes, default to axis=0
                            axis = 0
                        break
                    else:
                        # Default to axis=0 for other operators
                        axis = 0
            else:
                # For 1D or smaller tensors, use per-tensor quantization
                axis = None

            # Perform INT8 quantization
            if axis is not None:
                # Calculate scale per specified axis
                amax = np.amax(np.abs(arr), axis=axis, keepdims=True)
                scale_keepdims  = amax / 127.0  # INT8 range is [-128, 127]

                # Avoid division by zero
                scale_keepdims = np.where(scale_keepdims == 0, 1.0, scale_keepdims)

                arr_int8 = np.round(arr / scale_keepdims).astype(np.int8)

                 # For most operators (except LSTM/GRU), create 1D scale for storage
                if not any(op in ['LSTM', 'GRU'] for op in consumer_ops):
                    # Calculate scale per axis but flatten to 1D for storage
                    amax_1d = np.amax(np.abs(arr), axis=axis, keepdims=False)  # Flatten scale
                    scale = amax_1d / 127.0

                    # Avoid division by zero
                    scale = np.where(scale == 0, 1.0, scale)

                    # For different operator types, ensure correct shape
                    if 'Conv' in consumer_ops and len(arr.shape) == 4:
                        # Conv: [C_out, C_in, K, K], axis=(1,2,3) -> scale should be [C_out]
                        scale = scale.reshape(arr.shape[0])
                    elif 'ConvTranspose' in consumer_ops and len(arr.shape) == 4:
                        # ConvTranspose: [C_in, C_out, K, K], axis=(0,2,3) -> scale should be [C_out]
                        scale = scale.reshape(arr.shape[1])
                    elif 'Gemm' in consumer_ops and len(arr.shape) == 2:
                        # Gemm: [out, in], axis=1 -> scale should be [out]
                        scale = scale.reshape(arr.shape[0])
                    elif 'MatMul' in consumer_ops and len(arr.shape) == 2:
                        # MatMul: [in, out], axis=0 -> scale should be [out]
                        scale = scale.reshape(arr.shape[1])
                    else:
                        # For other cases, flatten to 1D
                        scale = scale.flatten()
                else:
                    # For LSTM/GRU, keep the reduced scale but not the original shape
                    scale = amax_1d / 127.0
                    scale = np.where(scale == 0, 1.0, scale)
            else:
                # For 1D or smaller tensors, use per-tensor quantization
                amax = np.amax(np.abs(arr))
                scale_keepdims = amax / 127.0 if amax != 0 else 1.0
                scale = scale_keepdims  # For per-tensor, scale is scalar
                arr_int8 = np.round(arr / scale_keepdims).astype(np.int8)

            # === 反量化: INT8 -> FP32 ===
            if axis is not None:
                # Broadcast scale back to original shape for dequantization
                scale_broadcast = np.expand_dims(scale, axis=axis) if np.ndim(scale) > 0 else scale
                dequantized = arr_int8.astype(np.float32) * scale_broadcast
            else:
                dequantized = arr_int8.astype(np.float32) * scale
            # === 计算误差指标 ===
            diff = dequantized - original_fp32
            mse = np.mean(diff ** 2)
            mae = np.mean(np.abs(diff))
            max_abs_error = np.max(np.abs(diff))
            # Relative error (avoid division by zero)
            rel_error = np.abs(diff) / (np.abs(original_fp32) + 1e-8)
            mean_rel_error = np.mean(rel_error)
            max_rel_error = np.max(rel_error)
            print(f"Quantized '{name}':")
            print(f"  Shape: {original_fp32.shape}")
            print(f"  Scale shape: {scale.shape if hasattr(scale, 'shape') else 'scalar'}")
            print(f"  MSE: {mse:.6e}, MAE: {mae:.6e}")
            print(f"  Max Abs Error: {max_abs_error:.6e}")
            print(f"  Mean Rel Error: {mean_rel_error:.2%}, Max Rel Error: {max_rel_error:.2%}")

            # Create INT8 initializer for quantized weights
            int8_initializer = numpy_helper.from_array(arr_int8.astype(np.int8), name)
            int8_initializer.data_type = onnx.TensorProto.INT8

            # Create scale initializer
            scale_name = f"{name}_scale"
            scale_initializer = numpy_helper.from_array(scale, scale_name)
            scale_initializer.data_type = onnx.TensorProto.FLOAT

            # Update the model: replace original with INT8 weights and add scale
            dag_model.initializers[name] = int8_initializer

            # Determine if scale should be stored as attribute or as input
            # If scale is a scalar or small array, store as attribute; otherwise, store as input
            scale_size = scale.size if hasattr(scale, 'size') else 1
            scale_is_scalar = np.isscalar(scale) or scale_size == 1

            scale_name = f"{name}_scale"
            scale_initializer = numpy_helper.from_array(scale, scale_name)
            scale_initializer.data_type = onnx.TensorProto.FLOAT
            dag_model.initializers[scale_name] = scale_initializer

            # Add scale as input to the nodes that consume the original initializer
            for node in consumers:
                # Add scale tensor as an additional input
                scale_input = {
                    'name': scale_name,
                    'shape': list(scale.shape) if hasattr(scale, 'shape') else []
                }
                node.inputs.append(scale_input)

            print(f"Converted FP32 tensor '{name}' to INT8 with scale tensor '{scale_name}' ({reason})")
            print(f"Original shape: {initializer.dims}, scale shape: {scale_initializer.dims}")
            converted_count += 1
        else:
            skipped_count += 1

    print(f"Converted {converted_count} FP32 tensors to INT8 with scale information")
    print(f"Preserved {skipped_count} tensors")
    print(f"Total initializers after quantization: {len(dag_model.initializers)}")


def move_input_tensor_to_attr(dag_model):
    """
    将一些算子input中包含的仅rank长度的tensor转换为attribute。
    比如resize算子中的scales、sizes, pad算子中的pads等, 这些都是小型一维张量，
    对于这类tensor可以直接转为attribute, 同时将node inputs中对应tensor置为空,
    对应的initializer也删除.
    """
    initializers_to_remove = set()

    SPECIAL_OPS = {
        'Resize': [(2, 'scales'), (3, 'sizes')],  # inputs[2]=scales, inputs[3]=sizes
        'Pad': [(1, 'pads')],  # inputs[1]=pads
        # 'Slice': [(1, 'starts'), (2, 'ends'), (3, 'axes'), (4, 'steps')],  # 多个参数
    }

    for node in dag_model.nodes.values():
        op_type = node.op_type
        if op_type not in SPECIAL_OPS:
            continue

        target_inputs = SPECIAL_OPS[op_type]

        for idx, attr_name in target_inputs:
            if idx >= len(node.inputs):
                continue

            input_tensor = node.inputs[idx]
            tensor_name = input_tensor['name']
            print("Checking input tensor: ", tensor_name)

            if not tensor_name or tensor_name not in dag_model.initializers:
                continue

            initializer = dag_model.initializers[tensor_name]

            # 检查是否为一维数组且长度较短（通常是rank长度，一般不超过8）
            if len(initializer.dims) == 1 and 0 < initializer.dims[0] <= 8:
                tensor_data = numpy_helper.to_array(initializer)

                if not hasattr(node, 'attributes') or node.attributes is None:
                    node.attributes = {}

                if tensor_data.dtype in [np.float32, np.float64]:
                    node.attributes[attr_name] = tensor_data.tolist()
                elif tensor_data.dtype in [np.int32, np.int64]:
                    node.attributes[attr_name] = tensor_data.tolist()
                else:
                    node.attributes[attr_name] = tensor_data.tolist()

                del node.inputs[idx]

                initializers_to_remove.add(tensor_name)
        print(node)

    for initializer_name in initializers_to_remove:
        if initializer_name in dag_model.initializers:
            del dag_model.initializers[initializer_name]

    print(f"Converted {len(initializers_to_remove)} tensor inputs to attributes")


class ModelConverter:
    """Main class for converting ONNX models to DAG-based format."""
    @staticmethod
    def parse_onnx_model(onnx_path, batch_size):
        model = onnx.load(onnx_path)

        # Optimize the ONNX model
        print("Optimizing ONNX model...")
        model = optimize_onnx_model(model, batch_size)
        # save optimized model
        onnx.save(model, "optimized_" + os.path.basename(onnx_path))

        dag_model = DAGBasedModel()

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

        # Initializers (parameters)
        for initializer in graph.initializer:
            name = initializer.name
            dag_model.initializers[name] = initializer

        initializer_names = {init.name for init in graph.initializer}
        # Inputs with shapes
        for inp in graph.input:
            if inp.name in initializer_names:
                continue
            tensor_type = inp.type.tensor_type
            shape_dims = [
                dim.dim_value if dim.HasField("dim_value") else 1
                for dim in tensor_type.shape.dim
            ]
            print("Graph input:", inp.name, "of shape:", shape_dims)
            dag_model.inputs.append({'name': inp.name, 'shape': shape_dims})

        # Outputs with shapes
        for out in graph.output:
            tensor_type = out.type.tensor_type
            shape_dims = [
                dim.dim_value if dim.HasField("dim_value") else 1
                for dim in tensor_type.shape.dim
            ]
            print("Graph output:", out.name, "of shape:", shape_dims)
            dag_model.outputs.append({'name': out.name, 'shape': shape_dims})

        modified_shapes = defaultdict(list)
        # Nodes with attributes and shapes of inputs/outputs
        ELEMWISE_OPS = {"Add", "And", "Div", "Equal", "Greater", "Less", "Max", "Mean",
            "Min", "Mul", "Or", "Pow", "Sub", "Sum", "Where", "Xor"}

        for i, node in enumerate(graph.node):
            if not node.name:
                # 使用 op_type + index 生成唯一名，避免冲突
                node.name = f"{node.op_type}_{i}"
        for node in graph.node:
            print("Processing node:", node.name, "of type:", node.op_type)
            attributes = {}
            for attr in node.attribute:
                # https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
                # message AttributeProto
                # enum AttributeType {
                #     UNDEFINED = 0;
                #     FLOAT = 1;
                #     INT = 2;
                #     STRING = 3;
                #     TENSOR = 4;
                #     GRAPH = 5;
                #     SPARSE_TENSOR = 11;
                #     TYPE_PROTO = 13;

                #     FLOATS = 6;
                #     INTS = 7;
                #     STRINGS = 8;
                #     TENSORS = 9;
                #     GRAPHS = 10;
                #     SPARSE_TENSORS = 12;
                #     TYPE_PROTOS = 14;
                # }

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
                #  GlobalAveragePool compression to 2d, so we can use storage instead of image
                if node.op_type == "GlobalAveragePool" and len(shape_dims) >= 2:
                    original_shape = shape_dims[:]
                    shape_dims = [original_shape[0], original_shape[1]]
                    print(f"Modified GlobalAveragePool shape from {original_shape} to {shape_dims}")
                    modified_shapes[output_name] = shape_dims

                outputs_with_shape.append(
                    {'name': output_name, 'shape': shape_dims}
                )

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
                if input_name in modified_shapes and len(modified_shapes[input_name]) > 0:
                    shape_dims = modified_shapes[input_name]
                    print(f"Modified shape of input {input_name} to {shape_dims}")
                inputs_with_shape.append(
                    {'name': input_name, 'shape': shape_dims}
                )

            if node.op_type in ELEMWISE_OPS and len(node.input) == 2:
                for i in range(len(inputs_with_shape)):
                    if inputs_with_shape[i]['name'] == node.input[0]:
                        id0 = i
                    if inputs_with_shape[i]['name'] == node.input[1]:
                        id1 = i
                if len(inputs_with_shape[id0]['shape']) != len(inputs_with_shape[id1]['shape']):
                    shape = broadcast_shapes(inputs_with_shape[id0]['shape'], inputs_with_shape[id1]['shape'])
                    print("  Elemwise operation with multiple inputs. Broadcasting shapes...", shape)
                    if len(inputs_with_shape[id0]['shape']) != len(shape):
                        inputs_with_shape[id0]['shape'] = shape
                        for initializer in graph.initializer:
                            if initializer.name == inputs_with_shape[id0]['name']:
                                orig_init = dag_model.initializers[initializer.name]
                                orig_array = numpy_helper.to_array(orig_init)
                                if orig_array.size == 1:
                                    scalar_val = orig_array.item()
                                    expanded_data = np.full(shape, scalar_val, dtype=orig_array.dtype)
                                else:
                                    expanded_data = np.broadcast_to(orig_array, shape).copy()
                                new_init = numpy_helper.from_array(expanded_data, name=new_name)
                                dag_model.initializers[initializer.name] = new_init
                    if len(inputs_with_shape[id1]['shape']) != len(shape):
                        inputs_with_shape[id1]['shape'] = shape
                        for initializer in graph.initializer:
                            if initializer.name == inputs_with_shape[id1]['name']:
                                orig_init = dag_model.initializers[initializer.name]
                                orig_array = numpy_helper.to_array(orig_init)
                                if orig_array.size == 1:
                                    scalar_val = orig_array.item()
                                    expanded_data = np.full(shape, scalar_val, dtype=orig_array.dtype)
                                else:
                                    expanded_data = np.broadcast_to(orig_array, shape).copy()
                                new_init = numpy_helper.from_array(expanded_data, name=initializer.name)
                                dag_model.initializers[initializer.name] = new_init

            new_node = Node(
                op_type=node.op_type,
                name=node.name,
                attributes=attributes,
                inputs=inputs_with_shape,
                outputs=outputs_with_shape
            )
            dag_model.add_node(new_node)

        # Build the DAG structure after adding all nodes
        dag_model.build_dependencies()

        return dag_model


def main():
    if len(sys.argv) < 2:
        print("Usage: python onnx2vkop.py <onnx_model_path>")
        sys.exit(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("-q","--quant", help="Override input_model")
    parser.add_argument("-i","--input", required=True, help="input_model file")
    parser.add_argument("-u","--unify", action='store_true', help="convert initializers to a single memory block")
    parser.add_argument("-b","--batch", help="batch size for inference")
    parser.add_argument("-r","--rgba", action='store_true', help="nchw to rgba conversion for initializers")
    args = parser.parse_args()
    if args.quant is not None:
        if args.quant not in ["fp16", "int8"]:
            print("Invalid quantization type. Please specify 'fp16' or 'int8'.")
            sys.exit(1)

    output_bin_path = os.path.splitext(args.input)[0] + ".vkopbin"

    print("Parsing ONNX model...")
    if args.batch is not None:
        batch_size = int(args.batch)
    else:
        batch_size = 1
    dag_model = ModelConverter.parse_onnx_model(args.input, batch_size)

    # Apply optimizations
    move_input_tensor_to_attr(dag_model)
    merge_initializers(dag_model)
    convert_flat_to_reshape(dag_model)
    remove_redundant_reshape(dag_model)
    fuse_conv_bn(dag_model)
    fuse_gated_conv(dag_model)
    fuse_conv_simple_activation(dag_model)
    
    # Rebuild DAG after optimizations
    dag_model.build_dependencies()
    
    if args.quant == "fp16":
        quantize_to_fp16_selective(dag_model)
    elif args.quant == "int8":
        quantize_to_int8_weight_only(dag_model)

    # Print concurrent execution levels
    concurrent_levels = dag_model.find_concurrent_nodes()
    print("\nConcurrent Execution Levels:")
    for level_idx, level in enumerate(concurrent_levels):
        print(f"Level {level_idx}: {level}")
    
    # Print operator statistics
    op_stats = {}
    for node in dag_model.nodes.values():
        op_type = node.op_type
        op_stats[op_type] = op_stats.get(op_type, 0) + 1

    print("\nOperator Statistics:")
    print("idx\ttype\t\tnum")
    for idx, (op_type, count) in enumerate(op_stats.items(), 1):
        print(f"{idx}\t{op_type}\t\t{count}")

    print("Saving to binary format...")
    dag_model.save_to_binary(output_bin_path)

    print("Model saved to:", output_bin_path)
    file_size = os.path.getsize(output_bin_path)
    print(f"File size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()
