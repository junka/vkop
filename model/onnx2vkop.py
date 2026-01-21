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
                f.write(struct.pack('f', item))  # For float attributes
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
                f.write(struct.pack('f', value))
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
        print("Array ",  arr.name, "shape:", arr.dims, "Data type:", data_type, "Total elements:", total_elements)
        arr = np.ascontiguousarray(numpy_helper.to_array(arr))
        data = arr.tobytes()
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
            assert shape_dims.size() == 4, "Input shape must be 4D"
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


def build_graph_index(nodes):
    # tensor_name -> producing node
    producer = {}
    # tensor_name -> list of (consumer_node, input_index)
    consumers = defaultdict(list)

    for node in nodes:
        for out in node['outputs']:
            producer[out['name']] = node
        for idx, inp in enumerate(node['inputs']):
            consumers[inp['name']].append((node, idx))

    return producer, consumers


def fuse_gated_conv(vk_model):
    producer, consumers = build_graph_index(vk_model.nodes)

    nodes = vk_model.nodes
    for idx, node in enumerate(nodes):
        node['_orig_idx'] = idx

    to_remove: Set[int] = set()
    replacements: Dict[int, Dict] = {}   

    for node in nodes:
        if node['op_type'] != 'Mul':
            continue

        inputs = node['inputs']
        if len(inputs) != 2:
            continue

        inp0, inp1 = inputs[0]['name'], inputs[1]['name']

        # 尝试两种组合: inp0 是 conv, inp1 是 sigmoid(conv)
        candidates = [
            (inp0, inp1),
            (inp1, inp0)
        ]

        matched = False
        for conv_out, sig_out in candidates:
            if conv_out not in producer or sig_out not in producer:
                continue

            conv_node = producer[conv_out]
            sig_node = producer[sig_out]

            if conv_node['op_type'] != 'Conv':
                continue
            if sig_node['op_type'] != 'Sigmoid':
                continue
            if len(sig_node['inputs']) != 1:
                continue
            if sig_node['inputs'][0]['name'] != conv_out:
                continue

            conv_consumers = consumers[conv_out]
            if len(conv_consumers) != 2:
                continue
            consumer_names = {c[0]['op_type'] for c in conv_consumers}
            if not (consumer_names == {'Sigmoid', 'Mul'}):
                continue
            
            fused = conv_node.copy()
            fused['attributes'] = dict(conv_node.get('attributes', {}))
            fused['attributes']['activation'] = 'Swish'
            fused['outputs'] = node['outputs']  # 输出为 Mul 的输出

            first_idx = conv_node['_orig_idx']
            replacements[first_idx] = fused
            to_remove.add(conv_node['_orig_idx'])
            to_remove.add(sig_node['_orig_idx'])
            to_remove.add(node['_orig_idx'])
            matched = True
            break

        if matched:
            continue

    new_nodes = []
    for idx, node in enumerate(nodes):
        if idx in to_remove:
            # 如果这是融合子图的第一个节点，就放 fused node
            if idx in replacements:
                new_nodes.append(replacements[idx])
        else:
            # 未被融合的节点，直接保留
            new_nodes.append(node)

    for node in new_nodes:
        node.pop('_orig_idx', None)

    vk_model.nodes = new_nodes


def fuse_conv_bn(vk_model):
    """
    Fuse Conv + BatchNormalization patterns into a single Conv node.
    This optimization reduces the number of operations and can improve performance.
    Works for both Conv+BN and Conv+BN+ReLU patterns.
    """
    producer, consumers = build_graph_index(vk_model.nodes)
    
    nodes = vk_model.nodes
    for idx, node in enumerate(nodes):
        node['_orig_idx'] = idx

    to_remove: Set[int] = set()
    replacements: Dict[int, Dict] = {}

    for node in nodes:
        if node['op_type'] != 'Conv':
            continue

        conv_out = node['outputs'][0]['name']

        conv_consumers = consumers.get(conv_out, [])
        if len(conv_consumers) != 1:
            continue

        bn_node, _ = conv_consumers[0]
        if bn_node['op_type'] != 'BatchNormalization':
            continue

        # inputs[1] are merged scale, bias, mean, variance respectively
        tensor_name = bn_node['inputs'][1]['name']
        if tensor_name not in vk_model.initializers:
            print(f"Warning: BatchNormalization parameters {tensor_name} not found, skipping fusion")
            continue

        tensor_array = numpy_helper.to_array(vk_model.initializers[tensor_name])
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

        eps = float(bn_node['attributes'].get('epsilon', 1e-5))

        has_conv_bias = len(node['inputs']) > 2  # 输入0是data, 输入1是weights, 输入2是bias(如果有)

        conv_weight_name = node['inputs'][1]['name']
        if conv_weight_name not in vk_model.initializers:
            print(f"Warning: Conv weight {conv_weight_name} not found, skipping fusion")
            continue

        conv_weight = numpy_helper.to_array(vk_model.initializers[conv_weight_name])

        if has_conv_bias:
            conv_bias_name = node['inputs'][2]['name']
            if conv_bias_name not in vk_model.initializers:
                print(f"Warning: Conv bias {conv_bias_name} not found, skipping fusion")
                continue
            conv_bias = numpy_helper.to_array(vk_model.initializers[conv_bias_name])
        else:
            conv_bias = np.zeros(scale_array.shape, dtype=scale_array.dtype)

        print("Fusing Conv node", node['name'], "with BN node", bn_node['name'])
        # 执行参数融合
        # 计算: gamma / sqrt(var + eps)
        inv_std = scale_array / np.sqrt(var_array + eps)

        # 新权重 = 旧权重 * (gamma / sqrt(var + eps))
        # 对于卷积，我们需要正确处理维度
        fused_weight = conv_weight * inv_std.reshape(-1, 1, 1, 1)  # reshape to match conv weight shape

        # 新偏置 = (旧偏置 - mean) * (gamma / sqrt(var + eps)) + beta
        fused_bias = (conv_bias - mean_array) * inv_std + bias_array

        # 直接更新vk_model.initializers中的权重和偏置
        # 更新权重为融合后的权重
        fused_weight_tensor = numpy_helper.from_array(fused_weight, conv_weight_name)
        vk_model.initializers[conv_weight_name] = fused_weight_tensor

        # 为偏置创建名称（如果原来没有bias，现在添加）
        if has_conv_bias:
            conv_bias_name = node['inputs'][2]['name']
        else:
            conv_bias_name = f"{node['name']}_fused_bias"

        fused_bias_tensor = numpy_helper.from_array(fused_bias, conv_bias_name)
        vk_model.initializers[conv_bias_name] = fused_bias_tensor

        fused = node.copy()
        fused['attributes'] = dict(node.get('attributes', {}))

        fused['outputs'] = bn_node['outputs']

        fused['inputs'] = [
            node['inputs'][0],  # 输入数据
            node['inputs'][1],  # 权重（名称不变，但数值已更新）
            {'name': conv_bias_name, 'shape': list(fused_bias.shape)}
        ]

        first_idx = node['_orig_idx']
        replacements[first_idx] = fused

        to_remove.add(node['_orig_idx'])
        to_remove.add(bn_node['_orig_idx'])

    new_nodes = []
    for idx, node in enumerate(nodes):
        if idx in to_remove:
            if idx in replacements:
                new_nodes.append(replacements[idx])
        else:
            new_nodes.append(node)

    for node in new_nodes:
        node.pop('_orig_idx', None)

    print("Fusing Conv+BN patterns...", len(to_remove))
    vk_model.nodes = new_nodes


def fuse_conv_simple_activation(vk_model):
    producer, consumers = build_graph_index(vk_model.nodes)
    ACTIVATIONS = {"Relu", "Sigmoid", "Tanh", "HardSwish", "Mish"}

    nodes = vk_model.nodes
    for idx, node in enumerate(nodes):
        node['_orig_idx'] = idx

    to_remove: Set[int] = set()
    replacements: Dict[int, Dict] = {}

    for node in nodes:
        if node['op_type'] in ['Conv', 'BatchNormalization', 'Add', 'Gemm'] and len(node['outputs']) == 1:
            out_name = node['outputs'][0]['name']
            outs = consumers.get(out_name, [])
            if len(outs) == 1:
                next_node, _ = outs[0]
                if next_node['op_type'] == 'Clip':
                    if len(next_node['inputs']) > 1:
                        min_input_name = next_node['inputs'][1]['name']
                        max_input_name = next_node['inputs'][2]['name']

                        if min_input_name in vk_model.initializers:
                            min_array = numpy_helper.to_array(vk_model.initializers[min_input_name])
                            if min_array.size == 1:
                                min_val = float(min_array.item())
                            del vk_model.initializers[min_input_name]
                                
                        if max_input_name in vk_model.initializers:
                            max_array = numpy_helper.to_array(vk_model.initializers[max_input_name])
                            if max_array.size == 1:
                                max_val = float(max_array.item())
                            del vk_model.initializers[max_input_name]
                    elif 'min' in next_node['attributes'] or 'max' in next_node['attributes']:
                        min_val = next_node['attributes'].get('min', -1.0)
                        max_val = next_node['attributes'].get('max', 1.0)
                    if min_val == 0.0 and max_val == 6.0:
                        fused = node.copy()
                        fused['attributes'] = dict(node.get('attributes', {}))
                        fused['attributes']['activation'] = 'Relu6'
                        fused['outputs'] = next_node['outputs']

                        first_idx = node['_orig_idx']
                        replacements[first_idx] = fused

                        to_remove.add(first_idx)
                        to_remove.add(next_node['_orig_idx'])
                elif (next_node['op_type'] in ACTIVATIONS and
                    len(next_node['inputs']) == 1 and
                    next_node['inputs'][0]['name'] == out_name):

                    fused = node.copy()
                    fused['attributes'] = dict(node.get('attributes', {}))
                    fused['attributes']['activation'] = next_node['op_type']
                    fused['outputs'] = next_node['outputs']

                    first_idx = node['_orig_idx']
                    replacements[first_idx] = fused

                    to_remove.add(first_idx)
                    to_remove.add(next_node['_orig_idx'])
                    continue

    new_nodes = []
    for idx, node in enumerate(nodes):
        if idx in to_remove:
            # 如果这是融合子图的第一个节点，就放 fused node
            if idx in replacements:
                new_nodes.append(replacements[idx])
            # 否则跳过（不添加）
        else:
            # 未被融合的节点，直接保留
            new_nodes.append(node)

    for node in new_nodes:
        node.pop('_orig_idx', None)

    vk_model.nodes = new_nodes


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


def merge_initializers(vk_model):
    """
    Merge batch normalization parameters into a single tensor [4, N]
    where N is the number of channels.

    Layout:
    Row 0: scale/weight (default: 1.0)
    Row 1: bias (default: 0.0)
    Row 2: mean (required)
    Row 3: variance (required)

    This modifies the vk_model by:
    1. Finding batch normalization nodes
    2. Merging their 4 input parameters into one
    3. Updating the nodes and initializers accordingly
    """

    # Find all batch normalization nodes
    bn_nodes = []
    for i, node in enumerate(vk_model.nodes):
        if node['op_type'] == 'BatchNormalization':
            bn_nodes.append((i, node))

    # Keep track of which initializers have been merged
    merged_initializers = set()

    for idx, node in bn_nodes:
        # BatchNormalization typically has 5 inputs:
        # input, scale, bias, mean, variance
        if len(node['inputs']) < 5:
            print(f"Warning: BatchNormalization node {node['name']} has less than 5 inputs")
            continue

        # Get the names of the parameters
        # input[0] is the actual input data
        # inputs[1-4] are scale, bias, mean, variance respectively
        input_data = node['inputs'][0]    # input data
        scale_name = node['inputs'][1]['name']  # scale/weight
        bias_name = node['inputs'][2]['name']   # bias
        mean_name = node['inputs'][3]['name']   # mean
        var_name = node['inputs'][4]['name']    # variance

        # Check if all required parameters exist
        required_params = [mean_name, var_name]
        for param_name in required_params:
            if param_name not in vk_model.initializers:
                print(f"Error: Required parameter {param_name} not found in initializers")
                continue

        # Get the parameter arrays
        try:
            mean_array = numpy_helper.to_array(vk_model.initializers[mean_name])
            var_array = numpy_helper.to_array(vk_model.initializers[var_name])

            # Validate that mean and variance have the same shape
            if mean_array.shape != var_array.shape:
                print(f"Error: Mean {mean_array.shape} and variance {var_array.shape} have different shapes")
                continue

            # Handle optional parameters with defaults
            if scale_name in vk_model.initializers:
                scale_array = numpy_helper.to_array(vk_model.initializers[scale_name])
            else:
                scale_array = np.ones_like(mean_array, dtype=mean_array.dtype)

            if bias_name in vk_model.initializers:
                bias_array = numpy_helper.to_array(vk_model.initializers[bias_name])
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
            merged_name = f"{node['name']}_bn_params"

            # Convert back to ONNX tensor
            merged_tensor = numpy_helper.from_array(merged_data, merged_name)

            # Add the merged tensor to initializers
            vk_model.initializers[merged_name] = merged_tensor
            
            # Update the node to use the merged parameter
            # Change inputs from 5 to 2: [input, merged_params]
            new_inputs = [
                input_data,  # Original input data (index 0)
                {'name': merged_name, 'shape': [4 * padded_N]}  # Merged parameters
            ]

            node['inputs'] = new_inputs

            # Mark original initializers for removal
            merged_initializers.update([scale_name, bias_name, mean_name, var_name])

            print(f"Merged batchnorm for {node['name']}: "
                  f"scale({scale_name}), bias({bias_name}), mean({mean_name}), var({var_name}) "
                  f"-> merged({merged_name})")

        except Exception as e:
            print(f"Error processing BatchNormalization node {node['name']}: {e}")
            continue

    # Remove the original individual initializers
    for initializer_name in merged_initializers:
        if initializer_name in vk_model.initializers:
            del vk_model.initializers[initializer_name]

    print(f"Merged {len(bn_nodes)} BatchNormalization nodes, removed {len(merged_initializers)} initializers")


def convert_flat_to_reshape(vk_model):
    """
    Convert Flat nodes to Reshape nodes with explicit shapes.

    Flatten operation flattens the input tensor into a 2D tensor, keeping dimensions
    up to axis-1 and flattening the rest into the second dimension.
    """
    nodes = vk_model.nodes
    new_nodes = []

    for node in nodes:
        if node['op_type'] == 'Flatten':
            # Get input shape
            if len(node['inputs']) > 0 and len(node['inputs'][0]['shape']) > 0:
                input_shape = node['inputs'][0]['shape']
                print(f"Flatten node {node['name']} input shape: {input_shape}")

                # Get axis attribute (default is 1 according to ONNX spec)
                axis = node['attributes'].get('axis', 1)

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
                reshape_node = {
                    'op_type': 'Reshape',
                    'name': node['name'],
                    'attributes': {},
                    'inputs': node['inputs'][:],  # Copy original inputs
                    'outputs': node['outputs'][:]  # Copy original outputs
                }

                # Add shape tensor as second input
                shape_tensor_name = node['name'] + '_shape'
                shape_tensor = np.array(output_shape, dtype=np.int64)
                shape_initializer = numpy_helper.from_array(shape_tensor, shape_tensor_name)
                vk_model.initializers[shape_tensor_name] = shape_initializer

                # Add the shape tensor as the second input to reshape
                reshape_node['inputs'].append({
                    'name': shape_tensor_name,
                    'shape': list(shape_tensor.shape)
                })

                new_nodes.append(reshape_node)
                print(f"Converted Flatten node '{node['name']}' to Reshape with shape {output_shape} (axis={axis})")
            else:
                # If we can't determine the shape, keep the original node
                new_nodes.append(node)
                print(f"Warning: Could not convert Flatten node '{node['name']}' - missing shape info")
        else:
            new_nodes.append(node)

    vk_model.nodes = new_nodes


def remove_redundant_reshape(vk_model):
    """
    Remove redundant reshape nodes where input and output shapes are the same.
    Updates connections so that the reshape's input becomes the next node's input.
    Also cleans up unused initializers.
    """
    nodes = vk_model.nodes
    # Build mapping of output names to producing nodes
    producer = {}
    for node in nodes:
        for out in node['outputs']:
            producer[out['name']] = node

    # Track which nodes to remove
    to_remove = set()
    # Map from reshape output names to their input names
    reshape_remap = {}
    # Track initializers used by redundant reshapes
    redundant_initializer_names = set()

    # First pass: identify redundant reshapes and build remapping
    for idx, node in enumerate(nodes):
        if node['op_type'] == 'Reshape':
            # Check if input and output shapes are the same
            if (len(node['inputs']) >= 1 and len(node['outputs']) >= 1 and
                node['inputs'][0]['shape'] == node['outputs'][0]['shape']):

                # This is a redundant reshape node
                input_name = node['inputs'][0]['name']
                output_name = node['outputs'][0]['name']

                # Record the mapping for remapping
                reshape_remap[output_name] = input_name
                # Mark this reshape node for removal
                to_remove.add(idx)

                # Collect initializers used by this reshape node (typically the shape tensor)
                for inp in node['inputs'][1:]:  # Skip the first input (data), consider the shape input
                    if inp['name'] in vk_model.initializers:
                        redundant_initializer_names.add(inp['name'])

                print(f"Identified redundant Reshape node: {node['name']}")

    # Check if the collected initializers are used by any other nodes
    # If not, they should be removed
    initializers_to_remove = set()
    if redundant_initializer_names:
        # Build a set of all tensor names used by all nodes (except the ones we're removing)
        used_tensors = set()
        for idx, node in enumerate(nodes):
            if idx not in to_remove:  # Skip nodes we're going to remove
                for inp in node['inputs']:
                    used_tensors.add(inp['name'])
                for out in node['outputs']:
                    used_tensors.add(out['name'])

        # Check if any of our redundant initializers are actually used elsewhere
        for initializer_name in redundant_initializer_names:
            if initializer_name not in used_tensors:
                initializers_to_remove.add(initializer_name)

    # Second pass: update all nodes that reference the removed reshape outputs
    for node in nodes:
        if node['name'] in [nodes[i]['name'] for i in to_remove]:
            continue  # Skip the nodes we're removing

        # Update inputs that reference removed reshape outputs
        for inp in node['inputs']:
            if inp['name'] in reshape_remap:
                old_name = inp['name']
                inp['name'] = reshape_remap[old_name]
                # Also update the shape if needed (should be the same)
                # Find the source node/input to get the correct shape
                print(f"Remapped input {old_name} to {inp['name']} in node {node['name']}")

    # Remove the marked nodes
    if to_remove:
        vk_model.nodes = [node for idx, node in enumerate(nodes) if idx not in to_remove]
        print(f"Removed {len(to_remove)} redundant reshape nodes")

    # Remove unused initializers
    if initializers_to_remove:
        for initializer_name in initializers_to_remove:
            if initializer_name in vk_model.initializers:
                del vk_model.initializers[initializer_name]
        print(f"Removed {len(initializers_to_remove)} unused initializers: {initializers_to_remove}")


def quantize_to_fp16_selective(vk_model):
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
    for node in vk_model.nodes:
        for inp in node['inputs']:
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

    for name, initializer in vk_model.initializers.items():
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
        consumer_ops = {node['op_type'] for node in consumers}

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
        # if len(arr.shape) == 1:  # Common bias tensor characteristics
        #     # Check if it's connected to a bias input in operations like Conv, Gemm, etc.
        #     is_bias = False
        #     for node in consumers:
        #         op_type = node['op_type']
        #         if op_type in ['Conv', 'ConvTranspose']:
        #             # Check if this tensor is the bias input (usually the 3rd input for Conv, 2nd for Gemm)
        #             for idx, inp in enumerate(node['inputs']):
        #                 if inp['name'] == name:
        #                     # For Conv: 0=inputs, 1=weights, 2=bias
        #                     if (op_type == 'Conv' and idx == 2) or \
        #                        (op_type == 'ConvTranspose' and idx == 2):
        #                         is_bias = True
        #                         break
        #     if is_bias:
        #         print(f"Preserving bias tensor '{name}' as FP32")
        #         skipped_count += 1
        #         should_quantize = False
        if should_quantize:
            # Convert to numpy array
            arr = numpy_helper.to_array(initializer)

            # Convert to FP16
            arr_fp16 = arr.astype(np.float16)

            # Convert back to ONNX tensor with FP16 data type
            fp16_initializer = numpy_helper.from_array(arr_fp16, name)

            # Update the initializer in the model
            vk_model.initializers[name] = fp16_initializer
            print(f"Converted FP32 tensor '{name}' to FP16 ({reason})")
            print(f"New shape: {fp16_initializer.dims}")
            converted_count += 1
        else:
            skipped_count += 1

    print(f"Converted {converted_count} FP32 tensors to FP16")
    print(f"Preserved {skipped_count} tensors")

def quantize_to_int8_weight_only(vk_model):
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
    for node in vk_model.nodes:
        for inp in node['inputs']:
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
    initializers_keys = list(vk_model.initializers.keys())
    
    for name in initializers_keys:
        initializer = vk_model.initializers[name]
        
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
        consumer_ops = {node['op_type'] for node in consumers}

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
                op_type = node['op_type']
                if op_type in ['Conv', 'ConvTranspose']:
                    # Check if this tensor is the bias input (usually the 3rd input for Conv, 2nd for Gemm)
                    for idx, inp in enumerate(node['inputs']):
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
                for op in consumer_ops:
                    if op == 'Conv':
                        # Conv weights: [C_out, C_in, K, K] - quantize per output channel
                        # Reduce along (C_in, K, K) dimensions -> axis=(1, 2, 3)
                        if len(arr.shape) == 4:
                            axis = (1, 2, 3)
                        elif len(arr.shape) == 3:
                            axis = (1, 2)
                        else:
                            axis = 0
                        break
                    elif op == 'ConvTranspose':
                        # ConvTranspose weights: [C_in, C_out, K, K] - quantize per output channel
                        # Reduce along (C_in, K, K) dimensions -> axis=(0, 2, 3)
                        if len(arr.shape) == 4:
                            axis = (0, 2, 3)
                        else:
                            # For other shapes, default to axis=1
                            axis = 1
                        break
                    elif op == 'Gemm':
                        # Gemm weights: [out, in] - quantize per output dimension
                        # Reduce along in dimension -> axis=1
                        if len(arr.shape) == 2:
                            axis = 1
                        else:
                            # For other shapes, default to axis=0
                            axis = 0
                        break
                    elif op == 'MatMul':
                        # MatMul weights: typically [in, out] - quantize per output dimension
                        # Reduce along in dimension -> axis=0
                        if len(arr.shape) == 2:
                            axis = 0
                        else:
                            # For other shapes, default to axis=1
                            axis = 1
                        break
                    elif op in ['LSTM', 'GRU']:
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
            vk_model.initializers[name] = int8_initializer

            # Determine if scale should be stored as attribute or as input
            # If scale is a scalar or small array, store as attribute; otherwise, store as input
            scale_size = scale.size if hasattr(scale, 'size') else 1
            scale_is_scalar = np.isscalar(scale) or scale_size == 1

            if scale_is_scalar:
                # For scalar scales, store as attribute
                for node in consumers:
                    if 'attributes' not in node:
                        node['attributes'] = {}

                    # Add scale as attribute to the node
                    scale_value = float(scale) if hasattr(scale, 'item') else float(scale)
            else:
                # For array scales, create initializer and add as input
                scale_name = f"{name}_scale"
                scale_initializer = numpy_helper.from_array(scale, scale_name)
                scale_initializer.data_type = onnx.TensorProto.FLOAT
                vk_model.initializers[scale_name] = scale_initializer

                # Add scale as input to the nodes that consume the original initializer
                for node in consumers:
                    # Add scale tensor as an additional input
                    scale_input = {
                        'name': scale_name,
                        'shape': list(scale.shape) if hasattr(scale, 'shape') else []
                    }
                    node['inputs'].append(scale_input)

            print(f"Converted FP32 tensor '{name}' to INT8 with scale tensor '{scale_name}' ({reason})")
            print(f"Original shape: {initializer.dims}, scale shape: {scale_initializer.dims}")
            converted_count += 1
        else:
            skipped_count += 1

    print(f"Converted {converted_count} FP32 tensors to INT8 with scale information")
    print(f"Preserved {skipped_count} tensors")
    print(f"Total initializers after quantization: {len(vk_model.initializers)}")


def move_input_tensor_to_attr(vk_model):
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

    for node in vk_model.nodes:
        op_type = node['op_type']
        if op_type not in SPECIAL_OPS:
            continue

        target_inputs = SPECIAL_OPS[op_type]

        for idx, attr_name in target_inputs:
            if idx >= len(node['inputs']):
                continue

            input_tensor = node['inputs'][idx]
            tensor_name = input_tensor['name']
            print("Checking input tensor: ", tensor_name)

            if not tensor_name or tensor_name not in vk_model.initializers:
                continue

            initializer = vk_model.initializers[tensor_name]

            # 检查是否为一维数组且长度较短（通常是rank长度，一般不超过8）
            if len(initializer.dims) == 1 and 0 < initializer.dims[0] <= 8:
                tensor_data = numpy_helper.to_array(initializer)

                if 'attributes' not in node:
                    node['attributes'] = {}

                if tensor_data.dtype in [np.float32, np.float64]:
                    node['attributes'][attr_name] = tensor_data.tolist()
                elif tensor_data.dtype in [np.int32, np.int64]:
                    node['attributes'][attr_name] = tensor_data.tolist()
                else:
                    node['attributes'][attr_name] = tensor_data.tolist()

                del node['inputs'][idx]

                initializers_to_remove.add(tensor_name)
        print(node)

    for initializer_name in initializers_to_remove:
        if initializer_name in vk_model.initializers:
            del vk_model.initializers[initializer_name]

    print(f"Converted {len(initializers_to_remove)} tensor inputs to attributes")

def unsqueeze_initializers(vk_model):
    """Optimize unsqueeze operations on initializers by pre-computing the result"""
    print("Unsqueezing initializers...")

    nodes_to_remove = []
    initializers_to_add = []
    initializers_to_remove = []

    # 遍历所有节点查找Unsqueeze操作
    for node in vk_model.nodes:
        if node['op_type'] == 'Unsqueeze':
            # 检查输入是否为initializer
            input_name = node.input[0]
            initializer = None

            # 查找对应的initializer
            for init in vk_model.initializers:
                if init.name == input_name:
                    initializer = init
                    break

            if initializer is not None:
                axes = None
                for attr in node['attribute']:
                    if attr['name'] == 'axes':
                        axes = list(attr.ints)
                        break

                if axes is None and len(node.input) > 1:
                    axes_input_name = node.input[1]
                    for init in vk_model.initializer:
                        if init.name == axes_input_name:
                            if init.data_type == 7:  # INT64
                                axes = list(init.int64_data)
                            elif init.data_type == 6:  # INT32
                                axes = list(init.int32_data)
                            break

                if axes is not None:
                    original_shape = list(initializer.dims)
                    new_shape = original_shape[:]
                    axes_sorted = sorted([int(axis) if axis >= 0 else axis + len(original_shape) + 1 for axis in axes], reverse=True)

                    for axis in axes_sorted:
                        actual_axis = axis if axis >= 0 else axis + len(new_shape) + 1
                        new_shape.insert(actual_axis, 1)

                    import copy
                    new_initializer = copy.deepcopy(initializer)
                    new_initializer.name = node.output[0]
                    new_initializer.dims[:] = new_shape

                    initializers_to_add.append(new_initializer)
                    initializers_to_remove.append(initializer.name)
                    nodes_to_remove.append(node)

                    print(f"Pre-computed unsqueeze for initializer '{input_name}' with axes {axes}")

    for node in nodes_to_remove:
        vk_model.nodes.remove(node)

    for init_name in initializers_to_remove:
        for i, init in enumerate(vk_model.initializers):
            if init.name == init_name:
                vk_model.initializers.remove(init)
                break

    for init in initializers_to_add:
        vk_model.initializers.append(init)

    if nodes_to_remove:
        print(f"Removed {len(nodes_to_remove)} unsqueeze nodes that operated on initializers")

    return vk_model


class ModelConverter:
    """Main class for converting ONNX models to VKOP format."""
    @staticmethod
    def parse_onnx_model(onnx_path, batch_size):
        model = onnx.load(onnx_path)

        # Optimize the ONNX model
        print("Optimizing ONNX model...")
        model = optimize_onnx_model(model, batch_size)
        # save optimized model
        onnx.save(model, "optimized_" + os.path.basename(onnx_path))

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

        # Initializers (parameters)
        for initializer in graph.initializer:
            name = initializer.name
            vk_model.initializers[name] = initializer

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
                                orig_init = vk_model.initializers[initializer.name]
                                orig_array = numpy_helper.to_array(orig_init)
                                if orig_array.size == 1:
                                    scalar_val = orig_array.item()
                                    expanded_data = np.full(shape, scalar_val, dtype=orig_array.dtype)
                                else:
                                    expanded_data = np.broadcast_to(orig_array, shape).copy()
                                new_init = numpy_helper.from_array(expanded_data, name=new_name)
                                vk_model.initializers[initializer.name] = new_init
                    if len(inputs_with_shape[id1]['shape']) != len(shape):
                        inputs_with_shape[id1]['shape'] = shape
                        for initializer in graph.initializer:
                            if initializer.name == inputs_with_shape[id1]['name']:
                                orig_init = vk_model.initializers[initializer.name]
                                orig_array = numpy_helper.to_array(orig_init)
                                if orig_array.size == 1:
                                    scalar_val = orig_array.item()
                                    expanded_data = np.full(shape, scalar_val, dtype=orig_array.dtype)
                                else:
                                    expanded_data = np.broadcast_to(orig_array, shape).copy()
                                new_init = numpy_helper.from_array(expanded_data, name=initializer.name)
                                vk_model.initializers[initializer.name] = new_init

            vk_model.nodes.append({
                'op_type': node.op_type,
                'name': node.name,
                'attributes': attributes,
                'inputs': inputs_with_shape,
                'outputs': outputs_with_shape
            })

        return vk_model


def unified_initializers(vk_model):
    """
    Consolidate 1D and 2D tensors from initializers into a large memory block with 16-byte padding.
    This function:
    1. Identifies 1D and 2D tensors in initializers
    2. Concatenates them into a single memory block with 16-byte alignment
    3. Keeps track of shapes and offsets for each tensor
    """
    print("Consolidating 1D and 2D initializers...")

    initializer_consumers = defaultdict(list)
    for node in vk_model.nodes:
        for inp in node['inputs']:
            initializer_consumers[inp['name']].append(node)

    # Identify 1D and 2D tensors
    tensors_to_consolidate = []
    tensor_info = []  # Store tensor name, shape, data, and offset

    for name, initializer in vk_model.initializers.items():
        dims = list(initializer.dims)
        if len(dims) in [1, 2]:  # Only 1D or 2D tensors
            consumers = initializer_consumers.get(name, [])
            consumer_ops = {node['op_type'] for node in consumers}

            # Only process if it's consumed by Conv or BatchNormalization
            if any(op in ['Conv', 'BatchNormalization'] for op in consumer_ops):
                tensor_data = numpy_helper.to_array(initializer)
                tensors_to_consolidate.append((name, dims, initializer.data_type, tensor_data))
            else:
                print(f"Skipping tensor {name} as it's not consumed by Conv or BatchNormalization")

    if not tensors_to_consolidate:
        print("No 1D or 2D tensors found to consolidate.")
        return

    print(f"Found {len(tensors_to_consolidate)} 1D and 2D tensors to consolidate.")

    # Calculate total size with 16-byte padding
    current_offset = 0
    unified_data = b""

    for name, shape, datatype, data in tensors_to_consolidate:
        # Get the size in bytes for this tensor
        element_size = data.itemsize
        tensor_size_bytes = data.nbytes

        # Calculate padding to align to 16-byte boundary,
        # Min TexelBuffer Alignment requirement, usually 16 bytes
        padding_needed = (16 - (current_offset % 16)) % 16
        if padding_needed > 0:
            unified_data += b"\x00" * padding_needed
            current_offset += padding_needed

        # Record tensor info with offset
        tensor_info.append({
            'name': name,
            'shape': shape,
            'offset': current_offset,
            'size_bytes': tensor_size_bytes,
            'type' : datatype
        })
        print(f"Tensor {name} shape: {shape}, size: {tensor_size_bytes} bytes, offset: {current_offset}")

        # Add the tensor data
        tensor_bytes = data.tobytes()
        unified_data += tensor_bytes
        current_offset += tensor_size_bytes

    # Create a new unified initializer
    unified_name = "unified_tensors"
    unified_initializer = numpy_helper.from_array(
        np.frombuffer(unified_data, dtype=np.float32), 
        unified_name
    )
    unified_initializer.data_type = onnx.TensorProto.FLOAT

    # Update the model: remove old tensors and add unified one
    for name, _, _, _ in tensors_to_consolidate:
        del vk_model.initializers[name]

    vk_model.initializers[unified_name] = unified_initializer

    # Store metadata about the consolidation in model attributes
    # We'll create a metadata tensor with shape and offset information
    metadata_list = []
    for idx, info in enumerate(tensor_info):
        # Add name length, offset, and size_bytes as integers
        name_length = len(info['name'])

        # Add to metadata list
        metadata_list.append(info['type'])
        metadata_list.append(name_length)
        metadata_list.append(info['offset'])
        metadata_list.append(info['size_bytes'])

        # Add shape dimensions, padding to 4 dimensions with 1s
        shape_dims = [1] * (2 - len(info['shape'])) + info['shape']
        pad_dims = [0] * 2
        metadata_list.extend(shape_dims)
        metadata_list.extend(pad_dims)

    metadata_array = np.array(metadata_list, dtype=np.int32)
    metadata_initializer = numpy_helper.from_array(metadata_array, "unified_metadata")
    vk_model.initializers["unified_metadata"] = metadata_initializer

    all_names_bytes = b""
    for info in tensor_info:
        name_bytes = info['name'].encode('utf-8')
        all_names_bytes += name_bytes

    names_array = np.frombuffer(all_names_bytes, dtype=np.uint8)
    names_initializer = numpy_helper.from_array(names_array, "unified_names")
    vk_model.initializers["unified_names"] = names_initializer

    print(f"Consolidated {len(tensors_to_consolidate)} tensors into a single tensor of size {len(unified_data)} bytes")
    print(f"Added metadata tensor with shape information for {len(tensor_info)} tensors")

    return {
        'tensor_info': tensor_info,
        'total_size': len(unified_data),
        'metadata_size': len(metadata_list)
    }


def convert_initializers_nchw_to_rgba(vk_model):
    """
    Convert 3D and 4D tensors in initializers from NCHW format to RGBA format
    following the same logic as the C++ convertTensorToRGBA function in Tensor.hpp
    First, reshape all tensors to NCHW 4D format, then convert to RGBA.
    Also preserve original shape information in metadata using RGBAConversion struct format.
    """
    print("Converting 3D and 4D initializers from NCHW to RGBA format...")

    initializer_consumers = defaultdict(list)
    for node in vk_model.nodes:
        for inp in node['inputs']:
            initializer_consumers[inp['name']].append(node)

    converted_count = 0
    shape_metadata = []
    name_list = []

    for name, initializer in vk_model.initializers.items():
        dims = list(initializer.dims)

        # Only process 3D and 4D tensors
        if len(dims) in [3, 4]:
            tensor_data = numpy_helper.to_array(initializer)

            consumers = initializer_consumers.get(name, [])
            consumer_ops = {node['op_type'] for node in consumers}
            transpose = 'Conv' in consumer_ops

            if len(dims) == 4:
                orig_n, orig_c, h, w = dims
            else:  # 3D: [C, H, W] → treat as [1, C, H, W]
                orig_c, h, w = dims
                orig_n = 1
                tensor_data = tensor_data.reshape(orig_n, orig_c, h, w)

            # Determine logical batch and channel based on transpose
            if transpose:
                batch = orig_c      # logical batch = C
                channel = orig_n    # logical channel = N
            else:
                batch = orig_n      # logical batch = N
                channel = orig_c    # logical channel = C

            c4 = (channel + 3) // 4  # UP_DIV equivalent for channels
            pack = (h == 1 and w == 1)

            total_elements = w * h * batch * c4 * 4
            rgba_flat = np.zeros(total_elements, dtype=tensor_data.dtype)

            row_pitch = w * 4
            layer_stride = batch * h * w * 4

            # Perform the conversion following the C++ logic
            for c4_idx in range(c4):  # channel group index
                for n_idx in range(batch):  # batch index
                    for h_idx in range(h):  # height index
                        for w_idx in range(w):  # width index
                            if pack:
                                # ((n*H + h) * (W * C4) + (w + c4 * W)) * 4
                                base_linear = ((n_idx * h + h_idx) * (w * c4) + (w_idx + c4_idx * w)) * 4
                            else:
                                # c4 * layer_stride + (n*H + h) * row_pitch + w * 4
                                base_linear = c4_idx * layer_stride + (n_idx * h + h_idx) * row_pitch + w_idx * 4

                            # Now fill RGBA channels (k=0..3)
                            for k in range(4):
                                ch = c4_idx * 4 + k
                                if ch < channel:
                                    if transpose:
                                        src_val = tensor_data[ch, n_idx, h_idx, w_idx]
                                    else:
                                        src_val = tensor_data[n_idx, ch, h_idx, w_idx]
                                else:
                                     src_val = 0

                                rgba_flat[base_linear + k] = src_val

            new_initializer = numpy_helper.from_array(rgba_flat, name)
            new_initializer.data_type = initializer.data_type
            vk_model.initializers[name] = new_initializer

            dtype = initializer.data_type  # ONNX data type
            name_len = len(name)
            offset = 0
            size = rgba_flat.size
            padded_dims = [1] * (4 - len(dims)) + dims

            shape_metadata.append([dtype, name_len, offset, size] + padded_dims)
            name_bytes = name.encode('utf-8')
            name_list.append(name_bytes)

            converted_count += 1
            print(f"Converted {name} from shape {dims} to RGBA format with shape {rgba_flat.shape}")

    # Store shape metadata as a new initializer using RGBAConversion format
    if shape_metadata:
        # Flatten the metadata list
        flattened_metadata = []

        for entry in shape_metadata:
            flattened_metadata.extend(entry)

        # Create metadata array
        metadata_array = np.array(flattened_metadata, dtype=np.int32)
        metadata_initializer = numpy_helper.from_array(metadata_array, "rgba_conversion_metadata")
        vk_model.initializers["rgba_conversion_metadata"] = metadata_initializer

        all_names_bytes = b"".join(name_list)
        names_array = np.frombuffer(all_names_bytes, dtype=np.uint8)
        names_initializer = numpy_helper.from_array(names_array, "rgba_conversion_names")
        vk_model.initializers["rgba_conversion_names"] = names_initializer

    print(f"Converted {converted_count} initializers from NCHW to RGBA format")
    print(f"Preserved original shape information in metadata using RGBAConversion format")
    return converted_count


def replace_globalaveragepool_conv_with_gemm(vk_model):
    """
    Replace Conv that follows GlobalAveragePool with GEMM when conditions are met.
    This optimization works when:
    1. GlobalAveragePool output shape is [N, C, 1, 1] (after GAP)
    2. Conv has kernel size 1x1 and stride 1x1
    3. Conv has no padding or padding that maintains the 1x1 output
    4. The operations can be equivalently represented as a matrix multiplication
    """
    print("Replacing Conv after GlobalAveragePool with GEMM...")

    producer, consumers = build_graph_index(vk_model.nodes)

    nodes = vk_model.nodes
    for idx, node in enumerate(nodes):
        node['_orig_idx'] = idx

    to_remove: Set[int] = set()
    replacements: Dict[int, Dict] = {}

    for node in nodes:
        # Check if it's a Conv node
        if node['op_type'] != 'Conv':
            continue

        conv_input_name = node['inputs'][0]['name']

        # Find the producer of this Conv's input
        if conv_input_name not in producer:
            continue

        gap_node = producer[conv_input_name]
        # Check if the producer is a GlobalAveragePool
        if gap_node['op_type'] != 'GlobalAveragePool':
            continue

        # Check if Conv has kernel size 1x1 (typical for 1x1 convolutions after GAP)
        kernel_shape = node['attributes'].get('kernel_shape', [1, 1])
        strides = node['attributes'].get('strides', [1, 1])
        pads = node['attributes'].get('pads', [0, 0, 0, 0])  # [pad_top, pad_left, pad_bottom, pad_right]

        # Only proceed if kernel is 1x1 and stride is 1x1 (common case after GAP)
        if kernel_shape == [1, 1] and strides == [1, 1]:
            # Verify that padding doesn't change the 1x1 nature of the operation
            if pads == [0, 0, 0, 0] or (pads[0] == pads[2] and pads[1] == pads[3] and pads[0] <= 1 and pads[1] <= 1):
                print(f"Found Conv '{node['name']}' after GlobalAveragePool '{gap_node['name']}' with 1x1 kernel, replacing with GEMM")

                weight_input = node['inputs'][1]
                weight_name = weight_input['name']

                if weight_name in vk_model.initializers:
                    # Get the original weight tensor
                    weight_tensor = vk_model.initializers[weight_name]
                    original_shape = list(weight_tensor.dims)

                    # For 1x1 conv, weight shape is [out_channels, in_channels, 1, 1]
                    # For GEMM, we need [out_channels, in_channels]
                    if len(original_shape) == 4 and original_shape[2] == 1 and original_shape[3] == 1:
                        new_shape = [original_shape[0], original_shape[1]]  # [out_channels, in_channels]

                        # Get the weight data and reshape it
                        weight_data = numpy_helper.to_array(weight_tensor)
                        reshaped_weight = weight_data.reshape(new_shape)

                        # Create a new initializer with the reshaped data
                        new_weight_tensor = numpy_helper.from_array(reshaped_weight, weight_name)
                        new_weight_tensor.data_type = weight_tensor.data_type

                        # Update the initializer
                        vk_model.initializers[weight_name] = new_weight_tensor
                        print(f"Reshaped weight '{weight_name}' from {original_shape} to {new_shape}")

                # Create GEMM node
                gemm_node = {
                    'op_type': 'Gemm',
                    'name': f"Gemm_after_GAP_{node['name']}",
                    'attributes': {
                        'alpha': 1.0,
                        'beta': 1.0,
                        'transA': 0,  # Don't transpose A
                        'transB': 0   # Don't transpose B
                    },
                    'inputs': [
                        gap_node['outputs'][0],  # Output from GlobalAveragePool (becomes matrix A in GEMM)
                        {
                            'name': weight_input['name'],
                            'shape': weight_input['shape'][:2]
                        }  # Conv weights (becomes matrix B in GEMM)
                    ],
                    'outputs': []
                }

                # If Conv has bias, add it as the third input to GEMM
                if len(node['inputs']) > 2:
                    gemm_node['inputs'].append(node['inputs'][2])  # Conv bias

                for output in node['outputs']:
                    # Original shape from Conv
                    orig_shape = output['shape']

                    # If the shape ends with [1, 1] (H, W), remove them to keep only [N, C]
                    new_shape = orig_shape[:]
                    if len(new_shape) >= 2 and new_shape[-2:] == [1, 1]:
                        new_shape = new_shape[:-2]  # Remove the last two dimensions (H, W)

                    gemm_node['outputs'].append({
                        'name': output['name'],
                        'shape': new_shape
                    })

                    # Also update the model's outputs if this output is in the global outputs
                    for model_output in vk_model.outputs:
                        if model_output['name'] == output['name']:
                            model_output['shape'] = new_shape
                            print(f"Updated model output '{output['name']}' shape to {new_shape}")

                # Find the index of the Conv node to replace it
                conv_idx = node['_orig_idx']

                # Mark the Conv node for removal
                to_remove.add(conv_idx)

                # Add the new GEMM node
                replacements[conv_idx] = gemm_node

    # Build new nodes list
    new_nodes = []
    for idx, node in enumerate(nodes):
        if idx in to_remove:
            # Replace the Conv node with GEMM
            if idx in replacements:
                new_nodes.append(replacements[idx])
        else:
            new_nodes.append(node)

    for node in new_nodes:
        if '_orig_idx' in node:
            del node['_orig_idx']

    replaced_count = len([idx for idx in to_remove if idx in replacements])
    print(f"Replaced {replaced_count} Conv nodes after GlobalAveragePool with GEMM nodes")
    vk_model.nodes = new_nodes
    return vk_model


def replace_reducemean_reshape_with_globalaveragepool(vk_model):
    """
    Replace ReduceMean + Reshape pattern with GlobalAveragePool when conditions are met.
    This optimization works when:
    1. ReduceMean operates on HW dimensions (axes=[2, 3] or [-2, -1] for NCHW format)
    2. Either:
       - Reshape removes the trailing 1x1 dimensions (when keepdims=1), OR
       - No reshape needed because keepdims=0 produces desired output shape
    3. The operations can be equivalently represented as a GlobalAveragePool
    
    This is functionally equivalent to GlobalAveragePool for NCHW format.
    Handles both opset13 (axes in attributes) and opset18+ (axes in inputs) formats.
    """
    print("Replacing ReduceMean + Reshape with GlobalAveragePool...")

    producer, consumers = build_graph_index(vk_model.nodes)

    nodes = vk_model.nodes
    for idx, node in enumerate(nodes):
        node['_orig_idx'] = idx

    to_remove: Set[int] = set()
    replacements: Dict[int, Dict] = {}

    for node in nodes:
        # Check if it's a ReduceMean node
        if node['op_type'] != 'ReduceMean':
            continue

        reducemean_output_name = node['outputs'][0]['name']

        # Get axes - could be in attributes (opset13) or inputs (opset18+)
        axes = []
        keepdims = node['attributes'].get('keepdims', 1)  # Default is 1 in ONNX
        noop_with_empty_axes = node['attributes'].get('noop_with_empty_axes', 0)  # Default is 0 in ONNX

        # Check if axes is in attributes (opset13 style)
        if 'axes' in node['attributes']:
            axes = node['attributes']['axes']
        else:
            # Check if axes is in inputs (opset18+ style)
            # Axes would be the second input (after the data input)
            if len(node['inputs']) > 1:
                axes_input_name = node['inputs'][1]['name']
                if axes_input_name in vk_model.initializers:
                    axes_tensor = vk_model.initializers[axes_input_name]
                    axes_array = numpy_helper.to_array(axes_tensor)
                    axes = axes_array.tolist() if hasattr(axes_array, 'tolist') else list(axes_array)

        # Handle empty axes condition based on noop_with_empty_axes
        if len(axes) == 0 and noop_with_empty_axes == 1:
            # If noop_with_empty_axes is true and axes is empty, this performs no reduction
            continue
        elif len(axes) == 0 and noop_with_empty_axes == 0:
            # If noop_with_empty_axes is false and axes is empty, this reduces all axes
            continue  # This case is not relevant for our optimization

        # Normalize negative axes to positive (for 4D tensors, -1 becomes 3, -2 becomes 2)
        normalized_axes = []
        for ax in axes:
            if ax < 0:
                # Determine tensor rank from input shape to normalize axes properly
                input_shape = node['inputs'][0]['shape']
                normalized_axes.append(len(input_shape) + ax)
            else:
                normalized_axes.append(ax)
        
        # For NCHW format, axes [2, 3] or [-2, -1] correspond to H and W dimensions
        if sorted(normalized_axes) == [2, 3]:
            # Check if the reshape removes the 1x1 dimensions (i.e., changes [..., 1, 1] to [...])
            input_shape = node['inputs'][0]['shape']  # Original input shape before ReduceMean
            output_after_reduce = node['outputs'][0]['shape']  # Shape after ReduceMean

            if keepdims == 1:
                reducemean_consumers = consumers.get(reducemean_output_name, [])
                if len(reducemean_consumers) != 1:
                    continue
                reshape_node, _ = reducemean_consumers[0]
                # Check if consumer is a Reshape node
                if reshape_node['op_type'] != 'Reshape':
                    continue

                output_after_reshape = reshape_node['outputs'][0]['shape']

                # After ReduceMean with keepdims=1, the shape should be [N, C, 1, 1]
                # After Reshape, it should remove these trailing 1x1 dimensions to [N, C]
                if (len(output_after_reduce) == 4 and 
                    output_after_reduce[2] == 1 and output_after_reduce[3] == 1 and
                    len(output_after_reshape) == len(input_shape) - 2 and  # Removed 2 dimensions
                    output_after_reshape[:2] == output_after_reduce[:2]):  # First two dims (N, C) match

                    print(f"Found ReduceMean '{node['name']}' with axes {axes} (normalized to {normalized_axes}) followed by Reshape '{reshape_node['name']}', replacing with GlobalAveragePool")

                    # Create GlobalAveragePool node
                    globalavgpool_node = {
                        'op_type': 'GlobalAveragePool',
                        'name': f"GlobalAveragePool_from_{node['name']}_to_{reshape_node['name']}",
                        'attributes': {},
                        'inputs': [
                            node['inputs'][0]  # Original input to ReduceMean
                        ],
                        'outputs': [{
                            'name': reshape_node['outputs'][0]['name'],  # Output from Reshape
                            'shape': output_after_reshape  # Final shape after reshape
                        }]
                    }

                    # Find the index of the ReduceMean node to replace the sequence
                    reducemean_idx = node['_orig_idx']
                    reshape_idx = reshape_node['_orig_idx']

                    # Mark both nodes for removal
                    to_remove.add(reducemean_idx)
                    to_remove.add(reshape_idx)

                    # Add the new GlobalAveragePool node in place of the first removed node
                    replacements[reducemean_idx] = globalavgpool_node
            elif keepdims == 0:
                # Check if the output shape after ReduceMean is what we expect for GlobalAveragePool
                # Input shape is [N, C, H, W], expected output is [N, C] (with keepdims=0)
                if (len(input_shape) == 4 and
                    len(output_after_reduce) == 2 and
                    input_shape[0] == output_after_reduce[0] and  # Batch dimension matches
                    input_shape[1] == output_after_reduce[1]):   # Channel dimension matches

                    print(f"Found ReduceMean '{node['name']}' with axes {axes} (normalized to {normalized_axes}) and keepdims=0, replacing with GlobalAveragePool")

                    # Create GlobalAveragePool node
                    globalavgpool_node = {
                        'op_type': 'GlobalAveragePool',
                        'name': f"GlobalAveragePool_from_{node['name']}_keepdims0",
                        'attributes': {},
                        'inputs': [
                            node['inputs'][0]  # Original input to ReduceMean
                        ],
                        'outputs': [
                            node['outputs'][0]  # Same output as the ReduceMean node
                        ]
                    }

                    # Find the index of the ReduceMean node to replace
                    reducemean_idx = node['_orig_idx']

                    # Mark the ReduceMean node for removal
                    to_remove.add(reducemean_idx)

                    # Add the new GlobalAveragePool node in place of the ReduceMean
                    replacements[reducemean_idx] = globalavgpool_node
    # Build new nodes list
    new_nodes = []
    for idx, node in enumerate(nodes):
        if idx in to_remove:
            # If this is the first node in the sequence, add the replacement
            if idx in replacements:
                new_nodes.append(replacements[idx])
            # Otherwise skip (second node in sequence)
        else:
            new_nodes.append(node)

    for node in new_nodes:
        if '_orig_idx' in node:
            del node['_orig_idx']

    replaced_count = len([idx for idx in to_remove if idx in replacements])
    print(f"Replaced {replaced_count} ReduceMean+Reshape sequences with GlobalAveragePool nodes")
    vk_model.nodes = new_nodes
    return vk_model


def unify_reduce_operators(vk_model):
    """Unify all reduceXX operators into a single reduce operator with the specific operation stored in attributes."""
    producer, consumers = build_graph_index(vk_model.nodes)

    nodes = vk_model.nodes
    for idx, node in enumerate(nodes):
        node['_orig_idx'] = idx

    to_remove: Set[int] = set()
    replacements: Dict[int, Dict] = {}

    # Define mapping from ONNX reduce operations to internal representation
    reduce_ops_map = {
        'ReduceSum': 'sum',
        'ReduceMean': 'mean',
        'ReduceMax': 'max',
        'ReduceMin': 'min',
        'ReduceProd': 'prod',
        'ReduceSumSquare': 'sum_square',
        'ReduceL1': 'l1_norm',
        'ReduceL2': 'l2_norm',
        'ReduceLogSum': 'log_sum',
        'ReduceLogSumExp': 'log_sum_exp'
    }

    for node in nodes:
        op_type = node['op_type']
        
        # Check if this is a reduce operation
        if op_type not in reduce_ops_map:
            continue

        # Extract axes - could be in attributes (opset13) or inputs (opset18+)
        axes = []
        noop_with_empty_axes = node['attributes'].get('noop_with_empty_axes', 0)  # Default is 0 in ONNX
        
        # Check if axes is in attributes (opset13 style)
        if 'axes' in node['attributes']:
            axes = node['attributes']['axes']
        else:
            # Check if axes is in inputs (opset18+ style)
            # Axes would be the second input (after the data input)
            if len(node['inputs']) > 1:
                axes_input_name = node['inputs'][1]['name']
                if axes_input_name in vk_model.initializers:
                    axes_tensor = vk_model.initializers[axes_input_name]
                    axes_array = numpy_helper.to_array(axes_tensor)
                    axes = axes_array.tolist() if hasattr(axes_array, 'tolist') else list(axes_array)
        print(f"Unifying {op_type} '{node['name']}' with axes {axes} and noop_with_empty_axes={noop_with_empty_axes}")
        # Handle empty axes according to ONNX specification
        if len(axes) == 0:
            if noop_with_empty_axes == 1:
                # If noop_with_empty_axes is true and axes is empty, this performs no reduction
                # We keep the original node as it is a no-op
                original_idx = node['_orig_idx']
                to_remove.add(node['_orig_idx'])
                continue
            elif noop_with_empty_axes == 0:
                # If noop_with_empty_axes is false and axes is empty, this reduces all axes
                # Get input shape to determine all axes
                input_shape = node['inputs'][0]['shape']
                axes = list(range(len(input_shape)))  # Reduce over all dimensions
        
        # Normalize negative axes to positive
        input_shape = node['inputs'][0]['shape']
        normalized_axes = []
        for ax in axes:
            if ax < 0:
                normalized_axes.append(len(input_shape) + ax)
            else:
                normalized_axes.append(ax)

        # Create unified reduce node
        reduce_node = {
            'op_type': 'Reduce',
            'name': node['name'],
            'attributes': {
                'reduce_op': reduce_ops_map[op_type],
                'axes': normalized_axes,
                'keepdims': node['attributes'].get('keepdims', 1),
            },
            'inputs': [
                node['inputs'][0]  # Data input remains the same
            ],
            'outputs': node['outputs'][:]  # Copy outputs
        }

        # Remove axes from inputs if it was there (opset18+)
        # Keep only the data input
        if len(node['inputs']) > 1:
            # Remove the axes input, keeping only the data input
            reduce_node['inputs'] = [node['inputs'][0]]

        # Mark original node for removal and add replacement
        original_idx = node['_orig_idx']
        to_remove.add(original_idx)
        replacements[original_idx] = reduce_node

    # Build new nodes list
    new_nodes = []
    for idx, node in enumerate(nodes):
        if idx in to_remove:
            # If this is a node to be replaced, add the replacement
            if idx in replacements:
                new_nodes.append(replacements[idx])
        else:
            new_nodes.append(node)

    for node in new_nodes:
        if '_orig_idx' in node:
            del node['_orig_idx']

    replaced_count = len([idx for idx in to_remove if idx in replacements])
    print(f"Unified {replaced_count} reduceXX operations into Reduce nodes")
    vk_model.nodes = new_nodes
    return vk_model


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
    vk_model = ModelConverter.parse_onnx_model(args.input, batch_size)

    # unsqueeze_initializers(vk_model) # 不是必要，前序sim等fuse实现了
    move_input_tensor_to_attr(vk_model)
    merge_initializers(vk_model)
    convert_flat_to_reshape(vk_model)
    remove_redundant_reshape(vk_model)
    fuse_conv_bn(vk_model)
    fuse_gated_conv(vk_model)
    fuse_conv_simple_activation(vk_model)
    replace_reducemean_reshape_with_globalaveragepool(vk_model)
    unify_reduce_operators(vk_model)
    replace_globalaveragepool_conv_with_gemm(vk_model)
    if args.quant == "fp16":
        quantize_to_fp16_selective(vk_model)
    elif args.quant == "int8":
        quantize_to_int8_weight_only(vk_model)
    if args.unify is True:
        unified_initializers(vk_model)
    if args.rgba is True:
        convert_initializers_nchw_to_rgba(vk_model)

    op_stats = {}
    for node in vk_model.nodes:
        op_type = node['op_type']
        op_stats[op_type] = op_stats.get(op_type, 0) + 1

    print("\nOperator Statistics:")
    print("idx\ttype\t\tnum")
    for idx, (op_type, count) in enumerate(op_stats.items(), 1):
        print(f"{idx}\t{op_type}\t\t{count}")

    print("Saving to binary format...")
    vk_model.save_to_binary(output_bin_path)

    print("Model saved to:", output_bin_path)
    file_size = os.path.getsize(output_bin_path)
    print(f"File size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()