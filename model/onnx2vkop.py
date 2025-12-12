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
                # 动态维度（如 "N", "batch"）→ 固定为 1
                fixed_shape.append(1)
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

    # 记录要删除的节点
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

            # 检查 conv_out 是否只被这两个节点使用（避免副作用）
            conv_consumers = consumers[conv_out]
            if len(conv_consumers) != 2:
                continue
            consumer_names = {c[0]['op_type'] for c in conv_consumers}
            if not (consumer_names == {'Sigmoid', 'Mul'}):
                continue
            
            fused = conv_node.copy()
            fused['attributes'] = dict(conv_node.get('attributes', {}))
            fused['attributes']['activation'] = 'GateSigmoid'  # 自定义标记
            fused['outputs'] = node['outputs']  # 输出为 Mul 的输出

            first_idx = conv_node['_orig_idx']
            replacements[first_idx] = fused
            to_remove.add(conv_node['_orig_idx'])
            to_remove.add(sig_node['_orig_idx'])
            to_remove.add(node['_orig_idx'])
            matched = True
            break

        if matched:
            continue  # 避免重复处理

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

    print("Fusing gated convs...", len(to_remove))
    vk_model.nodes = new_nodes


def fuse_conv_bn_relu(vk_model):
    """
    Fuse Conv + BatchNormalization + ReLU patterns into a single Conv node.
    This optimization reduces the number of operations and can improve performance.
    """
    producer, consumers = build_graph_index(vk_model.nodes)
    
    nodes = vk_model.nodes
    for idx, node in enumerate(nodes):
        node['_orig_idx'] = idx

    # 记录要删除的节点
    to_remove: Set[int] = set()
    replacements: Dict[int, Dict] = {}

    for node in nodes:
        # 只处理Conv节点
        if node['op_type'] != 'Conv':
            continue

        conv_out = node['outputs'][0]['name']
        
        # 检查Conv的输出是否被单一节点消费
        conv_consumers = consumers.get(conv_out, [])
        if len(conv_consumers) != 1:
            continue
            
        bn_node, _ = conv_consumers[0]
        # 检查消费者是否为BatchNormalization
        if bn_node['op_type'] != 'BatchNormalization':
            continue
            
        bn_out = bn_node['outputs'][0]['name']
        # 检查BN的输出是否被单一节点消费
        bn_consumers = consumers.get(bn_out, [])
        if len(bn_consumers) != 1:
            continue
            
        relu_node, _ = bn_consumers[0]
        # 检查消费者是否为ReLU
        if relu_node['op_type'] != 'Relu':
            continue

        # 创建融合后的节点
        fused = node.copy()
        fused['attributes'] = dict(node.get('attributes', {}))
        
         # 复制BN节点的所有属性，这些属性包含了合并后的BN参数信息
        if 'attributes' in bn_node:
            for key, value in bn_node['attributes'].items():
                fused['attributes'][key] = value

        # 将BN和ReLU的信息合并到Conv中
        fused['attributes']['fused_bn'] = 1
        fused['attributes']['activation'] = 'Relu'
        fused['outputs'] = relu_node['outputs']  # 输出为 Relu 的输出

        if len(bn_node['inputs']) >= 2:
            # 添加BN的参数输入到融合后的节点
            # 第一个是来自Conv的输入，第二个是合并后的BN参数
            bn_params_input = bn_node['inputs'][1]  # 合并后的BN参数
            fused['inputs'] = node['inputs'] + [bn_params_input]

        first_idx = node['_orig_idx']
        replacements[first_idx] = fused
        
        # 标记要删除的节点
        to_remove.add(node['_orig_idx'])
        to_remove.add(bn_node['_orig_idx'])
        to_remove.add(relu_node['_orig_idx'])

    # 构建新的节点列表
    new_nodes = []
    for idx, node in enumerate(nodes):
        if idx in to_remove:
            # 如果这是融合子图的第一个节点，就放 fused node
            if idx in replacements:
                new_nodes.append(replacements[idx])
        else:
            new_nodes.append(node)

    for node in new_nodes:
        node.pop('_orig_idx', None)

    print("Fusing Conv+BN+ReLU patterns...", len(to_remove))
    vk_model.nodes = new_nodes


def fuse_conv_simple_activation(vk_model):
    producer, consumers = build_graph_index(vk_model.nodes)
    ACTIVATIONS = {"Relu", "Sigmoid", "Tanh", "HardSwish", "Mish", "Relu6"}
    
    nodes = vk_model.nodes
    for idx, node in enumerate(nodes):
        node['_orig_idx'] = idx

    to_remove: Set[int] = set()
    replacements: Dict[int, Dict] = {}

    for node in nodes:
        if node['op_type'] in ['Conv', 'BatchNormalization'] and len(node['outputs']) == 1:
            out_name = node['outputs'][0]['name']
            outs = consumers.get(out_name, [])
            if len(outs) == 1:
                next_node, _ = outs[0]
                if (next_node['op_type'] in ACTIVATIONS and
                    len(next_node['inputs']) == 1 and
                    next_node['inputs'][0]['name'] == out_name):

                    fused = node.copy()
                    # fused['attributes'] = node['attributes'].copy()
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
                scale_array = np.ones_like(mean_array, dtype=np.float32)

            if bias_name in vk_model.initializers:
                bias_array = numpy_helper.to_array(vk_model.initializers[bias_name])
            else:
                bias_array = np.zeros_like(mean_array, dtype=np.float32)

            # Create merged tensor with shape [4, N]
            # where N is the number of elements in each parameter
            N = mean_array.size
            padN = ((N+3)//4) * 4
            merged_data = np.zeros((4, padN), dtype=np.float32)

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
                merged_data[0, base_idx:base_idx+4] = scale_flat[base_idx:base_idx+4]  # scale
                merged_data[1, base_idx:base_idx+4] = bias_flat[base_idx:base_idx+4]   # bias
                merged_data[2, base_idx:base_idx+4] = mean_flat[base_idx:base_idx+4]   # mean
                merged_data[3, base_idx:base_idx+4] = var_flat[base_idx:base_idx+4]    # variance            
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
                {'name': merged_name, 'shape': [4, N]}  # Merged parameters
            ]

            node['inputs'] = new_inputs

            # Mark original initializers for removal
            merged_initializers.update([scale_name, bias_name, mean_name, var_name])

            print(f"Merged BN parameters for node {node['name']}: "
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


def parse_onnx_model(onnx_path):
    model = onnx.load(onnx_path)

    # Optimize the ONNX model
    print("Optimizing ONNX model...")
    model = optimize_onnx_model(model)
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
            print("  Elemwise operation with multiple inputs. Broadcasting shapes...")
            for i in range(len(inputs_with_shape)):
                if inputs_with_shape[i]['name'] == node.input[0]:
                    id0 = i
                if inputs_with_shape[i]['name'] == node.input[1]:
                    id1 = i
            if len(inputs_with_shape[id0]['shape']) != len(inputs_with_shape[id1]['shape']):
                shape = broadcast_shapes(inputs_with_shape[id0]['shape'], inputs_with_shape[id1]['shape'])
                if len(inputs_with_shape[id0]['shape']) == 0:
                    inputs_with_shape[id0]['shape'] = shape
                    for initializer in graph.initializer:
                        if initializer.name == inputs_with_shape[id0]['name']:
                            orig_init = vk_model.initializers[initializer.name]
                            scalar_val = numpy_helper.to_array(orig_init).item()
                            expanded_data = np.full(shape, scalar_val, dtype=np.float32)
                            new_init = numpy_helper.from_array(expanded_data, name=new_name)
                            vk_model.initializers[initializer.name] = new_init
                if len(inputs_with_shape[id1]['shape']) == 0:
                    inputs_with_shape[id1]['shape'] = shape
                    for initializer in graph.initializer:
                        if initializer.name == inputs_with_shape[id1]['name']:
                            orig_init = vk_model.initializers[initializer.name]
                            scalar_val = numpy_helper.to_array(orig_init).item()
                            expanded_data = np.full(shape, scalar_val, dtype=np.float32)
                            new_init = numpy_helper.from_array(expanded_data, name=initializer.name)
                            vk_model.initializers[initializer.name] = new_init
    
        

        vk_model.nodes.append({
            'op_type': node.op_type,
            'name': node.name,
            'attributes': attributes,
            'inputs': inputs_with_shape,
            'outputs': outputs_with_shape
        })

    merge_initializers(vk_model)
    remove_redundant_reshape(vk_model)
    fuse_conv_bn_relu(vk_model)
    fuse_gated_conv(vk_model)
    fuse_conv_simple_activation(vk_model)

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
