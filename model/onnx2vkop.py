#!/bin/env python3

import os
import sys
import struct
import onnx
import numpy as np
from onnx import numpy_helper
import onnxoptimizer as optimizer

class VkModel:
    def __init__(self):
        """
        Initializes the class with the following attributes:

        Attributes:
            inputs (list): A list to store the input nodes of the graph.
            outputs (list): A list to store the output nodes of the graph.
            nodes (list): A list to store all the nodes in the graph.
            initializers (dict): A dictionary to store the initial values of the graph's parameters.
            graph_edges (list): A list to store the relationships or edges between nodes in the graph.
        """
        self.inputs = []
        self.outputs = []
        self.nodes = []
        self.initializers = {}
        self.graph_edges = []  # 新增字段，用于保存图关系

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

            # # Save graph edges (optional, for better visualization/debugging)
            # f.write(struct.pack('I', len(self.graph_edges)))
            # for edge in self.graph_edges:
            #     self._write_string(f, edge['from'])
            #     self._write_string(f, edge['to'])

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

    @staticmethod
    def _write_array(f, arr):
        arr = np.ascontiguousarray(arr)
        dtype = arr.dtype.name
        shape = list(arr.shape)
        VkModel._write_string(f, dtype)
        # print("Writing array of dtype:", dtype, "and shape:", shape)
        f.write(struct.pack('I', len(shape)))
        for dim in shape:
            f.write(struct.pack('I', dim))
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


def parse_onnx_model(onnx_path):
    model = onnx.load(onnx_path)

    # Optimize the ONNX model
    print("Optimizing ONNX model...")
    model = optimize_onnx_model(model)

    vk_model = VkModel()
    graph = model.graph

    # Inputs with shapes
    for inp in graph.input:
        tensor_type = inp.type.tensor_type
        shape_dims = [
            dim.dim_value if dim.HasField("dim_value") else 1
            for dim in tensor_type.shape.dim
        ]
        vk_model.inputs.append({'name': inp.name, 'shape': shape_dims})

    # Outputs with shapes
    for out in graph.output:
        tensor_type = out.type.tensor_type
        shape_dims = [
            dim.dim_value if dim.HasField("dim_value") else 1
            for dim in tensor_type.shape.dim
        ]
        vk_model.outputs.append({'name': out.name, 'shape': shape_dims})

    for node in graph.input:
        print("Graph input:", node.name)

    # Nodes with attributes and shapes of inputs/outputs
    for node in graph.node:
        print("Processing node:", node.name, "of type:", node.op_type)
        attributes = {}
        for attr in node.attribute:
            if attr.HasField('i'):
                attributes[attr.name] = attr.i
            elif attr.HasField('f'):
                attributes[attr.name] = attr.f
            elif attr.HasField('s'):
                attributes[attr.name] = attr.s.decode('utf-8')

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
                print(f"Warning: Input tensor {input_name} not found in graph.")
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

        # Save graph edges
        for input_name in node.input:
            vk_model.graph_edges.append({'from': input_name, 'to': node.name})
        for output_name in node.output:
            vk_model.graph_edges.append({'from': node.name, 'to': output_name})

    # Initializers (parameters)
    for initializer in graph.initializer:
        name = initializer.name
        arr = numpy_helper.to_array(initializer)
        vk_model.initializers[name] = arr

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
